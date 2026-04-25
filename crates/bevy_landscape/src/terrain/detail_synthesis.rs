//! Real-time procedural detail synthesis for fine clipmap LOD levels.
//!
//! A compute pass runs each frame writing full height (bilinear source +
//! fBM detail) into the fine layers of the R32Float clipmap height array.
//! The vertex shader reads height directly — no separate detail texture.
//!
//! Architecture
//! ─────────────
//! Main world:
//!   `DetailSynthesisConfig`  – tweakable noise parameters
//!   `DetailSynthesisState`   – per-frame per-LOD params (extracted to render world)
//!
//! Render world:
//!   `DetailSynthesisPipeline`   – cached descriptor + pipeline id
//!   `DetailSynthesisBindGroups` – per-LOD bind groups rebuilt each frame
//!   `DetailSynthesisNode`       – render graph node (runs before StartMainPass)

use std::num::NonZeroU64;

use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
            BufferInitDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, SamplerBindingType,
            ShaderStages, StorageTextureAccess, TextureAspect, TextureFormat, TextureSampleType,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

use crate::terrain::{
    clipmap_texture::TerrainClipmapState, config::TerrainConfig, material::TerrainMaterial,
    source_heightmap::SourceHeightmapState,
};

// ── Main-world types ──────────────────────────────────────────────────────────

/// Tweakable procedural noise parameters, live-reloaded each frame.
#[derive(Resource, Clone, Reflect, PartialEq)]
#[reflect(Resource)]
pub struct DetailSynthesisConfig {
    /// Maximum height residual in world-space metres.
    pub max_amplitude: f32,
    /// fBM lacunarity (frequency multiplier per octave).
    pub lacunarity: f32,
    /// fBM gain (amplitude multiplier per octave, aka persistence).
    pub gain: f32,
    /// Erosion shaping: 0 = plain fBM, 1 = fully gradient-attenuated.
    pub erosion_strength: f32,
    /// Per-terrain noise domain offset (avoids default-seed repetition).
    pub seed: Vec2,
    /// Disable synthesis without removing the plugin.
    pub enabled: bool,
    /// Slope angle (degrees) at which the detail amplitude starts fading.
    /// Suppresses fBM on cliff faces where fractal noise looks unrealistic.
    pub slope_mask_threshold_deg: f32,
    /// Angular width (degrees) of the fade band above the threshold.
    pub slope_mask_falloff_deg: f32,
    /// Per-fragment normal perturbation strength.  Independent of
    /// `max_amplitude` so the surface can carry visible noise lighting even
    /// when the displacement amplitude is zero (or vice versa).  Evaluated
    /// directly in the fragment shader against the same fBM field used by
    /// the displacement compute pass, so perturbed normals stay coherent.
    pub normal_detail_strength: f32,
}

impl Default for DetailSynthesisConfig {
    fn default() -> Self {
        Self {
            max_amplitude: 50.0,
            lacunarity: 2.1,
            gain: 0.45,
            erosion_strength: 0.55,
            seed: Vec2::ZERO,
            enabled: true,
            slope_mask_threshold_deg: 40.0,
            slope_mask_falloff_deg: 20.0,
            normal_detail_strength: 30.0,
        }
    }
}

// ── Extracted state (main → render world each frame) ─────────────────────────

/// Per-LOD compute dispatch parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PerLodSynthParams {
    /// Actual clipmap LOD level (used to index the height texture array layer).
    pub lod_level: u32,
    pub clip_center_x: f32,
    pub clip_center_z: f32,
    pub texel_world_size: f32,
    pub octave_count: u32,
}

/// Extracted every frame; drives the render-world prepare + dispatch systems.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct DetailSynthesisState {
    /// One entry per LOD level that actually needs synthesis.
    pub lod_params: Vec<PerLodSynthParams>,
    /// Handle to the R32Float clipmap height array (compute writes here).
    pub clipmap_height_handle: Option<Handle<Image>>,
    /// Handle to the R16Unorm source heightmap (compute reads for base height).
    pub source_heightmap_handle: Option<Handle<Image>>,
    /// World-space XZ of texel (0, 0) in the source heightmap.
    pub source_origin: Vec2,
    /// World-space XZ size of the source heightmap.
    pub source_extent: Vec2,
    /// [0,1] → world-space metres multiplier.
    pub height_scale: f32,
    /// World-space texel size of the source heightmap (~30 m).
    pub source_spacing: f32,
    /// Snapshot of the current config.
    pub config: DetailSynthesisConfig,
    /// Texture resolution (texels per edge, e.g. 512).
    pub resolution: u32,
}

#[derive(Default)]
pub(crate) struct DetailSynthesisCache {
    initialized: bool,
    lod_params: Vec<PerLodSynthParams>,
    source_origin: Vec2,
    source_extent: Vec2,
    height_scale: f32,
    source_spacing: f32,
    config: DetailSynthesisConfig,
    resolution: u32,
}

// ── GPU uniform struct (must match SynthesisParams in detail_synthesis.wgsl) ──

/// WGSL std140-compatible layout: 20 scalars × 4 bytes = 80 bytes, 16-aligned.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SynthesisParamsGpu {
    clip_center_x: f32,        //  0
    clip_center_z: f32,        //  4
    texel_world_size: f32,     //  8
    clipmap_res: f32,          // 12
    octave_count: u32,         // 16
    source_lod_spacing: f32,   // 20
    max_amplitude: f32,        // 24
    lacunarity: f32,           // 28
    gain: f32,                 // 32
    erosion_strength: f32,     // 36
    seed_x: f32,               // 40
    seed_z: f32,               // 44
    source_origin_x: f32,      // 48
    source_origin_z: f32,      // 52
    source_extent_x: f32,      // 56
    source_extent_z: f32,      // 60
    height_scale: f32,         // 64
    slope_mask_threshold: f32, // 68 (degrees)
    slope_mask_falloff: f32,   // 72 (degrees)
    _pad: f32,                 // 76
}

const PARAMS_SIZE: u64 = std::mem::size_of::<SynthesisParamsGpu>() as u64;

// ── Render-world: pipeline resource ──────────────────────────────────────────

#[derive(Resource)]
struct DetailSynthesisPipeline {
    layout_desc: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
}

fn make_layout_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: "detail_synthesis_layout".into(),
        entries: vec![
            // binding 0: uniform SynthesisParams
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(PARAMS_SIZE).unwrap()),
                },
                count: None,
            },
            // binding 1: per-layer D2 storage write to clipmap height array
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // binding 2: source heightmap (R16Unorm, sampled)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // binding 3: source heightmap sampler
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    }
}

impl FromWorld for DetailSynthesisPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout_desc = make_layout_desc();
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/detail_synthesis.wgsl");

        let pipeline_id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("detail_synthesis_pipeline".into()),
                    layout: vec![layout_desc.clone()],
                    push_constant_ranges: vec![],
                    shader,
                    shader_defs: vec![],
                    entry_point: Some("synthesize".into()),
                    zero_initialize_workgroup_memory: false,
                });

        DetailSynthesisPipeline {
            layout_desc,
            pipeline_id,
        }
    }
}

// ── Render-world: per-frame bind groups ──────────────────────────────────────

#[derive(Resource, Default)]
struct DetailSynthesisBindGroups {
    groups: Vec<BindGroup>,
    _bufs: Vec<Buffer>,
}

// ── Render-world: prepare system ─────────────────────────────────────────────

fn prepare_detail_synthesis_bind_groups(
    mut commands: Commands,
    pipeline: Res<DetailSynthesisPipeline>,
    pipeline_cache: Res<PipelineCache>,
    state: Option<Res<DetailSynthesisState>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
) {
    let Some(state) = state else { return };
    if state.lod_params.is_empty() {
        commands.insert_resource(DetailSynthesisBindGroups::default());
        return;
    }

    let Some(clipmap_handle) = &state.clipmap_height_handle else {
        return;
    };
    let Some(clipmap_gpu) = gpu_images.get(clipmap_handle) else {
        return;
    };

    let Some(source_handle) = &state.source_heightmap_handle else {
        return;
    };
    let Some(source_gpu) = gpu_images.get(source_handle) else {
        return;
    };

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout_desc);
    let mut groups: Vec<BindGroup> = Vec::with_capacity(state.lod_params.len());
    let mut bufs: Vec<Buffer> = Vec::with_capacity(state.lod_params.len());

    for lp in &state.lod_params {
        let gpu_params = SynthesisParamsGpu {
            clip_center_x: lp.clip_center_x,
            clip_center_z: lp.clip_center_z,
            texel_world_size: lp.texel_world_size,
            clipmap_res: state.resolution as f32,
            octave_count: lp.octave_count,
            source_lod_spacing: state.source_spacing,
            max_amplitude: state.config.max_amplitude,
            lacunarity: state.config.lacunarity,
            gain: state.config.gain,
            erosion_strength: state.config.erosion_strength,
            seed_x: state.config.seed.x,
            seed_z: state.config.seed.y,
            source_origin_x: state.source_origin.x,
            source_origin_z: state.source_origin.y,
            source_extent_x: state.source_extent.x,
            source_extent_z: state.source_extent.y,
            height_scale: state.height_scale,
            slope_mask_threshold: state.config.slope_mask_threshold_deg,
            slope_mask_falloff: state.config.slope_mask_falloff_deg,
            _pad: 0.0,
        };

        let uniform_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("detail_synthesis_uniform"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // D2 view targeting the specific clipmap layer for this LOD.
        let layer_view = clipmap_gpu.texture.create_view(&TextureViewDescriptor {
            label: Some("detail_synthesis_layer_view"),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: lp.lod_level,
            array_layer_count: Some(1),
            usage: None,
        });

        let bind_group = render_device.create_bind_group(
            Some("detail_synthesis_bind_group"),
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &uniform_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&layer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&source_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&source_gpu.sampler),
                },
            ],
        );

        groups.push(bind_group);
        bufs.push(uniform_buf);
    }

    commands.insert_resource(DetailSynthesisBindGroups {
        groups,
        _bufs: bufs,
    });
}

// ── Render graph node ─────────────────────────────────────────────────────────

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DetailSynthesisLabel;

#[derive(Default)]
pub struct DetailSynthesisNode;

impl Node for DetailSynthesisNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline = world.resource::<DetailSynthesisPipeline>();
        let Some(bind_groups) = world.get_resource::<DetailSynthesisBindGroups>() else {
            return Ok(());
        };
        if bind_groups.groups.is_empty() {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id) {
            CachedPipelineState::Ok(_) => {}
            _ => return Ok(()),
        }
        let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let state = world.resource::<DetailSynthesisState>();
        let workgroups = state.resolution.div_ceil(8);

        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("detail_synthesis_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute_pipeline);

        for bind_group in &bind_groups.groups {
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        Ok(())
    }
}

// ── Main-world: update system ─────────────────────────────────────────────────

/// Updates `DetailSynthesisState` from the current clipmap + source heightmap state.
/// Runs in `Update`, after `update_terrain_view_state`.
pub fn update_synthesis_state(
    clipmap_state: Res<TerrainClipmapState>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    source_state: Option<Res<SourceHeightmapState>>,
    view: Res<crate::terrain::resources::TerrainViewState>,
    terrain_config: Res<TerrainConfig>,
    detail_config: Res<DetailSynthesisConfig>,
    mut cache: Local<DetailSynthesisCache>,
    mut commands: Commands,
) {
    let Some(source) = source_state else { return };
    if view.clip_centers.is_empty() {
        return;
    }
    let Some(mat) = materials.get_mut(&clipmap_state.material_handle) else {
        return;
    };

    let source_spacing = source.texel_size;
    // Mirror the synthesis params into the material uniform so the fragment
    // shader can re-evaluate the same fBM field for normal perturbation.
    // octave_count = 0 (or normal_strength = 0) disables the perturbation.
    let base_freq = 2.0 / source_spacing.max(0.001);
    let normal_octaves = if detail_config.enabled && detail_config.normal_detail_strength > 0.0 {
        6.0
    } else {
        0.0
    };
    mat.params.synthesis_norm = Vec4::new(
        detail_config.seed.x,
        detail_config.seed.y,
        base_freq,
        normal_octaves,
    );
    mat.params.synthesis_norm2 = Vec4::new(
        detail_config.lacunarity,
        detail_config.gain,
        detail_config.erosion_strength,
        detail_config.normal_detail_strength,
    );
    mat.params.source_meta = Vec4::new(
        source.world_origin.x,
        source.world_origin.y,
        source.world_extent.x,
        source.world_extent.y,
    );
    let active_levels = terrain_config.active_clipmap_levels() as usize;
    let clip_levels = crate::terrain::clipmap_texture::compute_clip_levels(
        &terrain_config,
        &view.clip_centers,
        &view.level_scales,
    );
    let mut all_lod_params: Vec<PerLodSynthParams> = Vec::with_capacity(active_levels);

    for (lod, clip) in clip_levels.iter().copied().enumerate().take(active_levels) {
        let texel_ws = clip.w;

        // Coarse LODs (texel spacing ≥ source) get 0 octaves: pure source
        // sampling, no fBM detail.  Fine LODs add octaves of fBM up to 6.
        let octave_count = if !detail_config.enabled || texel_ws >= source_spacing {
            0
        } else {
            let base_wl = source_spacing * 0.5;
            let fine_wl = (texel_ws * 2.0).max(0.01);
            let raw = (base_wl / fine_wl).log2().floor() as i32;
            raw.clamp(1, 6) as u32
        };

        all_lod_params.push(PerLodSynthParams {
            lod_level: lod as u32,
            clip_center_x: clip.x,
            clip_center_z: clip.y,
            texel_world_size: texel_ws,
            octave_count,
        });
    }

    // Only log when the synthesis configuration changes, not every frame.
    if !cache.initialized
        || cache.lod_params.len() != all_lod_params.len()
        || cache.source_spacing != source_spacing
    {
        if all_lod_params.is_empty() {
            warn!(
                "[DetailSynthesis] No LODs qualify for synthesis \
                 (source_spacing={source_spacing:.1}m, active_levels={active_levels}). \
                 Check max_mip_level in landscape.toml."
            );
        } else {
            info!(
                "[DetailSynthesis] Synthesizing {} LOD(s), source_spacing={source_spacing:.1}m",
                all_lod_params.len()
            );
        }
    }

    let global_dirty = !cache.initialized
        || cache.source_origin != source.world_origin
        || cache.source_extent != source.world_extent
        || cache.height_scale != terrain_config.height_scale
        || cache.source_spacing != source_spacing
        || cache.config != *detail_config
        || cache.resolution != terrain_config.clipmap_resolution()
        || cache.lod_params.len() != all_lod_params.len();

    let lod_params = if global_dirty {
        all_lod_params.clone()
    } else {
        all_lod_params
            .iter()
            .zip(cache.lod_params.iter())
            .filter_map(|(new, old)| (new != old).then_some(new.clone()))
            .collect()
    };

    cache.initialized = true;
    cache.lod_params = all_lod_params;
    cache.source_origin = source.world_origin;
    cache.source_extent = source.world_extent;
    cache.height_scale = terrain_config.height_scale;
    cache.source_spacing = source_spacing;
    cache.config = detail_config.clone();
    cache.resolution = terrain_config.clipmap_resolution();

    commands.insert_resource(DetailSynthesisState {
        lod_params,
        clipmap_height_handle: Some(clipmap_state.height_texture_handle.clone()),
        source_heightmap_handle: Some(source.handle.clone()),
        source_origin: source.world_origin,
        source_extent: source.world_extent,
        height_scale: terrain_config.height_scale,
        source_spacing,
        config: detail_config.clone(),
        resolution: terrain_config.clipmap_resolution(),
    });
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct DetailSynthesisPlugin;

impl Plugin for DetailSynthesisPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<DetailSynthesisConfig>()
            .init_resource::<DetailSynthesisConfig>()
            .init_resource::<DetailSynthesisState>()
            .add_plugins(ExtractResourcePlugin::<DetailSynthesisState>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DetailSynthesisBindGroups>()
            .add_systems(
                Render,
                prepare_detail_synthesis_bind_groups.in_set(RenderSystems::PrepareResources),
            )
            .add_render_graph_node::<DetailSynthesisNode>(Core3d, DetailSynthesisLabel)
            .add_render_graph_edge(Core3d, DetailSynthesisLabel, Node3d::StartMainPass);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<DetailSynthesisPipeline>();
    }
}

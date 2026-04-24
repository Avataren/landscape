//! Real-time procedural detail synthesis for fine clipmap LOD levels.
//!
//! A compute pass runs each frame writing a height residual (world-space metres)
//! into a per-layer R32Float texture array.  The terrain vertex shader adds the
//! residual on top of the coarse source heightmap, producing sub-metre detail
//! without pre-baked tiles.
//!
//! Architecture
//! ─────────────
//! Main world:
//!   `DetailSynthesisConfig`  – tweakable noise parameters
//!   `DetailTexture`          – R32Float handle shared with TerrainMaterial
//!   `DetailSynthesisState`   – per-frame per-LOD params (extracted to render world)
//!
//! Render world:
//!   `DetailSynthesisPipeline`   – cached descriptor + pipeline id
//!   `DetailSynthesisBindGroups` – per-LOD bind groups rebuilt each frame
//!   `DetailSynthesisNode`       – render graph node (runs before StartMainPass)

use std::num::NonZeroU64;

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{
            Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
        },
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
            BufferInitDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineCache,
            ShaderStages, StorageTextureAccess, TextureAspect, TextureDimension, TextureFormat,
            TextureUsages, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

use crate::terrain::{
    clipmap_texture::TerrainClipmapState,
    config::TerrainConfig,
    material::TerrainMaterial,
    source_heightmap::SourceHeightmapState,
};

// ── Main-world types ──────────────────────────────────────────────────────────

/// Tweakable procedural noise parameters, live-reloaded each frame.
#[derive(Resource, Clone, Reflect)]
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
}

impl Default for DetailSynthesisConfig {
    fn default() -> Self {
        Self {
            max_amplitude: 6.0,
            lacunarity: 2.1,
            gain: 0.45,
            erosion_strength: 0.55,
            seed: Vec2::ZERO,
            enabled: true,
        }
    }
}

/// Handle to the R32Float detail residual texture array (one layer per LOD).
/// Inserted by `setup_terrain`; passed to `TerrainMaterial` for vertex reads.
#[derive(Resource, Clone)]
pub struct DetailTexture {
    pub handle: Handle<Image>,
    pub resolution: u32,
    pub levels: u32,
}

// ── Extracted state (main → render world each frame) ─────────────────────────

/// Per-LOD compute dispatch parameters.
#[derive(Clone, Debug)]
pub struct PerLodSynthParams {
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
    /// Handle to the R32Float detail array (for per-layer view creation).
    pub detail_texture: Option<Handle<Image>>,
    /// World-space texel size of the source heightmap (~30 m).
    pub source_spacing: f32,
    /// Snapshot of the current config.
    pub config: DetailSynthesisConfig,
    /// Texture resolution (number of texels per edge, e.g. 512).
    pub resolution: u32,
}

// ── GPU uniform struct (must match SynthesisParams in detail_synthesis.wgsl) ──

/// WGSL std140-compatible layout: 12 scalars × 4 bytes = 48 bytes, 16-aligned.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SynthesisParamsGpu {
    clip_center_x: f32,      //  0
    clip_center_z: f32,      //  4
    texel_world_size: f32,   //  8
    clipmap_res: f32,        // 12
    octave_count: u32,       // 16
    source_lod_spacing: f32, // 20
    max_amplitude: f32,      // 24
    lacunarity: f32,         // 28
    gain: f32,               // 32
    erosion_strength: f32,   // 36
    seed_x: f32,             // 40
    seed_z: f32,             // 44
}

const PARAMS_SIZE: u64 = std::mem::size_of::<SynthesisParamsGpu>() as u64;

// ── Render-world: pipeline resource ──────────────────────────────────────────

#[derive(Resource)]
struct DetailSynthesisPipeline {
    /// Stored so `prepare` can retrieve the BindGroupLayout from PipelineCache.
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
            // binding 1: per-layer D2 storage write
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
        ],
    }
}

impl FromWorld for DetailSynthesisPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout_desc = make_layout_desc();
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/detail_synthesis.wgsl");

        let pipeline_id = world
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

        DetailSynthesisPipeline { layout_desc, pipeline_id }
    }
}

// ── Render-world: per-frame bind groups ──────────────────────────────────────

#[derive(Resource, Default)]
struct DetailSynthesisBindGroups {
    groups: Vec<BindGroup>,
    // Uniform buffers kept alive until the next prepare run.
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
    if !state.config.enabled || state.lod_params.is_empty() {
        commands.insert_resource(DetailSynthesisBindGroups::default());
        return;
    }
    let Some(handle) = &state.detail_texture else { return };
    let Some(gpu_image) = gpu_images.get(handle) else { return };

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout_desc);
    let mut groups: Vec<BindGroup> = Vec::with_capacity(state.lod_params.len());
    let mut bufs: Vec<Buffer> = Vec::with_capacity(state.lod_params.len());

    for (lod_idx, lp) in state.lod_params.iter().enumerate() {
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
        };

        let uniform_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("detail_synthesis_uniform"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create a D2 view targeting exactly this LOD layer of the D2Array.
        let layer_view = gpu_image.texture.create_view(&TextureViewDescriptor {
            label: Some("detail_synthesis_layer_view"),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: lod_idx as u32,
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
            ],
        );

        groups.push(bind_group);
        bufs.push(uniform_buf);
    }

    commands.insert_resource(DetailSynthesisBindGroups { groups, _bufs: bufs });
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
        let Some(compute_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.pipeline_id)
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

// ── Main-world helper functions ───────────────────────────────────────────────

/// Creates the R32Float detail texture array. Called once from `setup_terrain`.
pub fn create_detail_texture(config: &TerrainConfig, images: &mut Assets<Image>) -> DetailTexture {
    let res = config.clipmap_resolution();
    let levels = config.active_clipmap_levels();

    // 4 bytes per texel (R32Float); zero-initialised → no residual until compute runs.
    let mut image = Image::new(
        Extent3d { width: res, height: res, depth_or_array_layers: levels },
        TextureDimension::D2,
        vec![0u8; (res * res * levels * 4) as usize],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });

    let handle = images.add(image);
    DetailTexture { handle, resolution: res, levels }
}

/// Updates `DetailSynthesisState` from the current clipmap material uniforms.
/// Runs in `Update`, after `update_terrain_view_state`.
pub fn update_synthesis_state(
    clipmap_state: Res<TerrainClipmapState>,
    materials: Res<Assets<TerrainMaterial>>,
    source_state: Option<Res<SourceHeightmapState>>,
    detail_texture: Option<Res<DetailTexture>>,
    config: Res<DetailSynthesisConfig>,
    mut commands: Commands,
) {
    let (Some(source), Some(detail_tex)) = (source_state, detail_texture) else {
        return;
    };
    let Some(mat) = materials.get(&clipmap_state.material_handle) else {
        return;
    };

    let source_spacing = source.texel_size;
    let active_levels = mat.params.num_lod_levels as usize;
    let mut lod_params: Vec<PerLodSynthParams> = Vec::new();

    for lod in 0..active_levels {
        let clip = mat.params.clip_levels[lod];
        let texel_ws = clip.w;

        // Skip LODs whose texel spacing is already at or coarser than the source —
        // the source heightmap already contains those frequencies.
        if texel_ws >= source_spacing {
            continue;
        }

        // Octave count: span from source half-Nyquist down to 2 × texel_ws.
        //   base_wavelength   = source_spacing / 2
        //   finest_wavelength = 2 * texel_ws
        //   octaves ≈ log2(base / finest)
        let base_wl = source_spacing * 0.5;
        let fine_wl = (texel_ws * 2.0).max(0.01);
        let octave_count = (base_wl / fine_wl).log2().floor() as u32;
        let octave_count = octave_count.clamp(1, 6);

        lod_params.push(PerLodSynthParams {
            clip_center_x: clip.x,
            clip_center_z: clip.y,
            texel_world_size: texel_ws,
            octave_count,
        });
    }

    commands.insert_resource(DetailSynthesisState {
        lod_params,
        detail_texture: Some(detail_tex.handle.clone()),
        source_spacing,
        config: config.clone(),
        resolution: detail_tex.resolution,
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
            .init_resource::<DetailSynthesisPipeline>()
            .init_resource::<DetailSynthesisBindGroups>()
            .add_systems(
                Render,
                prepare_detail_synthesis_bind_groups.in_set(RenderSystems::PrepareResources),
            )
            .add_render_graph_node::<DetailSynthesisNode>(Core3d, DetailSynthesisLabel)
            .add_render_graph_edge(Core3d, DetailSynthesisLabel, Node3d::StartMainPass);
    }
}

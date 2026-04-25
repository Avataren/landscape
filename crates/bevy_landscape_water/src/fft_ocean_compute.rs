//! Tessendorf FFT ocean — render-world compute pipeline.
//!
//! Per frame this dispatches:
//!   1. `animate`            (1 dispatch) — fills freq_a / freq_dz_a from the
//!                            CPU-baked H₀ and ω textures plus current time.
//!   2. `ifft_pass` ×2·log₂N  — Stockham OOP radix-2 IFFT, log₂N horizontal
//!                              passes followed by log₂N vertical passes.
//!                              Ping-pongs between freq_{a,b} (and the dz
//!                              ping-pong).  Pre-built bind groups alternate.
//!   3. `compose`            (1 dispatch) — packs (h, dx, dz, jacobian) into
//!                            the water-sampled `displacement` texture.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingResource, BindingType, BufferBindingType, CachedComputePipelineId,
            CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor,
            DynamicUniformBuffer, PipelineCache, ShaderStages, ShaderType,
            StorageTextureAccess, TextureFormat, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};
use crate::fft_ocean::{OceanFftBuffers, OceanFftSettings, NUM_CASCADES};

const WG: u32 = 8;

// ---------------------------------------------------------------------------
// Per-pass uniform parameters (must match WGSL `PassParams`).
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, ShaderType)]
pub struct PassParams {
    pub stage: u32,
    pub direction: u32,
    pub log_n: u32,
    pub n: u32,
    pub inverse: u32,
    pub pingpong: u32,
    pub choppy: f32,
    pub time_seconds: f32,
    pub cascade_world_sizes: Vec4,
}

// ---------------------------------------------------------------------------
// Render-world resources.
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub(crate) struct OceanFftPassUniform {
    pub buffer: DynamicUniformBuffer<PassParams>,
    /// Indices in the dynamic buffer, set during prepare.
    pub indices: Vec<u32>,
}

#[derive(Resource)]
pub(crate) struct OceanFftBindGroup {
    pub group: BindGroup,
    /// Asset-id fingerprint of the seven textures bound.  Bind group is
    /// rebuilt when any of them changes (e.g. spectrum-rebuild creates new
    /// images for wind/amplitude/world_size changes).
    pub fingerprint: [bevy::asset::AssetId<Image>; 7],
}

#[derive(Resource)]
pub(crate) struct OceanFftPipeline {
    layout: BindGroupLayoutDescriptor,
    animate: CachedComputePipelineId,
    ifft_pass: CachedComputePipelineId,
    compose: CachedComputePipelineId,
}

impl FromWorld for OceanFftPipeline {
    fn from_world(world: &mut World) -> Self {
        let storage_rw_rgba32 = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::ReadWrite,
                format: TextureFormat::Rgba32Float,
                view_dimension: TextureViewDimension::D2Array,
            },
            count: None,
        };
        let storage_r_rgba32 = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::ReadOnly,
                format: TextureFormat::Rgba32Float,
                view_dimension: TextureViewDimension::D2Array,
            },
            count: None,
        };
        let storage_w_rgba16 = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D2Array,
            },
            count: None,
        };

        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(PassParams::min_size()),
                },
                count: None,
            },
            storage_r_rgba32(1),  // init_h0
            storage_r_rgba32(2),  // init_omega_kvec
            storage_rw_rgba32(3), // freq_a
            storage_rw_rgba32(4), // freq_b
            storage_rw_rgba32(5), // freq_dz_a
            storage_rw_rgba32(6), // freq_dz_b
            storage_w_rgba16(7),  // displacement
        ];

        let layout = BindGroupLayoutDescriptor::new("ocean_fft_layout", &entries);

        let shader = load_embedded_asset!(world, "shaders/ocean_fft.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let make_pl = |entry: &str| -> CachedComputePipelineId {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                zero_initialize_workgroup_memory: false,
                label: Some(Cow::Owned(format!("ocean_fft_{entry}"))),
                layout: vec![layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(Cow::Owned(entry.to_string())),
            })
        };

        let animate = make_pl("animate");
        let ifft_pass = make_pl("ifft_pass");
        let compose = make_pl("compose");

        Self {
            layout,
            animate,
            ifft_pass,
            compose,
        }
    }
}

// ---------------------------------------------------------------------------
// Render schedule: prepare uniform + bind group every frame.
// ---------------------------------------------------------------------------

fn prepare_pass_uniform(
    mut uniform: ResMut<OceanFftPassUniform>,
    settings: Res<OceanFftSettings>,
    buffers: Option<Res<OceanFftBuffers>>,
    time: Res<Time>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(buf) = buffers else {
        return;
    };
    let n = buf.n;
    let log_n = buf.log_n;
    let choppy = settings.choppy;
    let t = time.elapsed_secs();
    let cw = buf.cascade_world_sizes.as_slice();
    let cascade_world_sizes = Vec4::new(
        cw.first().copied().unwrap_or(0.0),
        cw.get(1).copied().unwrap_or(0.0),
        cw.get(2).copied().unwrap_or(0.0),
        cw.get(3).copied().unwrap_or(0.0),
    );

    // Slot list:
    //   0                        animate                (writes freq_a)
    //   1..=log_n                horizontal IFFT passes (alternating ping/pong)
    //   log_n+1..=2*log_n        vertical   IFFT passes
    //   2*log_n + 1              compose
    let total_slots = (2 * log_n + 2) as usize;
    uniform.indices.clear();
    uniform.buffer.clear();

    // animate.
    let mut idx = uniform.buffer.push(&PassParams {
        stage: 0,
        direction: 0,
        log_n,
        n,
        inverse: 0,
        pingpong: 0,
        choppy,
        time_seconds: t,
        cascade_world_sizes,
        ..Default::default()
    });
    uniform.indices.push(idx);

    // After animate the data is in *_a (pingpong = 0).
    // Each butterfly pass flips the pingpong slot.
    let mut pingpong: u32 = 0;
    for stage in 1..=log_n {
        idx = uniform.buffer.push(&PassParams {
            stage,
            direction: 0, // horizontal
            log_n,
            n,
            inverse: 1,
            pingpong,
            choppy,
            time_seconds: t,
            cascade_world_sizes,
            ..Default::default()
        });
        uniform.indices.push(idx);
        pingpong ^= 1;
    }
    for stage in 1..=log_n {
        idx = uniform.buffer.push(&PassParams {
            stage,
            direction: 1, // vertical
            log_n,
            n,
            inverse: 1,
            pingpong,
            choppy,
            time_seconds: t,
            cascade_world_sizes,
            ..Default::default()
        });
        uniform.indices.push(idx);
        pingpong ^= 1;
    }
    // compose: reads from whichever buffer holds the final spatial fields.
    idx = uniform.buffer.push(&PassParams {
        stage: 0,
        direction: 0,
        log_n,
        n,
        inverse: 0,
        pingpong,
        choppy,
        time_seconds: t,
        cascade_world_sizes,
        ..Default::default()
    });
    uniform.indices.push(idx);

    debug_assert_eq!(uniform.indices.len(), total_slots);

    uniform
        .buffer
        .write_buffer(&render_device, &render_queue);
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<OceanFftPipeline>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<OceanFftBuffers>>,
    images: Res<RenderAssets<GpuImage>>,
    uniform: Res<OceanFftPassUniform>,
    render_device: Res<RenderDevice>,
    existing: Option<Res<OceanFftBindGroup>>,
) {
    let Some(buf) = buffers else {
        return;
    };
    let handles = [
        &buf.init_h0,
        &buf.init_omega_kvec,
        &buf.freq_a,
        &buf.freq_b,
        &buf.freq_dz_a,
        &buf.freq_dz_b,
        &buf.displacement,
    ];
    let images = handles
        .iter()
        .map(|h| images.get(*h))
        .collect::<Option<Vec<&GpuImage>>>();
    let Some(images) = images else {
        return;
    };

    let fingerprint: [bevy::asset::AssetId<Image>; 7] = std::array::from_fn(|i| handles[i].id());
    if let Some(existing) = existing.as_ref() {
        if existing.fingerprint == fingerprint {
            // Same set of textures — bind group is still valid.  The uniform
            // buffer was updated this frame in `prepare_pass_uniform`.
            return;
        }
    }

    let Some(uniform_binding) = uniform.buffer.binding() else {
        return;
    };

    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: uniform_binding,
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::TextureView(&images[0].texture_view),
        },
        BindGroupEntry {
            binding: 2,
            resource: BindingResource::TextureView(&images[1].texture_view),
        },
        BindGroupEntry {
            binding: 3,
            resource: BindingResource::TextureView(&images[2].texture_view),
        },
        BindGroupEntry {
            binding: 4,
            resource: BindingResource::TextureView(&images[3].texture_view),
        },
        BindGroupEntry {
            binding: 5,
            resource: BindingResource::TextureView(&images[4].texture_view),
        },
        BindGroupEntry {
            binding: 6,
            resource: BindingResource::TextureView(&images[5].texture_view),
        },
        BindGroupEntry {
            binding: 7,
            resource: BindingResource::TextureView(&images[6].texture_view),
        },
    ];
    let bg = render_device.create_bind_group(
        Some("ocean_fft_bg"),
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &entries,
    );
    commands.insert_resource(OceanFftBindGroup {
        group: bg,
        fingerprint,
    });
}

// ---------------------------------------------------------------------------
// Render graph node.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct OceanFftLabel;

enum OceanFftNodeState {
    Loading,
    Ready,
}

pub(crate) struct OceanFftNode {
    state: OceanFftNodeState,
}

impl Default for OceanFftNode {
    fn default() -> Self {
        Self {
            state: OceanFftNodeState::Loading,
        }
    }
}

impl Node for OceanFftNode {
    fn update(&mut self, world: &mut World) {
        if let Some(pipeline) = world.get_resource::<OceanFftPipeline>() {
            let cache = world.resource::<PipelineCache>();
            let ok = |id: CachedComputePipelineId| {
                matches!(
                    cache.get_compute_pipeline_state(id),
                    CachedPipelineState::Ok(_)
                )
            };
            if matches!(self.state, OceanFftNodeState::Loading)
                && ok(pipeline.animate)
                && ok(pipeline.ifft_pass)
                && ok(pipeline.compose)
            {
                self.state = OceanFftNodeState::Ready;
            }
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if matches!(self.state, OceanFftNodeState::Loading) {
            return Ok(());
        }

        let Some(settings) = world.get_resource::<OceanFftSettings>() else {
            return Ok(());
        };
        if !settings.enabled {
            return Ok(());
        }

        let Some(buffers) = world.get_resource::<OceanFftBuffers>() else {
            return Ok(());
        };
        let Some(bg) = world.get_resource::<OceanFftBindGroup>() else {
            return Ok(());
        };
        let Some(uniform) = world.get_resource::<OceanFftPassUniform>() else {
            return Ok(());
        };
        if uniform.indices.is_empty() {
            return Ok(());
        }

        let cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<OceanFftPipeline>();
        let Some(p_animate) = cache.get_compute_pipeline(pipeline.animate) else {
            return Ok(());
        };
        let Some(p_ifft) = cache.get_compute_pipeline(pipeline.ifft_pass) else {
            return Ok(());
        };
        let Some(p_compose) = cache.get_compute_pipeline(pipeline.compose) else {
            return Ok(());
        };

        let n = buffers.n;
        let wg_x = n.div_ceil(WG);
        let wg_y = n.div_ceil(WG);
        // The shader uses gid.z as the cascade index — dispatch one z slot
        // per cascade so all cascades run in parallel within each pass.
        let wg_z = NUM_CASCADES as u32;
        let log_n = buffers.log_n;
        let indices = &uniform.indices;

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("ocean_fft"),
                ..default()
            });

        let mut slot = 0usize;
        let dispatch = |pass: &mut bevy::render::render_resource::ComputePass<'_>,
                        pl: &bevy::render::render_resource::ComputePipeline,
                        offset: u32| {
            pass.set_bind_group(0, &bg.group, &[offset]);
            pass.set_pipeline(pl);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        };

        // animate
        dispatch(&mut pass, p_animate, indices[slot]);
        slot += 1;

        // 2·log_n IFFT passes
        for _ in 0..(2 * log_n) {
            dispatch(&mut pass, p_ifft, indices[slot]);
            slot += 1;
        }

        // compose
        dispatch(&mut pass, p_compose, indices[slot]);

        // Track that we successfully ran at least once so the water material
        // can confidently sample without a one-frame stale window.
        OCEAN_FFT_HAS_RUN.store(true, Ordering::Relaxed);

        Ok(())
    }
}

pub(crate) static OCEAN_FFT_HAS_RUN: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Plugin.
// ---------------------------------------------------------------------------

pub(crate) struct OceanFftComputePlugin;

impl Plugin for OceanFftComputePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/ocean_fft.wgsl");

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<OceanFftPassUniform>()
            .add_systems(
                Render,
                prepare_pass_uniform.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(OceanFftLabel, OceanFftNode::default());
        render_graph.add_node_edge(OceanFftLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<OceanFftPipeline>();
    }
}

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            binding_types::uniform_buffer, BindGroup, BindGroupEntries, BindGroupEntry,
            BindGroupLayoutDescriptor, BindGroupLayoutEntries, BindGroupLayoutEntry,
            BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
            StorageTextureAccess, TextureFormat, TextureViewDimension, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

use crate::erosion_images::ErosionBuffers;
use crate::erosion_params::{ErosionControlState, ErosionParams, ErosionUniform, TICKS_PER_FRAME};
use crate::images::NormalizationImage;
use crate::uniforms::GeneratorUniform;

const WG: u32 = 8;

// ---------------------------------------------------------------------------
// Render-world resources
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct ErosionUniformBindGroup(BindGroup);

#[derive(Resource)]
struct ErosionTexturesBindGroup {
    bind_group: BindGroup,
    resolution: u32,
}

#[derive(Resource)]
pub(crate) struct ErosionDeltaHeightBuffer {
    pub buffer: Buffer,
    pub resolution: u32,
}

#[derive(Resource, Default)]
struct ErosionUniformBuffer {
    buffer: UniformBuffer<ErosionUniform>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

struct ErosionPipelines {
    copy_in: CachedComputePipelineId,
    init_hardness: CachedComputePipelineId,
    clear_buffers: CachedComputePipelineId,
    copy_out: CachedComputePipelineId,
    water_add: CachedComputePipelineId,
    flux_update: CachedComputePipelineId,
    water_velocity: CachedComputePipelineId,
    erode_deposit: CachedComputePipelineId,
    sediment_transport: CachedComputePipelineId,
    copy_b_to_sediment: CachedComputePipelineId,
    evaporate: CachedComputePipelineId,
    clear_delta: CachedComputePipelineId,
    thermal_compute: CachedComputePipelineId,
    thermal_apply: CachedComputePipelineId,
    particle_erode: CachedComputePipelineId,
    particle_apply: CachedComputePipelineId,
}

#[derive(Resource)]
struct ErosionPipeline {
    uniform_layout: BindGroupLayoutDescriptor,
    textures_layout: BindGroupLayoutDescriptor,
    pipelines: ErosionPipelines,
}

impl FromWorld for ErosionPipeline {
    fn from_world(world: &mut World) -> Self {
        let uniform_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<ErosionUniform>(false),),
        );
        let uniform_layout =
            BindGroupLayoutDescriptor::new("erosion_uniform_layout", &uniform_entries);

        let r32rw = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::ReadWrite,
                format: TextureFormat::R32Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        };
        let rgba32rw = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::ReadWrite,
                format: TextureFormat::Rgba32Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        };
        let buf_rw = BindGroupLayoutEntry {
            binding: 7,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(4),
            },
            count: None,
        };

        let textures_layout = BindGroupLayoutDescriptor::new(
            "erosion_textures_layout",
            &[
                r32rw(0),    // height_a
                r32rw(1),    // height_b
                r32rw(2),    // water
                r32rw(3),    // sediment
                rgba32rw(4), // flux
                rgba32rw(5), // velocity
                r32rw(6),    // hardness
                buf_rw,      // delta_height (binding 7)
                r32rw(8),    // raw_heights
            ],
        );

        let shader = load_embedded_asset!(world, "shaders/erosion.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let make_pl = |entry: &str| -> CachedComputePipelineId {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                zero_initialize_workgroup_memory: false,
                label: Some(Cow::Owned(format!("erosion_{entry}"))),
                layout: vec![uniform_layout.clone(), textures_layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(Cow::Owned(entry.to_string())),
            })
        };

        let pipelines = ErosionPipelines {
            copy_in: make_pl("erosion_copy_in"),
            init_hardness: make_pl("erosion_init_hardness"),
            clear_buffers: make_pl("erosion_clear_buffers"),
            copy_out: make_pl("erosion_copy_out"),
            water_add: make_pl("hydro_water_add"),
            flux_update: make_pl("hydro_flux_update"),
            water_velocity: make_pl("hydro_water_velocity"),
            erode_deposit: make_pl("hydro_erode_deposit"),
            sediment_transport: make_pl("hydro_sediment_transport"),
            copy_b_to_sediment: make_pl("copy_b_to_sediment"),
            evaporate: make_pl("hydro_evaporate"),
            clear_delta: make_pl("clear_delta_height"),
            thermal_compute: make_pl("thermal_compute"),
            thermal_apply: make_pl("thermal_apply"),
            particle_erode: make_pl("particle_erode"),
            particle_apply: make_pl("particle_apply"),
        };

        Self {
            uniform_layout,
            textures_layout,
            pipelines,
        }
    }
}

// ---------------------------------------------------------------------------
// Prepare systems
// ---------------------------------------------------------------------------

fn prepare_erosion_uniform_bind_group(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
    erosion_params: Option<Res<ErosionParams>>,
    erosion_ctrl: Option<Res<ErosionControlState>>,
    gen_uniform: Option<Res<GeneratorUniform>>,
    mut buf: ResMut<ErosionUniformBuffer>,
) {
    let Some(ep) = erosion_params else {
        return;
    };
    let (resolution, base_seed) = gen_uniform
        .map(|u| (u.resolution.x, u.seed))
        .unwrap_or((1024, 42));
    // Mix ticks_done into the particle seed so each frame's batch uses fresh
    // starting positions rather than replaying the same paths every iteration.
    let ticks = erosion_ctrl.map(|c| c.ticks_done()).unwrap_or(0);
    let seed = base_seed.wrapping_add(ticks.wrapping_mul(2654435761));
    *buf.buffer.get_mut() = ErosionUniform::from_params(&ep, resolution, seed);
    buf.buffer.write_buffer(&render_device, &render_queue);

    let bg = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.uniform_layout),
        &BindGroupEntries::single(buf.buffer.binding().unwrap()),
    );
    commands.insert_resource(ErosionUniformBindGroup(bg));
}

fn prepare_delta_height_buffer(
    mut commands: Commands,
    erosion_buffers: Option<Res<ErosionBuffers>>,
    existing: Option<Res<ErosionDeltaHeightBuffer>>,
    render_device: Res<RenderDevice>,
) {
    let Some(eb) = erosion_buffers else {
        return;
    };
    let resolution = eb.resolution;
    if existing
        .map(|b| b.resolution == resolution)
        .unwrap_or(false)
    {
        return;
    }
    let size = (resolution as u64) * (resolution as u64) * 4;
    let buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("erosion_delta_height"),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(ErosionDeltaHeightBuffer { buffer, resolution });
}

fn prepare_erosion_textures_bind_group(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    erosion_buffers: Option<Res<ErosionBuffers>>,
    norm_image: Option<Res<NormalizationImage>>,
    delta_buf: Option<Res<ErosionDeltaHeightBuffer>>,
    existing: Option<Res<ErosionTexturesBindGroup>>,
) {
    let (Some(eb), Some(ni), Some(db)) = (erosion_buffers, norm_image, delta_buf) else {
        return;
    };
    if existing
        .map(|bg| bg.resolution == eb.resolution)
        .unwrap_or(false)
    {
        return;
    }

    let Some(ha) = gpu_images.get(&eb.height_a) else {
        return;
    };
    let Some(hb) = gpu_images.get(&eb.height_b) else {
        return;
    };
    let Some(wt) = gpu_images.get(&eb.water) else {
        return;
    };
    let Some(sed) = gpu_images.get(&eb.sediment) else {
        return;
    };
    let Some(flux) = gpu_images.get(&eb.flux) else {
        return;
    };
    let Some(vel) = gpu_images.get(&eb.velocity) else {
        return;
    };
    let Some(hard) = gpu_images.get(&eb.hardness) else {
        return;
    };
    let Some(raw) = gpu_images.get(&ni.raw_heights) else {
        return;
    };

    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&ha.texture_view),
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::TextureView(&hb.texture_view),
        },
        BindGroupEntry {
            binding: 2,
            resource: BindingResource::TextureView(&wt.texture_view),
        },
        BindGroupEntry {
            binding: 3,
            resource: BindingResource::TextureView(&sed.texture_view),
        },
        BindGroupEntry {
            binding: 4,
            resource: BindingResource::TextureView(&flux.texture_view),
        },
        BindGroupEntry {
            binding: 5,
            resource: BindingResource::TextureView(&vel.texture_view),
        },
        BindGroupEntry {
            binding: 6,
            resource: BindingResource::TextureView(&hard.texture_view),
        },
        BindGroupEntry {
            binding: 7,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &db.buffer,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 8,
            resource: BindingResource::TextureView(&raw.texture_view),
        },
    ];

    let bg = render_device.create_bind_group(
        Some("erosion_textures_bg"),
        &pipeline_cache.get_bind_group_layout(&pipeline.textures_layout),
        &entries,
    );
    commands.insert_resource(ErosionTexturesBindGroup {
        bind_group: bg,
        resolution: eb.resolution,
    });
}

// ---------------------------------------------------------------------------
// Render graph node
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct ErosionLabel;

enum ErosionNodeState {
    Loading,
    Ready,
}

pub(crate) struct ErosionNode {
    state: ErosionNodeState,
}

impl Default for ErosionNode {
    fn default() -> Self {
        Self {
            state: ErosionNodeState::Loading,
        }
    }
}

impl Node for ErosionNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ErosionPipeline>();
        let cache = world.resource::<PipelineCache>();
        let ok = |id: CachedComputePipelineId| {
            matches!(
                cache.get_compute_pipeline_state(id),
                CachedPipelineState::Ok(_)
            )
        };
        let pl = &pipeline.pipelines;
        if matches!(self.state, ErosionNodeState::Loading)
            && ok(pl.copy_in)
            && ok(pl.init_hardness)
            && ok(pl.clear_buffers)
            && ok(pl.copy_out)
            && ok(pl.water_add)
            && ok(pl.flux_update)
            && ok(pl.water_velocity)
            && ok(pl.erode_deposit)
            && ok(pl.sediment_transport)
            && ok(pl.copy_b_to_sediment)
            && ok(pl.evaporate)
            && ok(pl.clear_delta)
            && ok(pl.thermal_compute)
            && ok(pl.thermal_apply)
            && ok(pl.particle_erode)
            && ok(pl.particle_apply)
        {
            self.state = ErosionNodeState::Ready;
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if matches!(self.state, ErosionNodeState::Loading) {
            return Ok(());
        }

        let Some(ep) = world.get_resource::<ErosionParams>() else {
            return Ok(());
        };
        if !ep.enabled {
            return Ok(());
        }

        let Some(ctrl) = world.get_resource::<ErosionControlState>() else {
            return Ok(());
        };
        let Some(ubg) = world.get_resource::<ErosionUniformBindGroup>() else {
            return Ok(());
        };
        let Some(tbg) = world.get_resource::<ErosionTexturesBindGroup>() else {
            return Ok(());
        };

        let cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ErosionPipeline>();
        let pl = &pipeline.pipelines;

        macro_rules! get_pl {
            ($id:expr) => {
                match cache.get_compute_pipeline($id) {
                    Some(p) => p,
                    None => return Ok(()),
                }
            };
        }

        let p_copy_in = get_pl!(pl.copy_in);
        let p_init_hardness = get_pl!(pl.init_hardness);
        let p_clear_buffers = get_pl!(pl.clear_buffers);
        let p_copy_out = get_pl!(pl.copy_out);
        let p_water_add = get_pl!(pl.water_add);
        let p_flux_update = get_pl!(pl.flux_update);
        let p_water_velocity = get_pl!(pl.water_velocity);
        let p_erode_deposit = get_pl!(pl.erode_deposit);
        let p_sediment_transport = get_pl!(pl.sediment_transport);
        let p_copy_b_to_sediment = get_pl!(pl.copy_b_to_sediment);
        let p_evaporate = get_pl!(pl.evaporate);
        let p_clear_delta = get_pl!(pl.clear_delta);
        let p_thermal_compute = get_pl!(pl.thermal_compute);
        let p_thermal_apply = get_pl!(pl.thermal_apply);
        let p_particle_erode = get_pl!(pl.particle_erode);
        let p_particle_apply = get_pl!(pl.particle_apply);

        let res = tbg.resolution;
        let wg_xy = (res.div_ceil(WG), res.div_ceil(WG));

        let bg0 = &ubg.0;
        let bg1 = &tbg.bind_group;

        macro_rules! dispatch {
            ($label:expr, $pl:expr, $x:expr, $y:expr, $z:expr) => {{
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some($label),
                            ..default()
                        });
                pass.set_bind_group(0, bg0, &[]);
                pass.set_bind_group(1, bg1, &[]);
                pass.set_pipeline($pl);
                pass.dispatch_workgroups($x, $y, $z);
            }};
        }

        // Erosion is complete. GeneratorRawNode no longer overwrites raw_heights
        // every frame (it only dispatches when params change), so copy_out from
        // the last tick batch is still valid — nothing to do here.
        if !ctrl.is_dirty() {
            return Ok(());
        }

        let done = ctrl.ticks_done();

        // First tick of a new run: initialize buffers.
        if done == 0 {
            dispatch!("erosion_copy_in", p_copy_in, wg_xy.0, wg_xy.1, 1);
            dispatch!(
                "erosion_init_hardness",
                p_init_hardness,
                wg_xy.0,
                wg_xy.1,
                1
            );
            dispatch!(
                "erosion_clear_buffers",
                p_clear_buffers,
                wg_xy.0,
                wg_xy.1,
                1
            );
        }

        let ticks = TICKS_PER_FRAME.min(ep.iterations - done);
        for _tick in 0..ticks {
            dispatch!("hydro_water_add", p_water_add, wg_xy.0, wg_xy.1, 1);
            dispatch!("hydro_flux_update", p_flux_update, wg_xy.0, wg_xy.1, 1);
            dispatch!(
                "hydro_water_velocity",
                p_water_velocity,
                wg_xy.0,
                wg_xy.1,
                1
            );
            dispatch!("hydro_erode_deposit", p_erode_deposit, wg_xy.0, wg_xy.1, 1);
            dispatch!(
                "hydro_sediment_transport",
                p_sediment_transport,
                wg_xy.0,
                wg_xy.1,
                1
            );
            dispatch!(
                "copy_b_to_sediment",
                p_copy_b_to_sediment,
                wg_xy.0,
                wg_xy.1,
                1
            );
            dispatch!("hydro_evaporate", p_evaporate, wg_xy.0, wg_xy.1, 1);

            if ep.thermal_enabled {
                for _t in 0..ep.thermal_iterations {
                    dispatch!("clear_delta", p_clear_delta, wg_xy.0, wg_xy.1, 1);
                    dispatch!("thermal_compute", p_thermal_compute, wg_xy.0, wg_xy.1, 1);
                    dispatch!("thermal_apply", p_thermal_apply, wg_xy.0, wg_xy.1, 1);
                }
            }

            if ep.particle_enabled {
                let wg_p = ep.num_particles.div_ceil(64);
                dispatch!("clear_delta", p_clear_delta, wg_xy.0, wg_xy.1, 1);
                dispatch!("particle_erode", p_particle_erode, wg_p, 1, 1);
                dispatch!("particle_apply", p_particle_apply, wg_xy.0, wg_xy.1, 1);
            }
        }

        // Always write preview output after each batch.
        dispatch!("erosion_copy_out", p_copy_out, wg_xy.0, wg_xy.1, 1);

        let new_done = done + ticks;
        ctrl.ticks_done.store(new_done, Ordering::Release);
        if new_done >= ep.iterations {
            ctrl.dirty.store(false, Ordering::Release);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub(crate) struct ErosionComputePlugin;

impl Plugin for ErosionComputePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/erosion.wgsl");

        app.add_plugins(ExtractResourcePlugin::<ErosionParams>::default())
            .add_plugins(ExtractResourcePlugin::<ErosionBuffers>::default())
            .add_plugins(ExtractResourcePlugin::<ErosionControlState>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<ErosionUniformBuffer>()
            .add_systems(
                Render,
                (
                    prepare_delta_height_buffer,
                    prepare_erosion_uniform_bind_group,
                )
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                prepare_erosion_textures_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ErosionLabel, ErosionNode::default());
        render_graph.add_node_edge(crate::compute::GeneratorRawLabel, ErosionLabel);
        render_graph.add_node_edge(ErosionLabel, crate::compute::GeneratorNormLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ErosionPipeline>();
    }
}

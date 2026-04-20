use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
};

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingType, BufferBindingType, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineCache,
            ShaderStages, ShaderType, StorageTextureAccess, TextureDimension, TextureFormat,
            TextureSampleType, TextureUsages, TextureViewDimension, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

use crate::params::GeneratorParams;

const TILE_SIZE: u32 = 256;
const WORKGROUP_SIZE: u32 = 8;
pub const MAX_EXPORT_RESOLUTION: u32 = 16_384;

#[derive(Message, Clone)]
pub struct StartGeneratorExport {
    pub params: GeneratorParams,
    pub output_dir: PathBuf,
}

#[derive(Resource, Default)]
pub struct GeneratorExportState {
    pub active: bool,
    pub output_dir: Option<PathBuf>,
    pub log: Vec<String>,
    pub succeeded: bool,
    pub completed_generation: u64,
}

#[derive(Resource, Default)]
struct GeneratorExportRuntime {
    next_generation: u64,
    job: Option<GeneratorExportJob>,
}

struct GeneratorExportJob {
    generation: u64,
    params: GeneratorParams,
    output_dir: PathBuf,
    levels: u32,
    height_images: Vec<Handle<Image>>,
    normal_images: Vec<Handle<Image>>,
    gpu_dispatched: Arc<AtomicBool>,
    readback_started: bool,
    heights: Vec<Option<Vec<u8>>>,
    normals: Vec<Option<Vec<u8>>>,
    writer: Option<ExportWriterHandle>,
}

struct ExportWriterHandle {
    log_rx: Mutex<Receiver<String>>,
    done: Arc<AtomicBool>,
    succeeded: Arc<AtomicBool>,
}

#[derive(Component, Clone, Copy)]
struct PendingGeneratorReadback {
    generation: u64,
    lod: u32,
    width: u32,
    height: u32,
    kind: ReadbackKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadbackKind {
    Height,
    Normal,
}

#[derive(Resource, Clone, ExtractResource)]
struct ActiveGeneratorExport {
    generation: u64,
    params: GeneratorParams,
    levels: u32,
    height_images: Vec<Handle<Image>>,
    normal_images: Vec<Handle<Image>>,
    gpu_dispatched: Arc<AtomicBool>,
}

#[derive(Clone, ShaderType)]
struct ExportGeneratorUniform {
    resolution: UVec2,
    octaves: u32,
    seed: u32,
    offset: Vec2,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    height_scale: f32,
    continent_frequency: f32,
    continent_strength: f32,
    ridge_strength: f32,
    warp_frequency: f32,
    warp_strength: f32,
    erosion_strength: f32,
}

impl ExportGeneratorUniform {
    fn from_params(params: &GeneratorParams, resolution: u32) -> Self {
        Self {
            resolution: UVec2::splat(resolution),
            octaves: params.octaves,
            seed: params.seed,
            offset: params.offset,
            frequency: params.frequency,
            lacunarity: params.lacunarity,
            gain: params.gain,
            height_scale: params.height_scale,
            continent_frequency: params.continent_frequency,
            continent_strength: params.continent_strength,
            ridge_strength: params.ridge_strength,
            warp_frequency: params.warp_frequency,
            warp_strength: params.warp_strength,
            erosion_strength: params.erosion_strength,
        }
    }
}

#[derive(Clone, ShaderType)]
struct DownsampleUniform {
    src_resolution: UVec2,
    dst_resolution: UVec2,
}

#[derive(Clone, ShaderType)]
struct NormalUniform {
    resolution: UVec2,
    effective_height_scale: f32,
    lod_scale: f32,
}

struct GeneratePassResources {
    _uniform: UniformBuffer<ExportGeneratorUniform>,
    uniform_bind_group: BindGroup,
    image_bind_group: BindGroup,
}

struct DownsamplePassResources {
    _uniform: UniformBuffer<DownsampleUniform>,
    uniform_bind_group: BindGroup,
    image_bind_group: BindGroup,
    dst_resolution: u32,
}

struct NormalPassResources {
    _uniform: UniformBuffer<NormalUniform>,
    uniform_bind_group: BindGroup,
    image_bind_group: BindGroup,
    resolution: u32,
}

#[derive(Resource)]
struct GeneratorExportRenderResources {
    generation: u64,
    generate_pass: GeneratePassResources,
    downsample_passes: Vec<DownsamplePassResources>,
    normal_passes: Vec<NormalPassResources>,
}

#[derive(Resource)]
struct GeneratorExportPipeline {
    generate_uniform_layout: BindGroupLayoutDescriptor,
    generate_image_layout: BindGroupLayoutDescriptor,
    downsample_uniform_layout: BindGroupLayoutDescriptor,
    downsample_image_layout: BindGroupLayoutDescriptor,
    normal_uniform_layout: BindGroupLayoutDescriptor,
    normal_image_layout: BindGroupLayoutDescriptor,
    generate_pipeline: CachedComputePipelineId,
    downsample_pipeline: CachedComputePipelineId,
    normal_pipeline: CachedComputePipelineId,
}

impl FromWorld for GeneratorExportPipeline {
    fn from_world(world: &mut World) -> Self {
        let generate_uniform_layout = BindGroupLayoutDescriptor::new(
            "generator_export_generate_uniform_layout",
            &[BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(ExportGeneratorUniform::min_size()),
                },
                count: None,
            }],
        );
        let generate_image_layout = BindGroupLayoutDescriptor::new(
            "generator_export_generate_image_layout",
            &[BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        );
        let downsample_uniform_layout = BindGroupLayoutDescriptor::new(
            "generator_export_downsample_uniform_layout",
            &[BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(DownsampleUniform::min_size()),
                },
                count: None,
            }],
        );
        let downsample_image_layout = BindGroupLayoutDescriptor::new(
            "generator_export_downsample_image_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        );
        let normal_uniform_layout = BindGroupLayoutDescriptor::new(
            "generator_export_normal_uniform_layout",
            &[BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NormalUniform::min_size()),
                },
                count: None,
            }],
        );
        let normal_image_layout = BindGroupLayoutDescriptor::new(
            "generator_export_normal_image_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        // rg8snorm is NOT in the WGSL storage-texture format list;
                        // rgba8snorm is.  We store XZ in RG and discard BA on readback.
                        format: TextureFormat::Rgba8Snorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        );

        let shader = load_embedded_asset!(world, "shaders/generator.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let generate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("landscape_generator_export_generate".into()),
            layout: vec![
                generate_uniform_layout.clone(),
                generate_image_layout.clone(),
            ],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("generate_height".into()),
            ..default()
        });
        let downsample_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("landscape_generator_export_downsample".into()),
                layout: vec![
                    downsample_uniform_layout.clone(),
                    downsample_image_layout.clone(),
                ],
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some("downsample_height".into()),
                ..default()
            });
        let normal_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("landscape_generator_export_normals".into()),
            layout: vec![normal_uniform_layout.clone(), normal_image_layout.clone()],
            shader,
            shader_defs: vec![],
            entry_point: Some("derive_normals".into()),
            ..default()
        });

        Self {
            generate_uniform_layout,
            generate_image_layout,
            downsample_uniform_layout,
            downsample_image_layout,
            normal_uniform_layout,
            normal_image_layout,
            generate_pipeline,
            downsample_pipeline,
            normal_pipeline,
        }
    }
}

pub(crate) struct GeneratorExportPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GeneratorExportLabel;

enum GeneratorExportNodeState {
    Loading,
    Ready,
}

struct GeneratorExportNode {
    state: GeneratorExportNodeState,
}

impl Default for GeneratorExportNode {
    fn default() -> Self {
        Self {
            state: GeneratorExportNodeState::Loading,
        }
    }
}

impl Node for GeneratorExportNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GeneratorExportPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        if matches!(self.state, GeneratorExportNodeState::Loading)
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.generate_pipeline),
                CachedPipelineState::Ok(_)
            )
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.downsample_pipeline),
                CachedPipelineState::Ok(_)
            )
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.normal_pipeline),
                CachedPipelineState::Ok(_)
            )
        {
            self.state = GeneratorExportNodeState::Ready;
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if matches!(self.state, GeneratorExportNodeState::Loading) {
            return Ok(());
        }

        let Some(active_export) = world.get_resource::<ActiveGeneratorExport>() else {
            return Ok(());
        };
        if active_export.gpu_dispatched.load(Ordering::Acquire) {
            return Ok(());
        }
        let Some(resources) = world.get_resource::<GeneratorExportRenderResources>() else {
            return Ok(());
        };
        if resources.generation != active_export.generation {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GeneratorExportPipeline>();
        let Some(generate_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.generate_pipeline)
        else {
            return Ok(());
        };
        let Some(downsample_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.downsample_pipeline)
        else {
            return Ok(());
        };
        let Some(normal_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.normal_pipeline)
        else {
            return Ok(());
        };

        // Each stage gets its own compute pass so the GPU's implicit pass
        // boundary acts as a memory barrier. Within a single pass, WebGPU/wgpu
        // gives no visibility guarantee between dispatches — the downsample
        // for L1 could read stale L0 data, causing height discontinuities at
        // LOD ring boundaries (the "chunk seam" crack).
        let resolution = active_export.params.export_resolution;
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("generator_export_generate"),
                        ..default()
                    });
            pass.set_pipeline(generate_pipeline);
            pass.set_bind_group(0, &resources.generate_pass.uniform_bind_group, &[]);
            pass.set_bind_group(1, &resources.generate_pass.image_bind_group, &[]);
            pass.dispatch_workgroups(
                resolution.div_ceil(WORKGROUP_SIZE),
                resolution.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        // One pass per downsample level: each reads the previous level's output
        // so every pass boundary guarantees the previous write is visible.
        for downsample in &resources.downsample_passes {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("generator_export_downsample"),
                        ..default()
                    });
            pass.set_pipeline(downsample_pipeline);
            pass.set_bind_group(0, &downsample.uniform_bind_group, &[]);
            pass.set_bind_group(1, &downsample.image_bind_group, &[]);
            pass.dispatch_workgroups(
                downsample.dst_resolution.div_ceil(WORKGROUP_SIZE),
                downsample.dst_resolution.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        // All normal derivations can share one pass: they each read from a
        // different height level (all fully written above) and write to a
        // different normal texture, so there are no cross-dispatch dependencies.
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("generator_export_normals"),
                        ..default()
                    });
            pass.set_pipeline(normal_pipeline);
            for normal in &resources.normal_passes {
                pass.set_bind_group(0, &normal.uniform_bind_group, &[]);
                pass.set_bind_group(1, &normal.image_bind_group, &[]);
                pass.dispatch_workgroups(
                    normal.resolution.div_ceil(WORKGROUP_SIZE),
                    normal.resolution.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
        }

        active_export.gpu_dispatched.store(true, Ordering::Release);
        Ok(())
    }
}

impl Plugin for GeneratorExportPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/generator.wgsl");

        app.init_resource::<GeneratorExportState>()
            .init_resource::<GeneratorExportRuntime>()
            .add_message::<StartGeneratorExport>()
            .add_observer(handle_generator_readback_complete)
            .add_plugins(ExtractResourcePlugin::<ActiveGeneratorExport>::default())
            .add_systems(
                Update,
                (
                    start_generator_export,
                    begin_generator_readback,
                    drain_generator_writer_logs,
                )
                    .chain(),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_export_bind_groups.in_set(RenderSystems::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GeneratorExportLabel, GeneratorExportNode::default());
        render_graph.add_node_edge(GeneratorExportLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GeneratorExportPipeline>();
    }
}

fn start_generator_export(
    mut commands: Commands,
    mut requests: MessageReader<StartGeneratorExport>,
    mut runtime: ResMut<GeneratorExportRuntime>,
    mut state: ResMut<GeneratorExportState>,
    mut images: ResMut<Assets<Image>>,
) {
    for request in requests.read() {
        if runtime.job.is_some() {
            state
                .log
                .push("Export already running; wait for it to finish.".into());
            continue;
        }

        state.log.clear();
        let mut params = request.params.clone();
        if params.export_resolution > MAX_EXPORT_RESOLUTION {
            state.log.push(format!(
                "Export resolution {} exceeds the current 16k cap; clamping to {}.",
                params.export_resolution, MAX_EXPORT_RESOLUTION
            ));
            params.export_resolution = MAX_EXPORT_RESOLUTION;
        }

        let levels = match baked_level_count(params.export_resolution) {
            Ok(levels) => levels,
            Err(err) => {
                state.log.push(format!("Export failed: {err}"));
                state.active = false;
                state.succeeded = false;
                state.output_dir = Some(request.output_dir.clone());
                state.completed_generation = state.completed_generation.saturating_add(1);
                continue;
            }
        };

        runtime.next_generation += 1;
        let generation = runtime.next_generation;
        let gpu_dispatched = Arc::new(AtomicBool::new(false));

        let mut height_images = Vec::with_capacity(levels as usize);
        let mut normal_images = Vec::with_capacity(levels as usize);
        for lod in 0..levels {
            let resolution = params.export_resolution >> lod;
            height_images.push(build_export_image(
                &mut images,
                resolution,
                TextureFormat::R32Float,
            ));
            normal_images.push(build_export_image(
                &mut images,
                resolution,
                TextureFormat::Rgba8Snorm,
            ));
        }

        state.log.push(format!(
            "Export started → {} ({} levels)",
            request.output_dir.display(),
            levels
        ));
        state.active = true;
        state.succeeded = false;
        state.output_dir = Some(request.output_dir.clone());

        commands.insert_resource(ActiveGeneratorExport {
            generation,
            params: params.clone(),
            levels,
            height_images: height_images.clone(),
            normal_images: normal_images.clone(),
            gpu_dispatched: gpu_dispatched.clone(),
        });

        runtime.job = Some(GeneratorExportJob {
            generation,
            params,
            output_dir: request.output_dir.clone(),
            levels,
            height_images,
            normal_images,
            gpu_dispatched,
            readback_started: false,
            heights: vec![None; levels as usize],
            normals: vec![None; levels as usize],
            writer: None,
        });
    }
}

fn begin_generator_readback(
    mut commands: Commands,
    mut runtime: ResMut<GeneratorExportRuntime>,
    mut state: ResMut<GeneratorExportState>,
) {
    let Some(job) = runtime.job.as_mut() else {
        return;
    };
    if job.readback_started || !job.gpu_dispatched.load(Ordering::Acquire) {
        return;
    }

    job.readback_started = true;
    state
        .log
        .push("GPU bake finished; reading back baked levels…".into());

    for lod in 0..job.levels {
        let resolution = job.params.export_resolution >> lod;
        commands.spawn((
            PendingGeneratorReadback {
                generation: job.generation,
                lod,
                width: resolution,
                height: resolution,
                kind: ReadbackKind::Height,
            },
            Readback::texture(job.height_images[lod as usize].clone()),
        ));
        commands.spawn((
            PendingGeneratorReadback {
                generation: job.generation,
                lod,
                width: resolution,
                height: resolution,
                kind: ReadbackKind::Normal,
            },
            Readback::texture(job.normal_images[lod as usize].clone()),
        ));
    }
}

fn handle_generator_readback_complete(
    event: On<ReadbackComplete>,
    mut commands: Commands,
    pending: Query<&PendingGeneratorReadback>,
    mut runtime: ResMut<GeneratorExportRuntime>,
    mut state: ResMut<GeneratorExportState>,
) {
    let Ok(pending) = pending.get(event.entity) else {
        return;
    };
    commands.entity(event.entity).despawn();

    let Some(job) = runtime.job.as_mut() else {
        return;
    };
    if job.generation != pending.generation || job.writer.is_some() {
        return;
    }

    let bytes_per_pixel = match pending.kind {
        ReadbackKind::Height => 4,
        ReadbackKind::Normal => 4, // Rgba8Snorm: 4 bytes/texel; RG extracted in build_normal_tile_bytes
    };
    let compact = strip_padded_rows(&event.data, pending.width, pending.height, bytes_per_pixel);
    let lod = pending.lod as usize;
    match pending.kind {
        ReadbackKind::Height => job.heights[lod] = Some(compact),
        ReadbackKind::Normal => job.normals[lod] = Some(compact),
    }

    let completed = job.heights.iter().flatten().count() + job.normals.iter().flatten().count();
    let total = (job.levels as usize) * 2;
    if completed < total {
        return;
    }

    state
        .log
        .push("Readback complete; writing tile hierarchy…".into());

    let heights = job
        .heights
        .iter_mut()
        .map(|slot| slot.take().expect("missing height readback"))
        .collect();
    let normals = job
        .normals
        .iter_mut()
        .map(|slot| slot.take().expect("missing normal readback"))
        .collect();
    job.writer = Some(spawn_export_writer(
        job.params.clone(),
        job.output_dir.clone(),
        heights,
        normals,
    ));
}

fn drain_generator_writer_logs(
    mut commands: Commands,
    mut runtime: ResMut<GeneratorExportRuntime>,
    mut state: ResMut<GeneratorExportState>,
) {
    let Some(job) = runtime.job.as_mut() else {
        return;
    };
    let Some(writer) = job.writer.as_ref() else {
        return;
    };

    while let Ok(line) = writer.log_rx.lock().unwrap().try_recv() {
        state.log.push(line);
    }

    if !writer.done.load(Ordering::Acquire) {
        return;
    }

    state.active = false;
    state.succeeded = writer.succeeded.load(Ordering::Acquire);
    state.completed_generation = job.generation;
    commands.remove_resource::<ActiveGeneratorExport>();
    runtime.job = None;
}

fn prepare_export_bind_groups(
    mut commands: Commands,
    active_export: Option<Res<ActiveGeneratorExport>>,
    existing: Option<Res<GeneratorExportRenderResources>>,
    pipeline: Res<GeneratorExportPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(active_export) = active_export else {
        if existing.is_some() {
            commands.remove_resource::<GeneratorExportRenderResources>();
        }
        return;
    };
    if existing
        .as_ref()
        .map(|existing| existing.generation == active_export.generation)
        .unwrap_or(false)
    {
        return;
    }

    let Some(l0_height) = gpu_images.get(&active_export.height_images[0]) else {
        return;
    };

    let mut generate_uniform = UniformBuffer::from(ExportGeneratorUniform::from_params(
        &active_export.params,
        active_export.params.export_resolution,
    ));
    generate_uniform.write_buffer(&render_device, &render_queue);
    let generate_uniform_bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.generate_uniform_layout),
        &BindGroupEntries::with_indices(((1, generate_uniform.binding().unwrap()),)),
    );
    let generate_image_bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.generate_image_layout),
        &BindGroupEntries::with_indices(((1, &l0_height.texture_view),)),
    );

    let mut downsample_passes = Vec::with_capacity(active_export.levels.saturating_sub(1) as usize);
    for lod in 1..active_export.levels {
        let Some(src) = gpu_images.get(&active_export.height_images[(lod - 1) as usize]) else {
            return;
        };
        let Some(dst) = gpu_images.get(&active_export.height_images[lod as usize]) else {
            return;
        };
        let dst_resolution = active_export.params.export_resolution >> lod;
        let mut uniform = UniformBuffer::from(DownsampleUniform {
            src_resolution: UVec2::splat(active_export.params.export_resolution >> (lod - 1)),
            dst_resolution: UVec2::splat(dst_resolution),
        });
        uniform.write_buffer(&render_device, &render_queue);
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.downsample_uniform_layout),
            &BindGroupEntries::with_indices(((2, uniform.binding().unwrap()),)),
        );
        let image_bind_group = render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.downsample_image_layout),
            &BindGroupEntries::with_indices(((2, &src.texture_view), (3, &dst.texture_view))),
        );
        downsample_passes.push(DownsamplePassResources {
            _uniform: uniform,
            uniform_bind_group: bind_group,
            image_bind_group,
            dst_resolution,
        });
    }

    let mut normal_passes = Vec::with_capacity(active_export.levels as usize);
    let effective_height_scale =
        active_export.params.height_scale * active_export.params.world_scale;
    for lod in 0..active_export.levels {
        let resolution = active_export.params.export_resolution >> lod;
        let Some(src) = gpu_images.get(&active_export.height_images[lod as usize]) else {
            return;
        };
        let Some(dst) = gpu_images.get(&active_export.normal_images[lod as usize]) else {
            return;
        };
        let mut uniform = UniformBuffer::from(NormalUniform {
            resolution: UVec2::splat(resolution),
            effective_height_scale,
            lod_scale: active_export.params.world_scale * (1u32 << lod) as f32,
        });
        uniform.write_buffer(&render_device, &render_queue);
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.normal_uniform_layout),
            &BindGroupEntries::with_indices(((3, uniform.binding().unwrap()),)),
        );
        let image_bind_group = render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.normal_image_layout),
            &BindGroupEntries::with_indices(((4, &src.texture_view), (5, &dst.texture_view))),
        );
        normal_passes.push(NormalPassResources {
            _uniform: uniform,
            uniform_bind_group: bind_group,
            image_bind_group,
            resolution,
        });
    }

    commands.insert_resource(GeneratorExportRenderResources {
        generation: active_export.generation,
        generate_pass: GeneratePassResources {
            _uniform: generate_uniform,
            uniform_bind_group: generate_uniform_bind_group,
            image_bind_group: generate_image_bind_group,
        },
        downsample_passes,
        normal_passes,
    });
}

fn build_export_image(
    images: &mut Assets<Image>,
    resolution: u32,
    format: TextureFormat,
) -> Handle<Image> {
    let mut image = Image::new_uninit(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |=
        TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

fn baked_level_count(export_resolution: u32) -> Result<u32, String> {
    if export_resolution > MAX_EXPORT_RESOLUTION {
        return Err(format!(
            "export_resolution {export_resolution} exceeds the current maximum of {MAX_EXPORT_RESOLUTION}"
        ));
    }
    if !export_resolution.is_power_of_two() {
        return Err(format!(
            "export_resolution {export_resolution} must be a power of two"
        ));
    }
    if export_resolution < TILE_SIZE * 2 {
        return Err(format!(
            "export_resolution {export_resolution} is too small (min {})",
            TILE_SIZE * 2
        ));
    }
    if export_resolution % TILE_SIZE != 0 {
        return Err(format!(
            "export_resolution {export_resolution} must be a multiple of {TILE_SIZE}"
        ));
    }

    let mut sz = export_resolution as usize;
    let mut levels = 0u32;
    while sz / TILE_SIZE as usize >= 2 {
        levels += 1;
        sz /= 2;
    }
    Ok(levels)
}

fn strip_padded_rows(data: &[u8], width: u32, height: u32, bytes_per_pixel: usize) -> Vec<u8> {
    let tight_row_bytes = width as usize * bytes_per_pixel;
    let padded_row_bytes = tight_row_bytes.next_multiple_of(256);
    if padded_row_bytes == tight_row_bytes {
        return data.to_vec();
    }

    let mut compact = Vec::with_capacity(tight_row_bytes * height as usize);
    for row in 0..height as usize {
        let start = row * padded_row_bytes;
        compact.extend_from_slice(&data[start..start + tight_row_bytes]);
    }
    compact
}

fn spawn_export_writer(
    params: GeneratorParams,
    output_dir: PathBuf,
    heights: Vec<Vec<u8>>,
    normals: Vec<Vec<u8>>,
) -> ExportWriterHandle {
    let (log_tx, log_rx) = mpsc::channel::<String>();
    let done = Arc::new(AtomicBool::new(false));
    let succeeded = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();
    let succeeded_clone = succeeded.clone();

    std::thread::spawn(move || {
        let run = || -> Result<(), String> {
            write_export_tiles(&params, &output_dir, &heights, &normals, &log_tx)?;
            Ok(())
        };

        match run() {
            Ok(()) => succeeded_clone.store(true, Ordering::Release),
            Err(err) => {
                log_tx.send(format!("Export failed: {err}")).ok();
            }
        }
        done_clone.store(true, Ordering::Release);
    });

    ExportWriterHandle {
        log_rx: Mutex::new(log_rx),
        done,
        succeeded,
    }
}

fn write_export_tiles(
    params: &GeneratorParams,
    output_dir: &Path,
    heights: &[Vec<u8>],
    normals: &[Vec<u8>],
    log: &mpsc::Sender<String>,
) -> Result<(), String> {
    prepare_output_dir(output_dir)?;
    log.send(format!(
        "Writing {} levels to '{}'",
        heights.len(),
        output_dir.display()
    ))
    .ok();

    for (lod, (height_level, normal_level)) in heights.iter().zip(normals.iter()).enumerate() {
        let lod = lod as u32;
        let resolution = params.export_resolution >> lod;
        let tiles_per_side = (resolution / TILE_SIZE) as i32;
        let tile_start = -(tiles_per_side / 2);

        let height_dir = output_dir.join(format!("height/L{lod}"));
        let normal_dir = output_dir.join(format!("normal/L{lod}"));
        std::fs::create_dir_all(&height_dir).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&normal_dir).map_err(|e| e.to_string())?;

        for tile_y in 0..tiles_per_side as usize {
            for tile_x in 0..tiles_per_side as usize {
                let tx = tile_start + tile_x as i32;
                let ty = tile_start + tile_y as i32;

                let height_tile =
                    build_height_tile_bytes(height_level, resolution as usize, tile_x, tile_y);
                let normal_tile =
                    build_normal_tile_bytes(normal_level, resolution as usize, tile_x, tile_y);

                let height_path = height_dir.join(format!("{tx}_{ty}.bin"));
                std::fs::write(&height_path, height_tile).map_err(|e| {
                    format!(
                        "Failed to write height tile L{lod}/{tx}_{ty}.bin to '{}': {e}",
                        height_path.display()
                    )
                })?;

                let normal_path = normal_dir.join(format!("{tx}_{ty}.bin"));
                std::fs::write(&normal_path, normal_tile).map_err(|e| {
                    format!(
                        "Failed to write normal tile L{lod}/{tx}_{ty}.bin to '{}': {e}",
                        normal_path.display()
                    )
                })?;
            }
        }

        log.send(format!(
            "Level {lod}: {} tiles",
            tiles_per_side * tiles_per_side
        ))
        .ok();
    }

    log.send(format!("Done → '{}'", output_dir.display())).ok();
    Ok(())
}

fn prepare_output_dir(output_dir: &Path) -> Result<(), String> {
    for subdir in ["height", "normal"] {
        let path = output_dir.join(subdir);
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| format!("Failed to clear '{subdir}': {e}"))?;
        }
    }
    std::fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "Failed to create output dir '{}': {e}",
            output_dir.display()
        )
    })
}

fn build_height_tile_bytes(
    level_bytes: &[u8],
    resolution: usize,
    tile_x: usize,
    tile_y: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity((TILE_SIZE * TILE_SIZE * 2) as usize);
    let start_x = tile_x * TILE_SIZE as usize;
    let start_y = tile_y * TILE_SIZE as usize;

    for row in 0..TILE_SIZE as usize {
        let src_row = start_y + row;
        for col in 0..TILE_SIZE as usize {
            let src_col = start_x + col;
            let offset = (src_row * resolution + src_col) * 4;
            let sample = f32::from_le_bytes([
                level_bytes[offset],
                level_bytes[offset + 1],
                level_bytes[offset + 2],
                level_bytes[offset + 3],
            ]);
            let quantized = (sample.clamp(0.0, 1.0) * 65535.0).round() as u16;
            out.extend_from_slice(&quantized.to_le_bytes());
        }
    }

    out
}

fn build_normal_tile_bytes(
    level_bytes: &[u8],
    resolution: usize,
    tile_x: usize,
    tile_y: usize,
) -> Vec<u8> {
    // Source format is Rgba8Snorm (4 bytes/texel). Extract only the RG bytes
    // (XZ normal components) and discard BA, producing a compact Rg8Snorm tile.
    let mut out = Vec::with_capacity((TILE_SIZE * TILE_SIZE * 2) as usize);
    let start_x = tile_x * TILE_SIZE as usize;
    let start_y = tile_y * TILE_SIZE as usize;

    for row in 0..TILE_SIZE as usize {
        let src_row = start_y + row;
        for col in 0..TILE_SIZE as usize {
            let src_col = start_x + col;
            let offset = (src_row * resolution + src_col) * 4;
            out.push(level_bytes[offset]);     // R = nx (X normal component)
            out.push(level_bytes[offset + 1]); // G = nz (Z normal component)
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{
        baked_level_count, build_height_tile_bytes, build_normal_tile_bytes, strip_padded_rows,
        TILE_SIZE,
    };

    // ---------------------------------------------------------------------------
    // baked_level_count
    // ---------------------------------------------------------------------------

    #[test]
    fn baked_level_count_known_resolutions() {
        assert_eq!(baked_level_count(512).unwrap(), 1);
        assert_eq!(baked_level_count(1024).unwrap(), 2);
        assert_eq!(baked_level_count(2048).unwrap(), 3);
        assert_eq!(baked_level_count(4096).unwrap(), 4);
        assert_eq!(baked_level_count(8192).unwrap(), 5);
    }

    #[test]
    fn baked_level_count_rejects_invalid() {
        assert!(baked_level_count(32768).is_err()); // exceeds MAX_EXPORT_RESOLUTION
        assert!(baked_level_count(3000).is_err()); // not a power-of-two
        assert!(baked_level_count(256).is_err()); // < TILE_SIZE * 2
        assert!(baked_level_count(0).is_err());
    }

    // ---------------------------------------------------------------------------
    // strip_padded_rows
    // ---------------------------------------------------------------------------

    #[test]
    fn strip_padded_rows_removes_copy_padding() {
        // 1 pixel wide, 4 bytes/px → tight = 4, padded = 256
        let mut raw = vec![0u8; 512];
        raw[0..4].copy_from_slice(&[1, 2, 3, 4]);
        raw[256..260].copy_from_slice(&[5, 6, 7, 8]);
        let compact = strip_padded_rows(&raw, 1, 2, 4);
        assert_eq!(compact, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn strip_padded_rows_passthrough_when_already_aligned() {
        // 64 pixels, 4 bytes/px → tight = 256 = padded (no stripping needed)
        let data: Vec<u8> = (0u8..=255).collect();
        let result = strip_padded_rows(&data, 64, 1, 4);
        assert_eq!(result, data);
    }

    #[test]
    fn strip_padded_rows_snorm_narrow_width() {
        // 1 pixel wide, 2 bytes/px (Rg8Snorm) → tight = 2, padded = 256
        let mut raw = vec![0u8; 512];
        raw[0] = 10;
        raw[1] = 20;
        raw[256] = 30;
        raw[257] = 40;
        let compact = strip_padded_rows(&raw, 1, 2, 2);
        assert_eq!(compact, vec![10, 20, 30, 40]);
    }

    #[test]
    fn strip_padded_rows_power_of_two_widths_are_naturally_aligned() {
        // For power-of-two widths used in export (512..4096) with R32Float (4 bpp)
        // and Rgba8Snorm (4 bpp), tight rows are always multiples of 256.
        for log2_w in 9u32..=12 {
            let w = 1u32 << log2_w; // 512, 1024, 2048, 4096
            for bpp in [2usize, 4] {
                let tight = w as usize * bpp;
                assert_eq!(
                    tight % 256,
                    0,
                    "width={w} bpp={bpp} tight={tight} needs stripping"
                );
            }
        }
    }

    // ---------------------------------------------------------------------------
    // build_height_tile_bytes — pixel-level correctness
    // ---------------------------------------------------------------------------

    fn make_height_image(resolution: usize) -> Vec<u8> {
        let mut buf = vec![0u8; resolution * resolution * 4];
        for row in 0..resolution {
            for col in 0..resolution {
                let h = (row * resolution + col) as f32 / (resolution * resolution) as f32;
                let off = (row * resolution + col) * 4;
                buf[off..off + 4].copy_from_slice(&h.to_le_bytes());
            }
        }
        buf
    }

    /// Read back the f32 at position (col, row) inside a tile (u16 encoded).
    fn tile_height_at(tile: &[u8], row: usize, col: usize) -> f32 {
        let off = (row * TILE_SIZE as usize + col) * 2;
        let q = u16::from_le_bytes([tile[off], tile[off + 1]]);
        q as f32 / 65535.0
    }

    #[test]
    fn height_tile_first_pixel_matches_source() {
        let resolution = 512usize;
        let img = make_height_image(resolution);

        for tile_y in 0..2usize {
            for tile_x in 0..2usize {
                let tile = build_height_tile_bytes(&img, resolution, tile_x, tile_y);
                assert_eq!(tile.len(), (TILE_SIZE * TILE_SIZE * 2) as usize);

                let src_row = tile_y * TILE_SIZE as usize;
                let src_col = tile_x * TILE_SIZE as usize;
                let expected_h =
                    (src_row * resolution + src_col) as f32 / (resolution * resolution) as f32;
                let expected_q = (expected_h * 65535.0).round() as u16;
                let actual_q = u16::from_le_bytes([tile[0], tile[1]]);
                assert_eq!(actual_q, expected_q, "tile ({tile_x},{tile_y}) first pixel");
            }
        }
    }

    #[test]
    fn height_tile_last_pixel_matches_source() {
        let resolution = 512usize;
        let img = make_height_image(resolution);

        for tile_y in 0..2usize {
            for tile_x in 0..2usize {
                let tile = build_height_tile_bytes(&img, resolution, tile_x, tile_y);

                // Last pixel = (tile_x*256+255, tile_y*256+255) in source
                let src_row = tile_y * TILE_SIZE as usize + (TILE_SIZE as usize - 1);
                let src_col = tile_x * TILE_SIZE as usize + (TILE_SIZE as usize - 1);
                let expected_h =
                    (src_row * resolution + src_col) as f32 / (resolution * resolution) as f32;
                let expected_q = (expected_h * 65535.0).round() as u16;
                let last = (TILE_SIZE as usize - 1) * TILE_SIZE as usize + (TILE_SIZE as usize - 1);
                let off = last * 2;
                let actual_q = u16::from_le_bytes([tile[off], tile[off + 1]]);
                assert_eq!(actual_q, expected_q, "tile ({tile_x},{tile_y}) last pixel");
            }
        }
    }

    /// The last row of tile (tx, ty) and the first row of tile (tx, ty+1) must be
    /// different — they're ADJACENT rows in the source, not the same row.
    #[test]
    fn adjacent_tiles_have_no_repeated_rows() {
        let resolution = 512usize;
        let img = make_height_image(resolution);

        for tile_x in 0..2usize {
            let tile0 = build_height_tile_bytes(&img, resolution, tile_x, 0);
            let tile1 = build_height_tile_bytes(&img, resolution, tile_x, 1);

            // Last row of tile0 (row 255 = source row 255)
            let last_row_h = tile_height_at(&tile0, TILE_SIZE as usize - 1, 0);
            // First row of tile1 (row 0 = source row 256)
            let first_row_h = tile_height_at(&tile1, 0, 0);

            // Values must differ (they come from consecutive but distinct rows)
            assert_ne!(
                last_row_h, first_row_h,
                "tile_x={tile_x}: last row of tile 0 == first row of tile 1 (off-by-one?)"
            );

            // Verify each references the correct source row
            let expected_last =
                (255 * resolution + tile_x * TILE_SIZE as usize) as f32 / (resolution * resolution) as f32;
            let expected_first =
                (256 * resolution + tile_x * TILE_SIZE as usize) as f32 / (resolution * resolution) as f32;

            assert!(
                (last_row_h - expected_last).abs() < 2.0 / 65535.0,
                "tile_x={tile_x}: last row height wrong"
            );
            assert!(
                (first_row_h - expected_first).abs() < 2.0 / 65535.0,
                "tile_x={tile_x}: first row of next tile height wrong"
            );
        }
    }

    /// Every pixel in the full image appears in exactly one tile.
    #[test]
    fn tiles_cover_source_without_gaps() {
        let resolution = 512usize;
        let tiles_per_side = resolution / TILE_SIZE as usize;
        let img = make_height_image(resolution);
        let mut covered = vec![false; resolution * resolution];

        for tile_y in 0..tiles_per_side {
            for tile_x in 0..tiles_per_side {
                let tile = build_height_tile_bytes(&img, resolution, tile_x, tile_y);
                for row in 0..TILE_SIZE as usize {
                    for col in 0..TILE_SIZE as usize {
                        let src_row = tile_y * TILE_SIZE as usize + row;
                        let src_col = tile_x * TILE_SIZE as usize + col;
                        let px = src_row * resolution + src_col;
                        assert!(!covered[px], "pixel ({src_col},{src_row}) covered twice");
                        covered[px] = true;

                        // Also verify the actual value
                        let expected_h = px as f32 / (resolution * resolution) as f32;
                        let expected_q = (expected_h * 65535.0).round() as u16;
                        let off = (row * TILE_SIZE as usize + col) * 2;
                        let actual_q = u16::from_le_bytes([tile[off], tile[off + 1]]);
                        assert_eq!(
                            actual_q, expected_q,
                            "wrong value at src ({src_col},{src_row})"
                        );
                    }
                }
            }
        }

        assert!(covered.iter().all(|&c| c), "some source pixels not covered");
    }

    // ---------------------------------------------------------------------------
    // build_normal_tile_bytes — RG extraction from Rgba8Snorm source
    // ---------------------------------------------------------------------------

    fn make_normal_image_rgba(resolution: usize) -> Vec<u8> {
        // Rgba8Snorm: 4 bytes per pixel.  R = col & 0xff, G = row & 0xff, BA = 0.
        let mut buf = vec![0u8; resolution * resolution * 4];
        for row in 0..resolution {
            for col in 0..resolution {
                let off = (row * resolution + col) * 4;
                buf[off] = (col & 0xff) as u8;
                buf[off + 1] = (row & 0xff) as u8;
                // B and A are zero (should be discarded)
                buf[off + 2] = 0xff;
                buf[off + 3] = 0xff;
            }
        }
        buf
    }

    #[test]
    fn normal_tile_extracts_rg_discards_ba() {
        let resolution = 512usize;
        let img = make_normal_image_rgba(resolution);

        // Tile (1,1): source starts at (256, 256)
        let tile = build_normal_tile_bytes(&img, resolution, 1, 1);
        assert_eq!(tile.len(), (TILE_SIZE * TILE_SIZE * 2) as usize);

        // First pixel of tile (1,1) = source pixel (col=256, row=256)
        assert_eq!(tile[0], (256 & 0xff) as u8, "R of first pixel wrong");
        assert_eq!(tile[1], (256 & 0xff) as u8, "G of first pixel wrong");

        // Last pixel = (511, 511)
        let last = ((TILE_SIZE as usize - 1) * TILE_SIZE as usize + (TILE_SIZE as usize - 1)) * 2;
        assert_eq!(tile[last], (511 & 0xff) as u8, "R of last pixel wrong");
        assert_eq!(tile[last + 1], (511 & 0xff) as u8, "G of last pixel wrong");
    }

    #[test]
    fn normal_tile_ba_bytes_not_included() {
        let resolution = 512usize;
        let mut img = vec![0u8; resolution * resolution * 4];
        // Mark BA with 0xAB so that if they leak into the output the test fails
        for i in 0..resolution * resolution {
            img[i * 4 + 2] = 0xAB;
            img[i * 4 + 3] = 0xCD;
        }
        let tile = build_normal_tile_bytes(&img, resolution, 0, 0);
        // All output bytes should be 0 (from R and G channels), never 0xAB or 0xCD
        assert!(
            tile.iter().all(|&b| b == 0),
            "BA bytes leaked into normal tile output"
        );
    }

    // ---------------------------------------------------------------------------
    // Height quantization round-trip
    // ---------------------------------------------------------------------------

    #[test]
    fn height_quantize_roundtrip_max_error_half_ulp() {
        // Export: (h * 65535).round() as u16
        // Reload: u16 as f32 / 65535.0
        // Maximum round-trip error should be ≤ 0.5 / 65535
        let max_err = 0.5_f32 / 65535.0 + f32::EPSILON * 4.0;
        for i in 0u32..=65535 {
            let original = i as f32 / 65535.0;
            let quantized = (original.clamp(0.0, 1.0) * 65535.0).round() as u16;
            let reloaded = quantized as f32 / 65535.0;
            let err = (original - reloaded).abs();
            assert!(err <= max_err, "roundtrip error at i={i}: {err} > {max_err}");
        }
    }
}

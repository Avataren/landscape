use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            binding_types::uniform_buffer, AsBindGroup, BindGroup, BindGroupEntries, BindGroupEntry,
            BindGroupLayoutDescriptor, BindGroupLayoutEntries, BindGroupLayoutEntry, BindingResource,
            BindingType, Buffer, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess,
            TextureFormat, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

use crate::images::{GeneratorImage, NormalizationImage};
use crate::uniforms::{GeneratorDisplayGeneration, GeneratorParamGeneration, GeneratorUniform, GeneratorUniformBuffer};

const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource)]
struct GeneratorUniformBindGroup(BindGroup);

#[allow(dead_code)]
#[derive(Resource)]
struct GeneratorImageBindGroup(BindGroup);

/// GPU buffer holding two u32 values: [min_bits, max_bits].
#[derive(Resource)]
struct MinmaxBuffer(Buffer);

/// Combined bind group for the preview normalization passes.
#[derive(Resource)]
struct NormalizationBindGroup(BindGroup);

fn prepare_uniform_bind_group(
    mut commands: Commands,
    pipeline: Res<GeneratorPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut uniform_buffer: ResMut<GeneratorUniformBuffer>,
    uniform: Res<GeneratorUniform>,
    render_device: Res<RenderDevice>,
) {
    *uniform_buffer.buffer.get_mut() = uniform.clone();
    uniform_buffer.buffer.write_buffer(&render_device, &render_queue);

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.uniform_layout),
        &BindGroupEntries::single(uniform_buffer.buffer.binding().unwrap().clone()),
    );
    commands.insert_resource(GeneratorUniformBindGroup(bind_group));
}

fn prepare_image_bind_group(
    mut commands: Commands,
    pipeline: Res<GeneratorPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    generator_image: Option<Res<GeneratorImage>>,
    render_device: Res<RenderDevice>,
) {
    let Some(gen_img) = generator_image else { return; };
    let Some(view) = gpu_images.get(&gen_img.heightfield) else { return; };

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.image_layout),
        &BindGroupEntries::single(&view.texture_view),
    );
    commands.insert_resource(GeneratorImageBindGroup(bind_group));
}

fn prepare_normalization_bind_group(
    mut commands: Commands,
    pipeline: Res<GeneratorPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    generator_image: Option<Res<GeneratorImage>>,
    norm_image: Option<Res<NormalizationImage>>,
    minmax_buf: Option<Res<MinmaxBuffer>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(gen_img), Some(norm_img), Some(mm_buf)) =
        (generator_image, norm_image, minmax_buf)
    else {
        return;
    };
    let Some(preview_view) = gpu_images.get(&gen_img.heightfield) else { return; };
    let Some(raw_view) = gpu_images.get(&norm_img.raw_heights) else { return; };

    // Reset min/max buffer before each frame's reduction pass.
    let mut init_bytes = [0u8; 8];
    init_bytes[0..4].copy_from_slice(&0x7F7F_FFFFu32.to_le_bytes());
    init_bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
    render_queue.write_buffer(&mm_buf.0, 0, &init_bytes);

    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&preview_view.texture_view),
        },
        BindGroupEntry {
            binding: 6,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &mm_buf.0,
                offset: 0,
                size: None,
            }),
        },
        BindGroupEntry {
            binding: 7,
            resource: BindingResource::TextureView(&raw_view.texture_view),
        },
    ];
    let bind_group = render_device.create_bind_group(
        Some("generator_normalization_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.norm_image_layout),
        &entries,
    );
    commands.insert_resource(NormalizationBindGroup(bind_group));
}

#[derive(Resource)]
struct GeneratorPipeline {
    uniform_layout: BindGroupLayoutDescriptor,
    image_layout: BindGroupLayoutDescriptor,
    norm_image_layout: BindGroupLayoutDescriptor,
    #[allow(dead_code)]
    pipeline: CachedComputePipelineId,
    gen_raw_pipeline: CachedComputePipelineId,
    reduce_pipeline: CachedComputePipelineId,
    normalize_pipeline: CachedComputePipelineId,
}

impl FromWorld for GeneratorPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let image_layout = GeneratorImage::bind_group_layout_descriptor(render_device);

        let uniform_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<GeneratorUniform>(false),),
        );
        let uniform_layout =
            BindGroupLayoutDescriptor::new("generator_uniform_bind_group_layout", &uniform_entries);

        let norm_image_layout = BindGroupLayoutDescriptor::new(
            "generator_norm_image_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(8),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        );

        let shader = load_embedded_asset!(world, "shaders/generator.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_generator_compute".into()),
            layout: vec![uniform_layout.clone(), image_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("generate")),
        });

        let gen_raw_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_generator_preview_raw".into()),
            layout: vec![uniform_layout.clone(), norm_image_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("preview_generate_raw")),
        });

        let reduce_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_generator_preview_reduce".into()),
            layout: vec![uniform_layout.clone(), norm_image_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("preview_reduce_minmax")),
        });

        let normalize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_generator_preview_normalize".into()),
            layout: vec![uniform_layout.clone(), norm_image_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("preview_normalize_display")),
        });

        Self {
            uniform_layout,
            image_layout,
            norm_image_layout,
            pipeline,
            gen_raw_pipeline,
            reduce_pipeline,
            normalize_pipeline,
        }
    }
}

// ---------------------------------------------------------------------------
// Node: GeneratorRawNode — runs preview_generate_raw only
// ---------------------------------------------------------------------------

enum GeneratorState { Loading, Ready }

/// Generates raw heights into `raw_heights`. Erosion (if enabled) runs after this.
struct GeneratorRawNode {
    state: GeneratorState,
    /// Tracks which generation was last dispatched. Uses AtomicU64 for interior
    /// mutability since `Node::run` takes `&self`. Starts at MAX so the first
    /// generation (0) is always treated as new.
    dispatched_generation: AtomicU64,
}

impl Default for GeneratorRawNode {
    fn default() -> Self {
        Self { state: GeneratorState::Loading, dispatched_generation: AtomicU64::new(u64::MAX) }
    }
}

impl Node for GeneratorRawNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GeneratorPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if matches!(self.state, GeneratorState::Loading)
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.gen_raw_pipeline),
                CachedPipelineState::Ok(_)
            )
        {
            self.state = GeneratorState::Ready;
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if matches!(self.state, GeneratorState::Loading) { return Ok(()); }

        let gen = world
            .get_resource::<GeneratorParamGeneration>()
            .map(|g| g.0)
            .unwrap_or(0);
        if gen == self.dispatched_generation.load(Ordering::Relaxed) {
            return Ok(());
        }

        let Some(uniform_bg) = world.get_resource::<GeneratorUniformBindGroup>() else {
            return Ok(());
        };
        let Some(norm_bg) = world.get_resource::<NormalizationBindGroup>() else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_res = world.resource::<GeneratorPipeline>();
        let uniform = world.resource::<GeneratorUniform>();

        let Some(gen_raw_pl) =
            pipeline_cache.get_compute_pipeline(pipeline_res.gen_raw_pipeline)
        else {
            return Ok(());
        };

        let wg_x = uniform.resolution.x.div_ceil(WORKGROUP_SIZE);
        let wg_y = uniform.resolution.y.div_ceil(WORKGROUP_SIZE);

        let mut pass = render_context.command_encoder().begin_compute_pass(
            &ComputePassDescriptor {
                label: Some("generator_preview_raw"),
                ..default()
            },
        );
        pass.set_bind_group(0, &uniform_bg.0, &[]);
        pass.set_bind_group(1, &norm_bg.0, &[]);
        pass.set_pipeline(gen_raw_pl);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        self.dispatched_generation.store(gen, Ordering::Relaxed);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Node: GeneratorNormNode — runs reduce + normalize
// ---------------------------------------------------------------------------

/// Reads raw_heights (which may have been processed by erosion), normalises, writes display output.
struct GeneratorNormNode {
    state: GeneratorState,
    /// Last GeneratorParamGeneration value we normalised for.
    dispatched_params_gen: AtomicU64,
    /// Last erosion tick count we normalised for (proxy for copy_out having run).
    dispatched_erosion_ticks: AtomicU64,
}

impl Default for GeneratorNormNode {
    fn default() -> Self {
        Self {
            state: GeneratorState::Loading,
            dispatched_params_gen: AtomicU64::new(u64::MAX),
            dispatched_erosion_ticks: AtomicU64::new(u64::MAX),
        }
    }
}

impl Node for GeneratorNormNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GeneratorPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if matches!(self.state, GeneratorState::Loading)
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.reduce_pipeline),
                CachedPipelineState::Ok(_)
            )
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.normalize_pipeline),
                CachedPipelineState::Ok(_)
            )
        {
            self.state = GeneratorState::Ready;
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if matches!(self.state, GeneratorState::Loading) { return Ok(()); }

        let display_gen = world
            .get_resource::<GeneratorDisplayGeneration>()
            .map(|g| g.0)
            .unwrap_or(0);
        let erosion_ticks = world
            .get_resource::<crate::erosion_params::ErosionControlState>()
            .map(|c| c.ticks_done() as u64)
            .unwrap_or(0);

        let last_gen   = self.dispatched_params_gen.load(Ordering::Relaxed);
        let last_ticks = self.dispatched_erosion_ticks.load(Ordering::Relaxed);
        if display_gen == last_gen && erosion_ticks == last_ticks {
            return Ok(());
        }

        let Some(uniform_bg) = world.get_resource::<GeneratorUniformBindGroup>() else {
            return Ok(());
        };
        let Some(norm_bg) = world.get_resource::<NormalizationBindGroup>() else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_res = world.resource::<GeneratorPipeline>();
        let uniform = world.resource::<GeneratorUniform>();

        let (Some(reduce_pl), Some(norm_pl)) = (
            pipeline_cache.get_compute_pipeline(pipeline_res.reduce_pipeline),
            pipeline_cache.get_compute_pipeline(pipeline_res.normalize_pipeline),
        ) else {
            return Ok(());
        };

        let wg_x = uniform.resolution.x.div_ceil(WORKGROUP_SIZE);
        let wg_y = uniform.resolution.y.div_ceil(WORKGROUP_SIZE);

        {
            let mut pass = render_context.command_encoder().begin_compute_pass(
                &ComputePassDescriptor {
                    label: Some("generator_preview_reduce"),
                    ..default()
                },
            );
            pass.set_bind_group(0, &uniform_bg.0, &[]);
            pass.set_bind_group(1, &norm_bg.0, &[]);
            pass.set_pipeline(reduce_pl);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass = render_context.command_encoder().begin_compute_pass(
                &ComputePassDescriptor {
                    label: Some("generator_preview_normalize"),
                    ..default()
                },
            );
            pass.set_bind_group(0, &uniform_bg.0, &[]);
            pass.set_bind_group(1, &norm_bg.0, &[]);
            pass.set_pipeline(norm_pl);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.dispatched_params_gen.store(display_gen, Ordering::Relaxed);
        self.dispatched_erosion_ticks.store(erosion_ticks, Ordering::Relaxed);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub(crate) struct GeneratorComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct GeneratorRawLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct GeneratorNormLabel;

impl Plugin for GeneratorComputePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/generator.wgsl");

        app.add_plugins(ExtractResourcePlugin::<GeneratorImage>::default())
            .add_plugins(ExtractResourcePlugin::<GeneratorUniform>::default())
            .add_plugins(ExtractResourcePlugin::<GeneratorParamGeneration>::default())
            .add_plugins(ExtractResourcePlugin::<GeneratorDisplayGeneration>::default())
            .add_plugins(ExtractResourcePlugin::<NormalizationImage>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_uniform_bind_group.in_set(RenderSystems::PrepareResources),
        );
        render_app.add_systems(
            Render,
            prepare_image_bind_group.in_set(RenderSystems::PrepareResources),
        );
        render_app.add_systems(
            Render,
            prepare_normalization_bind_group.in_set(RenderSystems::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GeneratorRawLabel, GeneratorRawNode::default());
        render_graph.add_node(GeneratorNormLabel, GeneratorNormNode::default());
        // Default ordering: Raw → Norm → CameraDriver.
        // ErosionComputePlugin inserts itself between Raw and Norm.
        render_graph.add_node_edge(GeneratorRawLabel, GeneratorNormLabel);
        render_graph.add_node_edge(GeneratorNormLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GeneratorPipeline>();
        render_app.init_resource::<GeneratorUniformBuffer>();

        let minmax_buf = {
            let render_device = render_app.world().resource::<RenderDevice>();
            render_device.create_buffer(&BufferDescriptor {
                label: Some("generator_minmax_buf"),
                size: 8,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        render_app.world_mut().insert_resource(MinmaxBuffer(minmax_buf));
    }
}

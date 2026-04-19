use std::borrow::Cow;

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            binding_types::uniform_buffer, AsBindGroup, BindGroup, BindGroupEntries,
            BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
            CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
            ShaderStages,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

use crate::images::GeneratorImage;
use crate::uniforms::{GeneratorUniform, GeneratorUniformBuffer};

const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource)]
struct GeneratorUniformBindGroup(BindGroup);

#[derive(Resource)]
struct GeneratorImageBindGroup(BindGroup);

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
    uniform_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);

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
    let Some(gen_img) = generator_image else {
        return;
    };
    let Some(view) = gpu_images.get(&gen_img.heightfield) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.image_layout),
        &BindGroupEntries::single(&view.texture_view),
    );
    commands.insert_resource(GeneratorImageBindGroup(bind_group));
}

#[derive(Resource)]
struct GeneratorPipeline {
    uniform_layout: BindGroupLayoutDescriptor,
    image_layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
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

        let shader = load_embedded_asset!(world, "shaders/generator.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_generator_compute".into()),
            layout: vec![uniform_layout.clone(), image_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("generate")),
        });

        Self {
            uniform_layout,
            image_layout,
            pipeline,
        }
    }
}

enum GeneratorState {
    Loading,
    Ready,
}

struct GeneratorNode {
    state: GeneratorState,
}

impl Default for GeneratorNode {
    fn default() -> Self {
        Self {
            state: GeneratorState::Loading,
        }
    }
}

impl Node for GeneratorNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GeneratorPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if matches!(self.state, GeneratorState::Loading)
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.pipeline),
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
        if matches!(self.state, GeneratorState::Loading) {
            return Ok(());
        }

        let (Some(uniform_bg), Some(image_bg)) = (
            world.get_resource::<GeneratorUniformBindGroup>(),
            world.get_resource::<GeneratorImageBindGroup>(),
        ) else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_res = world.resource::<GeneratorPipeline>();
        let uniform = world.resource::<GeneratorUniform>();
        let resolution = uniform.resolution;

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline) else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_bind_group(0, &uniform_bg.0, &[]);
        pass.set_bind_group(1, &image_bg.0, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(
            resolution.x.div_ceil(WORKGROUP_SIZE),
            resolution.y.div_ceil(WORKGROUP_SIZE),
            1,
        );

        Ok(())
    }
}

pub(crate) struct GeneratorComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GeneratorLabel;

impl Plugin for GeneratorComputePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/generator.wgsl");

        app.add_plugins(ExtractResourcePlugin::<GeneratorImage>::default())
            .add_plugins(ExtractResourcePlugin::<GeneratorUniform>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_uniform_bind_group.in_set(RenderSystems::PrepareResources),
        );
        render_app.add_systems(
            Render,
            prepare_image_bind_group.in_set(RenderSystems::PrepareResources),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GeneratorLabel, GeneratorNode::default());
        render_graph.add_node_edge(GeneratorLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GeneratorPipeline>();
        render_app.init_resource::<GeneratorUniformBuffer>();
    }
}

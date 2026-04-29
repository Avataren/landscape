use std::borrow::Cow;

use bevy::{
    asset::load_embedded_asset,
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

use crate::images::{CLOUD_ATLAS_SIZE, CLOUD_WORLEY_SIZE};
use crate::uniforms::{CloudsImage, CloudsUniform, CloudsUniformBuffer};

const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource)]
struct CloudsUniformBindGroup(BindGroup);

#[derive(Resource)]
struct CloudsImageBindGroup(BindGroup);

fn prepare_uniforms_bind_group(
    mut commands: Commands,
    pipeline: Res<CloudsPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut clouds_uniform_buffer: ResMut<CloudsUniformBuffer>,
    clouds_uniform: Res<CloudsUniform>,
    render_device: Res<RenderDevice>,
) {
    *clouds_uniform_buffer.buffer.get_mut() = clouds_uniform.clone();
    clouds_uniform_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.uniform_bind_group_layout),
        &BindGroupEntries::single(clouds_uniform_buffer.buffer.binding().unwrap().clone()),
    );
    commands.insert_resource(CloudsUniformBindGroup(bind_group));
}

fn prepare_textures_bind_group(
    mut commands: Commands,
    pipeline: Res<CloudsPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    clouds_image: Res<CloudsImage>,
    render_device: Res<RenderDevice>,
) {
    let Some(cloud_render_view) = gpu_images.get(&clouds_image.cloud_render_image) else {
        return;
    };
    let Some(cloud_history_view) = gpu_images.get(&clouds_image.cloud_history_image) else {
        return;
    };
    let Some(cloud_atlas_view) = gpu_images.get(&clouds_image.cloud_atlas_image) else {
        return;
    };
    let Some(cloud_worley_view) = gpu_images.get(&clouds_image.cloud_worley_image) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.texture_bind_group_layout),
        &BindGroupEntries::sequential((
            &cloud_render_view.texture_view,
            &cloud_history_view.texture_view,
            &cloud_atlas_view.texture_view,
            &cloud_worley_view.texture_view,
        )),
    );
    commands.insert_resource(CloudsImageBindGroup(bind_group));
}

#[derive(Resource)]
struct CloudsPipeline {
    texture_bind_group_layout: BindGroupLayoutDescriptor,
    uniform_bind_group_layout: BindGroupLayoutDescriptor,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for CloudsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = CloudsImage::bind_group_layout_descriptor(render_device);
        let shader = load_embedded_asset!(world, "shaders/clouds_compute.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<CloudsUniform>(false),),
        );
        let uniform_bind_group_layout =
            BindGroupLayoutDescriptor::new("clouds_uniform_bind_group_layout", &entries);

        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_clouds_compute_init".into()),
            layout: vec![
                uniform_bind_group_layout.clone(),
                texture_bind_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("init")),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            zero_initialize_workgroup_memory: false,
            label: Some("landscape_clouds_compute".into()),
            layout: vec![
                uniform_bind_group_layout.clone(),
                texture_bind_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("update")),
        });

        Self {
            texture_bind_group_layout,
            uniform_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum CloudsState {
    Loading,
    Init,
    Update,
}

struct CloudsNode {
    state: CloudsState,
}

impl Default for CloudsNode {
    fn default() -> Self {
        Self {
            state: CloudsState::Loading,
        }
    }
}

impl Node for CloudsNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<CloudsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if matches!(self.state, CloudsState::Loading)
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline),
                CachedPipelineState::Ok(_)
            )
        {
            self.state = CloudsState::Init;
        } else if matches!(self.state, CloudsState::Init)
            && matches!(
                pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline),
                CachedPipelineState::Ok(_)
            )
        {
            self.state = CloudsState::Update;
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if matches!(self.state, CloudsState::Loading) {
            return Ok(());
        }

        let Some(texture_bind_group) = world.get_resource::<CloudsImageBindGroup>() else {
            return Ok(());
        };
        let Some(uniform_bind_group) = world.get_resource::<CloudsUniformBindGroup>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<CloudsPipeline>();
        let uniform = world.resource::<CloudsUniform>();
        let resolution = uniform.render_resolution.as_uvec2();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_bind_group(0, &uniform_bind_group.0, &[]);
        pass.set_bind_group(1, &texture_bind_group.0, &[]);

        match self.state {
            CloudsState::Loading => {}
            CloudsState::Init => {
                let Some(init_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.init_pipeline)
                else {
                    return Ok(());
                };
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(
                    CLOUD_ATLAS_SIZE.div_ceil(WORKGROUP_SIZE),
                    CLOUD_ATLAS_SIZE.div_ceil(WORKGROUP_SIZE),
                    CLOUD_WORLEY_SIZE,
                );
            }
            CloudsState::Update => {
                let Some(update_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.update_pipeline)
                else {
                    return Ok(());
                };
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(
                    resolution.x.div_ceil(WORKGROUP_SIZE),
                    resolution.y.div_ceil(WORKGROUP_SIZE),
                    1,
                );
            }
        }

        Ok(())
    }
}

pub(crate) struct CloudsComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct CloudsLabel;

impl Plugin for CloudsComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<CloudsImage>::default())
            .add_plugins(ExtractResourcePlugin::<CloudsUniform>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_textures_bind_group.in_set(RenderSystems::PrepareResources),
        );
        render_app.add_systems(
            Render,
            prepare_uniforms_bind_group.in_set(RenderSystems::PrepareResources),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(CloudsLabel, CloudsNode::default());
        render_graph.add_node_edge(CloudsLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<CloudsPipeline>();
        render_app.init_resource::<CloudsUniformBuffer>();
    }
}

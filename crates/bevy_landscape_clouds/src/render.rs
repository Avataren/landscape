use bevy::{
    asset::{embedded_asset, embedded_path, AssetPath},
    pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::{
        alpha::AlphaMode,
        render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError},
    },
    shader::ShaderRef,
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub(crate) struct CloudsDisplayMaterial {
    #[texture(100, visibility(vertex, fragment))]
    #[sampler(101, visibility(vertex, fragment))]
    pub cloud_render_image: Handle<Image>,
}

impl Material for CloudsDisplayMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("shaders/clouds_display.wgsl"))
                .with_source("embedded"),
        )
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("shaders/clouds_display.wgsl"))
                .with_source("embedded"),
        )
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub(crate) struct CloudsRenderPlugin;

impl Plugin for CloudsRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/clouds_display.wgsl");
        embedded_asset!(app, "shaders/clouds_compute.wgsl");

        app.add_plugins(MaterialPlugin::<CloudsDisplayMaterial>::default());
    }
}

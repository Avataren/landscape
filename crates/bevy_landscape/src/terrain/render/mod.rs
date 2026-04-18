pub mod extract;
pub mod pipelines;
pub mod prepare;
pub mod queue;

use crate::terrain::clipmap_texture::TerrainClipmapUploads;
use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        render_resource::{TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect},
        renderer::RenderQueue,
        texture::GpuImage,
        RenderApp, RenderSystems,
    },
};

/// Placeholder for the custom render pipeline (extract → prepare → queue → draw).
/// When implemented, this plugin will wire into Bevy's RenderApp with storage
/// buffers, partial texture uploads, and indirect draw commands.
pub struct TerrainRenderPlugin;

impl Plugin for TerrainRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<TerrainClipmapUploads>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            bevy::render::Render,
            apply_terrain_texture_uploads.in_set(RenderSystems::PrepareResources),
        );
    }
}

fn apply_terrain_texture_uploads(
    mut uploads: ResMut<TerrainClipmapUploads>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_queue: Res<RenderQueue>,
) {
    for upload in uploads.uploads.drain(..) {
        let Some(gpu_image) = gpu_images.get(&upload.texture) else {
            continue;
        };

        render_queue.write_texture(
            TexelCopyTextureInfo {
                texture: &gpu_image.texture,
                mip_level: 0,
                origin: upload.origin,
                aspect: TextureAspect::All,
            },
            &upload.data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload.bytes_per_row),
                rows_per_image: Some(upload.rows_per_image),
            },
            upload.size,
        );
    }
}

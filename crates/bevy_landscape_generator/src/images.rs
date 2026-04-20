use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
};

pub fn build_generator_image(images: &mut Assets<Image>, resolution: u32) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

pub fn build_normalization_image(images: &mut Assets<Image>, resolution: u32) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 4],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

/// Holds the GPU heightfield preview texture handle (Rgba32Float).
/// Shared between main world and render world via ExtractResource.
#[derive(Resource, Clone, ExtractResource, AsBindGroup)]
pub struct GeneratorImage {
    #[storage_texture(0, image_format = Rgba32Float, access = ReadWrite)]
    pub heightfield: Handle<Image>,
}

/// Intermediate R32Float texture holding one raw height value per texel.
/// Used by the preview normalization pipeline before tone-mapping.
#[derive(Resource, Clone, ExtractResource)]
pub struct NormalizationImage {
    pub raw_heights: Handle<Image>,
}

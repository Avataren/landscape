use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

pub const CLOUD_ATLAS_SIZE: u32 = 512;
pub const CLOUD_WORLEY_SIZE: u32 = 32;

fn storage_image_2d(images: &mut Assets<Image>, resolution: UVec2) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: resolution.x,
            height: resolution.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

fn storage_image_3d(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    let mut image = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        TextureDimension::D3,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    images.add(image)
}

pub fn build_cloud_images(
    images: &mut Assets<Image>,
    resolution: UVec2,
) -> (Handle<Image>, Handle<Image>, Handle<Image>, Handle<Image>) {
    (
        storage_image_2d(images, resolution),
        storage_image_2d(images, resolution),
        storage_image_2d(images, UVec2::splat(CLOUD_ATLAS_SIZE)),
        storage_image_3d(images, CLOUD_WORLEY_SIZE),
    )
}

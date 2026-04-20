use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
};

const EROSION_USAGES: TextureUsages = TextureUsages::from_bits_retain(
    TextureUsages::COPY_SRC.bits()
        | TextureUsages::COPY_DST.bits()
        | TextureUsages::STORAGE_BINDING.bits()
        | TextureUsages::TEXTURE_BINDING.bits(),
);

pub fn build_erosion_r32(images: &mut Assets<Image>, resolution: u32) -> Handle<Image> {
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
    image.texture_descriptor.usage = EROSION_USAGES;
    images.add(image)
}

pub fn build_erosion_rgba32(images: &mut Assets<Image>, resolution: u32) -> Handle<Image> {
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
    image.texture_descriptor.usage = EROSION_USAGES;
    images.add(image)
}

/// All GPU erosion simulation textures. Rebuilt when resolution changes.
#[derive(Resource, Clone, ExtractResource)]
pub struct ErosionBuffers {
    pub resolution: u32,
    /// Primary height buffer (ping).
    pub height_a: Handle<Image>,
    /// Scratch height buffer (pong / sediment swap target).
    pub height_b: Handle<Image>,
    /// Water column depth.
    pub water: Handle<Image>,
    /// Suspended sediment.
    pub sediment: Handle<Image>,
    /// Virtual pipe outflow flux (L, R, T, B).
    pub flux: Handle<Image>,
    /// 2-D velocity (vx, vy stored in R, G; BA unused).
    pub velocity: Handle<Image>,
    /// Per-cell erosion resistance (seeded from noise).
    pub hardness: Handle<Image>,
}

impl ErosionBuffers {
    pub fn new(images: &mut Assets<Image>, resolution: u32) -> Self {
        Self {
            resolution,
            height_a: build_erosion_r32(images, resolution),
            height_b: build_erosion_r32(images, resolution),
            water: build_erosion_r32(images, resolution),
            sediment: build_erosion_r32(images, resolution),
            flux: build_erosion_rgba32(images, resolution),
            velocity: build_erosion_rgba32(images, resolution),
            hardness: build_erosion_r32(images, resolution),
        }
    }
}

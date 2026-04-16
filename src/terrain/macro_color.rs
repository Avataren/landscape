use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use exr::prelude::read_first_rgba_layer_from_file;
use std::path::Path;

use crate::terrain::{config::TerrainConfig, world_desc::TerrainSourceDesc};

struct MacroColorAccumulator {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sums: Vec<[f32; 4]>,
    counts: Vec<u32>,
}

pub struct MacroColorLoadResult {
    pub image: Image,
    pub enabled: bool,
}

fn make_macro_color_image(width: u32, height: u32, data: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn fallback_macro_color_image() -> Image {
    make_macro_color_image(1, 1, vec![255, 255, 255, 255])
}

fn downsample_dimensions(
    src_width: usize,
    src_height: usize,
    max_resolution: u32,
) -> (usize, usize) {
    let max_resolution = max_resolution.max(1) as usize;
    let longest_edge = src_width.max(src_height);
    if longest_edge <= max_resolution {
        return (src_width, src_height);
    }

    let scale = longest_edge.div_ceil(max_resolution);
    (src_width.div_ceil(scale), src_height.div_ceil(scale))
}

fn load_macro_color_exr(path: &Path, max_resolution: u32) -> Result<Image, String> {
    let image = read_first_rgba_layer_from_file(
        path,
        move |resolution, _channels| {
            let src_width = resolution.width();
            let src_height = resolution.height();
            let (dst_width, dst_height) =
                downsample_dimensions(src_width, src_height, max_resolution);

            MacroColorAccumulator {
                src_width,
                src_height,
                dst_width,
                dst_height,
                sums: vec![[0.0; 4]; dst_width * dst_height],
                counts: vec![0; dst_width * dst_height],
            }
        },
        |acc: &mut MacroColorAccumulator, position, (r, g, b, a): (f32, f32, f32, f32)| {
            let dst_x = (position.x() * acc.dst_width / acc.src_width).min(acc.dst_width - 1);
            let dst_y = (position.y() * acc.dst_height / acc.src_height).min(acc.dst_height - 1);
            let idx = dst_y * acc.dst_width + dst_x;
            acc.sums[idx][0] += r;
            acc.sums[idx][1] += g;
            acc.sums[idx][2] += b;
            acc.sums[idx][3] += a;
            acc.counts[idx] += 1;
        },
    )
    .map_err(|err| format!("{err}"))?;

    let acc = image.layer_data.channel_data.pixels;
    let mut bytes = Vec::with_capacity(acc.dst_width * acc.dst_height * 4);

    for (sum, count) in acc.sums.iter().zip(acc.counts.iter()) {
        let inv_count = if *count > 0 { 1.0 / *count as f32 } else { 0.0 };
        bytes.push((sum[0].mul_add(inv_count, 0.0).clamp(0.0, 1.0) * 255.0).round() as u8);
        bytes.push((sum[1].mul_add(inv_count, 0.0).clamp(0.0, 1.0) * 255.0).round() as u8);
        bytes.push((sum[2].mul_add(inv_count, 0.0).clamp(0.0, 1.0) * 255.0).round() as u8);
        bytes.push((sum[3].mul_add(inv_count, 1.0).clamp(0.0, 1.0) * 255.0).round() as u8);
    }

    Ok(make_macro_color_image(
        acc.dst_width as u32,
        acc.dst_height as u32,
        bytes,
    ))
}

pub fn load_macro_color_texture(
    config: &TerrainConfig,
    desc: &TerrainSourceDesc,
) -> MacroColorLoadResult {
    if !config.use_macro_color_map {
        return MacroColorLoadResult {
            image: fallback_macro_color_image(),
            enabled: false,
        };
    }

    let Some(path) = desc.macro_color_root.as_deref() else {
        warn!("[Terrain] macro color map enabled but no diffuse EXR path is configured.");
        return MacroColorLoadResult {
            image: fallback_macro_color_image(),
            enabled: false,
        };
    };

    match load_macro_color_exr(Path::new(path), config.macro_color_resolution) {
        Ok(image) => {
            info!(
                "[Terrain] Loaded macro color EXR '{}' (downsampled to <= {} px).",
                path, config.macro_color_resolution
            );
            MacroColorLoadResult {
                image,
                enabled: true,
            }
        }
        Err(err) => {
            warn!(
                "[Terrain] failed to load macro color EXR '{}': {}",
                path, err
            );
            MacroColorLoadResult {
                image: fallback_macro_color_image(),
                enabled: false,
            }
        }
    }
}

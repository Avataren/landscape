use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use exr::prelude::read_first_rgba_layer_from_file;
use std::path::{Path, PathBuf};

use crate::terrain::material::MAX_SHADER_MATERIAL_SLOTS;
use crate::terrain::material_slots::MaterialSlot;

pub const PBR_TEX_RESOLUTION: u32 = 1024;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn make_pbr_array(data: Vec<u8>) -> Image {
    let layers = MAX_SHADER_MATERIAL_SLOTS as u32;
    let mut image = Image::new(
        Extent3d {
            width: PBR_TEX_RESOLUTION,
            height: PBR_TEX_RESOLUTION,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn fill_layer(fill: [u8; 4]) -> Vec<u8> {
    let n = (PBR_TEX_RESOLUTION * PBR_TEX_RESOLUTION) as usize;
    fill.repeat(n)
}

/// Load a raster image (JPG, PNG, …) and return RGBA bytes at `PBR_TEX_RESOLUTION`.
fn load_raster_rgba(path: &Path) -> Option<Vec<u8>> {
    let img = image::open(path)
        .map_err(|e| warn!("[PBR] failed to open '{}': {e}", path.display()))
        .ok()?;
    let img = img.resize_exact(
        PBR_TEX_RESOLUTION,
        PBR_TEX_RESOLUTION,
        image::imageops::FilterType::Lanczos3,
    );
    Some(img.to_rgba8().into_raw())
}

struct ExrRgbaAccumulator {
    src_w: usize,
    src_h: usize,
    data: Vec<[f32; 4]>,
}

/// Load an EXR and return RGBA bytes (float → u8) at `PBR_TEX_RESOLUTION`.
/// The EXR components are clamped to [0, 1] and encoded as `round(v * 255)`.
fn load_exr_rgba(path: &Path) -> Option<Vec<u8>> {
    let res = PBR_TEX_RESOLUTION as usize;

    let result = read_first_rgba_layer_from_file(
        path,
        move |resolution, _| ExrRgbaAccumulator {
            src_w: resolution.width(),
            src_h: resolution.height(),
            data: vec![[0.0; 4]; res * res],
        },
        move |acc: &mut ExrRgbaAccumulator, pos, (r, g, b, a): (f32, f32, f32, f32)| {
            let dx = (pos.x() * res / acc.src_w).min(res - 1);
            let dy = (pos.y() * res / acc.src_h).min(res - 1);
            acc.data[dy * res + dx] = [r, g, b, a];
        },
    )
    .map_err(|e| warn!("[PBR] failed to read EXR '{}': {e}", path.display()))
    .ok()?;

    let pixels = result.layer_data.channel_data.pixels;
    let bytes: Vec<u8> = pixels
        .data
        .iter()
        .flat_map(|[r, g, b, a]| {
            [
                (r.clamp(0.0, 1.0) * 255.0).round() as u8,
                (g.clamp(0.0, 1.0) * 255.0).round() as u8,
                (b.clamp(0.0, 1.0) * 255.0).round() as u8,
                (a.clamp(0.0, 1.0) * 255.0).round() as u8,
            ]
        })
        .collect();
    Some(bytes)
}

/// Load an EXR normal map and return Rgba8Unorm bytes with standard encoding:
///   R = nx*0.5+0.5, G = ny*0.5+0.5, B = nz*0.5+0.5, A = 255
fn load_exr_normal(path: &Path) -> Option<Vec<u8>> {
    let res = PBR_TEX_RESOLUTION as usize;

    let result = read_first_rgba_layer_from_file(
        path,
        move |resolution, _| ExrRgbaAccumulator {
            src_w: resolution.width(),
            src_h: resolution.height(),
            data: vec![[0.0; 4]; res * res],
        },
        move |acc: &mut ExrRgbaAccumulator, pos, (r, g, b, a): (f32, f32, f32, f32)| {
            let dx = (pos.x() * res / acc.src_w).min(res - 1);
            let dy = (pos.y() * res / acc.src_h).min(res - 1);
            acc.data[dy * res + dx] = [r, g, b, a];
        },
    )
    .map_err(|e| warn!("[PBR] failed to read normal EXR '{}': {e}", path.display()))
    .ok()?;

    let pixels = result.layer_data.channel_data.pixels;
    let bytes: Vec<u8> = pixels
        .data
        .iter()
        .flat_map(|[r, g, b, _a]| {
            [
                ((r * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8,
                ((g * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8,
                ((b * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8,
                255u8,
            ]
        })
        .collect();
    Some(bytes)
}

fn resolve(assets_dir: &Path, rel: &Path) -> PathBuf {
    assets_dir.join(rel)
}

// ---------------------------------------------------------------------------
// Public array builders
// ---------------------------------------------------------------------------

/// Build a texture_2d_array with one albedo layer per `MaterialSlot`.
/// Slots without an `albedo_path` get a neutral mid-grey layer.
pub fn build_albedo_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data =
        Vec::with_capacity(MAX_SHADER_MATERIAL_SLOTS * (PBR_TEX_RESOLUTION as usize).pow(2) * 4);

    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let layer = slots.get(i).and_then(|s| s.albedo_path.as_deref()).and_then(|rel| {
            load_raster_rgba(&resolve(assets_dir, rel))
        });
        data.extend(layer.unwrap_or_else(|| fill_layer([128, 128, 128, 255])));
    }

    make_pbr_array(data)
}

/// Build a texture_2d_array with one normal-map layer per `MaterialSlot`.
/// EXR normals are encoded as Rgba8Unorm (nx,ny,nz → R,G,B in [0,1]).
/// Slots without a `normal_path` get a flat normal `[128,128,255,255]`.
pub fn build_normal_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data =
        Vec::with_capacity(MAX_SHADER_MATERIAL_SLOTS * (PBR_TEX_RESOLUTION as usize).pow(2) * 4);

    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let layer = slots.get(i).and_then(|s| s.normal_path.as_deref()).and_then(|rel| {
            let path = resolve(assets_dir, rel);
            if path.extension().and_then(|e| e.to_str()) == Some("exr") {
                load_exr_normal(&path)
            } else {
                load_raster_rgba(&path)
            }
        });
        data.extend(layer.unwrap_or_else(|| fill_layer([128, 128, 255, 255])));
    }

    make_pbr_array(data)
}

/// Build a texture_2d_array with one ORM layer per `MaterialSlot`.
/// Roughness is stored in the G channel (R=255 AO, B=0 metallic, A=255).
/// Slots without an `orm_path` get `[255,128,0,255]` (neutral roughness).
pub fn build_orm_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data =
        Vec::with_capacity(MAX_SHADER_MATERIAL_SLOTS * (PBR_TEX_RESOLUTION as usize).pow(2) * 4);

    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let layer = slots.get(i).and_then(|s| s.orm_path.as_deref()).and_then(|rel| {
            let path = resolve(assets_dir, rel);
            let raw = if path.extension().and_then(|e| e.to_str()) == Some("exr") {
                load_exr_rgba(&path)
            } else {
                load_raster_rgba(&path)
            }?;
            // Re-pack: roughness is in R (grayscale source) or G (already ORM).
            // For Polyhaven roughness-only maps (grayscale JPG/EXR), R=G=B.
            // Store roughness in G; set R=255 (AO), B=0 (non-metallic).
            let packed: Vec<u8> = raw
                .chunks_exact(4)
                .flat_map(|p| [255u8, p[0], 0u8, 255u8])
                .collect();
            Some(packed)
        });
        data.extend(layer.unwrap_or_else(|| fill_layer([255, 128, 0, 255])));
    }

    make_pbr_array(data)
}

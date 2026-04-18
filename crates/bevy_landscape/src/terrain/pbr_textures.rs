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

fn mip_levels_for(res: u32) -> u32 {
    (res as f32).log2() as u32 + 1
}

// ---------------------------------------------------------------------------
// Mipmap generation
// ---------------------------------------------------------------------------

/// Generate the full mip chain for one RGBA layer.
/// mip0, mip1, …, mip_n are concatenated — wgpu expects this (LayerMajor default).
fn layer_mip_chain(base: &[u8], res: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(base.len() * 4 / 3 + 4);
    out.extend_from_slice(base);

    let mut cur = base.to_vec();
    let mut cur_res = res;
    while cur_res > 1 {
        let next_res = (cur_res / 2).max(1);
        let img = image::RgbaImage::from_raw(cur_res, cur_res, cur)
            .expect("mip generation: invalid dimensions");
        let small = image::imageops::resize(
            &img,
            next_res,
            next_res,
            image::imageops::FilterType::Triangle,
        );
        cur = small.into_raw();
        out.extend_from_slice(&cur);
        cur_res = next_res;
    }
    out
}


// ---------------------------------------------------------------------------
// Texture upload helpers
// ---------------------------------------------------------------------------

fn make_pbr_array(data: Vec<u8>) -> Image {
    let layers = MAX_SHADER_MATERIAL_SLOTS as u32;
    let mips = mip_levels_for(PBR_TEX_RESOLUTION);

    let mut image = Image::new(
        Extent3d {
            width: PBR_TEX_RESOLUTION,
            height: PBR_TEX_RESOLUTION,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        // RENDER_WORLD only — no CPU copy needed after upload.
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = mips;
    image.sampler = ImageSampler::Descriptor({
        let mut desc = ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..default()
        };
        desc.set_anisotropic_filter(8);
        desc
    });
    image
}

// ---------------------------------------------------------------------------
// File loaders
// ---------------------------------------------------------------------------

fn fill_layer(fill: [u8; 4]) -> Vec<u8> {
    let n = (PBR_TEX_RESOLUTION * PBR_TEX_RESOLUTION) as usize;
    fill.repeat(n)
}

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

struct ExrRgbaAcc {
    src_w: usize,
    src_h: usize,
    data:  Vec<[f32; 4]>,
}

fn load_exr_pixels(path: &Path, label: &str) -> Option<Vec<[f32; 4]>> {
    let res = PBR_TEX_RESOLUTION as usize;
    let result = read_first_rgba_layer_from_file(
        path,
        move |resolution, _| ExrRgbaAcc {
            src_w: resolution.width(),
            src_h: resolution.height(),
            data:  vec![[0.0; 4]; res * res],
        },
        move |acc: &mut ExrRgbaAcc, pos, (r, g, b, a): (f32, f32, f32, f32)| {
            let dx = (pos.x() * res / acc.src_w).min(res - 1);
            let dy = (pos.y() * res / acc.src_h).min(res - 1);
            acc.data[dy * res + dx] = [r, g, b, a];
        },
    )
    .map_err(|e| warn!("[PBR] failed to read {} EXR '{}': {e}", label, path.display()))
    .ok()?;
    Some(result.layer_data.channel_data.pixels.data)
}

fn f(v: f32) -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 }

fn load_exr_rgba(path: &Path) -> Option<Vec<u8>> {
    let pixels = load_exr_pixels(path, "color")?;
    Some(pixels.iter().flat_map(|[r, g, b, a]| [f(*r), f(*g), f(*b), f(*a)]).collect())
}

fn load_exr_normal(path: &Path) -> Option<Vec<u8>> {
    let pixels = load_exr_pixels(path, "normal")?;
    // Polyhaven EXR normals are already in [0, 1] (same encoding as PNG normal maps).
    // Store directly — the shader decodes with `sample * 2 - 1`.
    Some(pixels.iter().flat_map(|[r, g, b, _a]| [f(*r), f(*g), f(*b), 255u8]).collect())
}

fn resolve(assets_dir: &Path, rel: &Path) -> PathBuf {
    assets_dir.join(rel)
}

// ---------------------------------------------------------------------------
// Public array builders
// ---------------------------------------------------------------------------

pub fn build_albedo_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data = Vec::new();
    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let base = slots
            .get(i)
            .and_then(|s| s.albedo_path.as_deref())
            .and_then(|rel| load_raster_rgba(&resolve(assets_dir, rel)))
            .unwrap_or_else(|| fill_layer([128, 128, 128, 255]));
        data.extend(layer_mip_chain(&base, PBR_TEX_RESOLUTION));
    }
    make_pbr_array(data)
}

pub fn build_normal_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data = Vec::new();
    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let base = slots
            .get(i)
            .and_then(|s| s.normal_path.as_deref())
            .and_then(|rel| {
                let path = resolve(assets_dir, rel);
                if path.extension().and_then(|e| e.to_str()) == Some("exr") {
                    load_exr_normal(&path)
                } else {
                    load_raster_rgba(&path)
                }
            })
            .unwrap_or_else(|| fill_layer([128, 128, 255, 255]));
        data.extend(layer_mip_chain(&base, PBR_TEX_RESOLUTION));
    }
    make_pbr_array(data)
}

pub fn build_orm_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data = Vec::new();
    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let base = slots
            .get(i)
            .and_then(|s| s.orm_path.as_deref())
            .and_then(|rel| {
                let path = resolve(assets_dir, rel);
                let raw = if path.extension().and_then(|e| e.to_str()) == Some("exr") {
                    load_exr_rgba(&path)
                } else {
                    load_raster_rgba(&path)
                }?;
                // Roughness source is grayscale (R=G=B). Store in G channel:
                // R=255 (full AO), G=roughness, B=0 (non-metallic), A=255.
                let packed: Vec<u8> = raw
                    .chunks_exact(4)
                    .flat_map(|p| [255u8, p[0], 0u8, 255u8])
                    .collect();
                Some(packed)
            })
            .unwrap_or_else(|| fill_layer([255, 128, 0, 255]));
        data.extend(layer_mip_chain(&base, PBR_TEX_RESOLUTION));
    }
    make_pbr_array(data)
}


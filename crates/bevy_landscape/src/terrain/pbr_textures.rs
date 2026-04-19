use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    mpsc, Arc, Mutex,
};

use crate::terrain::material::{TerrainMaterial, MAX_SHADER_MATERIAL_SLOTS};
use crate::terrain::material_slots::{MaterialLibrary, MaterialSlot};
use crate::terrain::PatchEntities;

pub const PBR_TEX_RESOLUTION: u32 = 1024;

fn mip_levels_for(res: u32) -> u32 {
    (res as f32).log2() as u32 + 1
}

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

fn fill_layer(fill: [u8; 4]) -> Vec<u8> {
    let n = (PBR_TEX_RESOLUTION * PBR_TEX_RESOLUTION) as usize;
    fill.repeat(n)
}

fn resolve(assets_dir: &Path, rel: &Path) -> PathBuf {
    assets_dir.join(rel)
}

fn clamp_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Load an EXR file by reading every source pixel into a full-resolution RGBA8
/// buffer (converting f32→u8 in the callback), then Lanczos3-resize to target.
///
/// The previous approach did `pos.x() * 1024 / 4096` in integer arithmetic,
/// writing only 1 in 4 destination pixels; the other 3 stayed at [0,0,0,0].
/// Those zeros decode in the shader as ts_xy=(-1,-1) and mip-averaging them
/// collapses all normal detail to near-flat.
fn load_exr(path: &Path) -> Option<Vec<u8>> {
    use exr::prelude::read_first_rgba_layer_from_file;

    struct Acc {
        w: usize,
        h: usize,
        data: Vec<u8>,
    }

    let result = read_first_rgba_layer_from_file(
        path,
        |res, _| Acc {
            w: res.width(),
            h: res.height(),
            data: vec![128u8; res.width() * res.height() * 4],
        },
        |acc: &mut Acc, pos, (r, g, b, _a): (f32, f32, f32, f32)| {
            let idx = (pos.y() * acc.w + pos.x()) * 4;
            acc.data[idx] = clamp_u8(r);
            acc.data[idx + 1] = clamp_u8(g);
            acc.data[idx + 2] = clamp_u8(b);
            acc.data[idx + 3] = 255;
        },
    )
    .map_err(|e| warn!("[PBR] failed to read EXR '{}': {e}", path.display()))
    .ok()?;

    let acc = result.layer_data.channel_data.pixels;
    let img = image::RgbaImage::from_raw(acc.w as u32, acc.h as u32, acc.data)?;
    let out = image::imageops::resize(
        &img,
        PBR_TEX_RESOLUTION,
        PBR_TEX_RESOLUTION,
        image::imageops::FilterType::Lanczos3,
    );
    info!(
        "[PBR] loaded EXR '{}' ({}×{})",
        path.display(),
        acc.w,
        acc.h
    );
    Some(out.into_raw())
}

/// Load any raster format (jpg, png, …) supported by the `image` crate.
fn load_raster(path: &Path) -> Option<Vec<u8>> {
    let img = image::open(path)
        .map_err(|e| warn!("[PBR] failed to open '{}': {e}", path.display()))
        .ok()?;
    let img = img.resize_exact(
        PBR_TEX_RESOLUTION,
        PBR_TEX_RESOLUTION,
        image::imageops::FilterType::Lanczos3,
    );
    info!(
        "[PBR] loaded '{}' ({}×{})",
        path.display(),
        img.width(),
        img.height()
    );
    Some(img.to_rgba8().into_raw())
}

fn load_any(path: &Path) -> Option<Vec<u8>> {
    if path.extension().and_then(|e| e.to_str()) == Some("exr") {
        load_exr(path)
    } else {
        load_raster(path)
    }
}

pub fn build_albedo_array(slots: &[MaterialSlot], assets_dir: &Path) -> Image {
    let mut data = Vec::new();
    for i in 0..MAX_SHADER_MATERIAL_SLOTS {
        let base = slots
            .get(i)
            .and_then(|s| s.albedo_path.as_deref())
            .and_then(|rel| load_any(&resolve(assets_dir, rel)))
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
            .and_then(|rel| load_any(&resolve(assets_dir, rel)))
            .unwrap_or_else(|| {
                warn!("[PBR] using flat normal placeholder for layer {i}");
                fill_layer([128, 128, 255, 255])
            });
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
                let raw = load_any(&resolve(assets_dir, rel))?;
                // Roughness source is greyscale; pack into ORM:
                // R=255 (full AO), G=roughness, B=0 (non-metallic), A=255.
                Some(
                    raw.chunks_exact(4)
                        .flat_map(|p| [255u8, p[0], 0u8, 255u8])
                        .collect(),
                )
            })
            .unwrap_or_else(|| fill_layer([255, 128, 0, 255]));
        data.extend(layer_mip_chain(&base, PBR_TEX_RESOLUTION));
    }
    make_pbr_array(data)
}

// ─── Hot-reload support ───────────────────────────────────────────────────────

const REBUILD_STEPS: u32 = 3; // albedo, normal, ORM

/// Set this flag to `true` from any system to trigger a background PBR texture
/// rebuild.  The terrain plugin reads it and starts a background thread; the
/// flag is cleared once the rebuild is submitted.
#[derive(Resource, Default)]
pub struct PbrTexturesDirty(pub bool);

/// Live progress of an in-flight PBR texture rebuild, readable by the editor UI.
#[derive(Resource, Default)]
pub struct PbrRebuildProgress {
    pub active: bool,
    /// 0.0 = not started, 1.0 = complete.
    pub fraction: f32,
}

struct PbrRebuildResult {
    albedo: Image,
    normal: Image,
    orm: Image,
}

#[derive(Resource, Default)]
pub(crate) struct PbrRebuildState {
    rx: Option<Mutex<mpsc::Receiver<PbrRebuildResult>>>,
    /// Incremented by the background thread after each array is built.
    counter: Option<Arc<AtomicU32>>,
}

pub(crate) fn rebuild_pbr_textures_system(
    mut dirty: ResMut<PbrTexturesDirty>,
    library: Res<MaterialLibrary>,
    patch_entities: Res<PatchEntities>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut state: ResMut<PbrRebuildState>,
    mut progress: ResMut<PbrRebuildProgress>,
) {
    // Update progress from the shared atomic counter.
    if let Some(ref counter) = state.counter {
        let done = counter.load(Ordering::Relaxed);
        progress.active = true;
        progress.fraction = done as f32 / REBUILD_STEPS as f32;
    }

    // Poll for a completed background rebuild.
    let old_rx = state.rx.take();
    if let Some(rx_mutex) = old_rx {
        let received = rx_mutex.lock().ok().and_then(|g| g.try_recv().ok());
        match received {
            Some(result) => {
                if let Some(mat) = materials.get_mut(&patch_entities.material_handle) {
                    mat.pbr_albedo_array = images.add(result.albedo);
                    mat.pbr_normal_array = images.add(result.normal);
                    mat.pbr_orm_array = images.add(result.orm);
                }
                state.counter = None;
                progress.active = false;
                progress.fraction = 0.0;
                info!("[PBR] Texture arrays hot-reloaded.");
            }
            None => {
                // Not ready yet — put it back.
                state.rx = Some(rx_mutex);
            }
        }
    }

    // Start a rebuild if requested and no rebuild is already running.
    if dirty.0 && state.rx.is_none() {
        dirty.0 = false;
        let slots = library.slots.clone();
        let (tx, rx) = mpsc::channel();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_bg = counter.clone();
        std::thread::spawn(move || {
            let assets_dir = std::path::Path::new("assets");
            let albedo = build_albedo_array(&slots, assets_dir);
            counter_bg.fetch_add(1, Ordering::Relaxed);
            let normal = build_normal_array(&slots, assets_dir);
            counter_bg.fetch_add(1, Ordering::Relaxed);
            let orm = build_orm_array(&slots, assets_dir);
            counter_bg.fetch_add(1, Ordering::Relaxed);
            let _ = tx.send(PbrRebuildResult {
                albedo,
                normal,
                orm,
            });
        });
        state.rx = Some(Mutex::new(rx));
        state.counter = Some(counter);
        progress.active = true;
        progress.fraction = 0.0;
        info!("[PBR] Background texture rebuild started.");
    }
}

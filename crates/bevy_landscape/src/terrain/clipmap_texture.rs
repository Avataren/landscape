use crate::terrain::{
    config::{TerrainConfig, MAX_SUPPORTED_CLIPMAP_LEVELS},
    material::TerrainMaterial,
    math::level_scale,
    resources::{HeightTileCpu, TerrainResidency, TerrainViewState, TileKey, TileState},
    world_desc::TerrainSourceDesc,
};
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Extent3d, Origin3d, TextureDimension, TextureFormat, TextureUsages},
    },
};
// ---------------------------------------------------------------------------
// Layer generation  (R32Float — world-space metres)
// ---------------------------------------------------------------------------

/// Bytes per texel for our height texture format.
pub const HEIGHT_BYTES_PER_TEXEL: usize = 4; // R32Float
pub const NORMAL_BYTES_PER_TEXEL: usize = 4; // RGBA8Snorm: RG=fine XZ, BA=coarse XZ

// ---------------------------------------------------------------------------
// Texture array creation
// ---------------------------------------------------------------------------

/// Bytes consumed by one full clipmap layer.
fn bytes_per_layer(res: u32, bytes_per_texel: usize) -> usize {
    (res * res) as usize * bytes_per_texel
}

/// Builds the initial clipmap texture array with all levels generated at
/// (0, 0) clip centres.  Every frame after startup `update_clipmap_textures`
/// refines individual layers when clip centres change.
pub fn create_initial_clipmap_texture(config: &TerrainConfig) -> Image {
    let res = config.clipmap_resolution();
    let active_levels = config.active_clipmap_levels();
    // Always allocate the full maximum number of layers so the GPU texture array
    // never needs to be recreated when hot-reloading terrains with different mip
    // level counts.  Layers beyond active_levels stay zero-filled.
    let max_layers = MAX_SUPPORTED_CLIPMAP_LEVELS as u32;
    let bpl = bytes_per_layer(res, HEIGHT_BYTES_PER_TEXEL);
    let data = vec![0u8; bpl * max_layers as usize];
    let _ = active_levels;

    let mut image = Image::new(
        Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: max_layers,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Float,
        // Keep the CPU copy so `update_clipmap_textures` can patch layers.
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    // STORAGE_BINDING: compute pass writes full height to fine LOD layers.
    // COPY_DST: CPU tile uploads via queue.write_texture().
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

pub fn create_initial_normal_clipmap_texture(config: &TerrainConfig) -> Image {
    let res = config.clipmap_resolution();
    let active_levels = config.active_clipmap_levels();
    let max_layers = MAX_SUPPORTED_CLIPMAP_LEVELS as u32;
    let bpl = bytes_per_layer(res, NORMAL_BYTES_PER_TEXEL);
    let data = vec![0u8; bpl * max_layers as usize];
    let _ = active_levels;

    let mut image = Image::new(
        Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: max_layers,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Snorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

// ---------------------------------------------------------------------------
// Uniform helper
// ---------------------------------------------------------------------------

/// Computes `clip_levels` for `TerrainMaterialUniforms` from a view state.
///
/// Layout per entry: `(ring_center_x, ring_center_z, inv_ring_span, texel_world_size)`.
///
/// `.xy` stores the ring center in world space (used for morph-alpha distance).
/// `.z`  = 1/ring_span, used for toroidal UV: `uv = fract(world_xz * inv_span)`.
/// `.w`  = texel world size (used for finite-difference normal epsilon).
pub fn compute_clip_levels(
    config: &TerrainConfig,
    clip_centers: &[IVec2],
    level_scales: &[f32],
) -> [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS] {
    let mut levels = [Vec4::ZERO; MAX_SUPPORTED_CLIPMAP_LEVELS];

    for lod in 0..config.active_clipmap_levels() as usize {
        let center = clip_centers.get(lod).copied().unwrap_or(IVec2::ZERO);
        let scale = level_scales
            .get(lod)
            .copied()
            .unwrap_or_else(|| level_scale(config.lod0_mesh_spacing, lod as u32));

        let ring_span = config.clipmap_resolution() as f32 * scale;
        let inv_span = 1.0 / ring_span;
        let texel_ws = ring_span / config.clipmap_resolution() as f32;
        // Ring center: world-space position the clip center corresponds to.
        let ring_center_x = center.x as f32 * scale;
        let ring_center_z = center.y as f32 * scale;

        levels[lod] = Vec4::new(ring_center_x, ring_center_z, inv_span, texel_ws);
    }

    levels
}

/// Computes `clip_levels` when all clip centres are at the origin.
/// Used during startup before the first `update_terrain_view_state` runs.
pub fn compute_initial_clip_levels(config: &TerrainConfig) -> [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS] {
    let zeros: Vec<IVec2> = vec![IVec2::ZERO; config.active_clipmap_levels() as usize];
    let scales: Vec<f32> = (0..config.active_clipmap_levels())
        .map(|l| level_scale(config.lod0_mesh_spacing, l))
        .collect();
    compute_clip_levels(config, &zeros, &scales)
}

// ---------------------------------------------------------------------------
// Runtime state resource
// ---------------------------------------------------------------------------

/// Tracks the live clipmap texture array and material so
/// `update_clipmap_textures` can patch them cheaply.
#[derive(Resource)]
pub struct TerrainClipmapState {
    pub height_texture_handle: Handle<Image>,
    pub normal_texture_handle: Handle<Image>,
    pub material_handle: Handle<TerrainMaterial>,
    pub height_cpu_data: Vec<u8>,
    pub normal_cpu_data: Vec<u8>,
    /// Clip centres from the last procedural texture regeneration.
    pub last_clip_centers: Vec<IVec2>,
    /// Clip centres at which resident tiles were last written into the texture.
    /// Sentinel IVec2::MAX on startup forces a full tile re-apply on the first frame.
    pub tile_apply_centers: Vec<IVec2>,
    /// Incremented whenever height_cpu_data is written. Used by update_patch_aabbs
    /// to skip the per-frame AABB scan when no new height data has arrived.
    pub height_generation: u64,
}

impl TerrainClipmapState {
    /// Sample world-space height from the finest LOD that covers `world_xz`.
    ///
    /// Uses nearest-neighbour lookup into the CPU mirror of the R32Float clipmap
    /// array. Returns 0 if the clipmap buffer is not yet populated.
    pub fn sample_height_cpu(
        &self,
        world_xz: Vec2,
        clip_centers: &[IVec2],
        level_scales: &[f32],
        res: u32,
    ) -> f32 {
        let bpl = (res as usize) * (res as usize) * HEIGHT_BYTES_PER_TEXEL;
        for (lod, (center, &scale)) in clip_centers.iter().zip(level_scales.iter()).enumerate() {
            let layer_offset = lod * bpl;
            if layer_offset + bpl > self.height_cpu_data.len() {
                break;
            }
            let inv_span = 1.0 / (res as f32 * scale);
            let u = (world_xz.x + 0.5 * scale) * inv_span;
            let v = (world_xz.y + 0.5 * scale) * inv_span;
            let u = u.fract();
            let v = v.fract();
            let tx = ((u * res as f32) as u32).min(res - 1);
            let ty = ((v * res as f32) as u32).min(res - 1);
            // Verify this LOD's window actually covers world_xz (else skip to coarser).
            let ring_half = res as f32 * scale * 0.5;
            let cx = center.x as f32 * scale;
            let cz = center.y as f32 * scale;
            if (world_xz.x - cx).abs() > ring_half || (world_xz.y - cz).abs() > ring_half {
                continue;
            }
            let byte_idx =
                layer_offset + (ty as usize * res as usize + tx as usize) * HEIGHT_BYTES_PER_TEXEL;
            if byte_idx + HEIGHT_BYTES_PER_TEXEL > self.height_cpu_data.len() {
                break;
            }
            // R32Float: values stored directly in metres.
            return f32::from_le_bytes([
                self.height_cpu_data[byte_idx],
                self.height_cpu_data[byte_idx + 1],
                self.height_cpu_data[byte_idx + 2],
                self.height_cpu_data[byte_idx + 3],
            ]);
        }
        0.0
    }
}

#[derive(Clone, Debug)]
pub struct TerrainTextureUpload {
    pub texture: Handle<Image>,
    pub origin: Origin3d,
    pub size: Extent3d,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    pub data: Vec<u8>,
}

#[derive(Resource, Clone, Default, ExtractResource)]
pub struct TerrainClipmapUploads {
    pub uploads: Vec<TerrainTextureUpload>,
}

pub fn begin_terrain_upload_frame(mut uploads: ResMut<TerrainClipmapUploads>) {
    uploads.uploads.clear();
}

fn queue_texture_rect_upload(
    uploads: &mut TerrainClipmapUploads,
    texture: &Handle<Image>,
    layer: u32,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    data: Vec<u8>,
) {
    uploads.uploads.push(TerrainTextureUpload {
        texture: texture.clone(),
        origin: Origin3d {
            x: origin_x,
            y: origin_y,
            z: layer,
        },
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        bytes_per_row,
        rows_per_image: height,
        data,
    });
}

fn queue_full_layer_upload(
    uploads: &mut TerrainClipmapUploads,
    texture: &Handle<Image>,
    layer: u32,
    layer_bytes: &[u8],
    res: u32,
    bytes_per_texel: usize,
) {
    queue_texture_rect_upload(
        uploads,
        texture,
        layer,
        0,
        0,
        res,
        res,
        res * bytes_per_texel as u32,
        layer_bytes.to_vec(),
    );
}

fn extract_layer_rect_bytes(
    layer_bytes: &[u8],
    res: u32,
    bytes_per_texel: usize,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let row_bytes = width as usize * bytes_per_texel;
    let mut out = Vec::with_capacity(row_bytes * height as usize);

    for row in 0..height as usize {
        let src = ((origin_y as usize + row) * res as usize + origin_x as usize) * bytes_per_texel;
        out.extend_from_slice(&layer_bytes[src..src + row_bytes]);
    }

    out
}

fn wrapped_segments(start_world: i32, end_world: i32, n: i32) -> [(u32, u32); 2] {
    let len = (end_world - start_world).max(0);
    if len == 0 {
        return [(0, 0), (0, 0)];
    }

    let start = start_world.rem_euclid(n) as u32;
    let first_len = len.min(n - start as i32) as u32;
    let second_len = (len - first_len as i32).max(0) as u32;

    [(start, first_len), (0, second_len)]
}

fn queue_world_rect_upload(
    uploads: &mut TerrainClipmapUploads,
    texture: &Handle<Image>,
    layer: u32,
    layer_bytes: &[u8],
    res: u32,
    bytes_per_texel: usize,
    rect_min: IVec2,
    rect_max: IVec2,
) {
    if rect_min.x >= rect_max.x || rect_min.y >= rect_max.y {
        return;
    }

    let n = res as i32;
    let x_segments = wrapped_segments(rect_min.x, rect_max.x, n);
    let y_segments = wrapped_segments(rect_min.y, rect_max.y, n);

    for (origin_y, height) in y_segments {
        if height == 0 {
            continue;
        }
        for (origin_x, width) in x_segments {
            if width == 0 {
                continue;
            }
            let data = extract_layer_rect_bytes(
                layer_bytes,
                res,
                bytes_per_texel,
                origin_x,
                origin_y,
                width,
                height,
            );
            queue_texture_rect_upload(
                uploads,
                texture,
                layer,
                origin_x,
                origin_y,
                width,
                height,
                width * bytes_per_texel as u32,
                data,
            );
        }
    }
}

fn queue_new_strip_uploads(
    uploads: &mut TerrainClipmapUploads,
    texture: &Handle<Image>,
    layer: u32,
    layer_bytes: &[u8],
    res: u32,
    bytes_per_texel: usize,
    new_center: IVec2,
    old_center: IVec2,
) {
    let n = res as i32;
    let half = (res / 2) as i32;
    let delta = new_center - old_center;

    if delta == IVec2::ZERO {
        return;
    }

    if old_center.x == i32::MAX || delta.x.abs() >= n || delta.y.abs() >= n {
        queue_full_layer_upload(uploads, texture, layer, layer_bytes, res, bytes_per_texel);
        return;
    }

    if delta.x != 0 {
        let (x_lo, x_hi) = if delta.x > 0 {
            (old_center.x + half, new_center.x + half)
        } else {
            (new_center.x - half, old_center.x - half)
        };
        queue_world_rect_upload(
            uploads,
            texture,
            layer,
            layer_bytes,
            res,
            bytes_per_texel,
            IVec2::new(x_lo, new_center.y - half),
            IVec2::new(x_hi, new_center.y + half),
        );
    }

    if delta.y != 0 {
        let (z_lo, z_hi) = if delta.y > 0 {
            (old_center.y + half, new_center.y + half)
        } else {
            (new_center.y - half, old_center.y - half)
        };
        queue_world_rect_upload(
            uploads,
            texture,
            layer,
            layer_bytes,
            res,
            bytes_per_texel,
            IVec2::new(new_center.x - half, z_lo),
            IVec2::new(new_center.x + half, z_hi),
        );
    }
}

fn tile_window_intersection(
    tile: &HeightTileCpu,
    clip_center: IVec2,
    res: u32,
) -> Option<(IVec2, IVec2)> {
    let ts = tile.tile_size as i32;
    let half = (res / 2) as i32;
    let tile_min = IVec2::new(tile.key.x * ts, tile.key.y * ts);
    let tile_max = tile_min + IVec2::splat(ts);
    let rect_min = tile_min.max(clip_center - IVec2::splat(half));
    let rect_max = tile_max.min(clip_center + IVec2::splat(half));

    (rect_min.x < rect_max.x && rect_min.y < rect_max.y).then_some((rect_min, rect_max))
}

fn clip_window_touches_world_bounds(
    clip_center: IVec2,
    res: u32,
    scale: f32,
    world_min: Vec2,
    world_max: Vec2,
) -> bool {
    if !world_max.cmpgt(world_min).all() {
        return false;
    }

    let half = res as f32 * 0.5;
    let min_sample = Vec2::new(
        (clip_center.x as f32 - half + 0.5) * scale,
        (clip_center.y as f32 - half + 0.5) * scale,
    );
    let max_sample = Vec2::new(
        (clip_center.x as f32 + half - 0.5) * scale,
        (clip_center.y as f32 + half - 0.5) * scale,
    );

    min_sample.x < world_min.x
        || min_sample.y < world_min.y
        || max_sample.x >= world_max.x
        || max_sample.y >= world_max.y
}

fn write_height_tile_rect(
    img_data: &mut [u8],
    layer_base: usize,
    res: u32,
    tile: &HeightTileCpu,
    rect_min: IVec2,
    rect_max: IVec2,
    height_scale: f32,
) {
    let ts = tile.tile_size as i32;
    let tile_min = IVec2::new(tile.key.x * ts, tile.key.y * ts);
    let tile_max = tile_min + IVec2::splat(ts);
    let write_min = rect_min.max(tile_min);
    let write_max = rect_max.min(tile_max);

    if write_min.x >= write_max.x || write_min.y >= write_max.y {
        return;
    }

    let n = res as i32;
    for gz in write_min.y..write_max.y {
        let row = (gz - tile_min.y) as usize;
        for gx in write_min.x..write_max.x {
            let col = (gx - tile_min.x) as usize;
            let src = row * tile.tile_size as usize + col;
            let tx = gx.rem_euclid(n) as usize;
            let tz = gz.rem_euclid(n) as usize;
            let dst = layer_base + (tz * res as usize + tx) * HEIGHT_BYTES_PER_TEXEL;

            if dst + HEIGHT_BYTES_PER_TEXEL <= img_data.len() {
                // Tile stores [0,1] normalized height; convert to world-space metres.
                let v: f32 = tile.data[src] * height_scale;
                img_data[dst..dst + HEIGHT_BYTES_PER_TEXEL].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

fn write_normal_tile_rect(
    img_data: &mut [u8],
    layer_base: usize,
    res: u32,
    tile: &HeightTileCpu,
    rect_min: IVec2,
    rect_max: IVec2,
) {
    let ts = tile.tile_size as i32;
    let tile_min = IVec2::new(tile.key.x * ts, tile.key.y * ts);
    let tile_max = tile_min + IVec2::splat(ts);
    let write_min = rect_min.max(tile_min);
    let write_max = rect_max.min(tile_max);

    if write_min.x >= write_max.x || write_min.y >= write_max.y {
        return;
    }

    let n = res as i32;
    for gz in write_min.y..write_max.y {
        let row = (gz - tile_min.y) as usize;
        for gx in write_min.x..write_max.x {
            let col = (gx - tile_min.x) as usize;
            let src = row * tile.tile_size as usize + col;
            let tx = gx.rem_euclid(n) as usize;
            let tz = gz.rem_euclid(n) as usize;
            let dst = layer_base + (tz * res as usize + tx) * NORMAL_BYTES_PER_TEXEL;

            if dst + NORMAL_BYTES_PER_TEXEL <= img_data.len() {
                img_data[dst..dst + NORMAL_BYTES_PER_TEXEL].copy_from_slice(&tile.normal_data[src]);
            }
        }
    }
}

fn write_height_tile_window(
    img_data: &mut [u8],
    layer_base: usize,
    res: u32,
    tile: &HeightTileCpu,
    clip_center: IVec2,
    height_scale: f32,
) {
    let half = (res / 2) as i32;
    write_height_tile_rect(
        img_data,
        layer_base,
        res,
        tile,
        clip_center - IVec2::splat(half),
        clip_center + IVec2::splat(half),
        height_scale,
    );
}

fn write_normal_tile_window(
    img_data: &mut [u8],
    layer_base: usize,
    res: u32,
    tile: &HeightTileCpu,
    clip_center: IVec2,
) {
    let half = (res / 2) as i32;
    write_normal_tile_rect(
        img_data,
        layer_base,
        res,
        tile,
        clip_center - IVec2::splat(half),
        clip_center + IVec2::splat(half),
    );
}

fn write_height_tile_new_strips(
    img_data: &mut [u8],
    layer_base: usize,
    res: u32,
    tile: &HeightTileCpu,
    new_center: IVec2,
    old_center: IVec2,
    height_scale: f32,
) {
    let n = res as i32;
    let half = (res / 2) as i32;
    let delta = new_center - old_center;

    if delta == IVec2::ZERO {
        return;
    }

    if delta.x.abs() >= n || delta.y.abs() >= n || old_center.x == i32::MAX {
        write_height_tile_window(img_data, layer_base, res, tile, new_center, height_scale);
        return;
    }

    if delta.x != 0 {
        let (x_lo, x_hi) = if delta.x > 0 {
            (old_center.x + half, new_center.x + half)
        } else {
            (new_center.x - half, old_center.x - half)
        };
        write_height_tile_rect(
            img_data,
            layer_base,
            res,
            tile,
            IVec2::new(x_lo, new_center.y - half),
            IVec2::new(x_hi, new_center.y + half),
            height_scale,
        );
    }

    if delta.y != 0 {
        let (z_lo, z_hi) = if delta.y > 0 {
            (old_center.y + half, new_center.y + half)
        } else {
            (new_center.y - half, old_center.y - half)
        };
        write_height_tile_rect(
            img_data,
            layer_base,
            res,
            tile,
            IVec2::new(new_center.x - half, z_lo),
            IVec2::new(new_center.x + half, z_hi),
            height_scale,
        );
    }
}

fn write_normal_tile_new_strips(
    img_data: &mut [u8],
    layer_base: usize,
    res: u32,
    tile: &HeightTileCpu,
    new_center: IVec2,
    old_center: IVec2,
) {
    let n = res as i32;
    let half = (res / 2) as i32;
    let delta = new_center - old_center;

    if delta == IVec2::ZERO {
        return;
    }

    if delta.x.abs() >= n || delta.y.abs() >= n || old_center.x == i32::MAX {
        write_normal_tile_window(img_data, layer_base, res, tile, new_center);
        return;
    }

    if delta.x != 0 {
        let (x_lo, x_hi) = if delta.x > 0 {
            (old_center.x + half, new_center.x + half)
        } else {
            (new_center.x - half, old_center.x - half)
        };
        write_normal_tile_rect(
            img_data,
            layer_base,
            res,
            tile,
            IVec2::new(x_lo, new_center.y - half),
            IVec2::new(x_hi, new_center.y + half),
        );
    }

    if delta.y != 0 {
        let (z_lo, z_hi) = if delta.y > 0 {
            (old_center.y + half, new_center.y + half)
        } else {
            (new_center.y - half, old_center.y - half)
        };
        write_normal_tile_rect(
            img_data,
            layer_base,
            res,
            tile,
            IVec2::new(new_center.x - half, z_lo),
            IVec2::new(new_center.x + half, z_hi),
        );
    }
}

// ---------------------------------------------------------------------------
// Update system
// ---------------------------------------------------------------------------

/// Updates clipmap layers incrementally when clip centres change.
///
/// Instead of regenerating the full layer (which zeros all heights and forces
/// a complete tile re-apply), only the newly-exposed strip is written.
/// All other texels keep their existing data — which is already correct
/// because the toroidal UV layout is stable for fixed world positions.
///
/// Runs in `Update` after `update_terrain_view_state`.
pub fn update_clipmap_textures(
    config: Res<TerrainConfig>,
    view: Res<TerrainViewState>,
    state: Option<ResMut<TerrainClipmapState>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    let Some(mut state) = state else { return };
    if view.clip_centers.is_empty() {
        return;
    }

    let levels = config.active_clipmap_levels() as usize;

    // Pad the cached list so index comparisons don't go out of bounds.
    while state.last_clip_centers.len() < levels {
        state.last_clip_centers.push(IVec2::new(i32::MAX, i32::MAX));
    }

    // Fast path: nothing moved to a new grid cell.
    let dirty = (0..levels).any(|i| {
        view.clip_centers.get(i).copied().unwrap_or(IVec2::ZERO) != state.last_clip_centers[i]
    });
    if !dirty {
        return;
    }

    // Refresh the per-level origin uniforms.
    if let Some(mat) = materials.get_mut(&state.material_handle) {
        mat.params.clip_levels =
            compute_clip_levels(&config, &view.clip_centers, &view.level_scales);
    }

    state.last_clip_centers = view.clip_centers.clone();
}

// ---------------------------------------------------------------------------
// Tile upload system
// ---------------------------------------------------------------------------

/// Legacy CPU tile upload path for pre-baked height and normal tiles.
///
/// The normal runtime path is synthesis-only: `update_required_tiles` leaves
/// `required_now` empty, so this system usually just drains no pending work.
/// The code remains available for compatibility with tile-backed experiments.
///
/// Strategy:
///   1. Move any new tiles from `pending_upload` into `resident_cpu` (persistent
///      CPU cache) and mark them `ResidentGpu`.
///   2. If clip centers changed since the last tile write (or new tiles arrived),
///      re-write every cached tile that falls inside the current clipmap window.
///   3. Update `tile_apply_centers` so we skip the work on unchanged frames.
pub fn apply_tiles_to_clipmap(
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    view: Res<TerrainViewState>,
    mut state: Option<ResMut<TerrainClipmapState>>,
    mut uploads: ResMut<TerrainClipmapUploads>,
    mut residency: ResMut<TerrainResidency>,
) {
    if view.clip_centers.is_empty() {
        return;
    }

    // --- Step 1: absorb newly loaded tiles into the persistent CPU cache --------
    let new_tiles = std::mem::take(&mut residency.pending_upload);
    let new_tile_keys: std::collections::HashSet<TileKey> =
        new_tiles.iter().map(|tile| tile.key).collect();
    let has_new = !new_tiles.is_empty();
    for tile in new_tiles {
        residency
            .tiles
            .insert(tile.key, TileState::ResidentGpu { slot: 0 });
        residency.resident_cpu.insert(tile.key, tile);
    }

    let Some(ref mut state) = state else { return };

    // Grow sentinel vec to match level count.
    let levels = config.active_clipmap_levels() as usize;
    while state.tile_apply_centers.len() < levels {
        state
            .tile_apply_centers
            .push(IVec2::new(i32::MAX, i32::MAX));
    }

    // --- Step 2: early-out when nothing changed ---------------------------------
    let centers_changed = (0..levels).any(|i| {
        view.clip_centers.get(i).copied().unwrap_or(IVec2::ZERO) != state.tile_apply_centers[i]
    });
    let needs_rebuild = residency.clipmap_needs_rebuild;
    if !has_new && !centers_changed && !needs_rebuild {
        return;
    }

    // --- Step 3: write tiles into the GPU texture ------------------------------
    let res = config.clipmap_resolution();
    let height_bpl = bytes_per_layer(res, HEIGHT_BYTES_PER_TEXEL);
    let normal_bpl = bytes_per_layer(res, NORMAL_BYTES_PER_TEXEL);
    // Incremental strip updates assume existing texels remain valid after a pure
    // translation. That breaks once a clip window grazes terrain bounds: edge
    // texels can legitimately hold out-of-bounds zeros, and after the toroidal
    // window scrolls those stale zeros can drift into the interior. Rebuild any
    // bound-touching level from resident tiles instead of strip-patching it.
    let force_full_levels: std::collections::HashSet<usize> = if centers_changed {
        (0..levels)
            .filter(|&lod| {
                let center = view.clip_centers.get(lod).copied().unwrap_or(IVec2::ZERO);
                let scale = view
                    .level_scales
                    .get(lod)
                    .copied()
                    .unwrap_or_else(|| level_scale(config.lod0_mesh_spacing, lod as u32));
                clip_window_touches_world_bounds(center, res, scale, desc.world_min, desc.world_max)
            })
            .collect()
    } else {
        std::collections::HashSet::new()
    };
    let mut dirty_levels = std::collections::HashSet::new();
    // If eviction removed cached tiles, stale texels can remain in the clipmap.
    // Rebuild each layer from fallback once, then re-apply resident tiles.
    if needs_rebuild {
        for lod in 0..levels {
            let height_layer_offset = lod * height_bpl;
            if let Some(slice) = state
                .height_cpu_data
                .get_mut(height_layer_offset..height_layer_offset + height_bpl)
            {
                slice.fill(0);
            }

            let normal_layer_offset = lod * normal_bpl;
            if let Some(slice) = state
                .normal_cpu_data
                .get_mut(normal_layer_offset..normal_layer_offset + normal_bpl)
            {
                slice.fill(0);
            }

            dirty_levels.insert(lod);
        }
        residency.clipmap_needs_rebuild = false;
    }

    // Avoid rewriting every resident tile when only a clip-center strip moved.
    // Existing texels stay valid in the toroidal layout; only newly exposed
    // texels need resident tile data stamped back in.
    //
    // When centers moved but no rebuild is needed, pre-filter resident tiles to
    // only those whose LOD level actually had a center change — avoids iterating
    // all 256 tiles when only coarse levels scrolled.
    let changed_levels: std::collections::HashSet<usize> = if centers_changed && !needs_rebuild {
        (0..levels)
            .filter(|&i| {
                view.clip_centers.get(i).copied().unwrap_or(IVec2::ZERO)
                    != state.tile_apply_centers[i]
            })
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    // Snapshot the ring mapping now so we can borrow resident_cpu separately.
    let tile_rings: std::collections::HashMap<TileKey, Vec<u8>> =
        residency.tile_key_to_rings.clone();

    let snapshot_keys: Vec<TileKey> = if needs_rebuild {
        residency.resident_cpu.keys().copied().collect()
    } else if centers_changed {
        residency
            .resident_cpu
            .keys()
            .filter(|k| {
                tile_rings
                    .get(k)
                    .map(|rings| {
                        rings
                            .iter()
                            .any(|&r| changed_levels.contains(&(r as usize)))
                    })
                    .unwrap_or(false)
            })
            .copied()
            .collect()
    } else {
        new_tile_keys
            .iter()
            .filter(|k| residency.resident_cpu.contains_key(k))
            .copied()
            .collect()
    };

    // Height write pass
    for key in &snapshot_keys {
        let Some(tile) = residency.resident_cpu.get(key) else {
            continue;
        };
        let rings: &[u8] = tile_rings.get(key).map(Vec::as_slice).unwrap_or(&[]);
        for &ring_u8 in rings {
            let level = ring_u8 as usize;
            if level >= levels {
                continue;
            }
            let clip_center = match view.clip_centers.get(level) {
                Some(&c) => c,
                None => continue,
            };
            let layer_base = level * height_bpl;
            if needs_rebuild || force_full_levels.contains(&level) || new_tile_keys.contains(key) {
                write_height_tile_window(
                    &mut state.height_cpu_data,
                    layer_base,
                    res,
                    tile,
                    clip_center,
                    config.height_scale,
                );
            } else if changed_levels.contains(&level) {
                let old_center = state.tile_apply_centers[level];
                write_height_tile_new_strips(
                    &mut state.height_cpu_data,
                    layer_base,
                    res,
                    tile,
                    clip_center,
                    old_center,
                    config.height_scale,
                );
            }
            dirty_levels.insert(level);
        }
    }

    // Normal write pass
    for key in &snapshot_keys {
        let Some(tile) = residency.resident_cpu.get(key) else {
            continue;
        };
        let rings: &[u8] = tile_rings.get(key).map(Vec::as_slice).unwrap_or(&[]);
        for &ring_u8 in rings {
            let level = ring_u8 as usize;
            if level >= levels {
                continue;
            }
            let clip_center = match view.clip_centers.get(level) {
                Some(&c) => c,
                None => continue,
            };
            let layer_base = level * normal_bpl;
            if needs_rebuild || force_full_levels.contains(&level) || new_tile_keys.contains(key) {
                write_normal_tile_window(
                    &mut state.normal_cpu_data,
                    layer_base,
                    res,
                    tile,
                    clip_center,
                );
            } else if changed_levels.contains(&level) {
                let old_center = state.tile_apply_centers[level];
                write_normal_tile_new_strips(
                    &mut state.normal_cpu_data,
                    layer_base,
                    res,
                    tile,
                    clip_center,
                    old_center,
                );
            }
            dirty_levels.insert(level);
        }
    }

    drop(snapshot_keys);

    if needs_rebuild {
        for &lod in &dirty_levels {
            let height_layer_offset = lod * height_bpl;
            queue_full_layer_upload(
                &mut uploads,
                &state.height_texture_handle,
                lod as u32,
                &state.height_cpu_data[height_layer_offset..height_layer_offset + height_bpl],
                res,
                HEIGHT_BYTES_PER_TEXEL,
            );

            let normal_layer_offset = lod * normal_bpl;
            queue_full_layer_upload(
                &mut uploads,
                &state.normal_texture_handle,
                lod as u32,
                &state.normal_cpu_data[normal_layer_offset..normal_layer_offset + normal_bpl],
                res,
                NORMAL_BYTES_PER_TEXEL,
            );
        }
    } else if centers_changed {
        for &lod in &dirty_levels {
            let height_layer_offset = lod * height_bpl;
            let normal_layer_offset = lod * normal_bpl;
            if force_full_levels.contains(&lod) {
                queue_full_layer_upload(
                    &mut uploads,
                    &state.height_texture_handle,
                    lod as u32,
                    &state.height_cpu_data[height_layer_offset..height_layer_offset + height_bpl],
                    res,
                    HEIGHT_BYTES_PER_TEXEL,
                );
                queue_full_layer_upload(
                    &mut uploads,
                    &state.normal_texture_handle,
                    lod as u32,
                    &state.normal_cpu_data[normal_layer_offset..normal_layer_offset + normal_bpl],
                    res,
                    NORMAL_BYTES_PER_TEXEL,
                );
                continue;
            }

            let new_center = view.clip_centers.get(lod).copied().unwrap_or(IVec2::ZERO);
            let old_center = state.tile_apply_centers[lod];
            queue_new_strip_uploads(
                &mut uploads,
                &state.height_texture_handle,
                lod as u32,
                &state.height_cpu_data[height_layer_offset..height_layer_offset + height_bpl],
                res,
                HEIGHT_BYTES_PER_TEXEL,
                new_center,
                old_center,
            );
            queue_new_strip_uploads(
                &mut uploads,
                &state.normal_texture_handle,
                lod as u32,
                &state.normal_cpu_data[normal_layer_offset..normal_layer_offset + normal_bpl],
                res,
                NORMAL_BYTES_PER_TEXEL,
                new_center,
                old_center,
            );
        }
    }

    for key in &new_tile_keys {
        let Some(tile) = residency.resident_cpu.get(key) else {
            continue;
        };
        let level = tile.key.level as usize;
        if level >= levels {
            continue;
        }
        if needs_rebuild || force_full_levels.contains(&level) {
            continue;
        }
        let Some(&clip_center) = view.clip_centers.get(level) else {
            continue;
        };
        let Some((rect_min, rect_max)) = tile_window_intersection(tile, clip_center, res) else {
            continue;
        };
        let height_layer_offset = level * height_bpl;
        queue_world_rect_upload(
            &mut uploads,
            &state.height_texture_handle,
            level as u32,
            &state.height_cpu_data[height_layer_offset..height_layer_offset + height_bpl],
            res,
            HEIGHT_BYTES_PER_TEXEL,
            rect_min,
            rect_max,
        );

        let normal_layer_offset = level * normal_bpl;
        queue_world_rect_upload(
            &mut uploads,
            &state.normal_texture_handle,
            level as u32,
            &state.normal_cpu_data[normal_layer_offset..normal_layer_offset + normal_bpl],
            res,
            NORMAL_BYTES_PER_TEXEL,
            rect_min,
            rect_max,
        );
    }

    if !dirty_levels.is_empty() {
        state.height_generation += 1;
    }
    state.tile_apply_centers = view.clip_centers.clone();
    residency.evict_to_budget(config.max_resident_tiles);
}

#[cfg(test)]
mod tests {
    use super::clip_window_touches_world_bounds;
    use bevy::prelude::{IVec2, Vec2};

    #[test]
    fn clip_window_inside_bounds_does_not_force_full_reapply() {
        assert!(!clip_window_touches_world_bounds(
            IVec2::ZERO,
            512,
            1.0,
            Vec2::splat(-2048.0),
            Vec2::splat(2048.0),
        ));
    }

    #[test]
    fn clip_window_touching_edge_forces_full_reapply() {
        assert!(clip_window_touches_world_bounds(
            IVec2::new(1793, 0),
            512,
            1.0,
            Vec2::splat(-2048.0),
            Vec2::splat(2048.0),
        ));
    }
}

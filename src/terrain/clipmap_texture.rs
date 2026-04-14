use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use std::f32::consts::TAU;
use crate::terrain::{
    config::TerrainConfig,
    math::level_scale,
    material::TerrainMaterial,
    resources::{TerrainResidency, TerrainViewState, TileKey, TileState},
};

// ---------------------------------------------------------------------------
// Procedural height function
// ---------------------------------------------------------------------------

/// World-space height function. Returns a value in [0, 1].
///
/// Uses multi-octave sine waves normalized over a 2048 × 2048 world area
/// centred at the origin.  Matches the procedural stub in `streamer.rs` so
/// that live-generated clipmap textures agree with tile data.
pub fn height_at_world(x: f32, z: f32) -> f32 {
    let u = x * (1.0 / 2048.0) + 0.5;
    let v = z * (1.0 / 2048.0) + 0.5;

    let h = 0.50 * (u * TAU * 2.0).sin() * (v * TAU * 2.0).cos()
          + 0.25 * (u * TAU * 4.0 + 1.3).cos() * (v * TAU * 4.0 + 0.7).sin()
          + 0.12 * (u * TAU * 8.0 + 0.5).sin() * (v * TAU * 8.0 + 2.1).cos()
          + 0.06 * (u * TAU * 16.0).cos() * (v * TAU * 13.0).sin()
          + 0.03 * (u * TAU * 32.0 + 0.9).sin() * (v * TAU * 27.0 + 1.5).cos();

    ((h + 1.0) * 0.5).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Layer generation  (R16Unorm — 65 535 levels over height_scale)
// ---------------------------------------------------------------------------

/// Bytes per texel for our height texture format.
const BYTES_PER_TEXEL: usize = 2; // R16Unorm

/// Generates one R16Unorm layer for a clipmap level.
///
/// The layer covers the square region centred on `center * level_scale` with
/// a side length of `ring_patches * patch_resolution * level_scale` world units.
/// `clipmap_resolution` texels span that side length.
///
/// R16Unorm gives ~0.008 m precision over a 512 m height range, eliminating
/// the quantization-induced staircase normals that R8Unorm produces.
pub fn generate_clipmap_layer(
    center: IVec2,
    level_scale_ws: f32,
    ring_patches: u32,
    patch_resolution: u32,
    clipmap_resolution: u32,
    use_procedural: bool,
) -> Vec<u8> {
    let texels = (clipmap_resolution * clipmap_resolution) as usize;
    if !use_procedural {
        return vec![0u8; texels * BYTES_PER_TEXEL];
    }

    let ring_span = ring_patches as f32 * patch_resolution as f32 * level_scale_ws;
    let origin_x  = center.x as f32 * level_scale_ws - ring_span * 0.5;
    let origin_z  = center.y as f32 * level_scale_ws - ring_span * 0.5;
    let texel_ws  = ring_span / clipmap_resolution as f32;

    let mut data = Vec::with_capacity(texels * BYTES_PER_TEXEL);

    for row in 0..clipmap_resolution {
        for col in 0..clipmap_resolution {
            let wx = origin_x + (col as f32 + 0.5) * texel_ws;
            let wz = origin_z + (row as f32 + 0.5) * texel_ws;
            let h  = height_at_world(wx, wz);
            let v  = (h * 65535.0) as u16;
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Texture array creation
// ---------------------------------------------------------------------------

/// Bytes consumed by one full clipmap layer.
fn bytes_per_layer(res: u32) -> usize {
    (res * res) as usize * BYTES_PER_TEXEL
}

/// Builds the initial clipmap texture array with all levels generated at
/// (0, 0) clip centres.  Every frame after startup `update_clipmap_textures`
/// refines individual layers when clip centres change.
pub fn create_initial_clipmap_texture(config: &TerrainConfig) -> Image {
    let res    = config.clipmap_resolution;
    let layers = config.clipmap_levels;
    let bpl    = bytes_per_layer(res);
    let mut data = vec![0u8; bpl * layers as usize];

    for level in 0..layers {
        let scale      = level_scale(config.world_scale, level);
        let layer_data = generate_clipmap_layer(
            IVec2::ZERO, scale,
            config.ring_patches, config.patch_resolution, res,
            config.procedural_fallback,
        );
        let offset = level as usize * bpl;
        data[offset..offset + bpl].copy_from_slice(&layer_data);
    }

    Image::new(
        Extent3d {
            width: res, height: res,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R16Unorm,
        // Keep the CPU copy so `update_clipmap_textures` can patch layers.
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
}

// ---------------------------------------------------------------------------
// Uniform helper
// ---------------------------------------------------------------------------

/// Computes `clip_levels[8]` for `TerrainMaterialUniforms` from a view state.
///
/// Layout per entry: `(origin_x, origin_z, inv_ring_span, texel_world_size)`.
pub fn compute_clip_levels(
    config: &TerrainConfig,
    clip_centers: &[IVec2],
    level_scales: &[f32],
) -> [Vec4; 8] {
    let mut levels = [Vec4::ZERO; 8];

    for lod in 0..(config.clipmap_levels as usize).min(8) {
        let center = clip_centers.get(lod).copied().unwrap_or(IVec2::ZERO);
        let scale  = level_scales.get(lod).copied()
            .unwrap_or_else(|| level_scale(config.world_scale, lod as u32));

        let ring_span = config.ring_patches as f32 * config.patch_resolution as f32 * scale;
        let origin_x  = center.x as f32 * scale - ring_span * 0.5;
        let origin_z  = center.y as f32 * scale - ring_span * 0.5;
        let inv_span  = 1.0 / ring_span;
        let texel_ws  = ring_span / config.clipmap_resolution as f32;

        levels[lod] = Vec4::new(origin_x, origin_z, inv_span, texel_ws);
    }

    levels
}

/// Computes `clip_levels` when all clip centres are at the origin.
/// Used during startup before the first `update_terrain_view_state` runs.
pub fn compute_initial_clip_levels(config: &TerrainConfig) -> [Vec4; 8] {
    let zeros: Vec<IVec2> = vec![IVec2::ZERO; config.clipmap_levels as usize];
    let scales: Vec<f32> = (0..config.clipmap_levels)
        .map(|l| level_scale(config.world_scale, l))
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
    pub texture_handle:    Handle<Image>,
    pub material_handle:   Handle<TerrainMaterial>,
    /// Clip centres from the last procedural texture regeneration.
    pub last_clip_centers: Vec<IVec2>,
    /// Clip centres at which resident tiles were last written into the texture.
    /// Sentinel IVec2::MAX on startup forces a full tile re-apply on the first frame.
    pub tile_apply_centers: Vec<IVec2>,
}

// ---------------------------------------------------------------------------
// Update system
// ---------------------------------------------------------------------------

/// Re-generates any clipmap layers whose clip centre has changed since the
/// last frame, and refreshes the per-level origin uniforms in the material.
///
/// Runs in `Update` after `update_terrain_view_state`.
pub fn update_clipmap_textures(
    config:    Res<TerrainConfig>,
    view:      Res<TerrainViewState>,
    state:     Option<ResMut<TerrainClipmapState>>,
    mut images:    ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    let Some(mut state) = state else { return };
    if view.clip_centers.is_empty() { return; }

    let res    = config.clipmap_resolution;
    let bpl    = bytes_per_layer(res);
    let levels = config.clipmap_levels as usize;

    // Pad the cached list so index comparisons don't go out of bounds.
    while state.last_clip_centers.len() < levels {
        // Use a sentinel that guarantees the first run regenerates every layer.
        state.last_clip_centers.push(IVec2::new(i32::MAX, i32::MAX));
    }

    // Fast path: nothing moved to a new grid cell.
    let dirty = (0..levels).any(|i| {
        view.clip_centers.get(i).copied().unwrap_or(IVec2::ZERO)
            != state.last_clip_centers[i]
    });
    if !dirty { return; }

    let Some(image) = images.get_mut(&state.texture_handle) else { return };

    for lod in 0..levels {
        let center = view.clip_centers.get(lod).copied().unwrap_or(IVec2::ZERO);
        if center == state.last_clip_centers[lod] { continue; }

        let scale = view.level_scales.get(lod).copied()
            .unwrap_or_else(|| level_scale(config.world_scale, lod as u32));

        let layer_data = generate_clipmap_layer(
            center, scale,
            config.ring_patches, config.patch_resolution, res,
            config.procedural_fallback,
        );

        let offset = lod * bpl;
        if let Some(ref mut data) = image.data {
            if let Some(slice) = data.get_mut(offset..offset + bpl) {
                slice.copy_from_slice(&layer_data);
            }
        }
    }

    // Refresh the per-level origin uniforms.
    if let Some(mat) = materials.get_mut(&state.material_handle) {
        mat.params.clip_levels = compute_clip_levels(
            &config,
            &view.clip_centers,
            &view.level_scales,
        );
    }

    state.last_clip_centers = view.clip_centers.clone();
}

// ---------------------------------------------------------------------------
// Tile upload system
// ---------------------------------------------------------------------------

/// Applies pre-baked height tiles to the live clipmap texture array.
///
/// **Why re-apply on every clip-center shift**: `update_clipmap_textures` runs
/// first and regenerates entire layers from the procedural fallback whenever the
/// clip center moves.  Without re-applying tile data afterwards the real EXR
/// heights would be invisible — the procedural data would win every frame.
///
/// Strategy:
///   1. Move any new tiles from `pending_upload` into `resident_cpu` (persistent
///      CPU cache) and mark them `ResidentGpu`.
///   2. If clip centers changed since the last tile write (or new tiles arrived),
///      re-write every cached tile that falls inside the current clipmap window.
///   3. Update `tile_apply_centers` so we skip the work on unchanged frames.
pub fn apply_tiles_to_clipmap(
    config:    Res<TerrainConfig>,
    view:      Res<TerrainViewState>,
    mut state:     Option<ResMut<TerrainClipmapState>>,
    mut images:    ResMut<Assets<Image>>,
    mut residency: ResMut<TerrainResidency>,
) {
    if view.clip_centers.is_empty() { return; }

    // --- Step 1: absorb newly loaded tiles into the persistent CPU cache --------
    let new_tiles = std::mem::take(&mut residency.pending_upload);
    let has_new   = !new_tiles.is_empty();
    for tile in new_tiles {
        residency.tiles.insert(tile.key, TileState::ResidentGpu { slot: 0 });
        residency.resident_cpu.insert(tile.key, tile.data);
    }

    if residency.resident_cpu.is_empty() { return; }

    let Some(ref mut state) = state else { return };

    // Grow sentinel vec to match level count.
    let levels = config.clipmap_levels as usize;
    while state.tile_apply_centers.len() < levels {
        state.tile_apply_centers.push(IVec2::new(i32::MAX, i32::MAX));
    }

    // --- Step 2: early-out when nothing changed ---------------------------------
    let centers_changed = (0..levels).any(|i| {
        view.clip_centers.get(i).copied().unwrap_or(IVec2::ZERO)
            != state.tile_apply_centers[i]
    });
    if !has_new && !centers_changed { return; }

    // --- Step 3: write tiles into the GPU texture ------------------------------
    let Some(image) = images.get_mut(&state.texture_handle) else { return };
    let Some(ref mut img_data) = image.data else { return };

    let res  = config.clipmap_resolution;
    let bpl  = bytes_per_layer(res);
    let half = (res / 2) as i32;
    let ts   = config.tile_size;

    // Collect resident tiles into a Vec to avoid holding the HashMap borrow
    // while mutating img_data (both live in TerrainResidency but img_data is
    // from Assets<Image> — the compiler is happy; we just need a stable iter).
    let tile_snapshot: Vec<(TileKey, &Vec<f32>)> = residency.resident_cpu.iter()
        .map(|(k, v)| (*k, v))
        .collect();

    for (key, tile_pixels) in &tile_snapshot {
        let level = key.level as usize;
        if level >= levels { continue; }

        let clip_center = match view.clip_centers.get(level) {
            Some(&c) => c,
            None     => continue,
        };

        let layer_base = level * bpl;

        for row in 0..ts {
            for col in 0..ts {
                let gx = key.x * ts as i32 + col as i32;
                let gz = key.y * ts as i32 + row as i32;

                let cx = gx - clip_center.x + half;
                let cz = gz - clip_center.y + half;

                if cx < 0 || cx >= res as i32 || cz < 0 || cz >= res as i32 { continue; }

                let texel_off = (cz as usize * res as usize + cx as usize) * BYTES_PER_TEXEL;
                let dst       = layer_base + texel_off;
                let h         = tile_pixels[(row * ts + col) as usize];
                let v         = (h * 65535.0) as u16;

                if dst + BYTES_PER_TEXEL <= img_data.len() {
                    img_data[dst..dst + BYTES_PER_TEXEL].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }

    state.tile_apply_centers = view.clip_centers.clone();
    residency.evict_to_budget(config.max_resident_tiles);
}

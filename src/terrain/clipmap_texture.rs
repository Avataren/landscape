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
    resources::TerrainViewState,
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
// Layer generation
// ---------------------------------------------------------------------------

/// Generates one R8Unorm layer for a clipmap level.
///
/// The layer covers the square region centred on `center * level_scale` with
/// a side length of `ring_patches * patch_resolution * level_scale` world units.
/// `clipmap_resolution` texels span that side length.
pub fn generate_clipmap_layer(
    center: IVec2,
    level_scale_ws: f32,
    ring_patches: u32,
    patch_resolution: u32,
    clipmap_resolution: u32,
) -> Vec<u8> {
    let ring_span = ring_patches as f32 * patch_resolution as f32 * level_scale_ws;
    let origin_x  = center.x as f32 * level_scale_ws - ring_span * 0.5;
    let origin_z  = center.y as f32 * level_scale_ws - ring_span * 0.5;
    let texel_ws  = ring_span / clipmap_resolution as f32;

    let n = (clipmap_resolution * clipmap_resolution) as usize;
    let mut data = Vec::with_capacity(n);

    for row in 0..clipmap_resolution {
        for col in 0..clipmap_resolution {
            let wx = origin_x + (col as f32 + 0.5) * texel_ws;
            let wz = origin_z + (row as f32 + 0.5) * texel_ws;
            data.push((height_at_world(wx, wz) * 255.0) as u8);
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Texture array creation
// ---------------------------------------------------------------------------

/// Builds the initial clipmap texture array with all levels generated at
/// (0, 0) clip centres.  Every frame after startup `update_clipmap_textures`
/// refines individual layers when clip centres change.
pub fn create_initial_clipmap_texture(config: &TerrainConfig) -> Image {
    let res    = config.clipmap_resolution;
    let layers = config.clipmap_levels;
    let bytes_per_layer = (res * res) as usize;
    let mut data = vec![0u8; bytes_per_layer * layers as usize];

    for level in 0..layers {
        let scale      = level_scale(config.world_scale, level);
        let layer_data = generate_clipmap_layer(
            IVec2::ZERO, scale,
            config.ring_patches, config.patch_resolution, res,
        );
        let offset = level as usize * bytes_per_layer;
        data[offset..offset + bytes_per_layer].copy_from_slice(&layer_data);
    }

    Image::new(
        Extent3d {
            width: res, height: res,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
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
    /// Clip centres from the last texture regeneration.
    pub last_clip_centers: Vec<IVec2>,
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

    let res             = config.clipmap_resolution;
    let bytes_per_layer = (res * res) as usize;
    let levels          = config.clipmap_levels as usize;

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
        );

        let offset = lod * bytes_per_layer;
        if let Some(ref mut data) = image.data {
            if let Some(slice) = data.get_mut(offset..offset + bytes_per_layer) {
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

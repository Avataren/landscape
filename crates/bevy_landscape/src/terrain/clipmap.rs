use crate::terrain::{
    config::TerrainConfig,
    math::{build_block_origins, level_scale},
    resources::TerrainViewState,
};
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// CPU patch instance descriptor
// ---------------------------------------------------------------------------

/// CPU-side description of one canonical `m × m` terrain block instance.
///
/// The corresponding Bevy `Transform` is:
///   translation = (origin_ws.x, 0, origin_ws.y)
///   scale       = (level_scale_ws, 1, level_scale_ws)
///
/// The vertex shader derives the LOD level from the transform's X-scale:
///   lod = round(log2(level_scale_ws / world_scale))
#[derive(Clone, Debug)]
pub struct PatchInstanceCpu {
    /// Clipmap LOD level (0 = finest detail).
    pub lod_level: u32,
    /// World-space XZ position of the block's minimum corner.
    pub origin_ws: Vec2,
    /// World-space size of one grid unit at this LOD (= world_scale × 2^lod).
    /// Used as the Transform X/Z scale.
    pub level_scale_ws: f32,
    /// World-space size of the block side (= m × level_scale_ws).
    /// Used only for CPU-side bounds intersection; not sent to the GPU.
    pub block_world_size: f32,
}

// ---------------------------------------------------------------------------
// Clipmap builder
// ---------------------------------------------------------------------------

/// Builds all `PatchInstanceCpu` block descriptors for the current terrain view.
///
/// Each LOD level contributes either 16 blocks (level 0, full fill) or
/// 12 blocks (levels > 0, hollow ring) following the GPU Gems 2 layout.
pub fn build_patch_instances_for_view(
    config: &TerrainConfig,
    view: &TerrainViewState,
) -> Vec<PatchInstanceCpu> {
    let m = config.block_size();
    let mut out = Vec::new();

    for level in 0..config.active_clipmap_levels() {
        let lvl_scale = match view.level_scales.get(level as usize) {
            Some(&s) => s,
            None => level_scale(config.world_scale, level),
        };
        let center = match view.clip_centers.get(level as usize) {
            Some(&c) => c,
            None => IVec2::ZERO,
        };

        let has_hole = level > 0;
        let origins = build_block_origins(center, lvl_scale, m, has_hole);
        let block_ws = m as f32 * lvl_scale;

        for origin in origins {
            out.push(PatchInstanceCpu {
                lod_level: level,
                origin_ws: origin,
                level_scale_ws: lvl_scale,
                block_world_size: block_ws,
            });
        }
    }

    out
}

/// Filters instances to those that overlap the terrain footprint.
/// When bounds are degenerate (world_min == world_max) all instances pass.
pub fn build_patch_instances_for_view_in_bounds(
    config: &TerrainConfig,
    view: &TerrainViewState,
    world_min: Vec2,
    world_max: Vec2,
) -> Vec<PatchInstanceCpu> {
    let patches = build_patch_instances_for_view(config, view);
    if world_min == world_max {
        return patches;
    }
    patches
        .into_iter()
        .filter(|p| block_intersects_world_bounds(p, world_min, world_max))
        .collect()
}

fn block_intersects_world_bounds(
    patch: &PatchInstanceCpu,
    world_min: Vec2,
    world_max: Vec2,
) -> bool {
    let patch_min = patch.origin_ws;
    let patch_max = patch.origin_ws + Vec2::splat(patch.block_world_size);

    patch_max.x > world_min.x
        && patch_min.x < world_max.x
        && patch_max.y > world_min.y
        && patch_min.y < world_max.y
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::math::snap_camera_to_level_grid;

    fn make_view(config: &TerrainConfig, cam: Vec3) -> TerrainViewState {
        let mut view = TerrainViewState::default();
        view.camera_pos_ws = cam;
        for level in 0..config.active_clipmap_levels() {
            let scale = level_scale(config.world_scale, level);
            view.level_scales.push(scale);
            view.clip_centers
                .push(snap_camera_to_level_grid(cam.xz(), scale));
        }
        view
    }

    #[test]
    fn patch_ordering_stable() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::new(100.0, 0.0, 100.0));
        let patches = build_patch_instances_for_view(&config, &view);

        let levels: Vec<u32> = patches.iter().map(|p| p.lod_level).collect();
        let mut prev = 0u32;
        for &l in &levels {
            assert!(l >= prev, "levels must be non-decreasing");
            prev = l;
        }
    }

    #[test]
    fn level_0_has_full_grid() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let patches = build_patch_instances_for_view(&config, &view);

        // Level 0: 4×4 = 16 blocks (no inner hole).
        let count_l0 = patches.iter().filter(|p| p.lod_level == 0).count();
        assert_eq!(count_l0, 16);
    }

    #[test]
    fn level_1_has_ring() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let patches = build_patch_instances_for_view(&config, &view);

        // Level 1: 4×4 = 16 minus inner 2×2 = 12 blocks.
        let count_l1 = patches.iter().filter(|p| p.lod_level == 1).count();
        assert_eq!(count_l1, 12);
    }

    #[test]
    fn block_world_sizes_double_per_level() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let patches = build_patch_instances_for_view(&config, &view);

        for level in 1..config.active_clipmap_levels() {
            let prev = patches
                .iter()
                .find(|p| p.lod_level == level - 1)
                .map(|p| p.block_world_size)
                .unwrap();
            let curr = patches
                .iter()
                .find(|p| p.lod_level == level)
                .map(|p| p.block_world_size)
                .unwrap();
            assert!(
                (curr - prev * 2.0).abs() < 1e-4,
                "block world size must double each level"
            );
        }
    }

    #[test]
    fn bounds_filter_removes_off_world_patches() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let full = build_patch_instances_for_view(&config, &view);
        let filtered = build_patch_instances_for_view_in_bounds(
            &config,
            &view,
            Vec2::new(-64.0, -64.0),
            Vec2::new(64.0, 64.0),
        );

        assert!(!filtered.is_empty());
        assert!(filtered.len() < full.len());
    }
}

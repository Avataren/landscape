use crate::terrain::resources::TileKey;
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// LOD scale helpers
// ---------------------------------------------------------------------------

/// World-space texel spacing for a given LOD level.
/// Each level doubles the previous level's spacing.
pub fn level_scale(base_sample_spacing: f32, level: u32) -> f32 {
    base_sample_spacing * (1u32 << level) as f32
}

/// Snap a camera XZ position to the integer grid for a given level scale.
/// Returns grid-space integer coordinates (multiply by scale to get world).
pub fn snap_camera_to_level_grid(camera_xz: Vec2, level_scale: f32) -> IVec2 {
    let gx = (camera_xz.x / level_scale).floor() as i32;
    let gy = (camera_xz.y / level_scale).floor() as i32;
    IVec2::new(gx, gy)
}

/// Snap the finest clipmap center to a grid that preserves exact nesting for
/// all active LOD levels.
///
/// Each coarser level's center is derived by right-shifting the fine center by
/// L bits: `centers[L] = fine_center >> L`. This already gives the desired
/// cadence for each level (L0 every 1 texel, L1 every 2, L2 every 4, ...).
///
/// Do not quantize the finest center to a larger stride. Over-snapping it to
/// `2^(active_levels - 1)` makes the whole clipmap stack move in huge jumps,
/// which shows up as black flashes and visible LOD popping while textures and
/// geometry catch up.
pub fn snap_camera_to_nested_clipmap_grid(
    camera_xz: Vec2,
    base_level_scale: f32,
    _active_levels: u32,
) -> IVec2 {
    snap_camera_to_level_grid(camera_xz, base_level_scale)
}

// ---------------------------------------------------------------------------
// Ring patch layout
// ---------------------------------------------------------------------------

/// Builds world-space XZ origins for all patches in a clipmap ring at `level`.
///
/// * `center`          – snapped grid-space center for this level
/// * `scale`           – world-space size of one texel at this level
/// * `patch_resolution`– vertices per patch edge
/// * `ring_patches`    – number of patches per ring edge (e.g. 8 → 8×8 outer, minus inner)
/// * `has_inner_hole`  – true for level > 0 (inner region is covered by finer level)
pub fn build_ring_patch_origins(
    center: IVec2,
    scale: f32,
    patch_resolution: u32,
    ring_patches: u32,
    has_inner_hole: bool,
) -> Vec<Vec2> {
    let patch_size_grid = patch_resolution as i32; // grid cells per patch
    let half = (ring_patches / 2) as i32;

    // Inner hole half-size in grid cells (half of ring / 2)
    let inner_half = if has_inner_hole { half / 2 } else { i32::MIN };

    let mut origins = Vec::new();

    for py in -half..half {
        for px in -half..half {
            // Is this patch inside the inner hole that the next-finer level covers?
            if has_inner_hole {
                let in_hole_x = px >= -inner_half && px < inner_half;
                let in_hole_y = py >= -inner_half && py < inner_half;
                if in_hole_x && in_hole_y {
                    continue;
                }
            }

            let gx = center.x + px * patch_size_grid;
            let gy = center.y + py * patch_size_grid;

            origins.push(Vec2::new(gx as f32 * scale, gy as f32 * scale));
        }
    }

    origins
}

// ---------------------------------------------------------------------------
// Tile coverage
// ---------------------------------------------------------------------------

/// Returns the set of tile keys at `level` required to cover the camera's
/// visible ring.
pub fn compute_needed_tiles_for_level(
    center: IVec2,
    level_scale: f32,
    patch_resolution: u32,
    ring_patches: u32,
    tile_size: u32,
    level: u8,
) -> Vec<TileKey> {
    let half = (ring_patches / 2) as i32;
    let patch_size_grid = patch_resolution as i32;
    let tile_grid = tile_size as i32;

    // Compute world-space bounds of the ring.
    let min_grid = center + IVec2::splat(-half * patch_size_grid);
    let max_grid = center + IVec2::splat(half * patch_size_grid);

    let tx_min =
        (min_grid.x as f32 * level_scale / (tile_grid as f32 * level_scale)).floor() as i32;
    let ty_min =
        (min_grid.y as f32 * level_scale / (tile_grid as f32 * level_scale)).floor() as i32;
    let tx_max = (max_grid.x as f32 * level_scale / (tile_grid as f32 * level_scale)).ceil() as i32;
    let ty_max = (max_grid.y as f32 * level_scale / (tile_grid as f32 * level_scale)).ceil() as i32;

    let mut keys = Vec::new();
    for ty in ty_min..ty_max {
        for tx in tx_min..tx_max {
            keys.push(TileKey {
                level,
                x: tx,
                y: ty,
            });
        }
    }
    keys
}

// ---------------------------------------------------------------------------
// Morph factor
// ---------------------------------------------------------------------------

/// Returns a blend factor in [0, 1] that smoothly transitions from 0 (fine)
/// to 1 (coarse) as `distance` approaches `band_end`.
#[allow(dead_code)]
pub fn morph_factor(distance: f32, band_start: f32, band_end: f32) -> f32 {
    if distance <= band_start {
        return 0.0;
    }
    if distance >= band_end {
        return 1.0;
    }
    (distance - band_start) / (band_end - band_start)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_scale_doubles() {
        let base = 2.0_f32;
        assert_eq!(level_scale(base, 0), 2.0);
        assert_eq!(level_scale(base, 1), 4.0);
        assert_eq!(level_scale(base, 2), 8.0);
        assert_eq!(level_scale(base, 5), 64.0);
    }

    #[test]
    fn snap_only_changes_on_grid_crossing() {
        let scale = 4.0_f32;
        let c1 = snap_camera_to_level_grid(Vec2::new(0.5, 0.5), scale);
        let c2 = snap_camera_to_level_grid(Vec2::new(3.9, 3.9), scale);
        // Both are still inside the [0,4) grid cell -> same snap
        assert_eq!(c1, c2);

        let c3 = snap_camera_to_level_grid(Vec2::new(4.1, 4.1), scale);
        assert_ne!(c1, c3);
    }

    #[test]
    fn nested_snap_preserves_fine_level_cadence() {
        let c0 = snap_camera_to_nested_clipmap_grid(Vec2::new(0.1, 0.1), 1.0, 4);
        let c1 = snap_camera_to_nested_clipmap_grid(Vec2::new(7.9, 7.9), 1.0, 4);
        let c2 = snap_camera_to_nested_clipmap_grid(Vec2::new(8.1, 8.1), 1.0, 4);

        assert_eq!(c0, IVec2::ZERO);
        assert_eq!(c1, IVec2::splat(7));
        assert_eq!(c2, IVec2::splat(8));
    }

    #[test]
    fn nested_snap_still_derives_coarser_levels_by_shift() {
        let fine = snap_camera_to_nested_clipmap_grid(Vec2::new(13.9, 13.9), 1.0, 4);
        assert_eq!(fine, IVec2::splat(13));
        assert_eq!(fine >> 1, IVec2::splat(6));
        assert_eq!(fine >> 2, IVec2::splat(3));
        assert_eq!(fine >> 3, IVec2::splat(1));
    }

    #[test]
    fn ring_patch_count_with_hole() {
        let ring = 8u32;
        let origins = build_ring_patch_origins(IVec2::ZERO, 1.0, 64, ring, true);
        // Full 8x8 = 64, inner 4x4 = 16 removed -> 48
        assert_eq!(origins.len(), 48);
    }

    #[test]
    fn ring_patch_count_no_hole() {
        let ring = 8u32;
        let origins = build_ring_patch_origins(IVec2::ZERO, 1.0, 64, ring, false);
        assert_eq!(origins.len(), 64);
    }

    #[test]
    fn morph_factor_boundaries() {
        assert_eq!(morph_factor(0.0, 10.0, 20.0), 0.0);
        assert_eq!(morph_factor(10.0, 10.0, 20.0), 0.0);
        assert_eq!(morph_factor(15.0, 10.0, 20.0), 0.5);
        assert_eq!(morph_factor(20.0, 10.0, 20.0), 1.0);
        assert_eq!(morph_factor(30.0, 10.0, 20.0), 1.0);
    }
}

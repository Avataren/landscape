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
// GPU Gems 2 block layout
// ---------------------------------------------------------------------------

/// Returns the world-space XZ origins of all canonical `m × m` blocks for
/// one clipmap level, following the GPU Gems 2 nested-grid structure.
///
/// Ring geometry (level > 0 — hollow ring):
///   Outer boundary: ± 2m grid units from `center`
///   Inner hole:     ± m  grid units from `center` (filled by the finer level)
///   → 12 blocks arranged in a 4 × 4 pattern with the centre 2 × 2 removed.
///
/// Fill geometry (level 0 — no finer level below):
///   All 16 blocks in the full 4 × 4 grid.
///
/// Each block's world origin is the block's minimum (−x, −z) corner.
/// The block covers `[origin, origin + m * scale]` in world space.
pub fn build_block_origins(center: IVec2, scale: f32, m: u32, has_inner_hole: bool) -> Vec<Vec2> {
    let m = m as i32;
    // Column starts (in grid units, relative to center): -2m, -m, 0, m
    let cols = [-2 * m, -m, 0, m];

    let mut origins = Vec::with_capacity(if has_inner_hole { 12 } else { 16 });

    for &bz in &cols {
        for &bx in &cols {
            // Skip the inner 2×2 hole for ring levels (l > 0).
            if has_inner_hole && bx >= -m && bx < m && bz >= -m && bz < m {
                continue;
            }
            let gx = center.x + bx;
            let gz = center.y + bz;
            origins.push(Vec2::new(gx as f32 * scale, gz as f32 * scale));
        }
    }

    origins
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
    fn block_count_ring() {
        // Ring level: 4×4 = 16 minus inner 2×2 = 4 → 12 blocks.
        let origins = build_block_origins(IVec2::ZERO, 1.0, 128, true);
        assert_eq!(origins.len(), 12);
    }

    #[test]
    fn block_count_fill() {
        // Fill level (l=0): full 4×4 = 16 blocks.
        let origins = build_block_origins(IVec2::ZERO, 1.0, 128, false);
        assert_eq!(origins.len(), 16);
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

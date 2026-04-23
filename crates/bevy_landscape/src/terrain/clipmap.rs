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
// Interior trim strip instances  (GPU Gems 2 §2.3.3)
// ---------------------------------------------------------------------------

/// CPU-side descriptor for one (2m+1)×2 interior trim strip.
///
/// The trim is rendered at the **coarse** LOD scale for the ring boundary it
/// fills.  Each strip is one coarse quad wide (in the perpendicular direction)
/// and 2m coarse quads long (spanning the full inner-hole edge).
///
/// # Why coarse scale matters
/// A fine-scale strip (1 fine texel wide) has both vertex columns snap to the
/// **same** coarse grid position after the vertex-shader morph (floor division),
/// collapsing the strip to zero width and rendering no pixels.  A coarse-scale
/// strip has its two columns at neighbouring coarse grid positions and is never
/// collapsed.
#[derive(Clone, Debug)]
pub struct TrimInstanceCpu {
    /// LOD level of the **coarse** ring whose inner hole this strip borders.
    /// The vertex shader derives this from `Transform.scale.x = level_scale_ws`.
    pub lod_level: u32,
    /// World-space XZ of the strip's minimum (−x, −z) corner.
    pub origin_ws: Vec2,
    /// One coarse grid unit in world space at this LOD (`world_scale * 2^lod`).
    pub level_scale_ws: f32,
    /// `false` = vertical strip (LEFT or RIGHT inner boundary edge).
    /// `true`  = horizontal strip (BOTTOM or TOP inner boundary edge).
    pub is_horizontal: bool,
}

/// Builds the four (2m+1)×2 interior trim strips that border every ring-level
/// inner hole (GPU Gems 2 Figure 2-5, blue pieces).
///
/// Four strips — LEFT, RIGHT, BOTTOM, TOP — are emitted unconditionally for
/// every coarse ring level L ≥ 1, each placed 1 coarse quad inward from the
/// respective inner-hole edge.  Because they are at **coarse** scale the two
/// columns of each strip map to *different* coarse grid positions and are never
/// collapsed by the morph snap.
///
/// The strips fill both the steady-state seam and the 1-fine-texel parity gap
/// that appears when the fine ring's clip center is odd (the gap that would
/// otherwise show as an L-shaped crack when the camera crosses a fine-grid line).
pub fn build_trim_instances_for_view(
    config: &TerrainConfig,
    view: &TerrainViewState,
) -> Vec<TrimInstanceCpu> {
    let m = config.block_size() as i32;
    let mut instances = Vec::new();

    for level in 1..config.active_clipmap_levels() {
        let l = level as usize;
        if l >= view.level_scales.len() || l >= view.clip_centers.len() {
            break;
        }

        let s = view.level_scales[l]; // one coarse grid unit
        let c = view.clip_centers[l];

        // Inner-hole boundary corners in world space.
        let min_x = (c.x - m) as f32 * s;
        let min_z = (c.y - m) as f32 * s;
        let max_x = (c.x + m) as f32 * s;
        let max_z = (c.y + m) as f32 * s;

        // LEFT: x ∈ [min_x, min_x+s], z ∈ [min_z, max_z]
        instances.push(TrimInstanceCpu {
            lod_level: level,
            origin_ws: Vec2::new(min_x, min_z),
            level_scale_ws: s,
            is_horizontal: false,
        });
        // RIGHT: x ∈ [max_x-s, max_x], z ∈ [min_z, max_z]
        instances.push(TrimInstanceCpu {
            lod_level: level,
            origin_ws: Vec2::new(max_x - s, min_z),
            level_scale_ws: s,
            is_horizontal: false,
        });
        // BOTTOM: x ∈ [min_x, max_x], z ∈ [min_z, min_z+s]
        instances.push(TrimInstanceCpu {
            lod_level: level,
            origin_ws: Vec2::new(min_x, min_z),
            level_scale_ws: s,
            is_horizontal: true,
        });
        // TOP: x ∈ [min_x, max_x], z ∈ [max_z-s, max_z]
        instances.push(TrimInstanceCpu {
            lod_level: level,
            origin_ws: Vec2::new(min_x, max_z - s),
            level_scale_ws: s,
            is_horizontal: true,
        });
    }

    instances
}

/// Filters trim instances to those that overlap the terrain footprint.
/// When bounds are degenerate (world_min == world_max) all trims pass.
pub fn build_trim_instances_for_view_in_bounds(
    config: &TerrainConfig,
    view: &TerrainViewState,
    world_min: Vec2,
    world_max: Vec2,
) -> Vec<TrimInstanceCpu> {
    let trims = build_trim_instances_for_view(config, view);
    if world_min == world_max {
        return trims;
    }
    let block_size = config.block_size() as f32;
    trims
        .into_iter()
        .filter(|trim| trim_intersects_world_bounds(trim, block_size, world_min, world_max))
        .collect()
}

fn trim_intersects_world_bounds(
    trim: &TrimInstanceCpu,
    block_size: f32,
    world_min: Vec2,
    world_max: Vec2,
) -> bool {
    let trim_min = trim.origin_ws;
    let trim_extent = if trim.is_horizontal {
        Vec2::new(2.0 * block_size * trim.level_scale_ws, trim.level_scale_ws)
    } else {
        Vec2::new(trim.level_scale_ws, 2.0 * block_size * trim.level_scale_ws)
    };
    let trim_max = trim_min + trim_extent;

    trim_max.x > world_min.x
        && trim_min.x < world_max.x
        && trim_max.y > world_min.y
        && trim_min.y < world_max.y
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

    #[test]
    fn trim_bounds_filter_removes_off_world_trims() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);

        let trims = build_trim_instances_for_view_in_bounds(
            &config,
            &view,
            Vec2::new(-64.0, -64.0),
            Vec2::new(64.0, 64.0),
        );

        assert!(trims.is_empty());
    }

    #[test]
    fn trim_bounds_filter_keeps_partial_overlap() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);

        let trims = build_trim_instances_for_view_in_bounds(
            &config,
            &view,
            Vec2::new(-300.0, -64.0),
            Vec2::new(-200.0, 64.0),
        );

        assert_eq!(trims.len(), 1);
        assert!(!trims[0].is_horizontal, "expected left vertical trim");
        assert_eq!(trims[0].lod_level, 1);
    }
}

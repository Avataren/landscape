use bevy::prelude::*;
use crate::terrain::{
    config::TerrainConfig,
    math::{build_ring_patch_origins, level_scale},
    resources::TerrainViewState,
};

// ---------------------------------------------------------------------------
// CPU patch instance descriptor
// ---------------------------------------------------------------------------

/// CPU-side description of one terrain patch instance.
/// Built from the clipmap state every frame (or when the camera crosses a
/// snapped grid boundary) and submitted to the GPU as a storage buffer.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PatchInstanceCpu {
    /// Clipmap LOD level (0 = finest detail).
    pub lod_level: u32,
    /// Patch type (0 = normal, reserved for trim patches later).
    pub patch_kind: u32,
    /// World-space XZ position of the patch minimum corner.
    pub origin_ws: Vec2,
    /// World-space size of one patch side.
    pub patch_size_ws: f32,
    /// Camera distance at which morphing starts.
    pub morph_start: f32,
    /// Camera distance at which morphing reaches full blend (= coarse grid).
    pub morph_end: f32,
}

// ---------------------------------------------------------------------------
// Clipmap builder
// ---------------------------------------------------------------------------

/// Builds all `PatchInstanceCpu` descriptors for the current terrain view.
///
/// This runs on the CPU every frame. The result goes into a GPU storage
/// buffer via the render extract/prepare path.
pub fn build_patch_instances_for_view(
    config: &TerrainConfig,
    view: &TerrainViewState,
) -> Vec<PatchInstanceCpu> {
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
        let origins = build_ring_patch_origins(
            center,
            lvl_scale,
            config.patch_resolution,
            config.ring_patches,
            has_hole,
        );

        let patch_world_size = config.patch_resolution as f32 * lvl_scale;
        // The ring spans ring_patches * patch_size in each direction.
        // Morph starts at morph_start_ratio of that span from the inner edge.
        let ring_world_span = patch_world_size * config.ring_patches as f32;
        let band_near = ring_world_span * config.morph_start_ratio;
        let band_far = ring_world_span;

        for origin in origins {
            out.push(PatchInstanceCpu {
                lod_level: level,
                patch_kind: 0,
                origin_ws: origin,
                patch_size_ws: patch_world_size,
                morph_start: band_near,
                morph_end: band_far,
            });
        }
    }

    out
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

        // Collect level of each patch; verify they are grouped by level.
        let levels: Vec<u32> = patches.iter().map(|p| p.lod_level).collect();
        let mut prev = 0u32;
        for &l in &levels {
            assert!(l >= prev, "levels must be non-decreasing");
            prev = l;
        }
    }

    #[test]
    fn level_0_has_no_hole() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let patches = build_patch_instances_for_view(&config, &view);

        // Level 0 has no inner hole, so it should have ring_patches^2 patches.
        let count_l0 = patches.iter().filter(|p| p.lod_level == 0).count();
        assert_eq!(
            count_l0 as u32,
            config.ring_patches * config.ring_patches
        );
    }

    #[test]
    fn level_1_has_hole() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let patches = build_patch_instances_for_view(&config, &view);

        // Level 1 has an inner hole that removes the central quarter of patches.
        let count_l1 = patches.iter().filter(|p| p.lod_level == 1).count();
        let full = config.ring_patches * config.ring_patches;
        let hole_edge = config.ring_patches / 2;
        let hole = hole_edge * hole_edge;
        assert_eq!(count_l1 as u32, full - hole);
    }

    #[test]
    fn patch_sizes_double_per_level() {
        let config = TerrainConfig::default();
        let view = make_view(&config, Vec3::ZERO);
        let patches = build_patch_instances_for_view(&config, &view);

        for level in 1..config.active_clipmap_levels() {
            let prev_size = patches
                .iter()
                .find(|p| p.lod_level == level - 1)
                .map(|p| p.patch_size_ws)
                .unwrap();
            let this_size = patches
                .iter()
                .find(|p| p.lod_level == level)
                .map(|p| p.patch_size_ws)
                .unwrap();
            assert!(
                (this_size - prev_size * 2.0).abs() < 1e-4,
                "each level should double the patch size"
            );
        }
    }
}

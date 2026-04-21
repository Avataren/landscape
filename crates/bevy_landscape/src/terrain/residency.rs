use crate::terrain::{
    config::TerrainConfig,
    math::{compute_needed_tiles_for_level, level_scale, snap_camera_to_level_grid},
    resources::{TerrainResidency, TerrainViewState},
};
use bevy::prelude::*;

/// Recomputes the `required_now` set from the current view state and evicts
/// excess tiles.
pub fn update_required_tiles(
    config: Res<TerrainConfig>,
    view: Res<TerrainViewState>,
    mut residency: ResMut<TerrainResidency>,
) {
    residency.required_now.clear();

    for level in 0..config.active_clipmap_levels() {
        let scale = match view.level_scales.get(level as usize) {
            Some(&s) => s,
            None => level_scale(config.world_scale, level),
        };
        let center = match view.clip_centers.get(level as usize) {
            Some(&c) => c,
            None => snap_camera_to_level_grid(view.camera_pos_ws.xz(), scale),
        };

        let keys = compute_needed_tiles_for_level(
            center,
            scale,
            config.block_size(),
            config.tile_size,
            level as u8,
        );

        for key in keys {
            residency.required_now.insert(key);
        }
    }

    residency.evict_to_budget(config.max_resident_tiles);
}

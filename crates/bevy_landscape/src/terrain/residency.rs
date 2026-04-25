use crate::terrain::{
    config::TerrainConfig,
    resources::{TerrainResidency, TerrainViewState},
};
use bevy::prelude::*;

/// Recomputes the `required_now` set from the current view state and evicts
/// excess tiles.
///
/// The detail-synthesis compute pass now writes every clipmap layer (sourced
/// from the global source-heightmap texture, optionally with fBM detail), so
/// no per-ring CPU tile uploads are needed for rendering.  This function still
/// runs to keep eviction bounded, but produces an empty required_now set.
/// Collision tiles are loaded on-demand by the local-collider system.
pub fn update_required_tiles(
    config: Res<TerrainConfig>,
    _view: Res<TerrainViewState>,
    mut residency: ResMut<TerrainResidency>,
) {
    residency.required_now.clear();
    residency.tile_key_to_rings.clear();
    residency.evict_to_budget(config.max_resident_tiles);
}

use bevy::prelude::*;
use crate::terrain::{
    clipmap::{build_patch_instances_for_view, PatchInstanceCpu},
    config::TerrainConfig,
    resources::TerrainViewState,
};

// ---------------------------------------------------------------------------
// Extracted frame resource (Phase 0: lives in main world)
// ---------------------------------------------------------------------------

/// Snapshot of terrain state needed for rendering, built each frame.
/// In Phase 2+ this will be extracted into the render world via
/// `ExtractResource` and consumed by prepare/queue systems there.
#[derive(Resource, Clone, Default)]
pub struct ExtractedTerrainFrame {
    pub camera_pos_ws: Vec3,
    pub clip_centers: Vec<IVec2>,
    pub level_scales: Vec<f32>,
    pub patches: Vec<PatchInstanceCpu>,
    pub height_scale: f32,
    pub morph_start_ratio: f32,
    pub clipmap_levels: u32,
}

// ---------------------------------------------------------------------------
// Extract system (Phase 0: runs in main world Update)
// ---------------------------------------------------------------------------

/// Builds the extracted terrain frame from main-world resources.
/// Phase 2+: this moves to `bevy::render::Extract` schedule in the RenderApp.
pub fn extract_terrain_frame(
    config: Res<TerrainConfig>,
    view: Res<TerrainViewState>,
    mut extracted: ResMut<ExtractedTerrainFrame>,
) {
    let patches = build_patch_instances_for_view(&config, &view);

    extracted.camera_pos_ws = view.camera_pos_ws;
    extracted.clip_centers = view.clip_centers.clone();
    extracted.level_scales = view.level_scales.clone();
    extracted.patches = patches;
    extracted.height_scale = config.height_scale;
    extracted.morph_start_ratio = config.morph_start_ratio;
    extracted.clipmap_levels = config.clipmap_levels;
}

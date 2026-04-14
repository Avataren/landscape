use bevy::prelude::*;
use super::extract::ExtractedTerrainFrame;

/// Prepares GPU resources from the extracted frame.
/// Phase 0: no-op skeleton.
/// Phase 2+: uploads TerrainFrameUniform, patch descriptor storage buffer,
///           height clipmap textures, and refreshes bind groups.
#[allow(dead_code)]
pub fn prepare_terrain_gpu(_extracted: Res<ExtractedTerrainFrame>) {}

pub mod extract;
pub mod pipelines;
pub mod prepare;
pub mod queue;

use bevy::prelude::*;

/// Placeholder for the custom render pipeline (extract → prepare → queue → draw).
/// When implemented, this plugin will wire into Bevy's RenderApp with storage
/// buffers, partial texture uploads, and indirect draw commands.
pub struct TerrainRenderPlugin;

impl Plugin for TerrainRenderPlugin {
    fn build(&self, _app: &mut App) {}
}

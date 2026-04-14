pub mod extract;
pub mod gpu_types;
pub mod pipelines;
pub mod prepare;
pub mod queue;

use bevy::prelude::*;
use extract::ExtractedTerrainFrame;

// ---------------------------------------------------------------------------
// Render plugin (Phase 0 skeleton)
// ---------------------------------------------------------------------------
// The full custom render pipeline (extract → prepare → queue → draw) requires
// wiring into Bevy's render graph with storage buffers and custom shaders.
// That is Phase 2+ work.  For Phase 0/1 we store extracted data in a plain
// main-world resource and read it from update systems.

pub struct TerrainRenderPlugin;

impl Plugin for TerrainRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ExtractedTerrainFrame>();
        // Phase 2+: wire RenderApp extract/prepare/queue here.
    }
}

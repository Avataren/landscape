use serde::{Deserialize, Serialize};
use std::path::Path;

/// Per-tile-root metadata persisted alongside the clip-mipmap tiles.
///
/// Written as `{tile_root}/metadata.toml` by the generator export and the
/// heightmap bake tool.  Loaded automatically when a level is resolved through
/// `LevelDesc::into_runtime`.  Fields are all optional so the file is forward-
/// and backward-compatible as new metadata is added.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TerrainMetadata {
    /// Normalised sea level in [0, 1] relative to the tile height range.
    /// A tile value of 1.0 = `height_scale` world units; water_level maps to
    /// `water_level * height_scale * world_scale` world units on the Y axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_level: Option<f32>,
}

impl TerrainMetadata {
    pub fn load(tile_root: &Path) -> Self {
        let path = tile_root.join("metadata.toml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self, tile_root: &Path) -> Result<(), String> {
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(tile_root.join("metadata.toml"), text).map_err(|e| e.to_string())
    }
}

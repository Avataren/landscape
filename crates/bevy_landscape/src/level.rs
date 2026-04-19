//! Level descriptor — a serialisable snapshot of everything needed to load one
//! terrain level at runtime.
//!
//! A *level file* is a JSON document (typically `my_level.json`) that records
//! the tile paths, world-scale parameters, and material definitions.  It is the
//! natural save artifact for the upcoming UI-based heightmap import workflow and
//! the single file a user needs to hand another machine to reproduce a scene.
//!
//! # Typical use
//! ```ignore
//! // Save from the running editor:
//! let desc = LevelDesc::from_current(&config, &source_desc, &material_library);
//! save_level("my_level.json", &desc)?;
//!
//! // Load at startup:
//! let desc = load_level("my_level.json")?;
//! let (config, source_desc, material_library) = desc.into_runtime();
//! ```

use crate::terrain::{
    config::{TerrainConfig, MAX_SUPPORTED_CLIPMAP_LEVELS},
    material_slots::MaterialLibrary,
    world_desc::TerrainSourceDesc,
};
use bevy::prelude::Vec2;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// LevelDesc
// ---------------------------------------------------------------------------

/// All level-specific data that can be persisted to a JSON file and reloaded.
///
/// Fields mirror the relevant parts of `landscape.toml` + `MaterialLibrary`,
/// unified into one document so the whole level is self-contained.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelDesc {
    /// Root directory of pre-baked height / normal tiles produced by `bake_tiles`.
    pub tile_root: Option<String>,
    /// Pre-baked normal tile root directory.
    pub normal_root: Option<String>,
    /// Path to the world-aligned macro/diffuse colour EXR.
    pub diffuse_path: Option<String>,
    /// Highest LOD index that has baked tiles in the tile hierarchy.
    pub max_mip_level: u8,
    /// Mip level used to build the global collision heightfield.
    pub collision_mip_level: u8,
    /// Uniform world scale applied to both horizontal (X/Z) and vertical (Y)
    /// extents.  World bounds are derived from the tile grid at load time.
    pub world_scale: f32,
    /// Base height range before `world_scale` is applied: a fully-white height
    /// texel (R16Unorm = 1.0) maps to this many world units on the Y axis.
    pub height_scale: f32,
    /// Number of nested clipmap LOD levels. This is the saved landscape view
    /// distance control exposed in the editor toolbar.
    pub clipmap_levels: u32,
    /// Procedural material slot definitions.
    pub material_library: MaterialLibrary,
}

impl Default for LevelDesc {
    fn default() -> Self {
        let default_config = TerrainConfig::default();
        Self {
            tile_root: None,
            normal_root: None,
            diffuse_path: None,
            max_mip_level: 5,
            collision_mip_level: 2,
            world_scale: 1.0,
            height_scale: default_config.height_scale,
            clipmap_levels: default_config.clipmap_levels,
            material_library: MaterialLibrary::default(),
        }
    }
}

impl LevelDesc {
    /// Build a `LevelDesc` from the currently active runtime state.
    pub fn from_current(
        config: &TerrainConfig,
        source: &TerrainSourceDesc,
        library: &MaterialLibrary,
    ) -> Self {
        // Recover the base height_scale (before world_scale was folded in).
        let base_height_scale = if config.world_scale > 0.0 {
            config.height_scale / config.world_scale
        } else {
            config.height_scale
        };

        Self {
            tile_root: source
                .tile_root
                .as_deref()
                .and_then(|p| p.to_str())
                .map(String::from),
            normal_root: source.normal_root.clone(),
            diffuse_path: source.macro_color_root.clone(),
            max_mip_level: source.max_mip_level,
            collision_mip_level: source.collision_mip_level,
            world_scale: config.world_scale,
            height_scale: base_height_scale,
            clipmap_levels: config.clipmap_levels,
            material_library: library.clone(),
        }
    }

    /// Resolve this descriptor into runtime objects ready for `TerrainPlugin`.
    ///
    /// World bounds are computed by scanning the tile directory (same logic as
    /// the startup config path).  They are returned separately because the
    /// caller must apply them to `TerrainSourceDesc` after construction.
    pub fn into_runtime(
        self,
    ) -> (
        TerrainConfig,
        TerrainSourceDesc,
        MaterialLibrary,
        Vec2,
        Vec2,
    ) {
        const TILE_SIZE: u32 = 256;

        let mut config = TerrainConfig::default();
        config.world_scale = self.world_scale;
        config.height_scale = self.height_scale * self.world_scale;
        config.clipmap_levels = self
            .clipmap_levels
            .max(1)
            .min(MAX_SUPPORTED_CLIPMAP_LEVELS as u32);

        let tile_root = self.tile_root.as_deref().map(PathBuf::from);
        let (world_min, world_max) = tile_root
            .as_deref()
            .map(|p| scan_world_bounds(p, TILE_SIZE, self.world_scale))
            .unwrap_or_else(|| {
                let h = 8192.0 * self.world_scale;
                (Vec2::splat(-h), Vec2::splat(h))
            });

        let source = TerrainSourceDesc {
            tile_root,
            normal_root: self.normal_root,
            macro_color_root: self.diffuse_path,
            material_root: None,
            world_min,
            world_max,
            max_mip_level: self.max_mip_level,
            collision_mip_level: self.collision_mip_level,
        };

        (config, source, self.material_library, world_min, world_max)
    }
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Serialises a `LevelDesc` to pretty-printed JSON and writes it to `path`.
pub fn save_level(path: impl AsRef<Path>, desc: &LevelDesc) -> Result<(), String> {
    let json = serde_json::to_string_pretty(desc).map_err(|e| e.to_string())?;
    std::fs::write(path.as_ref(), json).map_err(|e| e.to_string())?;
    Ok(())
}

/// Reads and deserialises a `LevelDesc` from a JSON file at `path`.
pub fn load_level(path: impl AsRef<Path>) -> Result<LevelDesc, String> {
    let json = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
    serde_json::from_str(&json).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Internal: tile-directory scan (shared logic with app_config)
// ---------------------------------------------------------------------------

fn scan_world_bounds(tile_root: &Path, tile_size: u32, world_scale: f32) -> (Vec2, Vec2) {
    let fallback_half = 8192.0 * world_scale;
    let fallback = (Vec2::splat(-fallback_half), Vec2::splat(fallback_half));

    let l0_dir = tile_root.join("height").join("L0");
    let Ok(dir) = std::fs::read_dir(&l0_dir) else {
        return fallback;
    };

    let mut min_tx = i32::MAX;
    let mut min_ty = i32::MAX;
    let mut max_tx = i32::MIN;
    let mut max_ty = i32::MIN;
    let mut found = false;

    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(stem) = name.to_str().and_then(|s| s.strip_suffix(".bin")) else {
            continue;
        };
        let Some((tx_str, ty_str)) = stem.split_once('_') else {
            continue;
        };
        let (Ok(tx), Ok(ty)) = (tx_str.parse::<i32>(), ty_str.parse::<i32>()) else {
            continue;
        };
        min_tx = min_tx.min(tx);
        min_ty = min_ty.min(ty);
        max_tx = max_tx.max(tx);
        max_ty = max_ty.max(ty);
        found = true;
    }

    if !found {
        return fallback;
    }

    let ts = tile_size as f32 * world_scale;
    (
        Vec2::new(min_tx as f32 * ts, min_ty as f32 * ts),
        Vec2::new((max_tx + 1) as f32 * ts, (max_ty + 1) as f32 * ts),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_preserves_saved_view_distance() {
        let mut desc = LevelDesc::default();
        desc.max_mip_level = 3;
        desc.clipmap_levels = 12;

        let (config, _, _, _, _) = desc.into_runtime();

        assert_eq!(config.clipmap_levels, 12);
    }

    #[test]
    fn runtime_clamps_view_distance_to_shader_limit() {
        let mut desc = LevelDesc::default();
        desc.clipmap_levels = 99;

        let (config, _, _, _, _) = desc.into_runtime();

        assert_eq!(config.clipmap_levels, MAX_SUPPORTED_CLIPMAP_LEVELS as u32);
    }
}

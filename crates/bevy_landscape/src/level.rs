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

use crate::metadata::TerrainMetadata;
use crate::terrain::{
    config::{TerrainConfig, MAX_SUPPORTED_CLIPMAP_LEVELS},
    material_slots::MaterialLibrary,
    world_desc::TerrainSourceDesc,
};
use bevy::prelude::Vec2;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

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
    /// Root directory for foliage instance tiles and splatmaps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foliage_root: Option<String>,
    /// Highest LOD index that has baked tiles in the tile hierarchy.
    pub max_mip_level: u8,
    /// Mip level used to build the global collision heightfield.
    pub collision_mip_level: u8,
    /// Physical scale of the source tiles: world-space metres per tile texel.
    /// Used to compute source_spacing = world_scale × 2^max_mip_level.
    pub world_scale: f32,
    /// Finest clipmap mesh vertex spacing in world-space metres.
    /// Defaults to `world_scale` when absent (backward-compatible).
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub lod0_mesh_spacing: f32,
    /// Base height range before `world_scale` is applied: a fully-white height
    /// texel (R16Unorm = 1.0) maps to this many world units on the Y axis.
    pub height_scale: f32,
    /// Number of nested clipmap LOD levels.  When non-zero this overrides the
    /// world-bounds-derived value in `into_runtime()`; 0 means "derive from
    /// world bounds" (backward-compatible default for old level files).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub clipmap_levels: u32,
    /// Procedural material slot definitions.
    pub material_library: MaterialLibrary,
    /// Foliage generation and rendering configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foliage_config: Option<crate::FoliageConfig>,
    /// Cloud renderer settings. Stored as a raw JSON value so `bevy_landscape`
    /// does not need to depend on `bevy_landscape_clouds`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clouds: Option<serde_json::Value>,
    /// Sky / time-of-day settings. Stored as raw JSON for the same reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sky: Option<serde_json::Value>,
    /// Water renderer settings. Stored as raw JSON so `bevy_landscape` does
    /// not need to depend on `bevy_landscape_water`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water: Option<serde_json::Value>,
    /// Detail synthesis (fBM) settings. Stored as raw JSON so `bevy_landscape`
    /// can remain independent of the editor's synthesis configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<serde_json::Value>,
    /// Metadata loaded from `{tile_root}/metadata.toml` (water level, etc.).
    /// Merged from the sidecar file during `into_runtime`; also preserved here
    /// so a saved level.json is self-contained even without the tile directory.
    #[serde(default, skip_serializing_if = "is_default_metadata")]
    pub metadata: TerrainMetadata,
}

fn is_default_metadata(m: &TerrainMetadata) -> bool {
    m.water_level.is_none()
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

impl Default for LevelDesc {
    fn default() -> Self {
        let default_config = TerrainConfig::default();
        Self {
            tile_root: None,
            normal_root: None,
            diffuse_path: None,
            foliage_root: None,
            max_mip_level: 5,
            collision_mip_level: 2,
            world_scale: 1.0,
            lod0_mesh_spacing: 0.0, // 0 = use world_scale (backward-compat sentinel)
            height_scale: default_config.height_scale,
            clipmap_levels: 0, // 0 = derive from world bounds at runtime
            material_library: MaterialLibrary::default(),
            foliage_config: None,
            clouds: None,
            sky: None,
            water: None,
            synthesis: None,
            metadata: TerrainMetadata::default(),
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
            foliage_root: source.foliage_root.clone(),
            max_mip_level: source.max_mip_level,
            collision_mip_level: source.collision_mip_level,
            world_scale: config.world_scale,
            lod0_mesh_spacing: config.lod0_mesh_spacing,
            height_scale: base_height_scale,
            clipmap_levels: config.clipmap_levels,
            material_library: library.clone(),
            foliage_config: None,
            clouds: None,
            sky: None,
            water: None,
            synthesis: None,
            metadata: TerrainMetadata::default(),
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
        TerrainMetadata,
    ) {
        const TILE_SIZE: u32 = 256;

        let mut config = TerrainConfig::default();
        config.world_scale = self.world_scale;
        // 0.0 is the sentinel meaning "not set" — fall back to world_scale.
        config.lod0_mesh_spacing = if self.lod0_mesh_spacing > 0.0 {
            self.lod0_mesh_spacing
        } else {
            self.world_scale
        };
        config.height_scale = self.height_scale * self.world_scale;

        let tile_root = self.tile_root.as_deref().map(PathBuf::from);
        let (world_min, world_max) = tile_root
            .as_deref()
            .map(|p| scan_world_bounds(p, TILE_SIZE, self.world_scale))
            .unwrap_or_else(|| {
                let h = 8192.0 * self.world_scale;
                (Vec2::splat(-h), Vec2::splat(h))
            });

        // Honor the explicit `clipmap_levels` from the level file when set;
        // otherwise derive from world extent so every ring has real tile data.
        //
        // Ring L half-width = block_size × 2 × (lod0_mesh_spacing × 2^L).
        // We find the largest L where this fits within the world half-extent,
        // then set clipmap_levels = L + 1.
        config.clipmap_levels = if self.clipmap_levels > 0 {
            self.clipmap_levels
                .clamp(1, MAX_SUPPORTED_CLIPMAP_LEVELS as u32)
        } else {
            let block_size = config.block_size() as f32;
            let world_half = {
                let dx = (world_max.x - world_min.x) * 0.5;
                let dz = (world_max.y - world_min.y) * 0.5;
                dx.min(dz).max(1.0)
            };
            let min_ring_half = block_size * 2.0 * config.lod0_mesh_spacing;
            let max_level = if min_ring_half > 0.0 {
                (world_half / min_ring_half).log2().floor() as u32
            } else {
                0
            };
            (max_level + 1).clamp(1, MAX_SUPPORTED_CLIPMAP_LEVELS as u32)
        };

        // Merge metadata: sidecar file takes precedence over level.json field
        // so that re-exporting tiles always reflects the latest values.
        let metadata = if let Some(root) = tile_root.as_deref() {
            let sidecar = TerrainMetadata::load(root);
            TerrainMetadata {
                water_level: sidecar.water_level.or(self.metadata.water_level),
            }
        } else {
            self.metadata
        };

        let source = TerrainSourceDesc {
            tile_root,
            normal_root: self.normal_root,
            macro_color_root: self.diffuse_path,
            foliage_root: self.foliage_root.clone(),
            material_root: None,
            world_min,
            world_max,
            max_mip_level: self.max_mip_level,
            collision_mip_level: self.collision_mip_level,
        };

        (
            config,
            source,
            self.material_library,
            world_min,
            world_max,
            metadata,
        )
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
    fn runtime_derives_clipmap_levels_from_world_bounds() {
        // No tile_root → fallback world bounds = ±8192 (world_scale=1).
        // block_size = 128, min_ring_half = 256 m.
        // max_level = floor(log2(8192/256)) = floor(log2(32)) = 5.
        // Expected clipmap_levels = 6.
        let desc = LevelDesc::default();
        let (config, _, _, _, _, _) = desc.into_runtime();
        assert_eq!(config.clipmap_levels, 6);
    }

    #[test]
    fn runtime_clamps_view_distance_to_shader_limit() {
        // Verify the formula result stays within [1, MAX_SUPPORTED_CLIPMAP_LEVELS].
        // With the fallback bounds (±8192, world_scale=1) and default clipmap_n=511
        // (block_size=128, min_ring_half=256), ratio=32, max_level=5, levels=6.
        // The MAX clamp is a safety net for unusually large real-tile datasets.
        let desc = LevelDesc::default();
        let (config, _, _, _, _, _) = desc.into_runtime();
        assert!(config.clipmap_levels >= 1);
        assert!(config.clipmap_levels <= MAX_SUPPORTED_CLIPMAP_LEVELS as u32);
    }
}

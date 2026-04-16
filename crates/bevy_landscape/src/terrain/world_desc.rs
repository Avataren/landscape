use bevy::prelude::*;

/// Describes where terrain source data lives and the world extents.
#[derive(Resource, Clone, Debug)]
pub struct TerrainSourceDesc {
    /// Optional root path for precomputed normal tiles.
    pub normal_root: Option<String>,
    /// Optional root path for material/mask tiles.
    pub material_root: Option<String>,
    /// Optional path for a world-aligned macro/diffuse color map.
    pub macro_color_root: Option<String>,
    /// World-space XZ minimum (2D footprint).
    pub world_min: Vec2,
    /// World-space XZ maximum (2D footprint).
    pub world_max: Vec2,
    /// Maximum mip level available in the tile hierarchy.
    pub max_mip_level: u8,
    /// Mip level used to build the global collision heightfield.
    /// Level 2 = 4 m/cell for world_scale = 1.0 (matches per-tile resolution).
    /// Lower = finer (more RAM, slower startup); higher = coarser (less RAM).
    /// See physics_colliders.rs for the memory/tile-count trade-offs.
    pub collision_mip_level: u8,
    /// Root directory of pre-baked tile files produced by `bake_tiles`.
    /// Subdirectory layout: `height/L{n}/{tx}_{ty}.bin` (R16Unorm, 256×256 texels).
    /// Falls back to procedural generation when `None` or when a tile file is missing.
    pub tile_root: Option<std::path::PathBuf>,
}

impl Default for TerrainSourceDesc {
    fn default() -> Self {
        Self {
            normal_root: None,
            material_root: None,
            macro_color_root: None,
            world_min: Vec2::ZERO,
            world_max: Vec2::ZERO,
            max_mip_level: 0,
            collision_mip_level: 2,
            tile_root: None,
        }
    }
}

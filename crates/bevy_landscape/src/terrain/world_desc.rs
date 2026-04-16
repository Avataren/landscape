use bevy::prelude::*;

/// Describes where terrain source data lives and the world extents.
#[derive(Resource, Clone, Default, Debug)]
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
    /// Root directory of pre-baked tile files produced by `bake_tiles`.
    /// Subdirectory layout: `height/L{n}/{tx}_{ty}.bin` (R16Unorm, 256×256 texels).
    /// Falls back to procedural generation when `None` or when a tile file is missing.
    pub tile_root: Option<std::path::PathBuf>,
}

impl TerrainSourceDesc {}

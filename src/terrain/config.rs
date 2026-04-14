use bevy::prelude::*;

/// Central configuration for the terrain renderer.
/// Tweak these values to scale world size and quality.
#[derive(Resource, Clone, Debug)]
pub struct TerrainConfig {
    /// Number of nested clipmap LOD levels.
    pub clipmap_levels: u32,
    /// Vertex count per patch edge (NxN grid). Must be power-of-two friendly.
    pub patch_resolution: u32,
    /// Number of patches per ring edge side.
    pub ring_patches: u32,
    /// Height/material tile texel resolution (square).
    pub tile_size: u32,
    /// Resolution of each clipmap level texture (square).
    pub clipmap_resolution: u32,
    /// World-space units per texel at LOD 0.
    pub world_scale: f32,
    /// World-space units for maximum terrain height (maps [0,1] height -> [0, height_scale]).
    pub height_scale: f32,
    /// Distance ratio within a ring at which morphing begins (0..1).
    pub morph_start_ratio: f32,
    /// Maximum number of terrain tiles kept resident on GPU.
    pub max_resident_tiles: usize,
    /// Maximum view distance for terrain (used for LOD scale computation).
    pub max_view_distance: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            clipmap_levels: 6,
            patch_resolution: 64,
            ring_patches: 8,
            tile_size: 256,
            clipmap_resolution: 2048,
            world_scale: 1.0,
            height_scale: 512.0,
            morph_start_ratio: 0.6,
            max_resident_tiles: 256,
            max_view_distance: 65536.0,
        }
    }
}

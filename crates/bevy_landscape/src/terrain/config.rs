use bevy::prelude::*;

pub const MAX_SUPPORTED_CLIPMAP_LEVELS: usize = 16;

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
    /// Uniform terrain scale multiplier at LOD 0, expressed as world-space
    /// units per height texel in X/Z.
    ///
    /// This is the runtime value used by terrain rendering, streaming, and
    /// collision. In the app binary it is normally loaded from
    /// `landscape.toml` `[terrain_config] world_scale`; the `1.0` default
    /// below is only a fallback when no external config overrides it.
    pub world_scale: f32,
    /// Effective runtime world-space Y range: a fully-white height texel
    /// (R16Unorm = 1.0) maps to this many world units after the app folds in
    /// the uniform `world_scale` multiplier. The compile-time default below is
    /// only a fallback before external config is applied.
    ///
    /// **Must match the height_scale used during baking.**  bake_tiles reads
    /// the same toml key automatically, so keeping it in one place is enough.
    /// Baked normals are computed with this scale; a mismatch makes slopes
    /// appear too flat or too steep in lighting.
    pub height_scale: f32,
    /// Distance ratio within a ring at which morphing begins (0..1).
    pub morph_start_ratio: f32,
    /// Maximum number of terrain tiles kept resident on GPU.
    pub max_resident_tiles: usize,
    /// Maximum view distance for terrain (used for LOD scale computation).
    #[allow(dead_code)]
    pub max_view_distance: f32,
    /// Fill clipmap layers and missing tiles with procedural sine-wave heights.
    /// Disabled by default — leave false when real tile data is available so
    /// unloaded regions show as flat (height 0) rather than mismatched bumps.
    pub procedural_fallback: bool,
    /// When true, sample the world-aligned diffuse EXR as terrain albedo.
    /// Disable to fall back to the procedural slope/altitude shading.
    pub use_macro_color_map: bool,
    /// Maximum resolution of the loaded macro color texture after startup
    /// downsampling. Keeps the 16k diffuse map at a practical runtime size.
    pub macro_color_resolution: u32,
    /// Flip the V (world-Z) axis when sampling the macro color map.
    /// Set to true when the diffuse EXR was exported with V=0 at the bottom
    /// (OpenGL / Houdini convention) so that mountain colours align with terrain.
    pub macro_color_flip_v: bool,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            clipmap_levels: 12,
            patch_resolution: 64,
            ring_patches: 8,
            tile_size: 256,
            world_scale: 1.0,
            height_scale: 1024.0,
            morph_start_ratio: 0.6,
            max_resident_tiles: 256,
            max_view_distance: 65536.0,
            procedural_fallback: false,
            use_macro_color_map: true,
            macro_color_resolution: 16384,
            macro_color_flip_v: false,
        }
    }
}

impl TerrainConfig {
    /// Number of clipmap levels the current shader/material layout can represent.
    pub fn active_clipmap_levels(&self) -> u32 {
        self.clipmap_levels.min(MAX_SUPPORTED_CLIPMAP_LEVELS as u32)
    }

    /// Effective resolution of each clipmap level texture.
    ///
    /// This is derived from the number of ring patches and the patch mesh
    /// resolution so the clipmap texel grid always matches the terrain grid.
    pub fn clipmap_resolution(&self) -> u32 {
        self.ring_patches * self.patch_resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipmap_resolution_tracks_ring_geometry() {
        let config = TerrainConfig::default();
        assert_eq!(
            config.clipmap_resolution(),
            config.ring_patches * config.patch_resolution
        );
    }
}

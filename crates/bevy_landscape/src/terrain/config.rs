use bevy::prelude::*;

pub const MAX_SUPPORTED_CLIPMAP_LEVELS: usize = 32;

/// Central configuration for the terrain renderer.
/// Tweak these values to scale world size and quality.
#[derive(Resource, Clone, Debug)]
pub struct TerrainConfig {
    /// Number of nested clipmap LOD levels.
    pub clipmap_levels: u32,
    /// Clipmap grid size. Must be `2^k - 1` (e.g. 511).
    ///
    /// Drives the GPU Gems 2 nested-grid layout:
    ///   block_size  m = (clipmap_n + 1) / 4   (quads per canonical block edge)
    ///   ring width    = 4 m grid units
    ///   inner hole    = 2 m grid units
    ///   texture res   = clipmap_n + 1 texels per axis
    ///
    /// Default 511 → m = 128, texture 512 × 512 — matches the previous
    /// `ring_patches = 8, patch_resolution = 64` resolution.
    pub clipmap_n: u32,
    /// Height/material tile texel resolution (square).
    pub tile_size: u32,
    /// Physical scale of the source heightmap tiles: world-space metres per
    /// tile texel at mip-level 0.  Used to interpret source data dimensions
    /// and compute `source_spacing = world_scale × 2^max_mip_level`.
    ///
    /// Loaded from `landscape.toml` `[terrain_config] world_scale`.
    pub world_scale: f32,

    /// Finest vertex/texel spacing of the runtime clipmap mesh, in world-space
    /// metres.  Each successive LOD doubles this value.
    ///
    /// Decoupled from `world_scale` so the mesh resolution can be set
    /// independently of the source tile pixel size.  For example, with a 30 m
    /// source and `lod0_mesh_spacing = 2.0`, LOD 0 has 2 m vertices even
    /// though the source only has data every 30 m — the synthesis pass fills
    /// the gap.
    ///
    /// Defaults to `world_scale` if not set (backward-compatible).
    pub lod0_mesh_spacing: f32,
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
    /// Inner ratio of the ring half-extent at which LOD morphing begins (0..1).
    /// 0.6 = morphing starts at 60 % of half-ring from centre (outer 40 % zone).
    pub morph_start_ratio: f32,
    /// Maximum number of terrain tiles kept resident on GPU.
    pub max_resident_tiles: usize,
    /// Maximum view distance for terrain (used for LOD scale computation).
    #[allow(dead_code)]
    pub max_view_distance: f32,
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
            // 511 = 2^9 - 1 → m = 128, texture 512×512 (same resolution as
            // the old ring_patches=8 × patch_resolution=64 defaults).
            clipmap_n: 511,
            tile_size: 256,
            world_scale: 1.0,
            lod0_mesh_spacing: 1.0,
            height_scale: 1024.0,
            morph_start_ratio: 0.6,
            max_resident_tiles: 256,
            max_view_distance: 65536.0,
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

    /// Canonical block size: quads per block edge.
    ///
    /// In the GPU Gems 2 layout, each ring decomposes into 12 (or 16 for level 0)
    /// `m × m` canonical blocks where `m = (clipmap_n + 1) / 4`.
    pub fn block_size(&self) -> u32 {
        (self.clipmap_n + 1) / 4
    }

    /// Texel resolution of each clipmap level texture.
    ///
    /// Equals `clipmap_n + 1` — the number of grid vertices along one ring
    /// edge, which spans 4 block-widths = 4 m grid units.
    pub fn clipmap_resolution(&self) -> u32 {
        self.clipmap_n + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_divides_resolution() {
        let config = TerrainConfig::default();
        assert_eq!(config.clipmap_resolution(), 4 * config.block_size());
    }

    #[test]
    fn clipmap_n_is_valid() {
        let config = TerrainConfig::default();
        let n = config.clipmap_n;
        // n must be 2^k - 1 for the GPU Gems 2 odd-n constraint.
        assert_eq!(
            (n + 1).count_ones(),
            1,
            "clipmap_n + 1 must be a power of two"
        );
    }
}

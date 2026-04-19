use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct GeneratorParams {
    /// Resolution of the GPU preview texture (must be power of two, ≤2048).
    pub resolution:        u32,
    /// Resolution of the exported tile hierarchy (must be power of two, multiple of 256).
    pub export_resolution: u32,
    pub octaves:           u32,
    pub frequency:         f32,
    pub lacunarity:        f32,
    pub gain:              f32,
    pub height_scale:      f32,
    pub world_scale:       f32,
    pub offset:            Vec2,
    pub seed:              u32,
}

impl Default for GeneratorParams {
    fn default() -> Self {
        Self {
            resolution:        1024,
            export_resolution: 4096,
            octaves:           6,
            frequency:         2.5,
            lacunarity:        2.0,
            gain:              0.5,
            height_scale:      1024.0,
            world_scale:       2.0,
            offset:            Vec2::ZERO,
            seed:              42,
        }
    }
}

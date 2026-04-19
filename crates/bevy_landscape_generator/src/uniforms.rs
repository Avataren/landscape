use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{ShaderType, UniformBuffer},
    },
};

use crate::params::GeneratorParams;

/// GPU-side uniform mirroring the WGSL `GeneratorParams` struct.
/// Layout (48 bytes, 3 × 16-byte rows):
///   row 0: resolution(8) + octaves(4) + seed(4)
///   row 1: offset(8) + frequency(4) + lacunarity(4)
///   row 2: gain(4) + height_scale(4) + pad(8)
#[derive(Clone, Resource, ExtractResource, ShaderType)]
pub(crate) struct GeneratorUniform {
    pub resolution:   UVec2,
    pub octaves:      u32,
    pub seed:         u32,
    pub offset:       Vec2,
    pub frequency:    f32,
    pub lacunarity:   f32,
    pub gain:         f32,
    pub height_scale: f32,
    pub pad:          Vec2,
}

impl Default for GeneratorUniform {
    fn default() -> Self {
        Self::from_params(&GeneratorParams::default())
    }
}

impl GeneratorUniform {
    pub fn from_params(p: &GeneratorParams) -> Self {
        Self {
            resolution:   UVec2::splat(p.resolution),
            octaves:      p.octaves,
            seed:         p.seed,
            offset:       p.offset,
            frequency:    p.frequency,
            lacunarity:   p.lacunarity,
            gain:         p.gain,
            height_scale: p.height_scale,
            pad:          Vec2::ZERO,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct GeneratorUniformBuffer {
    pub buffer: UniformBuffer<GeneratorUniform>,
}

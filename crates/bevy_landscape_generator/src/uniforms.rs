use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{ShaderType, UniformBuffer},
    },
};

use crate::params::GeneratorParams;

/// GPU-side uniform mirroring the WGSL `GeneratorParams` struct.
/// Layout (80 bytes, 5 × 16-byte rows):
///   row 0: resolution(8) + octaves(4) + seed(4)
///   row 1: offset(8) + frequency(4) + lacunarity(4)
///   row 2: gain(4) + height_scale(4) + continent_frequency(4) + continent_strength(4)
///   row 3: ridge_strength(4) + warp_frequency(4) + warp_strength(4) + erosion_strength(4)
///   row 4: grayscale(4) + pad(12)
#[derive(Clone, Resource, ExtractResource, ShaderType)]
pub(crate) struct GeneratorUniform {
    pub resolution: UVec2,
    pub octaves: u32,
    pub seed: u32,
    pub offset: Vec2,
    pub frequency: f32,
    pub lacunarity: f32,
    pub gain: f32,
    pub height_scale: f32,
    pub continent_frequency: f32,
    pub continent_strength: f32,
    pub ridge_strength: f32,
    pub warp_frequency: f32,
    pub warp_strength: f32,
    pub erosion_strength: f32,
    pub grayscale: u32,
}

impl Default for GeneratorUniform {
    fn default() -> Self {
        Self::from_params(&GeneratorParams::default())
    }
}

impl GeneratorUniform {
    pub fn from_params(p: &GeneratorParams) -> Self {
        Self {
            resolution: UVec2::splat(p.resolution),
            octaves: p.octaves,
            seed: p.seed,
            offset: p.offset,
            frequency: p.frequency,
            lacunarity: p.lacunarity,
            gain: p.gain,
            height_scale: p.height_scale,
            continent_frequency: p.continent_frequency,
            continent_strength: p.continent_strength,
            ridge_strength: p.ridge_strength,
            warp_frequency: p.warp_frequency,
            warp_strength: p.warp_strength,
            erosion_strength: p.erosion_strength,
            grayscale: p.grayscale,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct GeneratorUniformBuffer {
    pub buffer: UniformBuffer<GeneratorUniform>,
}

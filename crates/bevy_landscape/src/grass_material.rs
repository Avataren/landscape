//! Grass material shader and GPU uniforms.
//!
//! Simple grass shader supporting:
//! - World-aligned normals for grass blade lighting
//! - Subtle ambient occlusion from vertex paint (placeholder)
//! - Support for 8 blade variants via vertex ID
//! - Simple wind animation (can be extended later)

use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;

/// GPU uniforms for grass rendering.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GrassMaterial {
    /// Base color for grass blades (RGB).
    #[uniform(0)]
    pub base_color: LinearRgba,
    /// Ambient occlusion multiplier (0..1).
    #[uniform(1)]
    pub ao_multiplier: f32,
    /// Wind strength (0..1); 0 = no animation.
    #[uniform(2)]
    pub wind_strength: f32,
    /// Wind speed (cycles per second).
    #[uniform(3)]
    pub wind_speed: f32,
}

impl Default for GrassMaterial {
    fn default() -> Self {
        Self {
            base_color: LinearRgba::rgb(0.3, 0.5, 0.2), // Green
            ao_multiplier: 0.8,
            wind_strength: 0.0, // No wind by default
            wind_speed: 1.0,
        }
    }
}

impl Material for GrassMaterial {
    fn fragment_shader() -> bevy::shader::ShaderRef {
        "shaders/grass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

// Note: The shader code is embedded via bevy's shader loader.
// See grass.wgsl for the actual shader source.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grass_material_default() {
        let mat = GrassMaterial::default();
        assert_eq!(mat.base_color, LinearRgba::rgb(0.3, 0.5, 0.2));
        assert!(mat.ao_multiplier >= 0.0 && mat.ao_multiplier <= 1.0);
        assert!(mat.wind_strength >= 0.0 && mat.wind_strength <= 1.0);
        assert!(mat.wind_speed > 0.0);
    }

    #[test]
    fn test_grass_material_custom() {
        let mat = GrassMaterial {
            base_color: LinearRgba::rgb(0.2, 0.6, 0.1),
            ao_multiplier: 0.7,
            wind_strength: 0.5,
            wind_speed: 2.0,
        };
        assert_eq!(mat.base_color, LinearRgba::rgb(0.2, 0.6, 0.1));
        assert_eq!(mat.ao_multiplier, 0.7);
        assert_eq!(mat.wind_strength, 0.5);
        assert_eq!(mat.wind_speed, 2.0);
    }

    #[test]
    fn test_grass_material_alpha_mode() {
        let mat = GrassMaterial::default();
        assert_eq!(mat.alpha_mode(), AlphaMode::Blend);
    }
}

use bevy::prelude::*;

#[derive(Resource, Clone, Copy)]
pub struct CloudsConfig {
    pub render_resolution: UVec2,
    pub cloud_view_steps: u32,
    pub cloud_shadow_steps: u32,
    pub cloud_shadow_step_size: f32,
    pub cloud_shadow_step_multiply: f32,
    pub planet_radius: f32,
    pub cloud_bottom_height: f32,
    pub cloud_top_height: f32,
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub cloud_base_scale: f32,
    pub cloud_detail_scale: f32,
    pub cloud_detail_strength: f32,
    pub cloud_base_edge_softness: f32,
    pub cloud_bottom_softness: f32,
    pub cloud_history_blend: f32,
    pub cloud_ambient_color_top: Vec4,
    pub cloud_ambient_color_bottom: Vec4,
    pub cloud_min_transmittance: f32,
    pub forward_scattering_g: f32,
    pub backward_scattering_g: f32,
    pub scattering_lerp: f32,
    pub wind_velocity: Vec3,
    pub sun_color: Vec4,
}

impl Default for CloudsConfig {
    fn default() -> Self {
        Self {
            render_resolution: UVec2::new(960, 540),
            cloud_view_steps: 16,
            cloud_shadow_steps: 6,
            cloud_shadow_step_size: 10.0,
            cloud_shadow_step_multiply: 1.3,
            planet_radius: 6_371_000.0,
            cloud_bottom_height: 1_250.0,
            cloud_top_height: 2_400.0,
            cloud_coverage: 0.62,
            cloud_density: 0.03,
            cloud_base_scale: 1.5,
            cloud_detail_scale: 42.0,
            cloud_detail_strength: 0.27,
            cloud_base_edge_softness: 0.10,
            cloud_bottom_softness: 0.25,
            cloud_history_blend: 0.9,
            cloud_ambient_color_top: Vec4::new(0.72, 0.80, 0.92, 0.0),
            cloud_ambient_color_bottom: Vec4::new(0.32, 0.42, 0.58, 0.0),
            cloud_min_transmittance: 0.08,
            forward_scattering_g: 0.72,
            backward_scattering_g: -0.22,
            scattering_lerp: 0.18,
            wind_velocity: Vec3::new(4.0, 0.0, 1.3),
            sun_color: Vec4::new(1.0, 0.96, 0.92, 1.25),
        }
    }
}

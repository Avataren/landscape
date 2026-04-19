use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{AsBindGroup, ShaderType, UniformBuffer},
    },
};

#[derive(Clone, Resource, ExtractResource, ShaderType)]
pub(crate) struct CloudsUniform {
    pub render_resolution: Vec2,
    pub time: f32,
    pub planet_radius: f32,
    pub camera_translation: Vec3,
    pub _pad1: f32,
    pub previous_camera_translation: Vec3,
    pub _pad_prev: f32,
    pub inverse_camera_view: Mat4,
    pub inverse_camera_projection: Mat4,
    pub wind_displacement: Vec3,
    pub _pad2: f32,
    pub sun_direction: Vec4,
    pub sun_color: Vec4,
    pub cloud_heights: Vec4,
    pub noise_scales: Vec4,
    pub softness: Vec4,
    pub ambient_top: Vec4,
    pub ambient_bottom: Vec4,
    pub phase_steps: Vec4,
    pub shadow_params: Vec4,
    pub march: UVec4,
}

impl Default for CloudsUniform {
    fn default() -> Self {
        Self {
            render_resolution: Vec2::new(960.0, 540.0),
            time: 0.0,
            planet_radius: 6_371_000.0,
            camera_translation: Vec3::ZERO,
            _pad1: 0.0,
            previous_camera_translation: Vec3::ZERO,
            _pad_prev: 0.0,
            inverse_camera_view: Mat4::IDENTITY,
            inverse_camera_projection: Mat4::IDENTITY,
            wind_displacement: Vec3::ZERO,
            _pad2: 0.0,
            sun_direction: Vec4::new(0.4, 0.7, 0.2, 0.0),
            sun_color: Vec4::new(1.0, 0.96, 0.92, 1.25),
            cloud_heights: Vec4::new(1250.0, 2400.0, 0.62, 0.03),
            noise_scales: Vec4::new(1.5, 42.0, 0.27, 0.10),
            softness: Vec4::new(0.25, 0.0, 0.0, 0.0),
            ambient_top: Vec4::new(0.72, 0.80, 0.92, 0.0),
            ambient_bottom: Vec4::new(0.32, 0.42, 0.58, 0.0),
            phase_steps: Vec4::new(0.72, -0.22, 0.18, 0.08),
            shadow_params: Vec4::new(10.0, 1.3, 1.0, 0.9),
            march: UVec4::new(16, 6, 0, 0),
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct CloudsUniformBuffer {
    pub buffer: UniformBuffer<CloudsUniform>,
}

#[derive(Resource, Clone, ExtractResource, AsBindGroup)]
pub(crate) struct CloudsImage {
    #[storage_texture(0, image_format = Rgba32Float, access = ReadWrite)]
    pub cloud_render_image: Handle<Image>,
    #[storage_texture(1, image_format = Rgba32Float, access = ReadWrite)]
    pub cloud_history_image: Handle<Image>,
    #[storage_texture(2, image_format = Rgba32Float, access = ReadWrite)]
    pub cloud_atlas_image: Handle<Image>,
    #[storage_texture(3, image_format = Rgba32Float, access = ReadWrite, dimension = "3d")]
    pub cloud_worley_image: Handle<Image>,
}

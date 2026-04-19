mod compute;
mod config;
mod images;
mod render;
mod uniforms;

use bevy::{
    light::{light_consts::lux, DirectionalLight, NotShadowCaster, NotShadowReceiver},
    prelude::*,
};

use compute::CloudsComputePlugin;
pub use config::CloudsConfig;
use images::build_cloud_images;
use render::{CloudsDisplayMaterial, CloudsRenderPlugin};
use uniforms::{CloudsImage, CloudsUniform};

#[derive(Component, Clone, Copy, Debug, Default)]
struct CloudDisplayLayer;

pub struct VolumetricCloudsPlugin;

impl Plugin for VolumetricCloudsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CloudsConfig>()
            .init_resource::<CloudsUniform>()
            .add_plugins((CloudsRenderPlugin, CloudsComputePlugin))
            .add_systems(Startup, setup_cloud_layer)
            .add_systems(
                PostUpdate,
                (sync_cloud_uniforms, follow_cloud_layers)
                    .chain()
                    .after(TransformSystems::Propagate),
            );
    }
}

fn setup_cloud_layer(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CloudsDisplayMaterial>>,
    config: Res<CloudsConfig>,
) {
    let (cloud_render_image, cloud_history_image, cloud_atlas_image, cloud_worley_image) =
        build_cloud_images(&mut images, config.render_resolution);
    let material = materials.add(CloudsDisplayMaterial {
        cloud_render_image: cloud_render_image.clone(),
    });

    commands.insert_resource(CloudsImage {
        cloud_render_image,
        cloud_history_image,
        cloud_atlas_image,
        cloud_worley_image,
    });

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(Vec3::ONE))),
        MeshMaterial3d(material),
        Transform::from_scale(Vec3::splat(120_000.0)),
        CloudDisplayLayer,
        NotShadowCaster,
        NotShadowReceiver,
    ));
}

/// Approximate Rayleigh+Mie atmospheric transmittance for the sun at a given
/// elevation. `sin_elevation` is the Y component of the normalised sun direction
/// (= sin of the geometric elevation angle).
///
/// Returns an RGB multiplier: near-white at zenith, orange-red at the horizon.
fn atmospheric_sun_tint(sin_elevation: f32) -> Vec3 {
    // Optical air mass: how many atmosphere thicknesses the sunlight traverses.
    // Increases rapidly toward the horizon (1 at zenith, ~38 at horizon).
    let air_mass = (1.0 / sin_elevation.max(0.025)).min(40.0);

    // Rayleigh extinction coefficients per air-mass unit (λ^-4 dependence).
    // Tuned to produce visually plausible white→yellow→orange→red progression.
    Vec3::new(
        (-0.010 * air_mass).exp(), // red   — least scattered
        (-0.026 * air_mass).exp(), // green
        (-0.085 * air_mass).exp(), // blue  — most scattered
    )
}

fn sync_cloud_uniforms(
    time: Res<Time>,
    config: Res<CloudsConfig>,
    camera: Single<(&GlobalTransform, &Camera), With<Camera3d>>,
    sun: Query<(&Transform, &DirectionalLight), Without<Camera3d>>,
    mut uniform: ResMut<CloudsUniform>,
) {
    let (camera_transform, camera) = *camera;
    let previous_camera_translation = uniform.camera_translation;
    uniform.render_resolution = config.render_resolution.as_vec2();
    uniform.time = time.elapsed_secs_wrapped();
    uniform.planet_radius = config.planet_radius;
    uniform.previous_camera_translation = previous_camera_translation;
    uniform.camera_translation = camera_transform.translation();

    // Compute previous_view_proj from stored matrices before overwriting them.
    // inverse_camera_view = view-to-world, so its inverse is world-to-view.
    // inverse_camera_projection = clip-to-view, so its inverse is view-to-clip (projection).
    let prev_view = uniform.inverse_camera_view.inverse();
    let prev_proj = uniform.inverse_camera_projection.inverse();
    uniform.previous_view_proj = prev_proj * prev_view;

    uniform.inverse_camera_view = camera_transform.to_matrix();
    uniform.inverse_camera_projection = camera.computed.clip_from_view.inverse();
    uniform.wind_displacement += config.wind_velocity * time.delta_secs();
    uniform.cloud_heights = Vec4::new(
        config.cloud_bottom_height,
        config.cloud_top_height,
        config.cloud_coverage,
        config.cloud_density,
    );
    uniform.noise_scales = Vec4::new(
        config.cloud_base_scale,
        config.cloud_detail_scale,
        config.cloud_detail_strength,
        config.cloud_base_edge_softness,
    );
    uniform.softness = Vec4::new(config.cloud_bottom_softness, config.cloud_evolution_speed, 0.0, 0.0);
    uniform.ambient_top = config.cloud_ambient_color_top;
    uniform.ambient_bottom = config.cloud_ambient_color_bottom;
    uniform.phase_steps = Vec4::new(
        config.forward_scattering_g,
        config.backward_scattering_g,
        config.scattering_lerp,
        config.cloud_min_transmittance,
    );
    uniform.shadow_params = Vec4::new(
        config.cloud_shadow_step_size,
        config.cloud_shadow_step_multiply,
        0.0,
        config.cloud_history_blend,
    );
    uniform.march = UVec4::new(config.cloud_view_steps, config.cloud_shadow_steps, 0, 0);
    uniform.sun_color = Vec4::ZERO;

    if let Ok((sun_transform, light)) = sun.single() {
        let toward_sun: Vec3 = (-sun_transform.forward()).into();
        let toward_sun = toward_sun.normalize_or_zero();
        uniform.sun_direction = toward_sun.extend(0.0);
        let linear = light.color.to_linear();
        let day_factor = (light.illuminance / lux::RAW_SUNLIGHT).clamp(0.0, 1.0);

        // Bevy's DirectionalLight.color is always white; the red/orange sunset
        // tint lives only in the atmosphere GPU shader. Approximate Rayleigh+Mie
        // scattering here so cloud lighting matches the sky at low sun angles.
        let atm = atmospheric_sun_tint(toward_sun.y);

        uniform.sun_color = Vec4::new(
            config.sun_color.x * linear.red * day_factor * atm.x,
            config.sun_color.y * linear.green * day_factor * atm.y,
            config.sun_color.z * linear.blue * day_factor * atm.z,
            day_factor,
        );
        uniform.shadow_params.z = day_factor;
    }
}

fn follow_cloud_layers(
    camera: Single<(&Transform, &Projection), (With<Camera3d>, Without<CloudDisplayLayer>)>,
    mut layers: Query<&mut Transform, With<CloudDisplayLayer>>,
) {
    let (camera_transform, projection) = *camera;
    let far = match projection {
        Projection::Perspective(perspective) => perspective.far,
        _ => 100_000.0,
    };
    let scale = far * 1.2;

    for mut transform in &mut layers {
        transform.translation = camera_transform.translation;
        transform.scale = Vec3::splat(scale);
    }
}

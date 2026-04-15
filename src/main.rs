mod app_config;
mod terrain;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    core_pipeline::tonemapping::Tonemapping,
    light::{light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder},
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium},
    post_process::bloom::Bloom,
    prelude::*,
    camera::Exposure,
    input::mouse::AccumulatedMouseMotion,
    window::PrimaryWindow,
};
use terrain::{
    components::TerrainCamera,
    config::TerrainConfig,
    TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc,
};

const WINDOW_TITLE: &str = "Landscape Renderer";

fn main() {
    let cfg = app_config::load();

    let terrain_config = {
        let mut tc = TerrainConfig::default();
        if let Some(v) = cfg.render.clipmap_levels    { tc.clipmap_levels    = v; }
        if let Some(v) = cfg.render.height_scale      { tc.height_scale      = v; }
        if let Some(v) = cfg.render.macro_color_flip_v { tc.macro_color_flip_v = v; }
        tc
    };

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: WINDOW_TITLE.into(),
                resolution: (640u32, 480u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(TerrainPlugin)
        .add_plugins(TerrainDebugPlugin)
        .insert_resource(terrain_config)
        .insert_resource(TerrainSourceDesc {
            tile_root: cfg.source.tile_root,
            normal_root: cfg.source.normal_root,
            macro_color_root: cfg.source.macro_color_root,
            world_min: cfg.source.world_min,
            world_max: cfg.source.world_max,
            max_mip_level: cfg.source.max_mip_level,
            ..default()
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (camera_move, camera_look).chain())
        .add_systems(Update, update_window_title)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    // 16k heightmap: terrain spans ±8192 wu, max height 4096 wu.
    // Average terrain near origin ~2400 wu; start well above that.
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 1.0,
            far: 10_000_000.0,
            ..default()
        }),
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        Transform::from_xyz(0.0, 6000.0, -8000.0).looking_at(Vec3::new(0.0, 2500.0, 0.0), Vec3::Y),
        TerrainCamera,
    ));

    commands.spawn((
        DirectionalLight {
            // Use raw sunlight so the atmosphere does the scattering itself.
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        // Low-angle sun (~14° elevation) casts long shadows across terrain features.
        // X rotates the light down from horizontal; Y sets the azimuth.
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.25, 0.8, 0.0)),
        // Cascade shadow bounds tuned to the terrain scale (world ±2048, camera
        // starts ~8 000 wu out).  Four cascades cover close detail through the
        // full visible range.
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 1.0,
            first_cascade_far_bound: 500.0,
            maximum_distance: 20_000.0,
            overlap_proportion: 0.2,
        }
        .build(),
    ));
}

// ---------------------------------------------------------------------------
// Keyboard movement
// WASD move in the camera's horizontal plane (projected onto XZ).
// Q/E move straight up/down.
// Hold Shift for 5× speed.
// ---------------------------------------------------------------------------
fn camera_move(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    let Ok(mut t) = query.single_mut() else {
        return;
    };

    let speed = if keys.pressed(KeyCode::ShiftLeft) {
        5000.0
    } else {
        500.0
    };
    let dt = time.delta_secs();

    // Project camera forward/right onto XZ so WASD always moves horizontally.
    let fwd3: Vec3 = t.forward().into();
    let forward = Vec3::new(fwd3.x, 0.0, fwd3.z).normalize_or_zero();
    let right = Vec3::new(-fwd3.z, 0.0, fwd3.x); // 90° CCW = right-hand cross(fwd, Y)

    if keys.pressed(KeyCode::KeyW) {
        t.translation += forward * speed * dt;
    }
    if keys.pressed(KeyCode::KeyS) {
        t.translation -= forward * speed * dt;
    }
    if keys.pressed(KeyCode::KeyA) {
        t.translation -= right * speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        t.translation += right * speed * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        t.translation.y += speed * dt;
    }
    if keys.pressed(KeyCode::KeyQ) {
        t.translation.y -= speed * dt;
    }
}

// ---------------------------------------------------------------------------
// Mouse look — hold right mouse button and drag to look around.
// Yaw rotates around the world Y axis; pitch rotates around the local X axis.
// ---------------------------------------------------------------------------
fn camera_look(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    if !mouse_buttons.pressed(MouseButton::Right) {
        return;
    }
    let delta = mouse_motion.delta;
    if delta == Vec2::ZERO {
        return;
    }
    let Ok(mut t) = query.single_mut() else {
        return;
    };

    let yaw = -delta.x * 0.002;
    let pitch = -delta.y * 0.002;

    // Rotate yaw around world Y (prevents roll accumulation).
    let yaw_rot = Quat::from_rotation_y(yaw);
    // Rotate pitch around the camera's local X axis.
    let right: Vec3 = t.right().into();
    let pitch_rot = Quat::from_axis_angle(right, pitch);

    t.rotation = (yaw_rot * pitch_rot * t.rotation).normalize();
}

fn update_window_title(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut update_timer: Local<Option<Timer>>,
) {
    let timer = update_timer.get_or_insert_with(|| Timer::from_seconds(0.25, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() {
        return;
    }

    let Ok(mut window) = windows.single_mut() else {
        return;
    };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed());

    window.title = match fps {
        Some(fps) => format!("{WINDOW_TITLE} - {:.0} FPS", fps),
        None => format!("{WINDOW_TITLE} - -- FPS"),
    };
}

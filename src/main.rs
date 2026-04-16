mod app_config;
mod player;

use bevy::{
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::AccumulatedMouseMotion,
    light::{light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder},
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium},
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        render_resource::WgpuFeatures,
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
    window::PrimaryWindow,
};
use bevy_landscape::{TerrainCamera, TerrainConfig, TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc};
use bevy_landscape_editor::LandscapeEditorPlugin;
use player::{CameraMode, PlayerPlugin};

const WINDOW_TITLE: &str = "Landscape Renderer";

fn main() {
    let cfg = app_config::load();

    let terrain_config = {
        let mut tc = TerrainConfig::default();
        if let Some(v) = cfg.render.clipmap_levels {
            tc.clipmap_levels = v;
        }
        if let Some(v) = cfg.render.height_scale {
            tc.height_scale = v;
        }
        if let Some(v) = cfg.render.macro_color_flip_v {
            tc.macro_color_flip_v = v;
        }
        tc
    };

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        // The terrain shader adds its own hemisphere ambient (sky/ground bounce),
        // so this is only a flat fill for non-terrain objects.  Keep it small.
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 500.0,
            ..default()
        })
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: WgpuFeatures::POLYGON_MODE_LINE,
                        ..default()
                    }),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: WINDOW_TITLE.into(),
                        present_mode: bevy::window::PresentMode::Immediate,
                        resolution: (640u32, 480u32).into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(TerrainPlugin {
            config: terrain_config,
            source: TerrainSourceDesc {
                tile_root: cfg.source.tile_root,
                normal_root: cfg.source.normal_root,
                macro_color_root: cfg.source.macro_color_root,
                world_min: cfg.source.world_min,
                world_max: cfg.source.world_max,
                max_mip_level: cfg.source.max_mip_level,
                ..default()
            },
        })
        .add_plugins(TerrainDebugPlugin)
        .add_plugins(LandscapeEditorPlugin)
        .add_plugins(PlayerPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (camera_move, camera_look).chain())
        .add_systems(Update, update_window_title)
        .run();
}

fn setup_scene(mut commands: Commands, mut scattering_mediums: ResMut<Assets<ScatteringMedium>>) {
    // Spawn camera above terrain centre, angled downward so terrain is visible
    // immediately in freecam mode.  preload_terrain_startup uses the XZ position
    // to decide which tiles to load first.
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 0.1,
            // Terrain world is ~4 096 m across; 100 km gives comfortable margin
            // without the 100 000 000:1 depth range that caused Z-fighting at
            // any non-trivial distance.
            far: 100_000.0,
            ..default()
        }),
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        // Position at a comfortable altitude and look at terrain centre so the
        // view is correct from the first frame in freecam mode.
        Transform::from_xyz(0.0, 800.0, 1200.0).looking_at(Vec3::ZERO, Vec3::Y),
        // TerrainCamera::default() turns on the forward-bias for clipmap
        // centers — the LOD rings shift along the camera's view direction so
        // fine geometry covers the visible foreground in both freecam and
        // walking views.  See `TerrainCamera::forward_bias_ratio`.
        TerrainCamera::default(),
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
            // 8 km covers meaningful terrain shadow distance without the sparse
            // texel coverage that made the 20 km last cascade useless.
            maximum_distance: 8_000.0,
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
    mode: Res<CameraMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    if *mode != CameraMode::Freecam {
        return;
    }
    let Ok(mut t) = query.single_mut() else {
        return;
    };

    let speed = if keys.pressed(KeyCode::ShiftLeft) {
        5000.0
    } else {
        500.0
    };
    let dt = time.delta_secs();

    let forward: Vec3 = t.forward().into();
    let right: Vec3   = t.right().into();

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
    mode: Res<CameraMode>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    if *mode != CameraMode::Freecam {
        return;
    }
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

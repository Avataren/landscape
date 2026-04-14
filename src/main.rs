mod terrain;

use bevy::{input::mouse::AccumulatedMouseMotion, prelude::*};
use terrain::{components::TerrainCamera, TerrainDebugPlugin, TerrainPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Landscape Renderer".into(),
                resolution: (1920u32, 1080u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(TerrainPlugin)
        .add_plugins(TerrainDebugPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (camera_move, camera_look).chain())
        .run();
}

fn setup_scene(mut commands: Commands) {
    // Height texture: ±512 world units from origin, max height ≈ 128 units.
    // Start 300 units above origin and 300 units south, looking across the terrain.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 300.0, -300.0)
            .looking_at(Vec3::new(0.0, 64.0, 300.0), Vec3::Y),
        TerrainCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.0, 0.5, 0.0)),
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
    let Ok(mut t) = query.single_mut() else { return };

    let speed = if keys.pressed(KeyCode::ShiftLeft) { 500.0 } else { 100.0 };
    let dt = time.delta_secs();

    // Project camera forward/right onto XZ so WASD always moves horizontally.
    let fwd3: Vec3 = t.forward().into();
    let forward = Vec3::new(fwd3.x, 0.0, fwd3.z).normalize_or_zero();
    let right = Vec3::new(-fwd3.z, 0.0, fwd3.x); // 90° CCW = right-hand cross(fwd, Y)

    if keys.pressed(KeyCode::KeyW) { t.translation += forward * speed * dt; }
    if keys.pressed(KeyCode::KeyS) { t.translation -= forward * speed * dt; }
    if keys.pressed(KeyCode::KeyA) { t.translation -= right   * speed * dt; }
    if keys.pressed(KeyCode::KeyD) { t.translation += right   * speed * dt; }
    if keys.pressed(KeyCode::KeyE) { t.translation.y += speed * dt; }
    if keys.pressed(KeyCode::KeyQ) { t.translation.y -= speed * dt; }
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
    let Ok(mut t) = query.single_mut() else { return };

    let yaw   = -delta.x * 0.002;
    let pitch = -delta.y * 0.002;

    // Rotate yaw around world Y (prevents roll accumulation).
    let yaw_rot = Quat::from_rotation_y(yaw);
    // Rotate pitch around the camera's local X axis.
    let right: Vec3 = t.right().into();
    let pitch_rot = Quat::from_axis_angle(right, pitch);

    t.rotation = (yaw_rot * pitch_rot * t.rotation).normalize();
}

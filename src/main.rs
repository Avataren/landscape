mod terrain;

use bevy::prelude::*;
use terrain::{TerrainPlugin, TerrainDebugPlugin, components::TerrainCamera};

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
        .add_systems(Update, camera_controller)
        .run();
}

fn setup_scene(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 200.0, 0.0).looking_at(Vec3::new(500.0, 0.0, 500.0), Vec3::Y),
        TerrainCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.0, 0.5, 0.0)),
    ));
}

fn camera_controller(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };
    let speed = if keys.pressed(KeyCode::ShiftLeft) { 500.0 } else { 100.0 };
    let dt = time.delta_secs();

    let forward: Vec3 = transform.forward().into();
    let right: Vec3 = transform.right().into();

    if keys.pressed(KeyCode::KeyW) { transform.translation += forward * speed * dt; }
    if keys.pressed(KeyCode::KeyS) { transform.translation -= forward * speed * dt; }
    if keys.pressed(KeyCode::KeyA) { transform.translation -= right * speed * dt; }
    if keys.pressed(KeyCode::KeyD) { transform.translation += right * speed * dt; }
    if keys.pressed(KeyCode::KeyQ) { transform.translation.y -= speed * dt; }
    if keys.pressed(KeyCode::KeyE) { transform.translation.y += speed * dt; }
}

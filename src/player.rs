// player.rs
// First-person walking controller + freecam, backed by Avian3d physics.
//
// Press F1 to toggle between Walking and Freecam.
//
// Walking
// -------
//   A Dynamic capsule body handles gravity and terrain collision.
//   WASD sets horizontal velocity each frame; Space jumps.
//   Mouse always rotates view (cursor is locked).
//   The scene Camera3d (TerrainCamera) is driven from the body position + eye
//   height in PostUpdate — no parent-child needed so terrain queries stay clean.
//
// Freecam
// -------
//   Physics body is frozen (velocity zeroed, gravity override).
//   Camera is moved directly with WASD / Q / E + hold RMB to look
//   (existing behaviour from main.rs camera_move / camera_look, gated on mode).
//
// Switching Walking → Freecam: cursor unlocked; body frozen in place.
// Switching Freecam → Walking: cursor locked; body teleported to camera XZ;
//   yaw/pitch synced from camera rotation so the view is continuous.

use avian3d::prelude::*;
use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_landscape::{TerrainCamera, TerrainCollisionCache, TerrainConfig};

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

const CAPSULE_RADIUS: f32 = 0.3;
/// Cylinder section length.  Total height = 2 × RADIUS + LENGTH = 1.8 m.
const CAPSULE_LENGTH: f32 = 1.2;
/// Distance from body centre up to the camera (eye height from centre).
/// Body centre sits 0.9 m above ground when settled → camera is 2 m above ground.
const EYE_OFFSET: f32 = 1.1;
const WALK_SPEED: f32 = 8.0;
const JUMP_SPEED: f32 = 6.0;
const MOUSE_SENSITIVITY: f32 = 0.002;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Active camera / movement mode.  Toggle with F1.
#[derive(Resource, Default, PartialEq, Clone, Copy, Debug)]
pub enum CameraMode {
    Walking,
    #[default]
    Freecam,
}

// ---------------------------------------------------------------------------
// Private components / resources
// ---------------------------------------------------------------------------

/// Marker on the Dynamic rigid-body entity representing the player.
#[derive(Component)]
pub struct PlayerBody;

/// Accumulated yaw and pitch from mouse input (Walking mode).
#[derive(Resource, Default)]
struct PlayerLook {
    yaw: f32,
    pitch: f32,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default())
            .insert_resource(CameraMode::default())
            .insert_resource(PlayerLook::default())
            .add_systems(
                Update,
                (
                    spawn_player_once,
                    toggle_mode,
                    player_look,
                    player_move.after(player_look),
                ),
            )
            // PostUpdate runs after FixedPostUpdate (where Avian writeback lives).
            // clamp_player_to_terrain must run before sync_camera_to_body so the
            // camera sees the corrected position on the same frame.
            .add_systems(
                PostUpdate,
                (clamp_player_to_terrain, sync_camera_to_body).chain(),
            );
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Spawns the physics body on the first Update frame where the terrain
/// collision cache has been populated.  Tile colliders (from `sync_tile_colliders`)
/// and the coarse global heightfield (from `spawn_coarse_global_heightfield`) are
/// managed by the terrain plugin; the player body simply falls onto whatever is present.
fn spawn_player_once(
    mut done: Local<bool>,
    cache: Res<TerrainCollisionCache>,
    config: Res<TerrainConfig>,
    mut commands: Commands,
) {
    if *done {
        return;
    }
    // world_scale > 0 once preload_terrain_startup (PostStartup) has run.
    if cache.world_scale <= 0.0 {
        return;
    }
    *done = true;

    let spawn_xz = Vec2::ZERO;
    let ground_y = cache
        .sample_height(spawn_xz)
        .unwrap_or(config.height_scale * 0.5);
    // Place body centre just above the surface so the first physics step settles it.
    let spawn_y = ground_y + CAPSULE_RADIUS + CAPSULE_LENGTH * 0.5 + 0.5;

    commands.spawn((
        PlayerBody,
        RigidBody::Dynamic,
        Collider::capsule(CAPSULE_RADIUS, CAPSULE_LENGTH),
        LockedAxes::ROTATION_LOCKED,
        LinearVelocity::default(),
        Transform::from_xyz(spawn_xz.x, spawn_y, spawn_xz.y),
    ));
}

/// Handles F1 toggle between Walking and Freecam.
///
/// Walking → Freecam: freeze body, unlock cursor.
/// Freecam → Walking: teleport body to camera XZ, sync look angles, lock cursor.
fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<CameraMode>,
    mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut body_q: Query<(&mut Position, &mut LinearVelocity), With<PlayerBody>>,
    cam_q: Query<&Transform, (With<TerrainCamera>, Without<PlayerBody>)>,
    mut look: ResMut<PlayerLook>,
) {
    if !keys.just_pressed(KeyCode::F1) {
        return;
    }
    let Ok(mut cursor) = cursor_q.single_mut() else {
        return;
    };

    match *mode {
        CameraMode::Walking => {
            *mode = CameraMode::Freecam;
            cursor.grab_mode = CursorGrabMode::None;
            cursor.visible = true;
            // Freeze body in place so it doesn't fall while flying.
            if let Ok((_, mut vel)) = body_q.single_mut() {
                *vel = LinearVelocity::ZERO;
            }
        }
        CameraMode::Freecam => {
            *mode = CameraMode::Walking;
            cursor.grab_mode = CursorGrabMode::Locked;
            cursor.visible = false;

            if let Ok(cam_t) = cam_q.single() {
                // Sync yaw/pitch so the walking view is continuous with freecam.
                let (yaw, pitch, _) = cam_t.rotation.to_euler(EulerRot::YXZ);
                look.yaw = yaw;
                look.pitch = pitch;

                // Teleport body to camera position; gravity will settle the Y.
                if let Ok((mut pos, mut vel)) = body_q.single_mut() {
                    pos.0 = cam_t.translation;
                    *vel = LinearVelocity::ZERO;
                }
            }
        }
    }
}

/// Accumulates mouse motion into yaw/pitch (Walking mode only).
fn player_look(
    mode: Res<CameraMode>,
    mut look: ResMut<PlayerLook>,
    mouse_motion: Res<AccumulatedMouseMotion>,
) {
    if *mode != CameraMode::Walking {
        return;
    }

    let delta = mouse_motion.delta;
    if delta != Vec2::ZERO {
        look.yaw -= delta.x * MOUSE_SENSITIVITY;
        look.pitch = (look.pitch - delta.y * MOUSE_SENSITIVITY).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.05,
            std::f32::consts::FRAC_PI_2 - 0.05,
        );
    }
}

/// WASD horizontal velocity + Space jump (Walking mode only).
fn player_move(
    mode: Res<CameraMode>,
    mut body_q: Query<&mut LinearVelocity, With<PlayerBody>>,
    keys: Res<ButtonInput<KeyCode>>,
    look: Res<PlayerLook>,
) {
    if *mode != CameraMode::Walking {
        return;
    }
    let Ok(mut vel) = body_q.single_mut() else {
        return;
    };

    let (sin_y, cos_y) = look.yaw.sin_cos();
    let forward = Vec3::new(-sin_y, 0.0, -cos_y);
    let right = Vec3::new(cos_y, 0.0, -sin_y);

    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += right;
    }

    let horiz = if dir.length_squared() > 0.0 {
        dir.normalize() * WALK_SPEED
    } else {
        Vec3::ZERO
    };
    // Override horizontal; let Avian gravity own vertical.
    vel.x = horiz.x;
    vel.z = horiz.z;

    if keys.just_pressed(KeyCode::Space) && vel.y.abs() < 1.0 {
        vel.y += JUMP_SPEED;
    }
}

/// Safety net: if physics tunnels through a heightfield hole, snap the player
/// back above terrain and cancel any downward velocity.  Runs before
/// `sync_camera_to_body` so the correction is visible on the same frame.
fn clamp_player_to_terrain(
    mode: Res<CameraMode>,
    cache: Res<TerrainCollisionCache>,
    mut body_q: Query<(&mut Position, &mut LinearVelocity, &mut Transform), With<PlayerBody>>,
) {
    if *mode != CameraMode::Walking {
        return;
    }
    let Ok((mut pos, mut vel, mut transform)) = body_q.single_mut() else {
        return;
    };
    let Some(ground_y) = cache.sample_height(pos.0.xz()) else {
        return;
    };
    let foot_y = pos.0.y - CAPSULE_RADIUS - CAPSULE_LENGTH * 0.5;
    if foot_y < ground_y {
        let corrected_y = ground_y + CAPSULE_RADIUS + CAPSULE_LENGTH * 0.5;
        pos.0.y = corrected_y;
        transform.translation.y = corrected_y;
        if vel.y < 0.0 {
            vel.y = 0.0;
        }
    }
}

/// Drives the Camera3d Transform from the physics body (Walking mode only).
/// Runs in PostUpdate, after FixedPostUpdate where Avian writeback completes.
fn sync_camera_to_body(
    mode: Res<CameraMode>,
    body_q: Query<&Transform, With<PlayerBody>>,
    mut cam_q: Query<&mut Transform, (With<TerrainCamera>, Without<PlayerBody>)>,
    look: Res<PlayerLook>,
) {
    if *mode != CameraMode::Walking {
        return;
    }
    let (Ok(body_t), Ok(mut cam_t)) = (body_q.single(), cam_q.single_mut()) else {
        return;
    };

    cam_t.translation = body_t.translation + Vec3::Y * EYE_OFFSET;
    cam_t.rotation = Quat::from_rotation_y(look.yaw) * Quat::from_rotation_x(look.pitch);
}

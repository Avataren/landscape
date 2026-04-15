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
use crate::terrain::{
    collision::TerrainCollisionCache,
    components::TerrainCamera,
    config::TerrainConfig,
};

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

const CAPSULE_RADIUS: f32 = 0.3;
/// Cylinder section length.  Total height = 2 × RADIUS + LENGTH = 1.8 m.
const CAPSULE_LENGTH: f32 = 1.2;
/// Distance from body centre up to the camera (eye height from centre).
const EYE_OFFSET: f32 = 0.7;
const WALK_SPEED: f32 = 8.0;
const JUMP_SPEED: f32 = 6.0;
const MOUSE_SENSITIVITY: f32 = 0.002;
/// Heightfield grid side point count (cells = RESOLUTION − 1 = 128).
const HF_RESOLUTION: u32 = 129;
/// World-space size of the square heightfield (metres per side).
const HF_SIZE: f32 = 256.0;
/// Rebuild the heightfield when the player moves further than this.
const HF_UPDATE_DIST: f32 = 64.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Active camera / movement mode.  Toggle with F1.
#[derive(Resource, Default, PartialEq, Clone, Copy, Debug)]
pub enum CameraMode {
    #[default]
    Walking,
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

/// Tracks the current Static heightfield entity and its XZ centre.
#[derive(Resource, Default)]
struct HeightfieldState {
    entity: Option<Entity>,
    center: Option<Vec2>,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PhysicsPlugins::default())
            .insert_resource(CameraMode::default())
            .insert_resource(PlayerLook::default())
            .insert_resource(HeightfieldState::default())
            .add_systems(Update, (
                spawn_player_once,
                toggle_mode,
                player_look,
                player_move.after(player_look),
                update_heightfield,
            ))
            // PostUpdate runs after FixedPostUpdate (where Avian writeback lives),
            // so sync_camera_to_body always sees the settled physics Transform.
            .add_systems(PostUpdate, sync_camera_to_body);
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Spawns the physics body and initial heightfield on the first Update frame
/// where the terrain collision cache has been populated.
fn spawn_player_once(
    mut done: Local<bool>,
    cache: Res<TerrainCollisionCache>,
    config: Res<TerrainConfig>,
    mut commands: Commands,
    mut hf_state: ResMut<HeightfieldState>,
    mut cursor_q: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if *done { return; }
    // world_scale > 0 once preload_terrain_startup (PostStartup) has run.
    if cache.world_scale <= 0.0 { return; }
    *done = true;

    // Lock cursor for walking mode (default starting mode).
    if let Ok(mut cursor) = cursor_q.single_mut() {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }

    let spawn_xz = Vec2::ZERO;
    let ground_y = cache.sample_height(spawn_xz)
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

    let hf = spawn_heightfield(&mut commands, &cache, spawn_xz);
    hf_state.entity = Some(hf);
    hf_state.center = Some(spawn_xz);
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
    if !keys.just_pressed(KeyCode::F1) { return; }
    let Ok(mut cursor) = cursor_q.single_mut() else { return };

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
    if *mode != CameraMode::Walking { return; }

    let delta = mouse_motion.delta;
    if delta != Vec2::ZERO {
        look.yaw  -= delta.x * MOUSE_SENSITIVITY;
        look.pitch = (look.pitch - delta.y * MOUSE_SENSITIVITY)
            .clamp(
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
    if *mode != CameraMode::Walking { return; }
    let Ok(mut vel) = body_q.single_mut() else { return };

    let (sin_y, cos_y) = look.yaw.sin_cos();
    let forward = Vec3::new(-sin_y, 0.0, -cos_y);
    let right   = Vec3::new( cos_y, 0.0, -sin_y);

    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) { dir += forward; }
    if keys.pressed(KeyCode::KeyS) { dir -= forward; }
    if keys.pressed(KeyCode::KeyA) { dir -= right;   }
    if keys.pressed(KeyCode::KeyD) { dir += right;   }

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

/// Drives the Camera3d Transform from the physics body (Walking mode only).
/// Runs in PostUpdate, after FixedPostUpdate where Avian writeback completes.
fn sync_camera_to_body(
    mode: Res<CameraMode>,
    body_q: Query<&Transform, With<PlayerBody>>,
    mut cam_q: Query<&mut Transform, (With<TerrainCamera>, Without<PlayerBody>)>,
    look: Res<PlayerLook>,
) {
    if *mode != CameraMode::Walking { return; }
    let (Ok(body_t), Ok(mut cam_t)) = (body_q.single(), cam_q.single_mut()) else { return };

    cam_t.translation = body_t.translation + Vec3::Y * EYE_OFFSET;
    cam_t.rotation    = Quat::from_rotation_y(look.yaw) * Quat::from_rotation_x(look.pitch);
}

/// Rebuilds the heightfield when the player moves more than HF_UPDATE_DIST
/// from the last centre (runs regardless of mode so it stays ready).
fn update_heightfield(
    body_q: Query<&Transform, With<PlayerBody>>,
    cache: Res<TerrainCollisionCache>,
    mut commands: Commands,
    mut hf_state: ResMut<HeightfieldState>,
) {
    let Ok(body_t) = body_q.single() else { return };
    let player_xz = Vec2::new(body_t.translation.x, body_t.translation.z);

    let Some(last) = hf_state.center else { return };
    if last.distance(player_xz) <= HF_UPDATE_DIST { return; }

    if let Some(old) = hf_state.entity.take() {
        commands.entity(old).despawn();
    }
    let new_entity = spawn_heightfield(&mut commands, &cache, player_xz);
    hf_state.entity = Some(new_entity);
    hf_state.center = Some(player_xz);
}

// ---------------------------------------------------------------------------
// Heightfield helper
// ---------------------------------------------------------------------------

fn spawn_heightfield(commands: &mut Commands, cache: &TerrainCollisionCache, center: Vec2) -> Entity {
    let step = HF_SIZE / (HF_RESOLUTION - 1) as f32;
    let half = HF_SIZE * 0.5;

    // heights[row][col]: row = X axis, col = Z axis (avian3d convention).
    let heights: Vec<Vec<f32>> = (0..HF_RESOLUTION)
        .map(|xi| {
            let x = center.x - half + xi as f32 * step;
            (0..HF_RESOLUTION)
                .map(|zi| {
                    let z = center.y - half + zi as f32 * step;
                    cache.sample_height(Vec2::new(x, z)).unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    commands
        .spawn((
            RigidBody::Static,
            Collider::heightfield(heights, Vec3::new(HF_SIZE, 1.0, HF_SIZE)),
            Transform::from_xyz(center.x, 0.0, center.y),
        ))
        .id()
}

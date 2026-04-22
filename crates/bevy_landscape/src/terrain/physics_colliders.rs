use avian3d::prelude::*;
use bevy::prelude::*;
use std::collections::HashMap;

use crate::terrain::{
    collision::TerrainCollisionCache,
    config::TerrainConfig,
    resources::{TerrainViewState, TileKey},
    streamer::load_tile_data,
    world_desc::TerrainSourceDesc,
};

#[derive(Component)]
pub struct GlobalTerrainHeightfield;

/// Marker for the local sliding heightfield that tracks the camera.
#[derive(Component)]
pub struct LocalTerrainCollider;

/// Tracks the center of the last local heightfield build so we can skip
/// frames where the camera hasn't moved far enough to warrant a rebuild.
#[derive(Resource)]
pub struct LocalColliderState {
    last_center: Vec2,
}

impl Default for LocalColliderState {
    fn default() -> Self {
        // Initialise to an unreachable position so the first frame always builds.
        Self {
            last_center: Vec2::splat(f32::MAX / 2.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Single static world heightfield
// ---------------------------------------------------------------------------

/// Spawns one heightfield collider that covers the entire terrain footprint.
///
/// Reads LOD tiles directly from disk at startup (no streaming dependency),
/// so the result is seamless — there are no per-tile boundaries for physics
/// objects to get stuck on. The collider is never rebuilt or despawned.
///
/// Resolution: `collision_mip_level` controls which pre-baked mip is used.
/// Level 3 = 8 m/cell for world_scale = 1.0. For a 16 384 m world this
/// produces a 2 049 × 2 049 heightfield from 64 tile reads (~8 MB), which
/// is fast, seamless, and accurate enough for both characters and projectiles.
/// Lower this value (e.g. 2 → 4 m/cell, 256 tile reads) for more precision.
pub fn spawn_global_heightfield(
    desc: Res<TerrainSourceDesc>,
    config: Res<TerrainConfig>,
    mut commands: Commands,
) {
    spawn_global_heightfield_for_desc(&desc, &config, &mut commands);
}

pub fn spawn_global_heightfield_for_desc(
    desc: &TerrainSourceDesc,
    config: &TerrainConfig,
    commands: &mut Commands,
) {
    let world_size_x = desc.world_max.x - desc.world_min.x;
    let world_size_z = desc.world_max.y - desc.world_min.y;

    if world_size_x <= 0.0 || world_size_z <= 0.0 {
        warn!("[Terrain] Global heightfield skipped: world bounds not set.");
        return;
    }

    let source_level = desc.collision_mip_level;
    let cell_size = config.world_scale * (1u32 << source_level as u32) as f32;
    let level_tile_world = config.tile_size as f32 * cell_size;

    let nx = (world_size_x / cell_size).round() as usize + 1;
    let nz = (world_size_z / cell_size).round() as usize + 1;

    let mut tile_cache: HashMap<(i32, i32), Vec<f32>> = HashMap::default();

    // heights[xi][zi] — keep the existing project convention.
    let heights: Vec<Vec<f32>> = (0..nx)
        .map(|xi| {
            (0..nz)
                .map(|zi| {
                    let wx = (desc.world_min.x + xi as f32 * cell_size)
                        .min(desc.world_max.x - cell_size * 0.5);
                    let wz = (desc.world_min.y + zi as f32 * cell_size)
                        .min(desc.world_max.y - cell_size * 0.5);

                    let tx = (wx / level_tile_world).floor() as i32;
                    let tz = (wz / level_tile_world).floor() as i32;

                    let data = tile_cache.entry((tx, tz)).or_insert_with(|| {
                        load_tile_data(
                            TileKey {
                                level: source_level,
                                x: tx,
                                y: tz,
                            },
                            config.tile_size,
                            config.world_scale,
                            1.0,
                            desc.max_mip_level,
                            desc.tile_root.as_deref(),
                            None,
                            Some((desc.world_min, desc.world_max)),
                            false,
                        )
                        .data
                    });

                    let local_x = ((wx - tx as f32 * level_tile_world) / cell_size)
                        .round()
                        .clamp(0.0, (config.tile_size - 1) as f32)
                        as usize;
                    let local_z = ((wz - tz as f32 * level_tile_world) / cell_size)
                        .round()
                        .clamp(0.0, (config.tile_size - 1) as f32)
                        as usize;

                    data[local_z * config.tile_size as usize + local_x] * config.height_scale
                })
                .collect()
        })
        .collect();

    let scale_x = (nx - 1) as f32 * cell_size;
    let scale_z = (nz - 1) as f32 * cell_size;
    let center_x = desc.world_min.x + scale_x * 0.5;
    let center_z = desc.world_min.y + scale_z * 0.5;

    commands.spawn((
        GlobalTerrainHeightfield,
        RigidBody::Static,
        Collider::heightfield(heights, Vec3::new(scale_x, 1.0, scale_z)),
        Transform::from_xyz(center_x, 0.0, center_z),
    ));

    info!(
        "[Terrain] Global heightfield: {}x{} samples at {:.0} m/cell, {:.0}x{:.0} m coverage ({} tiles read).",
        nx,
        nz,
        cell_size,
        scale_x,
        scale_z,
        tile_cache.len(),
    );
}

// ---------------------------------------------------------------------------
// Local sliding heightfield — high resolution near the player
// ---------------------------------------------------------------------------

/// Maintains a high-resolution heightfield collider centred on the camera.
///
/// The global heightfield uses a coarse LOD and can deviate several metres
/// from the rendered surface on steep or curved terrain — enough for a
/// character capsule to visibly clip through.  This local collider is built
/// at full LOD-0 resolution (`world_scale` metres per cell) and covers a
/// ~256 m square, which is more than enough for any character movement.
///
/// It is rebuilt (old despawned, new spawned) whenever the camera moves
/// more than 64 world units from the last build centre.  At a walk speed
/// of 8 m/s that is roughly once every 8 seconds — cheap.
pub fn update_local_terrain_collider(
    mut state: ResMut<LocalColliderState>,
    view: Res<TerrainViewState>,
    cache: Res<TerrainCollisionCache>,
    config: Res<TerrainConfig>,
    collider_q: Query<Entity, With<LocalTerrainCollider>>,
    mut commands: Commands,
) {
    // Wait until the collision cache has been initialised by at least one tile.
    if cache.tile_size == 0 {
        return;
    }

    let camera_xz = view.camera_pos_ws.xz();
    let cell_size = config.world_scale; // finest available resolution

    // Target ~256 world-unit coverage.  Cap at 257 samples per axis so
    // rebuilds stay cheap even on very fine world_scales.
    const COVERAGE: f32 = 256.0;
    const REBUILD_DIST: f32 = COVERAGE * 0.25; // 64 m at default coverage
    let n = ((COVERAGE / cell_size).ceil() as usize + 1).min(257).max(9);

    if (camera_xz - state.last_center).length() <= REBUILD_DIST {
        return;
    }

    // Snap centre to the cell grid so the heightfield doesn't jitter.
    let cx = (camera_xz.x / cell_size).round() * cell_size;
    let cz = (camera_xz.y / cell_size).round() * cell_size;
    let center = Vec2::new(cx, cz);
    let half_n = (n as f32 - 1.0) * 0.5;

    // Build heights[column][row] (Avian heightfield convention).
    let heights: Vec<Vec<f32>> = (0..n)
        .map(|xi| {
            (0..n)
                .map(|zi| {
                    let wx = center.x + (xi as f32 - half_n) * cell_size;
                    let wz = center.y + (zi as f32 - half_n) * cell_size;
                    cache.sample_height(Vec2::new(wx, wz)).unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    let span = (n - 1) as f32 * cell_size;

    // Despawn the previous local collider before spawning the new one.
    for entity in collider_q.iter() {
        commands.entity(entity).despawn();
    }

    commands.spawn((
        LocalTerrainCollider,
        RigidBody::Static,
        Collider::heightfield(heights, Vec3::new(span, 1.0, span)),
        Transform::from_xyz(center.x, 0.0, center.y),
    ));

    info!(
        "[Terrain] Local heightfield rebuilt: {}×{} @ {:.1} m/cell, centre ({:.0}, {:.0})",
        n, n, cell_size, center.x, center.y
    );

    state.last_center = center;
}

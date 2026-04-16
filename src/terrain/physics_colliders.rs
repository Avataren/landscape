use avian3d::prelude::*;
use bevy::prelude::*;
use std::collections::HashMap;

use crate::terrain::{
    collision::TerrainCollisionCache,
    config::TerrainConfig,
    resources::{HeightTileCpu, TerrainResidency, TileKey},
    world_desc::TerrainSourceDesc,
};

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Tracks the physics collider entity for each resident LOD-0 tile.
///
/// Spawned and despawned in lockstep with `TerrainResidency::resident_cpu`.
/// Wherever the game has LOD-0 terrain data, physics has matching collision.
///
/// Note on stability: tile collider lifetime is camera-driven.  Objects far
/// from the camera will lose their fine tile collider when the tile evicts, but
/// the coarse global heightfield (Phase 2) acts as a permanent floor of last
/// resort so they won't fall into the void.
#[derive(Resource, Default)]
pub struct TileColliders {
    entities: HashMap<TileKey, Entity>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resamples a CPU tile into the height grid expected by `Collider::heightfield`.
///
/// Avian3d convention: `heights[xi][zi]` where xi = X axis, zi = Z axis.
/// Tile data convention: `data[row * tile_size + col]` where row = Z, col = X.
///
/// `resolution` is the number of sample points per axis (e.g. 64 → 4 m/cell
/// for a 256-texel tile at world_scale = 1.0).
fn build_tile_heightfield(
    tile: &HeightTileCpu,
    height_scale: f32,
    resolution: usize,
) -> (Vec<Vec<f32>>, Vec3) {
    let ts = tile.tile_size as usize;
    let step = ts / resolution;

    let heights: Vec<Vec<f32>> = (0..resolution)
        .map(|xi| {
            (0..resolution)
                .map(|zi| tile.data[(zi * step) * ts + (xi * step)] * height_scale)
                .collect()
        })
        .collect();

    // scale.x/z = world-space extent; scale.y = 1.0 (heights already in metres).
    let tile_world = tile.tile_size as f32; // world_scale = 1.0 m/texel
    (heights, Vec3::new(tile_world, 1.0, tile_world))
}

// ---------------------------------------------------------------------------
// Phase 1 — per-tile heightfield sync
// ---------------------------------------------------------------------------

/// Spawns heightfield colliders for newly-resident LOD-0 tiles and despawns
/// colliders for evicted tiles.
///
/// Must run after `apply_tiles_to_clipmap` so that `resident_cpu` has been
/// updated with any tiles that arrived from the background stream this frame.
pub fn sync_tile_colliders(
    residency: Res<TerrainResidency>,
    config: Res<TerrainConfig>,
    mut colliders: ResMut<TileColliders>,
    mut commands: Commands,
) {
    // Spawn colliders for newly resident LOD-0 tiles.
    for (key, tile) in &residency.resident_cpu {
        if key.level != 0 {
            continue;
        }
        if colliders.entities.contains_key(key) {
            continue;
        }

        let (heights, scale) = build_tile_heightfield(tile, config.height_scale, 64);

        // World-space centre of tile (key.x, key.y).
        let tile_world = tile.tile_size as f32;
        let cx = key.x as f32 * tile_world + tile_world * 0.5;
        let cz = key.y as f32 * tile_world + tile_world * 0.5;

        let entity = commands
            .spawn((
                RigidBody::Static,
                Collider::heightfield(heights, scale),
                Transform::from_xyz(cx, 0.0, cz),
            ))
            .id();

        colliders.entities.insert(*key, entity);
    }

    // Despawn colliders whose tiles were evicted.
    colliders.entities.retain(|key, entity| {
        if residency.resident_cpu.contains_key(key) {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });
}

// ---------------------------------------------------------------------------
// Phase 2 — coarse full-world fallback
// ---------------------------------------------------------------------------

/// Spawns a single low-resolution heightfield that covers the entire terrain
/// footprint.  Acts as a permanent floor of last resort while fine tile
/// colliders stream in, and for physics objects whose tiles get evicted as the
/// camera moves away.  Never despawned.
///
/// Runs once in PostStartup after `preload_terrain_startup` so the collision
/// cache is already populated with the starting area's data.
pub fn spawn_coarse_global_heightfield(
    desc: Res<TerrainSourceDesc>,
    cache: Res<TerrainCollisionCache>,
    mut commands: Commands,
) {
    let world_min = desc.world_min;
    let world_max = desc.world_max;
    let world_size = world_max - world_min;

    if world_size.x <= 0.0 || world_size.y <= 0.0 {
        warn!("[Terrain] Coarse heightfield skipped: world bounds not set.");
        return;
    }

    // 32 m/cell — coarse enough to be cheap, fine enough to catch steep slopes.
    const CELL_SIZE: f32 = 32.0;

    let nx = (world_size.x / CELL_SIZE).ceil() as usize + 1;
    let nz = (world_size.y / CELL_SIZE).ceil() as usize + 1;

    let actual_size_x = (nx - 1) as f32 * CELL_SIZE;
    let actual_size_z = (nz - 1) as f32 * CELL_SIZE;

    // heights[xi][zi]: outer = X axis, inner = Z axis (Avian3d convention).
    let heights: Vec<Vec<f32>> = (0..nx)
        .map(|xi| {
            (0..nz)
                .map(|zi| {
                    let wx = world_min.x + xi as f32 * CELL_SIZE;
                    let wz = world_min.y + zi as f32 * CELL_SIZE;
                    cache.sample_height(Vec2::new(wx, wz)).unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    let center_x = world_min.x + actual_size_x * 0.5;
    let center_z = world_min.y + actual_size_z * 0.5;

    commands.spawn((
        RigidBody::Static,
        Collider::heightfield(heights, Vec3::new(actual_size_x, 1.0, actual_size_z)),
        Transform::from_xyz(center_x, 0.0, center_z),
    ));

    info!(
        "[Terrain] Coarse global heightfield: {}×{} samples at {:.0}m/cell, \
         {:.0}m×{:.0}m coverage.",
        nx, nz, CELL_SIZE, actual_size_x, actual_size_z
    );
}

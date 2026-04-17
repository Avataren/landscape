use avian3d::prelude::*;
use bevy::prelude::*;
use std::collections::HashMap;

use crate::terrain::{
    config::TerrainConfig, resources::TileKey, streamer::load_tile_data,
    world_desc::TerrainSourceDesc,
};

#[derive(Component)]
pub struct GlobalTerrainHeightfield;

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

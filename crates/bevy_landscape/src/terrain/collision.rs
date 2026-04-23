use crate::terrain::resources::{HeightTileCpu, TileKey};
use bevy::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CPU collision cache
// ---------------------------------------------------------------------------

/// CPU-side terrain height cache.
///
/// Responsibilities after Phase 3:
/// - `sample_height`: point height queries on the CPU (spawn placement, coarse
///   heightfield construction).  Answers at full tile resolution with no
///   physics overhead.
///
/// Runtime physics queries (raycasts, shape overlaps, ground detection) should
/// use Avian3d's `SpatialQuery` system parameter instead, which operates
/// against the live tile heightfield colliders:
///
/// ```rust,ignore
/// fn my_system(spatial: SpatialQuery) {
///     if let Some(hit) = spatial.cast_ray(
///         Vec3::new(x, 10_000.0, z),
///         Dir3::NEG_Y,
///         f32::MAX,
///         true,
///         &SpatialQueryFilter::default(),
///     ) {
///         // hit.time_of_impact is distance; origin + dir * toi = hit point
///     }
/// }
/// ```
#[derive(Resource, Default)]
pub struct TerrainCollisionCache {
    /// Tile data keyed by `TileKey`.  One entry per loaded tile.
    tiles: HashMap<TileKey, CollisionTile>,
    pub tile_size: u32,
    pub world_scale: f32,
    pub height_scale: f32,
}

struct CollisionTile {
    data: Vec<f32>,
    tile_size: u32,
}

impl TerrainCollisionCache {
    /// Insert or replace a tile from a CPU-decoded payload.
    pub fn upload_tile(&mut self, payload: &HeightTileCpu) {
        self.tiles.insert(
            payload.key,
            CollisionTile {
                data: payload.data.clone(),
                tile_size: payload.tile_size,
            },
        );
    }

    /// Sample terrain height at a world-space XZ position.
    /// Returns `None` if no tile covers that position.
    pub fn sample_height(&self, world_xz: Vec2) -> Option<f32> {
        let (key, local) = self.world_to_tile(world_xz)?;
        let tile = self.tiles.get(&key)?;
        Some(bilinear_sample(&tile.data, tile.tile_size, local) * self.height_scale)
    }

    /// Clone all level-0 tile data whose world footprint overlaps the given XZ rectangle.
    pub fn snapshot_tiles_for_region(
        &self,
        min_xz: Vec2,
        max_xz: Vec2,
    ) -> std::collections::HashMap<TileKey, Vec<f32>> {
        let mut snapshot = std::collections::HashMap::new();
        if self.tile_size == 0 || self.world_scale <= 0.0 {
            return snapshot;
        }
        let tile_world_size = self.tile_size as f32 * self.world_scale;
        let tx_min = (min_xz.x / tile_world_size).floor() as i32;
        let tx_max = (max_xz.x / tile_world_size).floor() as i32;
        let ty_min = (min_xz.y / tile_world_size).floor() as i32;
        let ty_max = (max_xz.y / tile_world_size).floor() as i32;
        for ty in ty_min..=ty_max {
            for tx in tx_min..=tx_max {
                let key = TileKey { level: 0, x: tx, y: ty };
                if let Some(tile) = self.tiles.get(&key) {
                    snapshot.insert(key, tile.data.clone());
                }
            }
        }
        snapshot
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn world_to_tile(&self, world_xz: Vec2) -> Option<(TileKey, Vec2)> {
        if self.tile_size == 0 || self.world_scale <= 0.0 {
            return None;
        }
        let tile_world_size = self.tile_size as f32 * self.world_scale;
        let tx = (world_xz.x / tile_world_size).floor() as i32;
        let ty = (world_xz.y / tile_world_size).floor() as i32;
        let local_x = (world_xz.x - tx as f32 * tile_world_size) / tile_world_size;
        let local_y = (world_xz.y - ty as f32 * tile_world_size) / tile_world_size;
        // Use LOD 0 for collision (highest detail available)
        Some((
            TileKey {
                level: 0,
                x: tx,
                y: ty,
            },
            Vec2::new(local_x, local_y),
        ))
    }
}

fn bilinear_sample(data: &[f32], size: u32, uv: Vec2) -> f32 {
    // Scale by `size` (not `size - 1`): world_to_tile normalises by
    // tile_world_size = tile_size * world_scale, so uv=0 → index 0 and
    // uv=(n/tile_size) → index n exactly for integer world positions n.
    // Using (size-1) would shift every sample by up to half a cell,
    // producing metre-scale height errors on steep terrain.
    let u = uv.x.clamp(0.0, 1.0) * size as f32;
    let v = uv.y.clamp(0.0, 1.0) * size as f32;
    let x0 = (u.floor() as usize).min(size as usize - 1);
    let y0 = (v.floor() as usize).min(size as usize - 1);
    let x1 = (x0 + 1).min(size as usize - 1);
    let y1 = (y0 + 1).min(size as usize - 1);
    let fx = u.fract();
    let fy = v.fract();

    let h00 = data[y0 * size as usize + x0];
    let h10 = data[y0 * size as usize + x1];
    let h01 = data[y1 * size as usize + x0];
    let h11 = data[y1 * size as usize + x1];

    let top = h00 * (1.0 - fx) + h10 * fx;
    let bot = h01 * (1.0 - fx) + h11 * fx;
    top * (1.0 - fy) + bot * fy
}

// ---------------------------------------------------------------------------
// System: apply loaded tiles to collision cache
// ---------------------------------------------------------------------------

pub fn update_collision_tiles(
    config: Res<crate::terrain::config::TerrainConfig>,
    residency: Res<crate::terrain::resources::TerrainResidency>,
    mut cache: ResMut<TerrainCollisionCache>,
) {
    cache.tile_size = config.tile_size;
    cache.world_scale = config.world_scale;
    cache.height_scale = config.height_scale;

    // Consume pending uploads (collision takes a copy).
    for tile in &residency.pending_upload {
        cache.upload_tile(tile);
    }
    // Don't clear pending_upload here; GPU path also consumes it.
}

use bevy::prelude::*;
use std::collections::HashMap;
use crate::terrain::resources::{HeightTileCpu, TileKey};

// ---------------------------------------------------------------------------
// Collision hit result
// ---------------------------------------------------------------------------

pub struct TerrainHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// CPU collision cache
// ---------------------------------------------------------------------------

/// A low-resolution CPU-side terrain height cache for gameplay queries.
///
/// Populated from the same decoded tile data as the GPU path, but maintained
/// independently so it never requires GPU readback.
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

    /// Approximate terrain normal at a world-space XZ position via finite differences.
    pub fn sample_normal(&self, world_xz: Vec2) -> Option<Vec3> {
        let eps = self.world_scale;
        let h0 = self.sample_height(world_xz)?;
        let hx = self
            .sample_height(world_xz + Vec2::new(eps, 0.0))
            .unwrap_or(h0);
        let hy = self
            .sample_height(world_xz + Vec2::new(0.0, eps))
            .unwrap_or(h0);
        let n = Vec3::new(h0 - hx, eps, h0 - hy).normalize();
        Some(n)
    }

    /// Step a ray through the terrain and return the first intersection.
    pub fn raycast_terrain(&self, ray: Ray3d, max_dist: f32) -> Option<TerrainHit> {
        let step = self.world_scale.max(0.5);
        let mut t = 0.0_f32;

        while t < max_dist {
            let p = ray.origin + *ray.direction * t;
            let xz = Vec2::new(p.x, p.z);
            if let Some(h) = self.sample_height(xz) {
                if p.y <= h {
                    let normal = self.sample_normal(xz).unwrap_or(Vec3::Y);
                    return Some(TerrainHit {
                        point: Vec3::new(p.x, h, p.z),
                        normal,
                        distance: t,
                    });
                }
            }
            t += step;
        }
        None
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
        Some((TileKey { level: 0, x: tx, y: ty }, Vec2::new(local_x, local_y)))
    }
}

fn bilinear_sample(data: &[f32], size: u32, uv: Vec2) -> f32 {
    let u = uv.x.clamp(0.0, 1.0) * (size - 1) as f32;
    let v = uv.y.clamp(0.0, 1.0) * (size - 1) as f32;
    let x0 = u.floor() as usize;
    let y0 = v.floor() as usize;
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

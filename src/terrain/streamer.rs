use crate::terrain::{
    clipmap_texture::{height_at_world, normal_at_world},
    resources::{HeightTileCpu, TerrainResidency, TerrainStreamQueue, TileKey, TileState},
    world_desc::TerrainSourceDesc,
};
use bevy::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    mpsc::{self, Sender},
    Arc, Mutex,
};

// ---------------------------------------------------------------------------
// Background job channel
// ---------------------------------------------------------------------------

/// Receiver wrapped in Arc<Mutex> so it can be a Bevy resource (needs Sync).
#[derive(Resource, Clone)]
pub struct TerrainTileReceiver(pub Arc<Mutex<std::sync::mpsc::Receiver<HeightTileCpu>>>);

/// Loads height data for a single tile synchronously (blocking).
///
/// Tries the pre-baked file first; falls back to procedural or zeros.
/// Shared by the background-thread streamer and the startup preloader.
pub(crate) fn load_tile_data(
    key: TileKey,
    tile_size: u32,
    world_scale: f32,
    height_scale: f32,
    max_mip_level: u8,
    tile_root: Option<&std::path::Path>,
    normal_root: Option<&std::path::Path>,
    world_bounds: Option<(Vec2, Vec2)>,
    use_procedural: bool,
) -> HeightTileCpu {
    let level_scale_ws = world_scale * (1u32 << (key.level as u32)) as f32;
    // Geometry can use more clipmap levels than the baked tile hierarchy.
    // Reuse the coarsest available mip for those far rings instead of treating
    // them as missing data, which makes terrain appear to rise from a flat floor
    // as regions move inward.
    let source_level = key.level.min(max_mip_level);
    let source_scale_ws = world_scale * (1u32 << (source_level as u32)) as f32;
    let bounds = world_bounds.and_then(|(world_min, world_max)| {
        grid_bounds_for_level(world_min, world_max, source_scale_ws)
    });
    let normal_root = normal_root
        .map(PathBuf::from)
        .or_else(|| tile_root.map(|root| root.join("normal")));

    let height_data = tile_root
        .and_then(|root| {
            if source_level == key.level {
                load_disk_height_tile(root, key, tile_size, bounds)
            } else {
                build_resampled_tile(
                    key,
                    tile_size,
                    level_scale_ws,
                    source_level,
                    source_scale_ws,
                    bounds,
                    |src_key| read_r16_tile(&height_tile_path(root, src_key), tile_size),
                )
            }
        })
        .unwrap_or_else(|| {
            let len = (tile_size * tile_size) as usize;
            if use_procedural {
                let mut pixels = Vec::with_capacity(len);
                for row in 0..tile_size {
                    for col in 0..tile_size {
                        let world_x =
                            ((key.x * tile_size as i32 + col as i32) as f32 + 0.5) * level_scale_ws;
                        let world_z =
                            ((key.y * tile_size as i32 + row as i32) as f32 + 0.5) * level_scale_ws;
                        pixels.push(height_at_world(world_x, world_z));
                    }
                }
                pixels
            } else {
                vec![0.0f32; len]
            }
        });

    let normal_data = normal_root
        .as_deref()
        .and_then(|root| {
            if source_level == key.level {
                load_disk_normal_tile(root, key, tile_size, bounds)
            } else {
                build_resampled_tile(
                    key,
                    tile_size,
                    level_scale_ws,
                    source_level,
                    source_scale_ws,
                    bounds,
                    |src_key| read_rg8_snorm_tile(&normal_tile_path(root, src_key), tile_size),
                )
            }
        })
        .unwrap_or_else(|| {
            if use_procedural {
                build_procedural_normal_tile(key, tile_size, level_scale_ws, height_scale)
            } else {
                vec![[0u8, 0u8]; (tile_size * tile_size) as usize]
            }
        });

    HeightTileCpu {
        key,
        data: height_data,
        normal_data,
        tile_size,
    }
}

/// Spawns a background OS thread to load and decode one height tile.
pub fn spawn_background_height_job(
    key: TileKey,
    tile_size: u32,
    world_scale: f32,
    height_scale: f32,
    max_mip_level: u8,
    tile_root: Option<std::path::PathBuf>,
    normal_root: Option<std::path::PathBuf>,
    world_bounds: Option<(Vec2, Vec2)>,
    use_procedural: bool,
    tx: Sender<HeightTileCpu>,
) {
    std::thread::spawn(move || {
        let data = load_tile_data(
            key,
            tile_size,
            world_scale,
            height_scale,
            max_mip_level,
            tile_root.as_deref(),
            normal_root.as_deref(),
            world_bounds,
            use_procedural,
        );
        let _ = tx.send(data);
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GridBounds {
    min: IVec2,
    max: IVec2,
}

fn height_tile_path(root: &Path, key: TileKey) -> PathBuf {
    root.join(format!("height/L{}/{}_{}.bin", key.level, key.x, key.y))
}

fn normal_tile_path(root: &Path, key: TileKey) -> PathBuf {
    root.join(format!("L{}/{}_{}.bin", key.level, key.x, key.y))
}

fn tile_grid_bounds(key: TileKey, tile_size: u32) -> GridBounds {
    let tile_size = tile_size as i32;
    let min = IVec2::new(key.x * tile_size, key.y * tile_size);
    GridBounds {
        min,
        max: min + IVec2::splat(tile_size - 1),
    }
}

fn grid_bounds_for_level(
    world_min: Vec2,
    world_max: Vec2,
    level_scale_ws: f32,
) -> Option<GridBounds> {
    if level_scale_ws <= 0.0 || !world_max.cmpgt(world_min).all() {
        return None;
    }

    let min = IVec2::new(
        (world_min.x / level_scale_ws).floor() as i32,
        (world_min.y / level_scale_ws).floor() as i32,
    );
    let max_exclusive = IVec2::new(
        (world_max.x / level_scale_ws).ceil() as i32,
        (world_max.y / level_scale_ws).ceil() as i32,
    );

    if max_exclusive.x <= min.x || max_exclusive.y <= min.y {
        return None;
    }

    Some(GridBounds {
        min,
        max: max_exclusive - IVec2::ONE,
    })
}

fn build_bounded_tile<T, F>(
    key: TileKey,
    tile_size: u32,
    bounds: GridBounds,
    mut fetch_tile: F,
) -> Option<Vec<T>>
where
    T: Copy + Default,
    F: FnMut(TileKey) -> Option<Vec<T>>,
{
    let requested = tile_grid_bounds(key, tile_size);
    if requested.min.x >= bounds.min.x
        && requested.max.x <= bounds.max.x
        && requested.min.y >= bounds.min.y
        && requested.max.y <= bounds.max.y
    {
        return fetch_tile(key);
    }

    let tile_size_i32 = tile_size as i32;
    let mut out = vec![T::default(); (tile_size * tile_size) as usize];
    let mut cache: HashMap<TileKey, Vec<T>> = HashMap::new();

    for row in 0..tile_size {
        for col in 0..tile_size {
            let gx = key.x * tile_size_i32 + col as i32;
            let gz = key.y * tile_size_i32 + row as i32;
            if gx < bounds.min.x || gx > bounds.max.x || gz < bounds.min.y || gz > bounds.max.y {
                continue;
            }

            let src_key = TileKey {
                level: key.level,
                x: gx.div_euclid(tile_size_i32),
                y: gz.div_euclid(tile_size_i32),
            };

            if !cache.contains_key(&src_key) {
                cache.insert(src_key, fetch_tile(src_key)?);
            }

            let src_tile = cache.get(&src_key)?;
            let local_x = gx.rem_euclid(tile_size_i32) as usize;
            let local_y = gz.rem_euclid(tile_size_i32) as usize;
            out[(row * tile_size + col) as usize] =
                src_tile[local_y * tile_size as usize + local_x];
        }
    }

    Some(out)
}

fn build_resampled_tile<T, F>(
    key: TileKey,
    tile_size: u32,
    requested_scale_ws: f32,
    source_level: u8,
    source_scale_ws: f32,
    source_bounds: Option<GridBounds>,
    mut fetch_tile: F,
) -> Option<Vec<T>>
where
    T: Copy + Default,
    F: FnMut(TileKey) -> Option<Vec<T>>,
{
    let tile_size_i32 = tile_size as i32;
    let mut out = vec![T::default(); (tile_size * tile_size) as usize];
    let mut cache: HashMap<TileKey, Vec<T>> = HashMap::new();

    for row in 0..tile_size {
        for col in 0..tile_size {
            let world_x = ((key.x * tile_size_i32 + col as i32) as f32 + 0.5) * requested_scale_ws;
            let world_z = ((key.y * tile_size_i32 + row as i32) as f32 + 0.5) * requested_scale_ws;

            let gx = (world_x / source_scale_ws).floor() as i32;
            let gz = (world_z / source_scale_ws).floor() as i32;

            if let Some(bounds) = source_bounds {
                if gx < bounds.min.x || gx > bounds.max.x || gz < bounds.min.y || gz > bounds.max.y
                {
                    continue;
                }
            }

            let src_key = TileKey {
                level: source_level,
                x: gx.div_euclid(tile_size_i32),
                y: gz.div_euclid(tile_size_i32),
            };

            if !cache.contains_key(&src_key) {
                cache.insert(src_key, fetch_tile(src_key)?);
            }

            let src_tile = cache.get(&src_key)?;
            let local_x = gx.rem_euclid(tile_size_i32) as usize;
            let local_y = gz.rem_euclid(tile_size_i32) as usize;
            out[(row * tile_size + col) as usize] =
                src_tile[local_y * tile_size as usize + local_x];
        }
    }

    Some(out)
}

fn load_disk_height_tile(
    root: &Path,
    key: TileKey,
    tile_size: u32,
    bounds: Option<GridBounds>,
) -> Option<Vec<f32>> {
    match bounds {
        Some(bounds) => build_bounded_tile(key, tile_size, bounds, |src_key| {
            read_r16_tile(&height_tile_path(root, src_key), tile_size)
        }),
        None => read_r16_tile(&height_tile_path(root, key), tile_size),
    }
}

fn load_disk_normal_tile(
    root: &Path,
    key: TileKey,
    tile_size: u32,
    bounds: Option<GridBounds>,
) -> Option<Vec<[u8; 2]>> {
    match bounds {
        Some(bounds) => build_bounded_tile(key, tile_size, bounds, |src_key| {
            read_rg8_snorm_tile(&normal_tile_path(root, src_key), tile_size)
        }),
        None => read_rg8_snorm_tile(&normal_tile_path(root, key), tile_size),
    }
}

/// Reads a pre-baked R16Unorm tile from disk and converts to f32 in [0, 1].
/// Returns `None` if the file does not exist or has an unexpected size.
fn read_r16_tile(path: &std::path::Path, tile_size: u32) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    let expected = (tile_size * tile_size * 2) as usize;
    if bytes.len() != expected {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0)
            .collect(),
    )
}

fn read_rg8_snorm_tile(path: &Path, tile_size: u32) -> Option<Vec<[u8; 2]>> {
    let bytes = std::fs::read(path).ok()?;
    let expected = (tile_size * tile_size * 2) as usize;
    if bytes.len() != expected {
        return None;
    }
    Some(bytes.chunks_exact(2).map(|b| [b[0], b[1]]).collect())
}

fn encode_normal_xz_snorm(normal: Vec3) -> [u8; 2] {
    [
        (normal.x.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
        (normal.z.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
    ]
}

fn build_procedural_normal_tile(
    key: TileKey,
    tile_size: u32,
    level_scale_ws: f32,
    height_scale: f32,
) -> Vec<[u8; 2]> {
    let len = (tile_size * tile_size) as usize;
    let mut pixels = Vec::with_capacity(len);
    for row in 0..tile_size {
        for col in 0..tile_size {
            let world_x = ((key.x * tile_size as i32 + col as i32) as f32 + 0.5) * level_scale_ws;
            let world_z = ((key.y * tile_size as i32 + row as i32) as f32 + 0.5) * level_scale_ws;
            pixels.push(encode_normal_xz_snorm(normal_at_world(
                world_x,
                world_z,
                level_scale_ws,
                height_scale,
            )));
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Requests background loads for tiles that are needed but not yet requested.
pub fn request_tile_loads(
    mut queue: ResMut<TerrainStreamQueue>,
    mut residency: ResMut<TerrainResidency>,
    desc: Res<TerrainSourceDesc>,
    config: Res<crate::terrain::config::TerrainConfig>,
    sender: Option<Res<TerrainTileSender>>,
) {
    let Some(sender) = sender else { return };

    // Collect keys, coarsest LOD first so far terrain fills in before close detail.
    let mut needed: Vec<TileKey> = residency.required_now.iter().copied().collect();
    needed.sort_by(|a, b| b.level.cmp(&a.level));

    for key in needed {
        if queue.pending_requests.contains(&key) {
            continue;
        }
        let already_loaded = matches!(
            residency.tiles.get(&key),
            Some(TileState::LoadedCpu | TileState::ResidentGpu { .. })
        );
        if already_loaded {
            continue;
        }

        queue.pending_requests.insert(key);
        residency.tiles.insert(key, TileState::Requested);
        spawn_background_height_job(
            key,
            config.tile_size,
            config.world_scale,
            config.height_scale,
            desc.max_mip_level,
            desc.tile_root.clone(),
            desc.normal_root.as_ref().map(PathBuf::from),
            Some((desc.world_min, desc.world_max)),
            config.procedural_fallback,
            sender.0.clone(),
        );
    }
}

/// Drains finished background jobs into the residency pending-upload list.
pub fn poll_tile_stream_jobs(
    receiver: Option<Res<TerrainTileReceiver>>,
    mut queue: ResMut<TerrainStreamQueue>,
    mut residency: ResMut<TerrainResidency>,
) {
    let Some(receiver) = receiver else { return };
    let Ok(rx) = receiver.0.try_lock() else {
        return;
    };

    while let Ok(tile) = rx.try_recv() {
        let key = tile.key;
        queue.pending_requests.remove(&key);
        residency.tiles.insert(key, TileState::LoadedCpu);
        residency.touch(key);
        residency.pending_upload.push(tile);
    }
}

// ---------------------------------------------------------------------------
// Sender resource
// ---------------------------------------------------------------------------

/// The `Sender` end of the tile channel, kept as a resource so systems can
/// hand it to spawned threads.
#[derive(Resource, Clone)]
pub struct TerrainTileSender(pub Sender<HeightTileCpu>);

/// Creates the channel and inserts both halves as resources.
pub fn setup_tile_channel(mut commands: Commands) {
    let (tx, rx) = mpsc::channel();
    commands.insert_resource(TerrainTileSender(tx));
    commands.insert_resource(TerrainTileReceiver(Arc::new(Mutex::new(rx))));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_bounds_match_heightmap_extent() {
        let bounds = grid_bounds_for_level(Vec2::splat(-8192.0), Vec2::splat(8192.0), 1.0).unwrap();
        assert_eq!(bounds.min, IVec2::splat(-8192));
        assert_eq!(bounds.max, IVec2::splat(8191));
    }

    #[test]
    fn out_of_bounds_pixels_zero_fill() {
        let mut source_tiles = HashMap::new();
        source_tiles.insert(
            TileKey {
                level: 0,
                x: 0,
                y: 0,
            },
            vec![1.0, 2.0, 3.0, 4.0],
        );

        let bounds = GridBounds {
            min: IVec2::ZERO,
            max: IVec2::ONE,
        };

        let tile = build_bounded_tile(
            TileKey {
                level: 0,
                x: 1,
                y: 0,
            },
            2,
            bounds,
            |src_key| source_tiles.get(&src_key).cloned(),
        )
        .unwrap();

        assert_eq!(tile, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn resamples_from_coarsest_available_mip() {
        let mut source_tiles = HashMap::new();
        source_tiles.insert(
            TileKey {
                level: 0,
                x: 0,
                y: 0,
            },
            vec![1.0, 2.0, 3.0, 4.0],
        );
        source_tiles.insert(
            TileKey {
                level: 0,
                x: 1,
                y: 0,
            },
            vec![5.0, 6.0, 7.0, 8.0],
        );
        source_tiles.insert(
            TileKey {
                level: 0,
                x: 0,
                y: 1,
            },
            vec![9.0, 10.0, 11.0, 12.0],
        );
        source_tiles.insert(
            TileKey {
                level: 0,
                x: 1,
                y: 1,
            },
            vec![13.0, 14.0, 15.0, 16.0],
        );

        let tile = build_resampled_tile(
            TileKey {
                level: 1,
                x: 0,
                y: 0,
            },
            2,
            2.0,
            0,
            1.0,
            None,
            |src_key| source_tiles.get(&src_key).cloned(),
        )
        .unwrap();

        assert_eq!(tile, vec![4.0, 8.0, 12.0, 16.0]);
    }
}

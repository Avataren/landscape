use bevy::prelude::*;
use std::sync::{Arc, Mutex, mpsc::{self, Sender}};
use crate::terrain::{
    clipmap_texture::height_at_world,
    resources::{HeightTileCpu, TerrainResidency, TerrainStreamQueue, TileKey, TileState},
    world_desc::TerrainSourceDesc,
};

// ---------------------------------------------------------------------------
// Background job channel
// ---------------------------------------------------------------------------

/// Receiver wrapped in Arc<Mutex> so it can be a Bevy resource (needs Sync).
#[derive(Resource, Clone)]
pub struct TerrainTileReceiver(pub Arc<Mutex<std::sync::mpsc::Receiver<HeightTileCpu>>>);

/// Spawns a background OS thread to load and decode one height tile.
///
/// In a full implementation this reads from disk/network and decodes
/// the compressed format. Here we generate synthetic height data so the
/// render pipeline can be exercised without on-disk assets.
pub fn spawn_background_height_job(
    key: TileKey,
    tile_size: u32,
    world_scale: f32,
    tile_root: Option<std::path::PathBuf>,
    use_procedural: bool,
    tx: Sender<HeightTileCpu>,
) {
    std::thread::spawn(move || {
        // Try to load a pre-baked tile file first.
        let data = tile_root
            .as_ref()
            .and_then(|root| {
                let path = root.join(format!(
                    "height/L{}/{}_{}.bin",
                    key.level, key.x, key.y
                ));
                read_r16_tile(&path, tile_size)
            })
            .unwrap_or_else(|| {
                let len = (tile_size * tile_size) as usize;
                if use_procedural {
                    // Procedural fallback: multi-octave sine waves.
                    let level_scale_ws = world_scale * (1u32 << (key.level as u32)) as f32;
                    let mut pixels = Vec::with_capacity(len);
                    for row in 0..tile_size {
                        for col in 0..tile_size {
                            let world_x = ((key.x * tile_size as i32 + col as i32) as f32 + 0.5)
                                * level_scale_ws;
                            let world_z = ((key.y * tile_size as i32 + row as i32) as f32 + 0.5)
                                * level_scale_ws;
                            pixels.push(height_at_world(world_x, world_z));
                        }
                    }
                    pixels
                } else {
                    // Flat fallback: tile file missing, show zero height.
                    vec![0.0f32; len]
                }
            });

        let _ = tx.send(HeightTileCpu { key, data, tile_size });
    });
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

    // Collect keys first to avoid borrowing residency while mutating it.
    let needed: Vec<TileKey> = residency.required_now.iter().copied().collect();

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
            desc.tile_root.clone(),
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
    let Ok(rx) = receiver.0.try_lock() else { return };

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

//! Phase 9a: Foliage generation backend.
//!
//! Drives the "Generate / Regenerate Foliage" pipeline:
//!   1. Scan the terrain tile hierarchy for all L0 height tiles.
//!   2. For each tile, compute procedural density masks and bake instances.
//!   3. Write LOD0/LOD1/LOD2 binary tile files to foliage_root.
//!   4. After generation, load the resulting tiles into FoliageStagingQueue
//!      so the renderer picks them up automatically.
//!
//! The heavy work runs on a background OS thread; the main thread polls a
//! channel for progress updates and the final completion signal.

use crate::{
    foliage::{FoliageConfig, FoliageLodTier},
    foliage_gpu::{FoliageStagingBatch, FoliageStagingQueue},
    foliage_instance_gen::bake_and_write_foliage_instances,
    foliage_plugin::FoliageMeshHandles,
    foliage_reload::FoliageConfigResource,
    foliage_tiles::read_foliage_tile,
    terrain::config::TerrainConfig,
    terrain::world_desc::TerrainSourceDesc,
    FoliageSourceDesc,
};
use bevy::prelude::*;
use std::{
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
};

// ---------------------------------------------------------------------------
// Events & resources
// ---------------------------------------------------------------------------

/// Send this message to trigger foliage (re)generation from the terrain heightmap.
#[derive(Message, Default, Clone)]
pub struct FoliageGenerateRequest;

/// Progress messages sent from the background thread to the main thread.
#[derive(Debug)]
pub enum GenerationProgress {
    /// A tile finished baking. `done` out of `total`.
    TileComplete { done: usize, total: usize },
    /// All tiles done, `total_instances` generated across all LOD0 tiles.
    Finished { total_instances: usize },
    /// Generation failed with an error message.
    Failed(String),
}

/// Tracks the state of an in-progress or completed generation run.
#[derive(Resource, Default)]
pub struct FoliageGenerationState {
    pub is_running: bool,
    pub progress_message: String,
    pub tiles_done: usize,
    pub tiles_total: usize,
    /// Receiver wrapped for Bevy resource compatibility.
    pub receiver: Option<Arc<Mutex<Receiver<GenerationProgress>>>>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// React to `FoliageGenerateRequest` messages and start the background thread.
pub fn start_foliage_generation(
    mut events: MessageReader<FoliageGenerateRequest>,
    mut state: ResMut<FoliageGenerationState>,
    terrain_source: Res<TerrainSourceDesc>,
    terrain_config: Res<TerrainConfig>,
    foliage_source: Res<FoliageSourceDesc>,
    foliage_config_res: Res<FoliageConfigResource>,
) {
    for _ in events.read() {
        if state.is_running {
            info!("Foliage generation already in progress, ignoring request");
            continue;
        }

        let Some(foliage_root) = foliage_source.foliage_root.clone() else {
            warn!("Cannot generate foliage: foliage_root not set in FoliageSourceDesc");
            continue;
        };

        let config = match &foliage_config_res.0 {
            Some(c) => c.clone(),
            None => FoliageConfig::default(),
        };

        let tile_root = terrain_source.tile_root.clone();
        let tile_size = terrain_config.tile_size;
        let world_scale = terrain_config.world_scale;
        let height_scale = terrain_config.height_scale;
        let max_mip = terrain_source.max_mip_level;

        let (tx, rx) = mpsc::channel::<GenerationProgress>();

        std::thread::spawn(move || {
            run_generation(
                foliage_root,
                tile_root,
                tile_size,
                world_scale,
                height_scale,
                max_mip,
                config,
                tx,
            );
        });

        state.is_running = true;
        state.progress_message = "Starting generation...".to_string();
        state.tiles_done = 0;
        state.tiles_total = 0;
        state.receiver = Some(Arc::new(Mutex::new(rx)));
        info!("Foliage generation thread started");
    }
}

/// Poll the generation channel and handle results.
pub fn poll_foliage_generation(
    mut state: ResMut<FoliageGenerationState>,
    foliage_source: Res<FoliageSourceDesc>,
    foliage_config_res: Res<FoliageConfigResource>,
    mut staging_queue: ResMut<FoliageStagingQueue>,
    _handles: Res<FoliageMeshHandles>,
) {
    if !state.is_running {
        return;
    }

    let Some(rx_arc) = state.receiver.clone() else {
        return;
    };

    let Ok(rx) = rx_arc.lock() else { return };
    loop {
        match rx.try_recv() {
            Ok(GenerationProgress::TileComplete { done, total }) => {
                state.tiles_done = done;
                state.tiles_total = total;
                state.progress_message = format!("Baking tiles {done}/{total}...");
            }
            Ok(GenerationProgress::Finished { total_instances }) => {
                state.is_running = false;
                state.receiver = None;
                state.progress_message = format!("Done — {total_instances} instances generated.");
                info!("Foliage generation complete: {} instances", total_instances);

                // Load generated tiles into the staging queue
                if let Some(foliage_root) = &foliage_source.foliage_root {
                    load_generated_tiles_into_queue(
                        foliage_root,
                        foliage_config_res
                            .0
                            .as_ref()
                            .unwrap_or(&FoliageConfig::default()),
                        &mut staging_queue,
                    );
                }
                break;
            }
            Ok(GenerationProgress::Failed(msg)) => {
                state.is_running = false;
                state.receiver = None;
                state.progress_message = format!("Generation failed: {msg}");
                error!("Foliage generation failed: {}", msg);
                break;
            }
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                state.is_running = false;
                state.receiver = None;
                break;
            }
        }
    }
}

/// At startup, if foliage tiles already exist on disk, load them.
pub fn load_existing_foliage_tiles(
    foliage_source: Res<FoliageSourceDesc>,
    foliage_config_res: Res<FoliageConfigResource>,
    mut staging_queue: ResMut<FoliageStagingQueue>,
) {
    let Some(foliage_root) = &foliage_source.foliage_root else {
        return;
    };

    let config = foliage_config_res.0.as_ref().cloned().unwrap_or_default();
    let lod0_dir = foliage_root.join("LOD0/L0");

    if !lod0_dir.exists() {
        return;
    }

    info!("Foliage: loading existing tiles from {:?}", foliage_root);
    load_generated_tiles_into_queue(foliage_root, &config, &mut staging_queue);
}

// ---------------------------------------------------------------------------
// Background generation thread
// ---------------------------------------------------------------------------

fn run_generation(
    foliage_root: PathBuf,
    tile_root: Option<PathBuf>,
    tile_size: u32,
    world_scale: f32,
    height_scale: f32,
    max_mip: u8,
    config: FoliageConfig,
    tx: Sender<GenerationProgress>,
) {
    // Discover all L0 tile files
    let tiles = discover_height_tiles(tile_root.as_deref(), max_mip);
    let total = tiles.len();

    if total == 0 {
        // No terrain tiles — generate a flat test area
        let _ = tx.send(GenerationProgress::Failed(
            "No terrain height tiles found; run bake_tiles first or load a level".to_string(),
        ));
        return;
    }

    let mut total_instances = 0usize;

    for (done, (tx_tile, ty_tile, height_data)) in tiles.into_iter().enumerate() {
        match bake_and_write_foliage_instances(
            &foliage_root,
            tile_size,
            0, // L0 only for now
            tx_tile,
            ty_tile,
            world_scale,
            height_scale,
            &height_data,
            &config,
        ) {
            Ok(()) => {
                // Count instances in the written LOD0 tile
                let lod0_path = crate::foliage::foliage_tile_path(
                    &foliage_root,
                    FoliageLodTier::Lod0,
                    0,
                    tx_tile,
                    ty_tile,
                );
                if let Ok(instances) = read_foliage_tile(&lod0_path) {
                    total_instances += instances.len();
                }
            }
            Err(e) => {
                let _ = tx.send(GenerationProgress::Failed(format!(
                    "Tile ({tx_tile},{ty_tile}): {e}"
                )));
                return;
            }
        }

        let _ = tx.send(GenerationProgress::TileComplete {
            done: done + 1,
            total,
        });
    }

    let _ = tx.send(GenerationProgress::Finished { total_instances });
}

/// Returns `(tx, ty, height_data_f32)` for every L0 height tile found on disk.
fn discover_height_tiles(tile_root: Option<&Path>, _max_mip: u8) -> Vec<(i32, i32, Vec<f32>)> {
    let Some(root) = tile_root else {
        return vec![];
    };

    let l0_dir = root.join("height/L0");
    if !l0_dir.exists() {
        return vec![];
    }

    let mut tiles = vec![];

    let Ok(entries) = std::fs::read_dir(&l0_dir) else {
        return tiles;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }

        // Parse tx_ty from filename like "3_-2.bin"
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let parts: Vec<&str> = stem.split('_').collect();
        if parts.len() < 2 {
            continue;
        }
        let tx = parts[0].parse::<i32>().unwrap_or(0);
        // Handle negative coordinates: filename might be "3_-2.bin" → stem "3_-2"
        let ty_str = parts[1..].join("_");
        let ty = ty_str.parse::<i32>().unwrap_or(0);

        // Read R16Unorm tile → f32 heights
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        // Each texel is 2 bytes (R16Unorm), 256×256 = 65536 texels
        if bytes.len() < 4 {
            continue;
        }
        let height_data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0)
            .collect();

        tiles.push((tx, ty, height_data));
    }

    tiles
}

// ---------------------------------------------------------------------------
// Load tiles from disk into staging queue
// ---------------------------------------------------------------------------

/// Scan foliage_root for all LOD0/L0 tiles and push them into the staging queue.
/// LOD1 and LOD2 tiles are loaded similarly.
pub fn load_generated_tiles_into_queue(
    foliage_root: &Path,
    _config: &FoliageConfig,
    staging_queue: &mut FoliageStagingQueue,
) {
    for lod in FoliageLodTier::all() {
        let lod_dir = foliage_root.join(format!("LOD{}/L0", *lod as u8));
        if !lod_dir.exists() {
            continue;
        }

        let Ok(entries) = std::fs::read_dir(&lod_dir) else {
            continue;
        };

        // Collect all instances for this LOD (across all tiles) grouped by variant
        let mut per_variant: [Vec<crate::foliage::FoliageInstance>; 8] =
            std::array::from_fn(|_| Vec::new());

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Ok(instances) = read_foliage_tile(&path) else {
                continue;
            };
            for inst in instances {
                let v = inst.variant_id.min(7) as usize;
                per_variant[v].push(inst);
            }
        }

        // Push one batch per variant
        for (variant_id, instances) in per_variant.iter().enumerate() {
            if instances.is_empty() {
                continue;
            }
            staging_queue.push(FoliageStagingBatch {
                lod: *lod,
                variant_id: variant_id as u8,
                offset: 0,
                instances: instances.clone(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_generated_tiles_empty_dir() {
        let dir = TempDir::new().unwrap();
        let mut queue = FoliageStagingQueue::default();
        let config = FoliageConfig::default();
        // No tiles present — should not panic, queue stays empty
        load_generated_tiles_into_queue(dir.path(), &config, &mut queue);
        assert!(queue.batches.is_empty());
    }

    #[test]
    fn test_discover_height_tiles_no_dir() {
        let tiles = discover_height_tiles(None, 0);
        assert!(tiles.is_empty());
    }
}

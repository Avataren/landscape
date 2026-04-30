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

use super::{FoliageConfig, FoliageLodTier, FoliageSourceDesc};
use super::gpu::{FoliageStagingBatch, FoliageStagingQueue};
use super::instance_gen::bake_and_write_foliage_instances;
use super::reload::FoliageConfigResource;
use super::tiles::read_foliage_tile;
use crate::metadata::TerrainMetadata;
use crate::terrain::config::TerrainConfig;
use crate::terrain::world_desc::TerrainSourceDesc;
use bevy::prelude::*;
use std::{
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
};

/// Send this message to trigger foliage (re)generation from the terrain heightmap.
#[derive(Message, Default, Clone)]
pub struct FoliageGenerateRequest;

/// Progress messages sent from the background thread to the main thread.
#[derive(Debug)]
pub enum GenerationProgress {
    TileComplete { done: usize, total: usize },
    Finished { total_instances: usize },
    Failed(String),
}

/// Tracks the state of an in-progress or completed generation run.
#[derive(Resource, Default)]
pub struct FoliageGenerationState {
    pub is_running: bool,
    pub progress_message: String,
    pub tiles_done: usize,
    pub tiles_total: usize,
    pub receiver: Option<Arc<Mutex<Receiver<GenerationProgress>>>>,
}

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
                foliage_root, tile_root, tile_size, world_scale, height_scale, max_mip, config, tx,
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
) {
    if !state.is_running {
        return;
    }
    let Some(rx_arc) = state.receiver.clone() else { return };
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
                if let Some(foliage_root) = &foliage_source.foliage_root {
                    load_generated_tiles_into_queue(
                        foliage_root,
                        foliage_config_res.0.as_ref().unwrap_or(&FoliageConfig::default()),
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
    let Some(foliage_root) = &foliage_source.foliage_root else { return };
    let config = foliage_config_res.0.as_ref().cloned().unwrap_or_default();
    let lod0_dir = foliage_root.join("LOD0/L0");
    if !lod0_dir.exists() {
        return;
    }
    info!("Foliage: loading existing tiles from {:?}", foliage_root);
    load_generated_tiles_into_queue(foliage_root, &config, &mut staging_queue);
}

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
    let tiles = discover_height_tiles(tile_root.as_deref(), max_mip);
    let total = tiles.len();

    if total == 0 {
        let _ = tx.send(GenerationProgress::Failed(
            "No terrain height tiles found; run bake_tiles first or load a level".to_string(),
        ));
        return;
    }

    // Load water level from tile metadata (normalised [0,1] × height_scale → world Y).
    let water_level = tile_root
        .as_deref()
        .map(TerrainMetadata::load)
        .and_then(|m| m.water_level)
        .map(|wl| wl * height_scale)
        .unwrap_or(f32::NEG_INFINITY);

    let mut total_instances = 0usize;

    for (done, (tx_tile, ty_tile, height_data)) in tiles.into_iter().enumerate() {
        match bake_and_write_foliage_instances(
            &foliage_root,
            tile_size,
            0,
            tx_tile,
            ty_tile,
            world_scale,
            height_scale,
            &height_data,
            &config,
            water_level,
        ) {
            Ok(()) => {
                let lod0_path = super::foliage_tile_path(
                    &foliage_root, FoliageLodTier::Lod0, 0, tx_tile, ty_tile,
                );
                if let Ok(instances) = read_foliage_tile(&lod0_path) {
                    total_instances += instances.len();
                }
            }
            Err(e) => {
                let _ = tx.send(GenerationProgress::Failed(
                    format!("Tile ({tx_tile},{ty_tile}): {e}"),
                ));
                return;
            }
        }
        let _ = tx.send(GenerationProgress::TileComplete { done: done + 1, total });
    }

    let _ = tx.send(GenerationProgress::Finished { total_instances });
}

fn discover_height_tiles(tile_root: Option<&Path>, _max_mip: u8) -> Vec<(i32, i32, Vec<f32>)> {
    let Some(root) = tile_root else { return vec![] };
    let l0_dir = root.join("height/L0");
    if !l0_dir.exists() {
        return vec![];
    }
    let Ok(entries) = std::fs::read_dir(&l0_dir) else { return vec![] };
    let mut tiles = vec![];

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let parts: Vec<&str> = stem.split('_').collect();
        if parts.len() < 2 {
            continue;
        }
        let tx = parts[0].parse::<i32>().unwrap_or(0);
        let ty_str = parts[1..].join("_");
        let ty = ty_str.parse::<i32>().unwrap_or(0);
        let Ok(bytes) = std::fs::read(&path) else { continue };
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
        let Ok(entries) = std::fs::read_dir(&lod_dir) else { continue };

        let mut per_variant: [Vec<super::FoliageInstance>; 8] =
            std::array::from_fn(|_| Vec::new());

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Ok(instances) = read_foliage_tile(&path) else { continue };
            for inst in instances {
                let v = inst.variant_id.min(7) as usize;
                per_variant[v].push(inst);
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_generated_tiles_empty_dir() {
        let dir = TempDir::new().unwrap();
        let mut queue = FoliageStagingQueue::default();
        let config = FoliageConfig::default();
        load_generated_tiles_into_queue(dir.path(), &config, &mut queue);
        assert!(queue.batches.is_empty());
    }

    #[test]
    fn test_discover_height_tiles_no_dir() {
        let tiles = discover_height_tiles(None, 0);
        assert!(tiles.is_empty());
    }
}

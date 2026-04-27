//! Background foliage tile streaming and GPU buffer management.
//!
//! Parallels the terrain streaming system but for foliage instances. Handles:
//! - Background tile loading from disk
//! - GPU buffer allocation and memory budgeting
//! - LRU eviction policy
//! - Hot-reload synchronization with generation counter

use crate::foliage::{FoliageInstance, FoliageLodTier};
use bevy::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{
    mpsc::{self, Sender},
    Arc, Mutex,
};

// ---------------------------------------------------------------------------
// View state for foliage LODs
// ---------------------------------------------------------------------------

/// Runtime camera-driven foliage view: updated every frame based on camera distance.
#[derive(Resource, Default, Debug)]
pub struct FoliageViewState {
    /// Last observed camera world position (Y is ignored; foliage is purely XZ-planar).
    pub camera_pos_ws: Vec3,
    /// Active LOD tier based on camera height or distance heuristic.
    /// Some implementations blend between LOD0 and LOD1 in a transition zone.
    pub active_lod: FoliageLodTier,
}

// ---------------------------------------------------------------------------
// Foliage tile key and state
// ---------------------------------------------------------------------------

/// Uniquely identifies one foliage tile in the streaming hierarchy.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FoliageTileKey {
    pub lod: FoliageLodTier,
    pub mip_level: u8,
    pub x: i32,
    pub y: i32,
}

/// Lifecycle state of a single foliage tile.
#[derive(Clone, Debug)]
pub enum FoliageTileState {
    Unloaded,
    Requested,
    LoadedCpu {
        instance_count: u32,
        generation: u64,
    },
    ResidentGpu {
        offset: u32,
        instance_count: u32,
    },
    Evicting,
}

// ---------------------------------------------------------------------------
// CPU-side tile payloads
// ---------------------------------------------------------------------------

/// Raw foliage instance data decoded on a background thread, ready to upload to GPU.
#[derive(Debug, Clone)]
pub struct FoliageTileCpu {
    pub key: FoliageTileKey,
    /// Flat array of instances for this tile.
    pub instances: Vec<FoliageInstance>,
    /// Reload generation this tile was requested for. Tiles with a stale
    /// generation are discarded by `poll_foliage_tile_jobs` after a hot-reload.
    pub generation: u64,
}

impl FoliageTileCpu {
    /// Size in bytes of this tile's instance data.
    pub fn size_bytes(&self) -> usize {
        self.instances.len() * std::mem::size_of::<FoliageInstance>()
    }
}

// ---------------------------------------------------------------------------
// Residency and memory tracking
// ---------------------------------------------------------------------------

/// Tracks which foliage tiles are needed, loaded, and evictable.
/// Separate tracker per LOD tier.
#[derive(Resource, Default)]
pub struct FoliageResidency {
    /// Per-(lod, mip, x, y) lifecycle state.
    pub tiles: HashMap<FoliageTileKey, FoliageTileState>,
    /// Tiles the current frame requires to be resident.
    pub required_now: Vec<FoliageTileKey>,
    /// LRU order (front = oldest). Used for eviction.
    pub lru: VecDeque<FoliageTileKey>,
    /// CPU instance data kept alive after GPU upload so tiles can be re-queried.
    pub resident_cpu: HashMap<FoliageTileKey, FoliageTileCpu>,
    /// Total bytes allocated across all GPU buffers for this LOD tier.
    pub resident_memory_bytes: usize,
    /// Set when memory budget exhausted and eviction removed tiles.
    pub memory_exhausted: bool,
}

impl FoliageResidency {
    /// Mark a tile as recently used (moves it to the back of the LRU).
    pub fn touch(&mut self, key: FoliageTileKey) {
        self.lru.retain(|k| *k != key);
        self.lru.push_back(key);
    }

    /// Evict tiles until `resident_memory_bytes <= budget`.
    pub fn evict_to_budget(&mut self, budget_bytes: usize) {
        self.memory_exhausted = false;
        while self.resident_memory_bytes > budget_bytes {
            // Evict from the front of LRU that are not currently required.
            let candidate = self.lru.iter().position(|k| !self.required_now.contains(k));

            if let Some(idx) = candidate {
                let key = self.lru.remove(idx).unwrap();
                if let Some(tile_cpu) = self.resident_cpu.remove(&key) {
                    self.resident_memory_bytes = self
                        .resident_memory_bytes
                        .saturating_sub(tile_cpu.size_bytes());
                }
                self.tiles.remove(&key);
            } else {
                // All cached tiles are required; cannot evict safely.
                self.memory_exhausted = true;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stream queue
// ---------------------------------------------------------------------------

/// Manages pending foliage tile load requests.
/// One instance per LOD tier.
#[derive(Resource, Default)]
pub struct FoliageStreamQueue {
    /// Tiles whose load has been requested but not yet completed.
    pub pending_requests: Vec<FoliageTileKey>,
    /// Completed CPU payloads waiting to be consumed by extract/prepare systems.
    pub finished_tiles: Vec<FoliageTileCpu>,
    /// Maximum concurrent background tasks (prevents thread spawn explosion).
    pub max_pending_tasks: usize,
    /// Incremented on every hot-reload. Background threads tag their result
    /// with the generation that was current when they were spawned; results
    /// with a stale generation are silently discarded.
    pub reload_generation: u64,
    /// Set to true when hot-reload is triggered. Cleared once GPU sync completes.
    pub pending_gpu_sync: bool,
}

impl FoliageStreamQueue {
    pub fn new(max_pending_tasks: usize) -> Self {
        Self {
            pending_requests: Vec::new(),
            finished_tiles: Vec::new(),
            max_pending_tasks,
            reload_generation: 0,
            pending_gpu_sync: false,
        }
    }

    /// Can we spawn more background tasks?
    pub fn can_spawn_task(&self) -> bool {
        self.pending_requests.len() < self.max_pending_tasks
    }

    /// Add a tile to the pending request queue.
    pub fn request_tile(&mut self, key: FoliageTileKey) {
        if !self.pending_requests.contains(&key) {
            self.pending_requests.push(key);
        }
    }

    /// Remove a tile from the pending request queue (after spawning a background task).
    pub fn mark_requested(&mut self, key: FoliageTileKey) {
        self.pending_requests.retain(|k| *k != key);
    }
}

// ---------------------------------------------------------------------------
// Background tile loading
// ---------------------------------------------------------------------------

/// Receiver wrapped in Arc<Mutex> so it can be a Bevy resource (needs Sync).
#[derive(Resource, Clone)]
pub struct FoliageTileReceiver(pub Arc<Mutex<mpsc::Receiver<FoliageTileCpu>>>);

/// Loads foliage instance data for a single tile synchronously (blocking).
///
/// Reads instances from the binary tile file, or returns empty if file not found.
pub(crate) fn load_tile_data(
    key: FoliageTileKey,
    foliage_root: Option<&std::path::Path>,
) -> FoliageTileCpu {
    use crate::foliage_tiles::deserialize_foliage_tile;

    let instances = foliage_root
        .and_then(|root| {
            let path = foliage_tile_path(root, key);
            std::fs::read(&path)
                .ok()
                .and_then(|bytes| deserialize_foliage_tile(&bytes).ok())
        })
        .unwrap_or_default();

    FoliageTileCpu {
        key,
        instances,
        generation: 0, // set by spawn_background_foliage_job before sending
    }
}

/// Path to a foliage instance tile file.
pub(crate) fn foliage_tile_path(root: &std::path::Path, key: FoliageTileKey) -> PathBuf {
    root.join(format!(
        "LOD{}/L{}/{}_{}.bin",
        key.lod as u8, key.mip_level, key.x, key.y
    ))
}

/// Spawns a background OS thread to load foliage instance data for one tile.
pub fn spawn_background_foliage_job(
    key: FoliageTileKey,
    foliage_root: Option<PathBuf>,
    generation: u64,
    tx: Sender<FoliageTileCpu>,
) {
    std::thread::spawn(move || {
        let mut data = load_tile_data(key, foliage_root.as_deref());
        data.generation = generation;
        let _ = tx.send(data);
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_tile_key_equality() {
        let key1 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 1,
            y: 2,
        };
        let key2 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 1,
            y: 2,
        };
        assert_eq!(key1, key2);

        let key3 = FoliageTileKey {
            lod: FoliageLodTier::Lod1,
            mip_level: 0,
            x: 1,
            y: 2,
        };
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_foliage_tile_cpu_size_bytes() {
        let instances = vec![
            FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0),
            FoliageInstance::new(Vec3::ONE, Quat::IDENTITY, Vec3::ONE, 1),
        ];
        let tile = FoliageTileCpu {
            key: FoliageTileKey {
                lod: FoliageLodTier::Lod0,
                mip_level: 0,
                x: 0,
                y: 0,
            },
            instances,
            generation: 0,
        };
        assert_eq!(
            tile.size_bytes(),
            2 * std::mem::size_of::<FoliageInstance>()
        );
    }

    #[test]
    fn test_foliage_stream_queue_can_spawn() {
        let mut queue = FoliageStreamQueue::new(2);
        assert!(queue.can_spawn_task());

        let key = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 0,
            y: 0,
        };
        queue.request_tile(key);
        queue.request_tile(key); // duplicate, should not add again
        assert_eq!(queue.pending_requests.len(), 1);

        let key2 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 1,
            y: 0,
        };
        queue.request_tile(key2);
        assert_eq!(queue.pending_requests.len(), 2);
        assert!(!queue.can_spawn_task());
    }

    #[test]
    fn test_foliage_residency_lru_touch() {
        let mut residency = FoliageResidency::default();
        let key1 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 0,
            y: 0,
        };
        let key2 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 1,
            y: 0,
        };

        residency.touch(key1);
        residency.touch(key2);
        assert_eq!(residency.lru.len(), 2);

        // Touch key1 again; it should move to the back
        residency.touch(key1);
        assert_eq!(residency.lru.len(), 2);
        assert_eq!(residency.lru.back(), Some(&key1));
        assert_eq!(residency.lru.front(), Some(&key2));
    }

    #[test]
    fn test_foliage_residency_evict_to_budget() {
        let mut residency = FoliageResidency::default();
        let key1 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 0,
            y: 0,
        };
        let key2 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 1,
            y: 0,
        };

        let tile1 = FoliageTileCpu {
            key: key1,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 100],
            generation: 0,
        };
        let tile2 = FoliageTileCpu {
            key: key2,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 200],
            generation: 0,
        };

        let tile1_size = tile1.size_bytes();
        let tile2_size = tile2.size_bytes();

        residency.touch(key1);
        residency.touch(key2);
        residency.resident_cpu.insert(key1, tile1);
        residency.resident_cpu.insert(key2, tile2);
        residency.resident_memory_bytes = tile1_size + tile2_size;

        // Evict to fit budget that is smaller than tile2 alone
        // This should evict both tiles since budget is smaller than either
        residency.evict_to_budget(100); // Much smaller than either tile

        assert!(residency.resident_cpu.is_empty());
        assert_eq!(residency.resident_memory_bytes, 0);
    }

    #[test]
    fn test_foliage_residency_evict_with_required_tiles() {
        let mut residency = FoliageResidency::default();
        let key1 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 0,
            y: 0,
        };
        let key2 = FoliageTileKey {
            lod: FoliageLodTier::Lod0,
            mip_level: 0,
            x: 1,
            y: 0,
        };

        let tile1 = FoliageTileCpu {
            key: key1,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 100],
            generation: 0,
        };
        let tile2 = FoliageTileCpu {
            key: key2,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 200],
            generation: 0,
        };

        let tile1_size = tile1.size_bytes();
        let tile2_size = tile2.size_bytes();

        residency.touch(key1);
        residency.touch(key2);
        residency.required_now.push(key1);
        residency.resident_cpu.insert(key1, tile1);
        residency.resident_cpu.insert(key2, tile2);
        residency.resident_memory_bytes = tile1_size + tile2_size;

        // Try to evict to 0 bytes, but key1 is required. Should stop.
        residency.evict_to_budget(0);

        assert!(residency.memory_exhausted);
        assert!(residency.resident_cpu.contains_key(&key1));
        assert_eq!(residency.resident_memory_bytes, tile1_size);
    }
}

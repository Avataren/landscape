//! Background foliage tile streaming and GPU buffer management.
//!
//! Handles background tile loading from disk, GPU buffer allocation, LRU eviction,
//! and hot-reload synchronization via a generation counter.

use super::{FoliageInstance, FoliageLodTier};
use bevy::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{
    mpsc::{self, Sender},
    Arc, Mutex,
};

/// Runtime camera-driven foliage view: updated every frame based on camera distance.
#[derive(Resource, Default, Debug)]
pub struct FoliageViewState {
    pub camera_pos_ws: Vec3,
    pub active_lod: FoliageLodTier,
}

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
    LoadedCpu { instance_count: u32, generation: u64 },
    ResidentGpu { offset: u32, instance_count: u32 },
    Evicting,
}

/// Raw foliage instance data decoded on a background thread, ready to upload to GPU.
#[derive(Debug, Clone)]
pub struct FoliageTileCpu {
    pub key: FoliageTileKey,
    pub instances: Vec<FoliageInstance>,
    /// Reload generation this tile was requested for; stale tiles are discarded.
    pub generation: u64,
}

impl FoliageTileCpu {
    pub fn size_bytes(&self) -> usize {
        self.instances.len() * std::mem::size_of::<FoliageInstance>()
    }
}

/// Tracks which foliage tiles are needed, loaded, and evictable.
#[derive(Resource, Default)]
pub struct FoliageResidency {
    pub tiles: HashMap<FoliageTileKey, FoliageTileState>,
    pub required_now: Vec<FoliageTileKey>,
    pub lru: VecDeque<FoliageTileKey>,
    pub resident_cpu: HashMap<FoliageTileKey, FoliageTileCpu>,
    pub resident_memory_bytes: usize,
    pub memory_exhausted: bool,
}

impl FoliageResidency {
    pub fn touch(&mut self, key: FoliageTileKey) {
        self.lru.retain(|k| *k != key);
        self.lru.push_back(key);
    }

    pub fn evict_to_budget(&mut self, budget_bytes: usize) {
        self.memory_exhausted = false;
        while self.resident_memory_bytes > budget_bytes {
            let candidate = self.lru.iter().position(|k| !self.required_now.contains(k));
            if let Some(idx) = candidate {
                let key = self.lru.remove(idx).unwrap();
                if let Some(tile_cpu) = self.resident_cpu.remove(&key) {
                    self.resident_memory_bytes =
                        self.resident_memory_bytes.saturating_sub(tile_cpu.size_bytes());
                }
                self.tiles.remove(&key);
            } else {
                self.memory_exhausted = true;
                break;
            }
        }
    }
}

/// Manages pending foliage tile load requests.
#[derive(Resource, Default)]
pub struct FoliageStreamQueue {
    pub pending_requests: Vec<FoliageTileKey>,
    pub finished_tiles: Vec<FoliageTileCpu>,
    pub max_pending_tasks: usize,
    /// Incremented on every hot-reload; stale-generation results are discarded.
    pub reload_generation: u64,
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

    pub fn can_spawn_task(&self) -> bool {
        self.pending_requests.len() < self.max_pending_tasks
    }

    pub fn request_tile(&mut self, key: FoliageTileKey) {
        if !self.pending_requests.contains(&key) {
            self.pending_requests.push(key);
        }
    }

    pub fn mark_requested(&mut self, key: FoliageTileKey) {
        self.pending_requests.retain(|k| *k != key);
    }
}

/// Receiver wrapped in Arc<Mutex> so it can be a Bevy resource (needs Sync).
#[derive(Resource, Clone)]
pub struct FoliageTileReceiver(pub Arc<Mutex<mpsc::Receiver<FoliageTileCpu>>>);

pub(crate) fn load_tile_data(
    key: FoliageTileKey,
    foliage_root: Option<&std::path::Path>,
) -> FoliageTileCpu {
    use super::tiles::deserialize_foliage_tile;

    let instances = foliage_root
        .and_then(|root| {
            let path = foliage_tile_path(root, key);
            std::fs::read(&path)
                .ok()
                .and_then(|bytes| deserialize_foliage_tile(&bytes).ok())
        })
        .unwrap_or_default();

    FoliageTileCpu { key, instances, generation: 0 }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_stream_queue_can_spawn() {
        let mut queue = FoliageStreamQueue::new(2);
        assert!(queue.can_spawn_task());
        let key = FoliageTileKey { lod: FoliageLodTier::Lod0, mip_level: 0, x: 0, y: 0 };
        queue.request_tile(key);
        queue.request_tile(key); // duplicate
        assert_eq!(queue.pending_requests.len(), 1);
        let key2 = FoliageTileKey { lod: FoliageLodTier::Lod0, mip_level: 0, x: 1, y: 0 };
        queue.request_tile(key2);
        assert!(!queue.can_spawn_task());
    }

    #[test]
    fn test_foliage_residency_lru_touch() {
        let mut residency = FoliageResidency::default();
        let key1 = FoliageTileKey { lod: FoliageLodTier::Lod0, mip_level: 0, x: 0, y: 0 };
        let key2 = FoliageTileKey { lod: FoliageLodTier::Lod0, mip_level: 0, x: 1, y: 0 };
        residency.touch(key1);
        residency.touch(key2);
        residency.touch(key1);
        assert_eq!(residency.lru.back(), Some(&key1));
        assert_eq!(residency.lru.front(), Some(&key2));
    }
}

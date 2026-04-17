use bevy::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

/// Runtime camera-driven terrain view: updated every frame before rendering.
#[derive(Resource, Default, Debug)]
pub struct TerrainViewState {
    /// Last observed camera world position.
    pub camera_pos_ws: Vec3,
    /// Snapped clipmap center (integer grid coords) for each LOD level.
    pub clip_centers: Vec<IVec2>,
    /// World-space texel spacing for each LOD level.
    pub level_scales: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Tile key and state
// ---------------------------------------------------------------------------

/// Uniquely identifies one terrain tile in the streaming hierarchy.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
}

/// Lifecycle state of a single terrain tile.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum TileState {
    Unloaded,
    Requested,
    LoadedCpu,
    ResidentGpu { slot: u32 },
    Evicting,
}

// ---------------------------------------------------------------------------
// CPU-side tile payloads
// ---------------------------------------------------------------------------

/// Raw height data decoded on a background thread, ready to upload to GPU.
#[derive(Debug)]
pub struct HeightTileCpu {
    pub key: TileKey,
    /// Row-major f32 heights, [tile_size * tile_size] elements.
    pub data: Vec<f32>,
    /// Row-major RG8Snorm XZ normals, [tile_size * tile_size] elements.
    pub normal_data: Vec<[u8; 2]>,
    pub tile_size: u32,
    /// Reload generation this tile was requested for.  Tiles with a stale
    /// generation are discarded by `poll_tile_stream_jobs` after a hot-reload,
    /// preventing old tile data from merging into the new terrain.
    pub generation: u64,
}

/// Raw material mask data, one byte per channel per texel.
#[derive(Debug)]
#[allow(dead_code)]
pub struct MaterialTileCpu {
    pub key: TileKey,
    /// RGBA material weights, [tile_size * tile_size * 4] elements.
    pub data: Vec<u8>,
    pub tile_size: u32,
}

// ---------------------------------------------------------------------------
// Residency tracker
// ---------------------------------------------------------------------------

/// Tracks which tiles are needed, loaded, and evictable.
#[derive(Resource, Default)]
pub struct TerrainResidency {
    /// Per-key lifecycle state.
    pub tiles: HashMap<TileKey, TileState>,
    /// Tiles the current frame requires to be resident.
    pub required_now: HashSet<TileKey>,
    /// LRU order (front = oldest). Used for eviction.
    pub lru: VecDeque<TileKey>,
    /// Tiles that have been loaded to CPU and are waiting for GPU upload.
    pub pending_upload: Vec<HeightTileCpu>,
    /// CPU pixel data kept alive after GPU upload so tiles can be re-applied
    /// when the clipmap shifts to a new clip-center position.
    pub resident_cpu: HashMap<TileKey, HeightTileCpu>,
    /// Set when eviction removed cached CPU height data and clipmap layers must
    /// be rebuilt from fallback before resident tiles are re-applied.
    pub clipmap_needs_rebuild: bool,
}

impl TerrainResidency {
    /// Mark a tile as recently used (moves it to the back of the LRU).
    pub fn touch(&mut self, key: TileKey) {
        self.lru.retain(|k| *k != key);
        self.lru.push_back(key);
    }

    /// Evict tiles until `tiles.len() <= budget`.
    pub fn evict_to_budget(&mut self, budget: usize) {
        while self.tiles.len() > budget {
            // Evict from the front of LRU that are not currently required.
            let candidate = self.lru.iter().position(|k| !self.required_now.contains(k));

            if let Some(idx) = candidate {
                let key = self.lru.remove(idx).unwrap();
                self.tiles.remove(&key);
                if self.resident_cpu.remove(&key).is_some() {
                    self.clipmap_needs_rebuild = true;
                }
            } else {
                break; // All cached tiles are required; cannot evict safely.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stream queue
// ---------------------------------------------------------------------------

/// Manages pending tile load requests.
#[derive(Resource, Default)]
pub struct TerrainStreamQueue {
    /// Tiles whose load has been requested but not yet completed.
    pub pending_requests: HashSet<TileKey>,
    /// Completed CPU payloads waiting to be consumed.
    #[allow(dead_code)]
    pub finished_heights: Vec<HeightTileCpu>,
    /// Incremented on every hot-reload.  Background threads tag their result
    /// with the generation that was current when they were spawned; results
    /// with a stale generation are silently discarded by `poll_tile_stream_jobs`.
    pub reload_generation: u64,
}

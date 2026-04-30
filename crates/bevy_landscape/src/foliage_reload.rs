//! Foliage hot-reload system.
//!
//! Coordinates foliage reloading with terrain hot-reload. When a ReloadTerrainRequest
//! includes foliage_config, this system:
//! 1. Bumps the generation counter (invalidates in-flight tiles)
//! 2. Clears residency and GPU state
//! 3. Sets GPU sync request (for explicit GPU barrier)
//! 4. Triggers tile re-streaming from new foliage root

use crate::foliage::{FoliageConfig, FoliageSourceDesc};
use crate::foliage_gpu::{FoliageGpuState, FoliageGpuSyncRequest};
use crate::foliage_stream_queue::{FoliageResidency, FoliageStreamQueue};
use crate::terrain::ReloadTerrainRequest;
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Hot-reload coordination
// ---------------------------------------------------------------------------

/// Wrapper for FoliageConfig as a Resource (since Option<T> isn't Resource by itself).
#[derive(Resource, Clone, Debug)]
pub struct FoliageConfigResource(pub Option<FoliageConfig>);

impl Default for FoliageConfigResource {
    fn default() -> Self {
        Self(Some(FoliageConfig::default()))
    }
}

/// Marker resource to track whether foliage is currently loaded.
#[derive(Resource, Default, Debug)]
pub struct FoliageLoadState {
    pub is_loaded: bool,
    pub last_reload_frame: u32,
}

/// System that processes foliage hot-reload requests.
pub fn reload_foliage_system(
    mut events: MessageReader<ReloadTerrainRequest>,
    mut foliage_config: ResMut<FoliageConfigResource>,
    mut foliage_source: ResMut<FoliageSourceDesc>,
    mut stream_queue: ResMut<FoliageStreamQueue>,
    mut residency: ResMut<FoliageResidency>,
    mut gpu_state: ResMut<FoliageGpuState>,
    mut gpu_sync: ResMut<FoliageGpuSyncRequest>,
    mut load_state: ResMut<FoliageLoadState>,
) {
    for event in events.read() {
        // Always sync foliage_root from the terrain source (present on every reload).
        if let Some(root) = &event.source.foliage_root {
            foliage_source.foliage_root = Some(std::path::PathBuf::from(root));
        }

        // Only do the heavy foliage reset when a new FoliageConfig is provided.
        let Some(new_config) = &event.foliage_config else {
            continue;
        };

        foliage_config.0 = Some(new_config.clone());

        // Bump generation counter (invalidates all in-flight tiles)
        stream_queue.reload_generation += 1;

        gpu_sync.needs_sync = true;

        // Clear residency (old tiles are no longer valid)
        residency.tiles.clear();
        residency.lru.clear();
        residency.required_now.clear();
        residency.resident_cpu.clear();
        residency.resident_memory_bytes = 0;

        // Clear GPU state (buffers will be rebuilt as tiles load)
        gpu_state.clear_all();

        // Update load state
        load_state.is_loaded = true;
        load_state.last_reload_frame += 1;

        debug!("Foliage hot-reload: generation bumped, residency cleared, GPU sync requested");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_config_resource_default() {
        let config = FoliageConfigResource::default();
        assert!(config.0.is_some());
    }

    #[test]
    fn test_foliage_config_resource_with_value() {
        let inner = FoliageConfig::default();
        let config = FoliageConfigResource(Some(inner.clone()));
        assert!(config.0.is_some());
    }

    #[test]
    fn test_gpu_sync_request_state_progression() {
        let mut sync = FoliageGpuSyncRequest::default();
        assert!(!sync.needs_sync);
        sync.needs_sync = true;
        assert!(sync.needs_sync);
        sync.needs_sync = false;
        assert!(!sync.needs_sync);
    }

    #[test]
    fn test_foliage_residency_clear_all_fields() {
        let mut residency = FoliageResidency::default();

        // Add some data
        residency.resident_memory_bytes = 1000;
        assert_eq!(residency.resident_memory_bytes, 1000);

        // Clear should reset all fields
        residency.tiles.clear();
        residency.lru.clear();
        residency.required_now.clear();
        residency.resident_cpu.clear();
        residency.resident_memory_bytes = 0;

        assert!(residency.tiles.is_empty());
        assert!(residency.lru.is_empty());
        assert!(residency.required_now.is_empty());
        assert!(residency.resident_cpu.is_empty());
        assert_eq!(residency.resident_memory_bytes, 0);
    }

    #[test]
    fn test_foliage_stream_queue_generation_increment() {
        let mut queue = FoliageStreamQueue::new(16);
        let initial_gen = queue.reload_generation;

        queue.reload_generation += 1;
        assert_eq!(queue.reload_generation, initial_gen + 1);

        queue.reload_generation += 1;
        assert_eq!(queue.reload_generation, initial_gen + 2);
    }

    #[test]
    fn test_foliage_gpu_state_clear_all() {
        let mut gpu_state = FoliageGpuState::default();

        gpu_state.lods[0].set_variant(0, 0, 100);
        gpu_state.lods[1].set_variant(0, 0, 200);
        gpu_state.lods[2].set_variant(0, 0, 300);

        assert_eq!(gpu_state.lods[0].resident_count, 100);
        assert_eq!(gpu_state.lods[1].resident_count, 200);
        assert_eq!(gpu_state.lods[2].resident_count, 300);

        gpu_state.clear_all();

        assert_eq!(gpu_state.lods[0].resident_count, 0);
        assert_eq!(gpu_state.lods[1].resident_count, 0);
        assert_eq!(gpu_state.lods[2].resident_count, 0);
    }
}

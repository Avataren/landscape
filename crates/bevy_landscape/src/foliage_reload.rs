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
#[derive(Resource, Clone, Debug, Default)]
pub struct FoliageConfigResource(pub Option<FoliageConfig>);

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
    mut lod0_stream_queue: ResMut<FoliageStreamQueue>,
    mut lod1_stream_queue: ResMut<FoliageStreamQueue>,
    mut lod2_stream_queue: ResMut<FoliageStreamQueue>,
    mut lod0_residency: ResMut<FoliageResidency>,
    mut lod1_residency: ResMut<FoliageResidency>,
    mut lod2_residency: ResMut<FoliageResidency>,
    mut gpu_state: ResMut<FoliageGpuState>,
    mut gpu_sync: ResMut<FoliageGpuSyncRequest>,
    mut load_state: ResMut<FoliageLoadState>,
) {
    for event in events.read() {
        // Check if this reload includes foliage data
        let Some(new_config) = &event.foliage_config else {
            continue;
        };

        // Update foliage configuration and source
        foliage_config.0 = Some(new_config.clone());
        foliage_source.foliage_root = event
            .source
            .foliage_root
            .as_ref()
            .map(|s| std::path::PathBuf::from(s));

        // Bump generation counter on all LOD tiers (invalidates in-flight tiles)
        lod0_stream_queue.reload_generation += 1;
        lod1_stream_queue.reload_generation += 1;
        lod2_stream_queue.reload_generation += 1;

        // Set GPU sync request (signals render graph to insert barrier)
        gpu_sync.needs_sync = true;
        gpu_sync.requested_frame = 0; // Will be set by render-app

        // Clear residency (old tiles are no longer valid)
        lod0_residency.tiles.clear();
        lod0_residency.lru.clear();
        lod0_residency.required_now.clear();
        lod0_residency.resident_cpu.clear();
        lod0_residency.resident_memory_bytes = 0;

        lod1_residency.tiles.clear();
        lod1_residency.lru.clear();
        lod1_residency.required_now.clear();
        lod1_residency.resident_cpu.clear();
        lod1_residency.resident_memory_bytes = 0;

        lod2_residency.tiles.clear();
        lod2_residency.lru.clear();
        lod2_residency.required_now.clear();
        lod2_residency.resident_cpu.clear();
        lod2_residency.resident_memory_bytes = 0;

        // Clear GPU state (buffers will be rebuilt as tiles load)
        gpu_state.clear_all();

        // Update load state
        load_state.is_loaded = true;
        load_state.last_reload_frame += 1;

        debug!("Foliage hot-reload: generation bumped, residency cleared, GPU sync requested");
    }
}

// ---------------------------------------------------------------------------
// GPU sync completion handler
// ---------------------------------------------------------------------------

/// Mark GPU sync as complete once render graph barrier executes.
/// This should be called from the render-app after the GPU sync barrier.
pub fn mark_gpu_sync_complete(mut gpu_sync: ResMut<FoliageGpuSyncRequest>) {
    if gpu_sync.needs_sync && gpu_sync.completed_frame.is_none() {
        gpu_sync.needs_sync = false;
        debug!("GPU sync complete, foliage buffers safe to reallocate");
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
        assert!(config.0.is_none());
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
        assert!(!sync.needs_sync); // Defaults to false
        assert_eq!(sync.requested_frame, 0);
        assert!(sync.completed_frame.is_none());

        // Simulate request
        sync.needs_sync = true;
        sync.requested_frame = 100;
        assert!(sync.needs_sync);
        assert_eq!(sync.requested_frame, 100);

        // Simulate completion
        sync.completed_frame = Some(101);
        sync.needs_sync = false;
        assert!(!sync.needs_sync);
        assert_eq!(sync.completed_frame, Some(101));
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

        // Add some data to each LOD
        gpu_state.lod0.set_variant(0, 0, 100);
        gpu_state.lod1.set_variant(0, 0, 200);
        gpu_state.lod2.set_variant(0, 0, 300);

        assert_eq!(gpu_state.lod0.resident_count, 100);
        assert_eq!(gpu_state.lod1.resident_count, 200);
        assert_eq!(gpu_state.lod2.resident_count, 300);

        // Clear all
        gpu_state.clear_all();

        assert_eq!(gpu_state.lod0.resident_count, 0);
        assert_eq!(gpu_state.lod1.resident_count, 0);
        assert_eq!(gpu_state.lod2.resident_count, 0);
    }
}

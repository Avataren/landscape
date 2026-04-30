//! Foliage hot-reload coordination.
//!
//! When a ReloadTerrainRequest includes foliage_config, bumps the generation
//! counter, clears residency and GPU state, and triggers tile re-streaming.

use super::{FoliageConfig, FoliageSourceDesc};
use super::gpu::{FoliageGpuState, FoliageGpuSyncRequest};
use super::stream_queue::{FoliageResidency, FoliageStreamQueue};
use crate::terrain::ReloadTerrainRequest;
use bevy::prelude::*;

/// Wrapper for FoliageConfig as a Resource.
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
        if let Some(root) = &event.source.foliage_root {
            foliage_source.foliage_root = Some(std::path::PathBuf::from(root));
        }

        let Some(new_config) = &event.foliage_config else {
            continue;
        };

        foliage_config.0 = Some(new_config.clone());
        stream_queue.reload_generation += 1;
        gpu_sync.needs_sync = true;

        residency.tiles.clear();
        residency.lru.clear();
        residency.required_now.clear();
        residency.resident_cpu.clear();
        residency.resident_memory_bytes = 0;

        gpu_state.clear_all();
        load_state.is_loaded = true;
        load_state.last_reload_frame += 1;

        debug!("Foliage hot-reload: generation bumped, residency cleared, GPU sync requested");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_config_resource_default() {
        let config = FoliageConfigResource::default();
        assert!(config.0.is_some());
    }

    #[test]
    fn test_gpu_sync_request_state_progression() {
        let mut sync = FoliageGpuSyncRequest::default();
        assert!(!sync.needs_sync);
        sync.needs_sync = true;
        assert!(sync.needs_sync);
    }
}

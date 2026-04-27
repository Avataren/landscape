//! Foliage render integration and indirect draw management.
//!
//! Handles GPU resource preparation and indirect draw command generation.
//! Designed for integration with Bevy's render-app (extract/prepare phases).
//!
//! Note: Full render-graph integration deferred until Phase 8+.
//! This module provides the infrastructure and placeholder systems.

use crate::foliage::{FoliageInstance, FoliageLodTier};
use crate::foliage_gpu::{FoliageGpuState, FoliageStagingQueue};
use crate::foliage_reload::FoliageConfigResource;
use crate::foliage_stream_queue::FoliageViewState;
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Indirect draw command metadata
// ---------------------------------------------------------------------------

/// Metadata for rendering a single (LOD, variant) pair.
#[derive(Clone, Copy, Debug)]
pub struct IndirectDrawCommand {
    /// Offset (in instances) within the LOD's shared GPU buffer.
    pub instance_offset: u32,
    /// Number of instances to render.
    pub instance_count: u32,
    /// Which LOD tier this command renders.
    pub lod: FoliageLodTier,
    /// Which variant this command renders.
    pub variant_id: u8,
}

impl IndirectDrawCommand {
    pub fn new(lod: FoliageLodTier, variant_id: u8, offset: u32, count: u32) -> Self {
        Self {
            instance_offset: offset,
            instance_count: count,
            lod,
            variant_id: variant_id.clamp(0, 7),
        }
    }
}

// ---------------------------------------------------------------------------
// Render-app integration markers
// ---------------------------------------------------------------------------

/// System set for foliage render integration (extract/prepare/render phases).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FoliageRenderSystemSet {
    /// Extract foliage data from main world to render world.
    Extract,
    /// Prepare GPU buffers and sync staging data.
    Prepare,
    /// Queue rendering commands.
    Queue,
}

// ---------------------------------------------------------------------------
// Placeholder systems (for Phase 8+ integration)
// ---------------------------------------------------------------------------

/// (Phase 8+) Extract foliage staging data to render world.
///
/// This system will:
/// 1. Copy FoliageStagingQueue to render world
/// 2. Convert CPU FoliageInstance to GPU format
/// 3. Build upload commands for RenderQueue
pub fn extract_foliage_staging(_world: &mut World) {
    // TODO: Phase 8 - Implement render-world extraction
    // This requires access to render context and will be implemented
    // after render-graph infrastructure is in place.
}

/// (Phase 8+) Prepare GPU buffers by uploading staging data.
///
/// This system will:
/// 1. Process staging queue batches
/// 2. Upload via RenderQueue::write_buffer
/// 3. Update FoliageGpuState offsets and counts
/// 4. Clear staging queue after upload
pub fn prepare_foliage_gpu_buffers(
    _gpu_state: ResMut<FoliageGpuState>,
    _staging_queue: ResMut<FoliageStagingQueue>,
) {
    // TODO: Phase 8 - Implement GPU buffer upload
    // This requires RenderContext and will be integrated with prepare phase.
}

/// (Phase 8+) Queue foliage indirect draw commands.
///
/// This system will:
/// 1. Collect visible foliage entities
/// 2. Build indirect draw commands per (LOD, variant)
/// 3. Queue multi-draw calls to render graph
pub fn queue_foliage_indirect_draws() {
    // TODO: Phase 8 - Implement render queueing
    // This requires RenderContext and will be integrated with queue phase.
}

// ---------------------------------------------------------------------------
// LOD visibility culling helpers
// ---------------------------------------------------------------------------

/// Determine which LOD tier should render based on camera distance.
pub fn get_active_lod(
    camera_pos: Vec3,
    config_lod0_distance: f32,
    config_lod1_distance: f32,
) -> FoliageLodTier {
    // Note: We check XZ distance only (ignore Y)
    let camera_xz = Vec2::new(camera_pos.x, camera_pos.z);
    let distance_to_origin = camera_xz.length();

    if distance_to_origin < config_lod0_distance {
        FoliageLodTier::Lod0
    } else if distance_to_origin < config_lod1_distance {
        FoliageLodTier::Lod1
    } else {
        FoliageLodTier::Lod2
    }
}

/// Update foliage view state based on camera position.
pub fn update_foliage_view_state(
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    config: Option<Res<FoliageConfigResource>>,
    mut view_state: ResMut<FoliageViewState>,
) {
    let Some(config_res) = config else {
        return;
    };

    let Some(config) = &config_res.0 else {
        return;
    };

    for camera_transform in camera_query.iter() {
        let camera_pos = camera_transform.translation();
        view_state.camera_pos_ws = camera_pos;
        view_state.active_lod =
            get_active_lod(camera_pos, config.lod0_distance, config.lod1_distance);
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Convert CPU foliage instances to GPU format with offset.
///
/// The GPU format is the same (48 bytes per instance), but we need to track
/// which buffer offset to write to.
pub fn prepare_instances_for_gpu(
    instances: &[FoliageInstance],
    buffer_offset: u32,
) -> Vec<(u32, FoliageInstance)> {
    instances
        .iter()
        .enumerate()
        .map(|(idx, inst)| (buffer_offset + idx as u32, *inst))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indirect_draw_command_new() {
        let cmd = IndirectDrawCommand::new(FoliageLodTier::Lod0, 3, 100, 500);
        assert_eq!(cmd.lod, FoliageLodTier::Lod0);
        assert_eq!(cmd.variant_id, 3);
        assert_eq!(cmd.instance_offset, 100);
        assert_eq!(cmd.instance_count, 500);
    }

    #[test]
    fn test_indirect_draw_command_clamps_variant() {
        let cmd = IndirectDrawCommand::new(FoliageLodTier::Lod1, 255, 0, 100);
        assert_eq!(cmd.variant_id, 7); // Clamped to 0-7
    }

    #[test]
    fn test_get_active_lod_origin() {
        let pos = Vec3::ZERO;
        let lod = get_active_lod(pos, 50.0, 200.0);
        assert_eq!(lod, FoliageLodTier::Lod0);
    }

    #[test]
    fn test_get_active_lod_near() {
        // 25 meters away: within LOD0 distance (50m)
        let pos = Vec3::new(25.0, 0.0, 0.0);
        let lod = get_active_lod(pos, 50.0, 200.0);
        assert_eq!(lod, FoliageLodTier::Lod0);
    }

    #[test]
    fn test_get_active_lod_mid() {
        // 100 meters away: between LOD0 (50m) and LOD1 (200m)
        let pos = Vec3::new(100.0, 0.0, 0.0);
        let lod = get_active_lod(pos, 50.0, 200.0);
        assert_eq!(lod, FoliageLodTier::Lod1);
    }

    #[test]
    fn test_get_active_lod_far() {
        // 300 meters away: beyond LOD1 distance (200m)
        let pos = Vec3::new(300.0, 0.0, 0.0);
        let lod = get_active_lod(pos, 50.0, 200.0);
        assert_eq!(lod, FoliageLodTier::Lod2);
    }

    #[test]
    fn test_get_active_lod_ignores_y() {
        // Distance is 100m in XZ; Y should be ignored
        let pos_low = Vec3::new(100.0, 0.0, 0.0);
        let pos_high = Vec3::new(100.0, 1000.0, 0.0);
        let lod_low = get_active_lod(pos_low, 50.0, 200.0);
        let lod_high = get_active_lod(pos_high, 50.0, 200.0);
        assert_eq!(lod_low, FoliageLodTier::Lod1);
        assert_eq!(lod_high, FoliageLodTier::Lod1);
    }

    #[test]
    fn test_prepare_instances_for_gpu() {
        let instances = vec![
            FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0),
            FoliageInstance::new(Vec3::ONE, Quat::IDENTITY, Vec3::ONE, 1),
        ];

        let prepared = prepare_instances_for_gpu(&instances, 100);
        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared[0].0, 100); // First offset
        assert_eq!(prepared[1].0, 101); // Second offset
    }

    #[test]
    fn test_prepare_instances_for_gpu_with_non_zero_offset() {
        let instances = vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 5];
        let prepared = prepare_instances_for_gpu(&instances, 1000);

        assert_eq!(prepared.len(), 5);
        for (idx, (offset, _)) in prepared.iter().enumerate() {
            assert_eq!(*offset, 1000 + idx as u32);
        }
    }
}

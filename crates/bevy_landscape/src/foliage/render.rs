//! Foliage render integration and indirect draw management.
//!
//! Note: Full render-graph integration (extract/prepare/queue) is deferred to
//! Phase 8+. The three placeholder systems below are intentionally NOT scheduled
//! — they exist as scaffolding for when RenderContext infrastructure is in place.
//! Do not add them to a SystemSet until they are implemented.

use super::{FoliageInstance, FoliageLodTier};
use super::gpu::{FoliageGpuState, FoliageStagingQueue};
use super::reload::FoliageConfigResource;
use super::stream_queue::FoliageViewState;
use bevy::prelude::*;

/// Metadata for rendering a single (LOD, variant) pair.
#[derive(Clone, Copy, Debug)]
pub struct IndirectDrawCommand {
    pub instance_offset: u32,
    pub instance_count: u32,
    pub lod: FoliageLodTier,
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

/// System set for foliage render integration (extract/prepare/render phases).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FoliageRenderSystemSet {
    Extract,
    Prepare,
    Queue,
}

// ---------------------------------------------------------------------------
// Phase 8+ placeholder systems — NOT scheduled, do not add to any SystemSet.
// ---------------------------------------------------------------------------

pub fn extract_foliage_staging(_world: &mut World) {
    // TODO: Phase 8 — implement render-world extraction
}

pub fn prepare_foliage_gpu_buffers(
    _gpu_state: ResMut<FoliageGpuState>,
    _staging_queue: ResMut<FoliageStagingQueue>,
) {
    // TODO: Phase 8 — implement GPU buffer upload via RenderQueue::write_buffer
}

pub fn queue_foliage_indirect_draws() {
    // TODO: Phase 8 — implement indirect multi-draw queueing
}

// ---------------------------------------------------------------------------
// Active systems
// ---------------------------------------------------------------------------

/// Determine which LOD tier should render based on camera distance (XZ only).
pub fn get_active_lod(
    camera_pos: Vec3,
    config_lod0_distance: f32,
    config_lod1_distance: f32,
) -> FoliageLodTier {
    let distance = Vec2::new(camera_pos.x, camera_pos.z).length();
    if distance < config_lod0_distance {
        FoliageLodTier::Lod0
    } else if distance < config_lod1_distance {
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
    let Some(config_res) = config else { return };
    let Some(config) = &config_res.0 else { return };
    for camera_transform in camera_query.iter() {
        let camera_pos = camera_transform.translation();
        view_state.camera_pos_ws = camera_pos;
        view_state.active_lod =
            get_active_lod(camera_pos, config.lod0_distance, config.lod1_distance);
    }
}

/// Convert CPU foliage instances to (buffer_offset, instance) pairs.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indirect_draw_command_clamps_variant() {
        let cmd = IndirectDrawCommand::new(FoliageLodTier::Lod1, 255, 0, 100);
        assert_eq!(cmd.variant_id, 7);
    }

    #[test]
    fn test_get_active_lod_tiers() {
        assert_eq!(get_active_lod(Vec3::ZERO, 50.0, 200.0), FoliageLodTier::Lod0);
        assert_eq!(get_active_lod(Vec3::new(100.0, 0.0, 0.0), 50.0, 200.0), FoliageLodTier::Lod1);
        assert_eq!(get_active_lod(Vec3::new(300.0, 0.0, 0.0), 50.0, 200.0), FoliageLodTier::Lod2);
    }

    #[test]
    fn test_get_active_lod_ignores_y() {
        let lod_low = get_active_lod(Vec3::new(100.0, 0.0, 0.0), 50.0, 200.0);
        let lod_high = get_active_lod(Vec3::new(100.0, 1000.0, 0.0), 50.0, 200.0);
        assert_eq!(lod_low, lod_high);
    }
}

//! GPU buffer management for foliage rendering.
//!
//! Handles allocation, uploading, and lifecycle of GPU buffers for foliage instance data.
//! Uses Bevy's RenderApp to manage GPU resources across hot-reloads.
//!
//! # Architecture
//!
//! Per-LOD tier structure:
//! - One shared GPU instance buffer containing all variants
//! - Separate indirect draw command buffer per variant
//! - Offset tracking: (variant_id) → (offset, count)
//!
//! Example for LOD0 with 8 variants:
//! ```text
//! GPU Buffer Layout:
//! [Var0: offset=0, count=1000]
//! [Var1: offset=1000, count=1500]
//! [Var2: offset=2500, count=900]
//! ...
//! [Var7: offset=?, count=...]
//! ```

use crate::foliage::{FoliageInstance, FoliageLodTier};
use bevy::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GPU buffer handles and metadata
// ---------------------------------------------------------------------------

/// Metadata for a foliage variant's GPU buffer region.
#[derive(Clone, Copy, Debug)]
pub struct FoliageVariantGpuMeta {
    /// Offset (in instances) within the LOD's shared buffer.
    pub offset: u32,
    /// Number of instances for this variant.
    pub instance_count: u32,
}

/// Per-LOD GPU resource state.
#[derive(Clone, Debug, Default)]
pub struct FoliageLodGpuState {
    /// Which variants have data and where.
    pub variant_offsets: HashMap<u8, FoliageVariantGpuMeta>,
    /// Total capacity (in instances) of the LOD's shared buffer.
    pub gpu_capacity: u32,
    /// Total resident instances.
    pub resident_count: u32,
}

impl FoliageLodGpuState {
    /// Add or update a variant's GPU metadata.
    pub fn set_variant(&mut self, variant_id: u8, offset: u32, count: u32) {
        // If variant already exists, subtract the old count
        if let Some(old_meta) = self.variant_offsets.get(&variant_id) {
            self.resident_count = self.resident_count.saturating_sub(old_meta.instance_count);
        }
        // Insert the new variant metadata
        self.variant_offsets.insert(
            variant_id,
            FoliageVariantGpuMeta {
                offset,
                instance_count: count,
            },
        );
        // Add the new count
        self.resident_count += count;
    }

    /// Clear all variant data (for hot-reload).
    pub fn clear(&mut self) {
        self.variant_offsets.clear();
        self.resident_count = 0;
        self.gpu_capacity = 0;
    }
}

/// Global GPU state tracking for all LODs.
#[derive(Resource, Default, Debug)]
pub struct FoliageGpuState {
    pub lods: [FoliageLodGpuState; 3],
}

impl FoliageGpuState {
    /// Get mutable LOD state by tier.
    pub fn get_lod_mut(&mut self, lod: FoliageLodTier) -> &mut FoliageLodGpuState {
        &mut self.lods[lod as usize]
    }

    /// Get LOD state by tier (read-only).
    pub fn get_lod(&self, lod: FoliageLodTier) -> &FoliageLodGpuState {
        &self.lods[lod as usize]
    }

    /// Clear all GPU state (for hot-reload).
    pub fn clear_all(&mut self) {
        for lod in &mut self.lods {
            lod.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// GPU buffer allocation strategy
// ---------------------------------------------------------------------------

/// Strategy for allocating GPU buffer space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AllocationStrategy {
    /// Pre-allocate a fixed amount upfront (simple, predictable).
    Fixed { capacity_per_lod: u32 },
    /// Dynamically grow buffer as tiles load (complex, wasteful reallocation).
    Dynamic {
        initial_capacity: u32,
        growth_factor: f32,
    },
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        // 160k instances per LOD = ~7.7 MB per tier at 48 bytes per instance
        Self::Fixed {
            capacity_per_lod: 160_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Staging buffer for CPU→GPU transfer
// ---------------------------------------------------------------------------

/// Batch of instances to upload to GPU in one command.
#[derive(Clone, Debug)]
pub struct FoliageStagingBatch {
    pub lod: FoliageLodTier,
    pub variant_id: u8,
    pub offset: u32,
    pub instances: Vec<FoliageInstance>,
}

impl FoliageStagingBatch {
    pub fn size_bytes(&self) -> usize {
        self.instances.len() * std::mem::size_of::<FoliageInstance>()
    }
}

/// Queue of staging batches waiting to be uploaded to GPU.
#[derive(Resource, Default)]
pub struct FoliageStagingQueue {
    pub batches: Vec<FoliageStagingBatch>,
}

impl FoliageStagingQueue {
    pub fn push(&mut self, batch: FoliageStagingBatch) {
        self.batches.push(batch);
    }

    pub fn clear(&mut self) {
        self.batches.clear();
    }

    pub fn total_size_bytes(&self) -> usize {
        self.batches.iter().map(|b| b.size_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// GPU sync marker for hot-reload
// ---------------------------------------------------------------------------

/// Set to true during a hot-reload to signal that GPU foliage buffers need clearing.
#[derive(Resource, Default, Debug)]
pub struct FoliageGpuSyncRequest {
    pub needs_sync: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_lod_gpu_state_set_variant() {
        let mut state = FoliageLodGpuState::default();
        state.set_variant(0, 0, 100);
        assert_eq!(state.resident_count, 100);
        assert_eq!(state.variant_offsets[&0].offset, 0);
        assert_eq!(state.variant_offsets[&0].instance_count, 100);

        state.set_variant(1, 100, 200);
        assert_eq!(state.resident_count, 300);
        assert_eq!(state.variant_offsets[&1].offset, 100);
        assert_eq!(state.variant_offsets[&1].instance_count, 200);
    }

    #[test]
    fn test_foliage_lod_gpu_state_update_variant() {
        let mut state = FoliageLodGpuState::default();
        state.set_variant(0, 0, 100);
        assert_eq!(state.resident_count, 100);

        state.set_variant(0, 0, 150);
        assert_eq!(state.resident_count, 150);
    }

    #[test]
    fn test_foliage_lod_gpu_state_clear() {
        let mut state = FoliageLodGpuState::default();
        state.set_variant(0, 0, 100);
        state.set_variant(1, 100, 200);
        assert_eq!(state.resident_count, 300);

        state.clear();
        assert_eq!(state.resident_count, 0);
        assert!(state.variant_offsets.is_empty());
        assert_eq!(state.gpu_capacity, 0);
    }

    #[test]
    fn test_foliage_gpu_state_get_lod() {
        let mut gpu_state = FoliageGpuState::default();
        gpu_state
            .get_lod_mut(FoliageLodTier::Lod0)
            .set_variant(0, 0, 100);
        gpu_state
            .get_lod_mut(FoliageLodTier::Lod1)
            .set_variant(0, 0, 200);

        assert_eq!(gpu_state.get_lod(FoliageLodTier::Lod0).resident_count, 100);
        assert_eq!(gpu_state.get_lod(FoliageLodTier::Lod1).resident_count, 200);
    }

    #[test]
    fn test_foliage_gpu_state_clear_all() {
        let mut gpu_state = FoliageGpuState::default();
        gpu_state
            .get_lod_mut(FoliageLodTier::Lod0)
            .set_variant(0, 0, 100);
        gpu_state
            .get_lod_mut(FoliageLodTier::Lod1)
            .set_variant(0, 0, 200);

        gpu_state.clear_all();
        assert_eq!(gpu_state.lods[0].resident_count, 0);
        assert_eq!(gpu_state.lods[1].resident_count, 0);
        assert_eq!(gpu_state.lods[2].resident_count, 0);
    }

    #[test]
    fn test_foliage_staging_batch_size_bytes() {
        let batch = FoliageStagingBatch {
            lod: FoliageLodTier::Lod0,
            variant_id: 0,
            offset: 0,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 10],
        };
        assert_eq!(
            batch.size_bytes(),
            10 * std::mem::size_of::<FoliageInstance>()
        );
    }

    #[test]
    fn test_foliage_staging_queue_push_and_clear() {
        let mut queue = FoliageStagingQueue::default();
        let batch = FoliageStagingBatch {
            lod: FoliageLodTier::Lod0,
            variant_id: 0,
            offset: 0,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 10],
        };
        queue.push(batch);
        assert_eq!(queue.batches.len(), 1);

        queue.clear();
        assert!(queue.batches.is_empty());
    }

    #[test]
    fn test_foliage_staging_queue_total_size() {
        let mut queue = FoliageStagingQueue::default();
        let batch1 = FoliageStagingBatch {
            lod: FoliageLodTier::Lod0,
            variant_id: 0,
            offset: 0,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 10],
        };
        let batch2 = FoliageStagingBatch {
            lod: FoliageLodTier::Lod0,
            variant_id: 1,
            offset: 10,
            instances: vec![FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 0); 20],
        };
        queue.push(batch1);
        queue.push(batch2);

        let expected_size = (10 + 20) * std::mem::size_of::<FoliageInstance>();
        assert_eq!(queue.total_size_bytes(), expected_size);
    }

    #[test]
    fn test_allocation_strategy_default() {
        let strategy = AllocationStrategy::default();
        assert_eq!(
            strategy,
            AllocationStrategy::Fixed {
                capacity_per_lod: 160_000
            }
        );
    }

    #[test]
    fn test_foliage_gpu_sync_request() {
        let sync = FoliageGpuSyncRequest { needs_sync: true };
        assert!(sync.needs_sync);
        let cleared = FoliageGpuSyncRequest::default();
        assert!(!cleared.needs_sync);
    }
}

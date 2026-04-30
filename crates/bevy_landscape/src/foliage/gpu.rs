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

use super::{FoliageInstance, FoliageLodTier};
use bevy::prelude::*;
use std::collections::HashMap;

/// Metadata for a foliage variant's GPU buffer region.
#[derive(Clone, Copy, Debug)]
pub struct FoliageVariantGpuMeta {
    pub offset: u32,
    pub instance_count: u32,
}

/// Per-LOD GPU resource state.
#[derive(Clone, Debug, Default)]
pub struct FoliageLodGpuState {
    pub variant_offsets: HashMap<u8, FoliageVariantGpuMeta>,
    pub gpu_capacity: u32,
    pub resident_count: u32,
}

impl FoliageLodGpuState {
    pub fn set_variant(&mut self, variant_id: u8, offset: u32, count: u32) {
        if let Some(old_meta) = self.variant_offsets.get(&variant_id) {
            self.resident_count = self.resident_count.saturating_sub(old_meta.instance_count);
        }
        self.variant_offsets.insert(
            variant_id,
            FoliageVariantGpuMeta { offset, instance_count: count },
        );
        self.resident_count += count;
    }

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
    pub fn get_lod_mut(&mut self, lod: FoliageLodTier) -> &mut FoliageLodGpuState {
        &mut self.lods[lod as usize]
    }

    pub fn get_lod(&self, lod: FoliageLodTier) -> &FoliageLodGpuState {
        &self.lods[lod as usize]
    }

    pub fn clear_all(&mut self) {
        for lod in &mut self.lods {
            lod.clear();
        }
    }
}

/// Strategy for allocating GPU buffer space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AllocationStrategy {
    Fixed { capacity_per_lod: u32 },
    Dynamic { initial_capacity: u32, growth_factor: f32 },
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::Fixed { capacity_per_lod: 160_000 }
    }
}

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

/// Set to true during a hot-reload to signal that GPU foliage buffers need clearing.
#[derive(Resource, Default, Debug)]
pub struct FoliageGpuSyncRequest {
    pub needs_sync: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_lod_gpu_state_set_variant() {
        let mut state = FoliageLodGpuState::default();
        state.set_variant(0, 0, 100);
        assert_eq!(state.resident_count, 100);

        state.set_variant(1, 100, 200);
        assert_eq!(state.resident_count, 300);
    }

    #[test]
    fn test_foliage_lod_gpu_state_update_variant() {
        let mut state = FoliageLodGpuState::default();
        state.set_variant(0, 0, 100);
        state.set_variant(0, 0, 150);
        assert_eq!(state.resident_count, 150);
    }

    #[test]
    fn test_foliage_lod_gpu_state_clear() {
        let mut state = FoliageLodGpuState::default();
        state.set_variant(0, 0, 100);
        state.set_variant(1, 100, 200);
        state.clear();
        assert_eq!(state.resident_count, 0);
        assert!(state.variant_offsets.is_empty());
    }

    #[test]
    fn test_foliage_gpu_state_get_lod() {
        let mut gpu_state = FoliageGpuState::default();
        gpu_state.get_lod_mut(FoliageLodTier::Lod0).set_variant(0, 0, 100);
        gpu_state.get_lod_mut(FoliageLodTier::Lod1).set_variant(0, 0, 200);
        assert_eq!(gpu_state.get_lod(FoliageLodTier::Lod0).resident_count, 100);
        assert_eq!(gpu_state.get_lod(FoliageLodTier::Lod1).resident_count, 200);
    }

    #[test]
    fn test_foliage_gpu_state_clear_all() {
        let mut gpu_state = FoliageGpuState::default();
        gpu_state.get_lod_mut(FoliageLodTier::Lod0).set_variant(0, 0, 100);
        gpu_state.get_lod_mut(FoliageLodTier::Lod1).set_variant(0, 0, 200);
        gpu_state.clear_all();
        assert_eq!(gpu_state.lods[0].resident_count, 0);
        assert_eq!(gpu_state.lods[1].resident_count, 0);
    }

    #[test]
    fn test_allocation_strategy_default() {
        assert_eq!(
            AllocationStrategy::default(),
            AllocationStrategy::Fixed { capacity_per_lod: 160_000 }
        );
    }
}

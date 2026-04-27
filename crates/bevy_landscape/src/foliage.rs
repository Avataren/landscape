//! Foliage instancing system — GPU-instanced grass, trees, rocks, etc.
//!
//! This module provides procedurally-generated and painted foliage placement on terrain.
//! Foliage is rendered as a large GPU instance buffer with indirect multi-draw.
//!
//! # Data Flow
//!
//! 1. User paints foliage density onto a splatmap (stored in `foliage_root/painted/`)
//! 2. Editor computes procedural density mask from terrain (stored in `foliage_root/procedural_masks/`)
//! 3. Blend: `final_density = painted × procedural_mask / 255` (per cell)
//! 4. Generate instances: iterate tiles, spawn N instances per cell
//! 5. Assign variant IDs (0-7) randomly
//! 6. Bake instances per LOD tier into `foliage_root/LOD{n}/L{level}/{tx}_{ty}.bin`
//! 7. At runtime: load instance buffers, render via GPU indirect multi-draw

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for foliage generation and rendering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoliageConfig {
    /// LOD0 distance (metres). Instances within this range use full density.
    pub lod0_distance: f32,
    /// LOD1 distance (metres). Instances within this range use LOD1 density.
    pub lod1_distance: f32,
    /// LOD2 distance (metres). Instances beyond this range use LOD2 density.
    pub lod2_distance: f32,
    /// Density multiplier for LOD0 (0..1, typically 1.0 = full density).
    pub lod0_density: f32,
    /// Density multiplier for LOD1 (0..1, typically 0.5 = half density).
    pub lod1_density: f32,
    /// Density multiplier for LOD2 (0..1, typically 0.1 = 10% density).
    pub lod2_density: f32,
    /// Number of instances to spawn per cell at full density (before LOD reduction).
    pub instances_per_cell: u32,
    /// Random seed for instance placement and variant selection.
    pub random_seed: u64,
    /// Slope threshold (0..1): steeper slopes reduce grass density.
    pub slope_threshold: f32,
    /// Altitude band: only spawn grass in this Y range (world units).
    pub altitude_min: f32,
    pub altitude_max: f32,
    /// Distance from water (metres): grass density falls off near water.
    pub water_proximity_falloff: f32,
}

impl Default for FoliageConfig {
    fn default() -> Self {
        Self {
            lod0_distance: 50.0,
            lod1_distance: 200.0,
            lod2_distance: 1000.0,
            lod0_density: 1.0,
            lod1_density: 0.5,
            lod2_density: 0.1,
            instances_per_cell: 100,
            random_seed: 42,
            slope_threshold: 0.8, // steeper than this = no grass
            altitude_min: -8192.0,
            altitude_max: 8192.0,
            water_proximity_falloff: 50.0,
        }
    }
}

/// Source descriptor for foliage data.
#[derive(Resource, Clone, Debug, Default)]
pub struct FoliageSourceDesc {
    /// Root path for foliage tiles: painted splatmap, procedural masks, and baked instances.
    /// Subdirectory layout:
    /// - `painted/L{n}/{tx}_{ty}.bin` — painted density (R8, 0-255)
    /// - `procedural_masks/L{n}/{tx}_{ty}.bin` — procedural density (R8, 0-255)
    /// - `LOD{lod_idx}/L{n}/{tx}_{ty}.bin` — baked instances for LOD tier
    pub foliage_root: Option<PathBuf>,
}

/// A single foliage instance: position, rotation, scale, and variant ID.
///
/// Memory layout (48 bytes per instance):
/// - position: Vec3 (12 bytes)
/// - rotation: Quat (16 bytes)
/// - scale: Vec3 (12 bytes)
/// - variant_id: u32 (4 bytes, one of 0-7)
/// - padding: u32 (4 bytes, for alignment)
#[derive(Clone, Copy, Debug)]
pub struct FoliageInstance {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub variant_id: u32,
}

impl FoliageInstance {
    /// Create a new foliage instance.
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3, variant_id: u32) -> Self {
        Self {
            position,
            rotation,
            scale,
            variant_id: variant_id % 8, // Clamp to valid variant range
        }
    }

    /// Serialize to binary format (48 bytes).
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];

        // Position (Vec3, 12 bytes, bytes 0-11)
        bytes[0..4].copy_from_slice(&self.position.x.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.position.y.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.position.z.to_le_bytes());

        // Rotation (Quat, 16 bytes, bytes 12-27)
        bytes[12..16].copy_from_slice(&self.rotation.x.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.rotation.y.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.rotation.z.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.rotation.w.to_le_bytes());

        // Scale (Vec3, 12 bytes, bytes 28-39)
        bytes[28..32].copy_from_slice(&self.scale.x.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.scale.y.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.scale.z.to_le_bytes());

        // Variant ID (u32, 4 bytes, bytes 40-43)
        bytes[40..44].copy_from_slice(&self.variant_id.to_le_bytes());

        // Padding (u32, 4 bytes, bytes 44-47) — set to 0
        // (already initialized to 0)

        bytes
    }

    /// Deserialize from binary format (48 bytes).
    pub fn from_bytes(bytes: &[u8; 48]) -> Self {
        let position = Vec3::new(
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[0..4]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[4..8]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[8..12]).unwrap()),
        );

        let rotation = Quat::from_xyzw(
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[12..16]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[16..20]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[20..24]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[24..28]).unwrap()),
        );

        let scale = Vec3::new(
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[28..32]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[32..36]).unwrap()),
            f32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[36..40]).unwrap()),
        );

        let variant_id =
            u32::from_le_bytes(*<&[u8; 4]>::try_from(&bytes[40..44]).unwrap()) % 8;

        Self {
            position,
            rotation,
            scale,
            variant_id,
        }
    }
}

/// LOD tier for foliage instancing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoliageLodTier {
    /// LOD0: 0 metres to `FoliageConfig::lod0_distance`
    Lod0 = 0,
    /// LOD1: `lod0_distance` to `lod1_distance`
    Lod1 = 1,
    /// LOD2: `lod1_distance` to `lod2_distance` (or beyond)
    Lod2 = 2,
}

impl FoliageLodTier {
    /// All LOD tiers.
    pub fn all() -> &'static [FoliageLodTier] {
        &[FoliageLodTier::Lod0, FoliageLodTier::Lod1, FoliageLodTier::Lod2]
    }

    /// Get the directory suffix for this LOD tier (e.g., "LOD0", "LOD1", "LOD2").
    pub fn dir_suffix(&self) -> &'static str {
        match self {
            FoliageLodTier::Lod0 => "LOD0",
            FoliageLodTier::Lod1 => "LOD1",
            FoliageLodTier::Lod2 => "LOD2",
        }
    }
}

/// Tile path for foliage instances at a specific LOD tier and mip level.
pub fn foliage_tile_path(
    foliage_root: &std::path::Path,
    lod_tier: FoliageLodTier,
    mip_level: u8,
    tx: i32,
    ty: i32,
) -> PathBuf {
    foliage_root.join(format!(
        "{}/L{}/{}_{}",
        lod_tier.dir_suffix(),
        mip_level,
        tx,
        ty
    ))
}

/// Painted splatmap tile path (R8 density, 0-255).
pub fn painted_splatmap_path(
    foliage_root: &std::path::Path,
    mip_level: u8,
    tx: i32,
    ty: i32,
) -> PathBuf {
    foliage_root.join(format!("painted/L{}/{:+}_{:+}.bin", mip_level, tx, ty))
}

/// Procedural density mask tile path (R8 density, 0-255).
pub fn procedural_mask_path(
    foliage_root: &std::path::Path,
    mip_level: u8,
    tx: i32,
    ty: i32,
) -> PathBuf {
    foliage_root.join(format!(
        "procedural_masks/L{}/{:+}_{:+}.bin",
        mip_level, tx, ty
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_instance_serialization() {
        let inst = FoliageInstance::new(
            Vec3::new(1.5, 2.5, 3.5),
            Quat::IDENTITY,
            Vec3::splat(1.0),
            5,
        );

        let bytes = inst.to_bytes();
        let restored = FoliageInstance::from_bytes(&bytes);

        assert!(restored.position.abs_diff_eq(inst.position, 1e-6));
        assert!(restored.rotation.abs_diff_eq(inst.rotation, 1e-6));
        assert!(restored.scale.abs_diff_eq(inst.scale, 1e-6));
        assert_eq!(restored.variant_id, 5);
    }

    #[test]
    fn test_foliage_instance_variant_clamping() {
        let inst = FoliageInstance::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, 255);
        assert_eq!(inst.variant_id, 255 % 8);
    }

    #[test]
    fn test_lod_tier_all() {
        assert_eq!(FoliageLodTier::all().len(), 3);
    }

    #[test]
    fn test_path_construction() {
        let root = PathBuf::from("/data/foliage");
        let path = foliage_tile_path(&root, FoliageLodTier::Lod0, 2, 5, -3);
        assert!(path.to_str().unwrap().contains("LOD0"));
        assert!(path.to_str().unwrap().contains("L2"));
    }
}

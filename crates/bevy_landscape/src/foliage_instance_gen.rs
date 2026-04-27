//! Foliage instance generation pipeline.
//!
//! Orchestrates the full pipeline:
//! 1. Load painted splatmap + procedural mask for each tile
//! 2. Blend: final_density = painted × procedural / 255
//! 3. Spawn instances per cell based on final density
//! 4. Assign positions (jittered), rotations, scales, and variant IDs
//! 5. Bake instances per LOD tier
//! 6. Write to tile files

use crate::foliage::{FoliageConfig, FoliageInstance, FoliageLodTier};
use crate::foliage_generation::{blend_density_masks, write_mask_tile};
use crate::foliage_tiles::FoliageTileWriter;
use crate::painted_splatmap::PaintedSplatmapManager;
use std::path::Path;

/// Random number generator (simple LCG for deterministic placement).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        // Linear congruential generator
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 32) as f32) / 4294967296.0
    }

    fn next_u32(&mut self) -> u32 {
        (self.next() * 4294967296.0) as u32
    }
}

/// Generate foliage instances for a single tile at a specific LOD tier.
///
/// # Arguments
///
/// * `tile_size` - Tile resolution (pixels per axis, typically 256)
/// * `mip_level` - Mip level in the hierarchy
/// * `painted_splatmap` - Painted density per cell (0-255)
/// * `procedural_mask` - Procedural density per cell (0-255)
/// * `world_scale` - Metres per texel at mip level 0
/// * `config` - Foliage configuration
/// * `lod_tier` - Which LOD tier to generate for
/// * `cell_world_size` - World size of one cell (texel) at this mip level
///
/// Returns a Vec of FoliageInstance for this LOD tier.
pub fn generate_tile_instances(
    tile_size: u32,
    mip_level: u8,
    painted_splatmap: &[u8],
    procedural_mask: &[u8],
    world_scale: f32,
    config: &FoliageConfig,
    lod_tier: FoliageLodTier,
    tile_x: i32,
    tile_y: i32,
) -> Vec<FoliageInstance> {
    let tile_size = tile_size as usize;
    let cell_world_size = world_scale * (1u32 << mip_level) as f32;

    // Compute LOD density multiplier
    let lod_multiplier = match lod_tier {
        FoliageLodTier::Lod0 => config.lod0_density,
        FoliageLodTier::Lod1 => config.lod1_density,
        FoliageLodTier::Lod2 => config.lod2_density,
    };

    let mut instances = Vec::new();
    let mut rng = SimpleRng::new(config.random_seed ^ ((mip_level as u64) << 32));

    for y in 0..tile_size {
        for x in 0..tile_size {
            let idx = y * tile_size + x;

            // Blend densities: painted × procedural / 255
            let painted = painted_splatmap[idx] as u32;
            let procedural = procedural_mask[idx] as u32;
            let blended = (painted * procedural) / 255;
            let final_density = ((blended as f32 * lod_multiplier) / 255.0).clamp(0.0, 1.0);

            // Compute instance count for this cell
            let count = (final_density * config.instances_per_cell as f32).floor() as u32;

            if count == 0 {
                continue;
            }

            // Cell world position (center)
            let cell_x = tile_x as f32 * tile_size as f32 + x as f32;
            let cell_y = tile_y as f32 * tile_size as f32 + y as f32;
            let cell_world_x = cell_x * cell_world_size;
            let cell_world_z = cell_y * cell_world_size;

            // Spawn instances within this cell
            for _ in 0..count {
                // Random jitter within cell (±0.5 cell size)
                let offset_x = (rng.next() - 0.5) * cell_world_size;
                let offset_z = (rng.next() - 0.5) * cell_world_size;

                // Random rotation around Y axis
                let angle = rng.next() * std::f32::consts::TAU;
                let rotation = bevy::prelude::Quat::from_rotation_y(angle);

                // Random scale variation (0.8 to 1.2)
                let scale_factor = 0.8 + rng.next() * 0.4;
                let scale = bevy::prelude::Vec3::splat(scale_factor);

                // Assign random variant (0-7)
                let variant_id = rng.next_u32() % 8;

                instances.push(FoliageInstance::new(
                    bevy::prelude::Vec3::new(
                        cell_world_x + offset_x,
                        0.0, // Y position is set by terrain height at runtime
                        cell_world_z + offset_z,
                    ),
                    rotation,
                    scale,
                    variant_id,
                ));
            }
        }
    }

    instances
}

/// Full pipeline: bake instances for all LOD tiers and write to disk.
pub fn bake_and_write_foliage_instances(
    foliage_root: impl AsRef<Path>,
    tile_size: u32,
    mip_level: u8,
    tx: i32,
    ty: i32,
    world_scale: f32,
    height_scale: f32,
    height_data: &[f32],
    config: &FoliageConfig,
) -> std::io::Result<()> {
    let foliage_root = foliage_root.as_ref();

    // Step 1: Compute procedural masks
    let (slope_mask, altitude_mask, water_mask) = {
        use crate::foliage_generation::*;

        let slope = compute_slope_density(
            height_data,
            tile_size,
            world_scale,
            height_scale,
            config.slope_threshold,
        );
        let altitude = compute_altitude_density(
            height_data,
            height_scale,
            config.altitude_min,
            config.altitude_max,
        );

        // Water mask: assume water_level from config (TODO: get from metadata)
        let water_level = -1000.0; // Placeholder: below most terrain
        let water = compute_water_proximity_density(
            height_data,
            height_scale,
            water_level,
            config.water_proximity_falloff,
        );

        (slope, altitude, water)
    };

    let procedural_mask = blend_density_masks(&slope_mask, &altitude_mask, &water_mask);

    // Write procedural mask to disk (optional, for inspection)
    let procedural_mask_path =
        crate::foliage::procedural_mask_path(foliage_root, mip_level, tx, ty);
    write_mask_tile(&procedural_mask_path, &procedural_mask)?;

    // Step 2: Load painted splatmap
    let painted_manager = PaintedSplatmapManager::new(foliage_root);
    let painted_splatmap = painted_manager.read_tile(mip_level, tx, ty, tile_size)?;

    // Step 3: Generate instances per LOD tier
    let tile_writer = FoliageTileWriter::new(foliage_root);

    for lod_tier in FoliageLodTier::all() {
        let instances = generate_tile_instances(
            tile_size,
            mip_level,
            &painted_splatmap,
            &procedural_mask,
            world_scale,
            config,
            *lod_tier,
            tx,
            ty,
        );

        if !instances.is_empty() {
            tile_writer.write_lod_tile(*lod_tier, mip_level, tx, ty, &instances)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v1 = rng.next();
        assert!(v1 >= 0.0 && v1 <= 1.0);

        let v2 = rng.next_u32();
        assert!(v2 <= u32::MAX);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert!((rng1.next() - rng2.next()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_generate_tile_instances_empty() {
        let config = FoliageConfig::default();
        let painted = vec![0u8; 256 * 256]; // No painted density
        let procedural = vec![0u8; 256 * 256]; // No procedural density

        let instances = generate_tile_instances(
            256,
            0,
            &painted,
            &procedural,
            1.0,
            &config,
            FoliageLodTier::Lod0,
            0,
            0,
        );

        assert_eq!(instances.len(), 0);
    }

    #[test]
    fn test_generate_tile_instances_some_density() {
        let config = FoliageConfig {
            instances_per_cell: 10,
            ..Default::default()
        };

        let mut painted = vec![0u8; 256 * 256];
        let mut procedural = vec![0u8; 256 * 256];

        // Set first cell to 100% density
        painted[0] = 255;
        procedural[0] = 255;

        let instances = generate_tile_instances(
            256,
            0,
            &painted,
            &procedural,
            1.0,
            &config,
            FoliageLodTier::Lod0,
            0,
            0,
        );

        // Should generate ~10 instances (one cell with 100% density × 10 instances_per_cell)
        assert!(instances.len() >= 9 && instances.len() <= 11);

        for inst in instances {
            assert_eq!(inst.position.y, 0.0); // Y position unset at generation time
            assert!(inst.variant_id < 8);
        }
    }

    #[test]
    fn test_generate_tile_instances_lod_multiplier() {
        let config = FoliageConfig {
            instances_per_cell: 100,
            lod0_density: 1.0,
            lod1_density: 0.5,
            lod2_density: 0.1,
            ..Default::default()
        };

        let painted = vec![255u8; 256 * 256]; // 100% painted everywhere
        let procedural = vec![255u8; 256 * 256]; // 100% procedural everywhere

        let lod0_instances = generate_tile_instances(
            256,
            0,
            &painted,
            &procedural,
            1.0,
            &config,
            FoliageLodTier::Lod0,
            0,
            0,
        );

        let lod1_instances = generate_tile_instances(
            256,
            0,
            &painted,
            &procedural,
            1.0,
            &config,
            FoliageLodTier::Lod1,
            0,
            0,
        );

        // LOD1 should have roughly 50% the instances of LOD0
        let ratio = lod1_instances.len() as f32 / lod0_instances.len() as f32;
        assert!(ratio >= 0.45 && ratio <= 0.55);
    }
}

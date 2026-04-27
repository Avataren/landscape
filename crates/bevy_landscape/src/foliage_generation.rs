//! Procedural foliage generation — density mask computation and instance spawning.
//!
//! This module computes procedural density masks based on terrain properties:
//! - Slope: steeper slopes → lower density
//! - Altitude: bands allow selective placement (e.g., grass in mid-range altitudes)
//! - Water proximity: sparse placement near water

use bevy::prelude::*;
use std::path::Path;

/// Compute a slope-based density mask (0-255) for a height tile.
///
/// Steeper slopes reduce density. The mask value is:
/// - 0 at slopes >= slope_threshold
/// - 255 at nearly-flat slopes (slope ≈ 0)
/// - Linear interpolation in between
///
/// # Arguments
///
/// * `height_data` - Height values (range 0-1 after decoding from R16Unorm)
/// * `tile_size` - Tile resolution (pixels per axis)
/// * `world_scale` - Physical metres per pixel at mip level 0
/// * `height_scale` - World Y range for fully-white pixel
/// * `slope_threshold` - Threshold above which density falls to zero (0..1, often ~0.8)
///
/// # Returns
///
/// A density mask (one u8 per pixel) where 255 = full density, 0 = no grass
pub fn compute_slope_density(
    height_data: &[f32],
    tile_size: u32,
    world_scale: f32,
    height_scale: f32,
    slope_threshold: f32,
) -> Vec<u8> {
    let tile_size = tile_size as usize;
    let mut density = vec![255u8; height_data.len()];

    for y in 0..tile_size {
        for x in 0..tile_size {
            let idx = y * tile_size + x;

            // Sample center and 4-neighbors for finite-difference slope
            let c = height_data[idx] * height_scale;
            let n = if y > 0 {
                height_data[(y - 1) * tile_size + x] * height_scale
            } else {
                c
            };
            let s = if y < tile_size - 1 {
                height_data[(y + 1) * tile_size + x] * height_scale
            } else {
                c
            };
            let e = if x < tile_size - 1 {
                height_data[y * tile_size + (x + 1)] * height_scale
            } else {
                c
            };
            let w = if x > 0 {
                height_data[y * tile_size + (x - 1)] * height_scale
            } else {
                c
            };

            // Gradient vectors (in world units)
            let grad_x = (e - w) / (2.0 * world_scale);
            let grad_y = (s - n) / (2.0 * world_scale);
            let slope = (grad_x * grad_x + grad_y * grad_y).sqrt();

            // Map slope to density: 0 at threshold, 255 at flat
            let normalized_slope = slope.clamp(0.0, slope_threshold) / slope_threshold;
            let d = ((1.0 - normalized_slope) * 255.0) as u8;
            density[idx] = d;
        }
    }

    density
}

/// Compute an altitude-based density mask (0-255) for a height tile.
///
/// Defines altitude bands where foliage is dense (e.g., mid-elevation forests).
/// Returns 255 within the band, 0 outside.
///
/// # Arguments
///
/// * `height_data` - Height values (0-1 after R16Unorm decoding)
/// * `height_scale` - World Y range for fully-white pixel
/// * `altitude_min` - Minimum altitude (world units) where density is 255
/// * `altitude_max` - Maximum altitude (world units) where density is 255
///
/// # Returns
///
/// A density mask (0 or 255) where 255 = within altitude band, 0 = outside
pub fn compute_altitude_density(
    height_data: &[f32],
    height_scale: f32,
    altitude_min: f32,
    altitude_max: f32,
) -> Vec<u8> {
    height_data
        .iter()
        .map(|&h| {
            let world_y = h * height_scale;
            if world_y >= altitude_min && world_y <= altitude_max {
                255
            } else {
                0
            }
        })
        .collect()
}

/// Compute a water-proximity-based density mask (0-255) for a height tile.
///
/// At a given water level, pixels near water (above it by less than falloff)
/// get lower density. This creates a sparse band near shorelines.
///
/// # Arguments
///
/// * `height_data` - Height values (0-1 after R16Unorm decoding)
/// * `height_scale` - World Y range for fully-white pixel
/// * `water_level` - World Y of the water surface
/// * `falloff` - Distance (metres) over which density fades to zero
///
/// # Returns
///
/// A density mask (0-255) where pixels above water but within falloff range
/// are faded from 255 → 0 as they approach water level
pub fn compute_water_proximity_density(
    height_data: &[f32],
    height_scale: f32,
    water_level: f32,
    falloff: f32,
) -> Vec<u8> {
    if falloff <= 0.0 {
        return vec![255u8; height_data.len()];
    }

    height_data
        .iter()
        .map(|&h| {
            let world_y = h * height_scale;
            if world_y < water_level {
                // Below water: no grass
                0
            } else {
                // Above water: fade from water_level + falloff (255) to water_level (0)
                let dist_from_water = world_y - water_level;
                if dist_from_water >= falloff {
                    255
                } else {
                    ((dist_from_water / falloff) * 255.0) as u8
                }
            }
        })
        .collect()
}

/// Blend three density masks into a final procedural density mask.
///
/// Combines slope, altitude, and water proximity by multiplication:
/// `final = slope × altitude × water_proximity / 256²`
///
/// All values are clamped to 0-255.
pub fn blend_density_masks(
    slope: &[u8],
    altitude: &[u8],
    water_proximity: &[u8],
) -> Vec<u8> {
    assert_eq!(slope.len(), altitude.len());
    assert_eq!(altitude.len(), water_proximity.len());

    slope
        .iter()
        .zip(altitude.iter())
        .zip(water_proximity.iter())
        .map(|((&s, &a), &w)| {
            // Blend by multiplication and renormalize to 0-255
            let blended = (s as u32 * a as u32 * w as u32) / (256u32 * 256u32);
            blended.min(255) as u8
        })
        .collect()
}

/// Write a density mask to a binary tile file.
///
/// Format: raw u8 values (one per pixel), row-major order.
pub fn write_mask_tile(path: &Path, density: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, density)?;
    Ok(())
}

/// Read a density mask from a binary tile file.
///
/// Expects `tile_size × tile_size` bytes, row-major order.
pub fn read_mask_tile(path: &Path, tile_size: u32) -> std::io::Result<Vec<u8>> {
    let expected_size = (tile_size * tile_size) as usize;
    let data = std::fs::read(path)?;
    if data.len() != expected_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "mask tile has {} bytes, expected {}",
                data.len(),
                expected_size
            ),
        ));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slope_density_flat_surface() {
        // Flat surface at height 0.5 → slope = 0 → density = 255
        let height = vec![0.5; 256 * 256];
        let density = compute_slope_density(&height, 256, 1.0, 1024.0, 0.8);
        assert!(density.iter().all(|&d| d == 255));
    }

    #[test]
    fn test_altitude_density_in_band() {
        // Heights within the altitude band get density 255
        let height = vec![0.5; 256 * 256]; // world_y = 0.5 * 1024 = 512
        let density = compute_altitude_density(&height, 1024.0, 400.0, 600.0);
        assert!(density.iter().all(|&d| d == 255));
    }

    #[test]
    fn test_altitude_density_out_of_band() {
        let height = vec![0.9; 256 * 256]; // world_y = 0.9 * 1024 = 921.6
        let density = compute_altitude_density(&height, 1024.0, 400.0, 600.0);
        assert!(density.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_water_proximity_above_water() {
        // Heights well above water get full density
        let height = vec![0.9; 256 * 256]; // world_y = 921.6
        let density = compute_water_proximity_density(&height, 1024.0, 100.0, 50.0);
        assert!(density.iter().all(|&d| d == 255));
    }

    #[test]
    fn test_water_proximity_below_water() {
        // Heights below water get zero density
        let height = vec![0.05; 256 * 256]; // world_y = 51.2
        let density = compute_water_proximity_density(&height, 1024.0, 100.0, 50.0);
        assert!(density.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_blend_density_masks() {
        let slope = vec![128u8; 16];
        let altitude = vec![200u8; 16];
        let water = vec![255u8; 16];
        let blended = blend_density_masks(&slope, &altitude, &water);

        // (128 * 200 * 255) / (256 * 256) ≈ 99
        for &b in &blended {
            assert_eq!(b, 99);
        }
    }
}

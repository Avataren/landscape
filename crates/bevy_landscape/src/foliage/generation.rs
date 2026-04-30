//! Procedural foliage generation — density mask computation and instance spawning.
//!
//! Computes procedural density masks based on terrain properties:
//! slope, altitude bands, and water proximity.

use bevy::prelude::*;
use std::path::Path;

/// Compute a slope-based density mask (0-255) for a height tile.
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
            let c = height_data[idx] * height_scale;
            let n = if y > 0 { height_data[(y - 1) * tile_size + x] * height_scale } else { c };
            let s = if y < tile_size - 1 { height_data[(y + 1) * tile_size + x] * height_scale } else { c };
            let e = if x < tile_size - 1 { height_data[y * tile_size + (x + 1)] * height_scale } else { c };
            let w = if x > 0 { height_data[y * tile_size + (x - 1)] * height_scale } else { c };

            let grad_x = (e - w) / (2.0 * world_scale);
            let grad_y = (s - n) / (2.0 * world_scale);
            let slope = (grad_x * grad_x + grad_y * grad_y).sqrt();
            let normalized_slope = slope.clamp(0.0, slope_threshold) / slope_threshold;
            density[idx] = ((1.0 - normalized_slope) * 255.0) as u8;
        }
    }

    density
}

/// Compute an altitude-based density mask (0-255) for a height tile.
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
            if world_y >= altitude_min && world_y <= altitude_max { 255 } else { 0 }
        })
        .collect()
}

/// Compute a water-proximity-based density mask (0-255) for a height tile.
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
                0
            } else {
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
pub fn blend_density_masks(slope: &[u8], altitude: &[u8], water_proximity: &[u8]) -> Vec<u8> {
    assert_eq!(slope.len(), altitude.len());
    assert_eq!(altitude.len(), water_proximity.len());

    slope
        .iter()
        .zip(altitude.iter())
        .zip(water_proximity.iter())
        .map(|((&s, &a), &w)| {
            let blended = (s as u32 * a as u32 * w as u32) / (256u32 * 256u32);
            blended.min(255) as u8
        })
        .collect()
}

/// Write a density mask to a binary tile file.
pub fn write_mask_tile(path: &Path, density: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, density)?;
    Ok(())
}

/// Read a density mask from a binary tile file.
pub fn read_mask_tile(path: &Path, tile_size: u32) -> std::io::Result<Vec<u8>> {
    let expected_size = (tile_size * tile_size) as usize;
    let data = std::fs::read(path)?;
    if data.len() != expected_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("mask tile has {} bytes, expected {}", data.len(), expected_size),
        ));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slope_density_flat_surface() {
        let height = vec![0.5; 256 * 256];
        let density = compute_slope_density(&height, 256, 1.0, 1024.0, 0.8);
        assert!(density.iter().all(|&d| d == 255));
    }

    #[test]
    fn test_altitude_density_in_band() {
        let height = vec![0.5; 256 * 256];
        let density = compute_altitude_density(&height, 1024.0, 400.0, 600.0);
        assert!(density.iter().all(|&d| d == 255));
    }

    #[test]
    fn test_altitude_density_out_of_band() {
        let height = vec![0.9; 256 * 256];
        let density = compute_altitude_density(&height, 1024.0, 400.0, 600.0);
        assert!(density.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_water_proximity_above_water() {
        let height = vec![0.9; 256 * 256];
        let density = compute_water_proximity_density(&height, 1024.0, 100.0, 50.0);
        assert!(density.iter().all(|&d| d == 255));
    }

    #[test]
    fn test_water_proximity_below_water() {
        let height = vec![0.05; 256 * 256];
        let density = compute_water_proximity_density(&height, 1024.0, 100.0, 50.0);
        assert!(density.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_blend_density_masks() {
        let slope = vec![128u8; 16];
        let altitude = vec![200u8; 16];
        let water = vec![255u8; 16];
        let blended = blend_density_masks(&slope, &altitude, &water);
        for &b in &blended {
            assert_eq!(b, 99);
        }
    }
}

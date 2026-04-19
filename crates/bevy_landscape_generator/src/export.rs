//! Direct tile-hierarchy generator — no PNG intermediate, supports arbitrary resolution.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{
    mpsc::{self, Receiver},
    Mutex,
};

use crate::params::GeneratorParams;
use crate::terrain_fn::sample_height;

/// Receiver is wrapped in Mutex so ExportHandle is Sync (required for Bevy Resource).
pub struct ExportHandle {
    pub log_rx: Mutex<Receiver<String>>,
    pub done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Set to true only on a clean successful export (not on error).
    pub succeeded: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Output directory for the completed export.
    pub output_dir: PathBuf,
}

pub fn start_export(params: GeneratorParams, output_dir: PathBuf) -> ExportHandle {
    let (log_tx, log_rx) = mpsc::channel::<String>();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let succeeded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone = done.clone();
    let succeeded_clone = succeeded.clone();
    let output_dir_thread = output_dir.clone();

    std::thread::spawn(move || {
        let output_dir = output_dir_thread;
        let run = || -> Result<(), String> {
            log_tx
                .send(format!(
                    "Generating {}×{} tile hierarchy …",
                    params.export_resolution, params.export_resolution,
                ))
                .ok();

            generate_tiles_direct(&params, &output_dir, &log_tx)?;
            Ok(())
        };

        match run() {
            Ok(()) => {
                succeeded_clone.store(true, std::sync::atomic::Ordering::Release);
            }
            Err(e) => {
                log_tx.send(format!("Export failed: {e}")).ok();
            }
        }
        done_clone.store(true, std::sync::atomic::Ordering::Release);
    });

    ExportHandle {
        log_rx: Mutex::new(log_rx),
        done,
        succeeded,
        output_dir,
    }
}

fn generate_tiles_direct(
    params: &GeneratorParams,
    output_dir: &std::path::Path,
    log: &mpsc::Sender<String>,
) -> Result<(), String> {
    const TILE: usize = 256;
    const BUF: usize = TILE + 2; // 258×258: 1px border for cross-tile Sobel normals

    let export_res = params.export_resolution;

    let levels = {
        let mut sz = export_res as usize;
        let mut n = 0u32;
        while sz / TILE >= 2 {
            n += 1;
            sz /= 2;
        }
        n
    };

    if levels == 0 {
        return Err(format!(
            "export_resolution {} is too small (min {})",
            export_res,
            TILE * 2
        ));
    }

    log.send(format!("  {} LOD levels, tile {}px", levels, TILE))
        .ok();

    for subdir in ["height", "normal"] {
        let path = output_dir.join(subdir);
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| format!("Failed to clear '{subdir}': {e}"))?;
        }
    }
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Failed to create output dir: {e}"))?;

    let effective_height_scale = params.height_scale * params.world_scale;

    for lod in 0..levels {
        let tile_scale = 1u32 << lod;
        let mip_size = export_res >> lod;
        let tiles_per_side = (mip_size / TILE as u32) as i32;
        let tile_start = -(tiles_per_side / 2);
        let tile_end = tile_start + tiles_per_side;
        let lod_scale = params.world_scale * tile_scale as f32;

        let height_dir = output_dir.join(format!("height/L{lod}"));
        let normal_dir = output_dir.join(format!("normal/L{lod}"));
        std::fs::create_dir_all(&height_dir).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&normal_dir).map_err(|e| e.to_string())?;

        let tile_coords: Vec<(i32, i32)> = (tile_start..tile_end)
            .flat_map(|ty| (tile_start..tile_end).map(move |tx| (tx, ty)))
            .collect();
        let tile_count = tile_coords.len();

        let thread_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (tile_count + thread_count - 1) / thread_count;

        std::thread::scope(|s| -> Result<(), String> {
            let mut handles = Vec::new();
            for chunk in tile_coords.chunks(chunk_size) {
                let height_dir = &height_dir;
                let normal_dir = &normal_dir;
                let output_dir = output_dir;
                handles.push(s.spawn(move || -> Result<(), String> {
                    for &(tx, ty) in chunk {
                        let buf = build_height_buf(params, output_dir, export_res, lod, tx, ty)?;

                        let mut tile_bytes = Vec::with_capacity(TILE * TILE * 2);
                        let mut normal_bytes = Vec::with_capacity(TILE * TILE * 2);

                        for row in 0..TILE {
                            for col in 0..TILE {
                                let cx = col + 1; // offset into the 258×258 buf
                                let cy = row + 1;

                                // R16Unorm height
                                let h = buf[cy * BUF + cx];
                                let v = (h.clamp(0.0, 1.0) * 65535.0) as u16;
                                tile_bytes.extend_from_slice(&v.to_le_bytes());

                                // Sobel-based world-space normal (XZ channels, RG8Snorm)
                                let hf = |dx: i32, dy: i32| -> f32 {
                                    buf[(cy as i32 + dy) as usize * BUF + (cx as i32 + dx) as usize]
                                };
                                let gx = (hf(1, -1) + 2.0 * hf(1, 0) + hf(1, 1)
                                    - hf(-1, -1)
                                    - 2.0 * hf(-1, 0)
                                    - hf(-1, 1))
                                    * (1.0 / 8.0)
                                    * effective_height_scale;
                                let gz = (hf(-1, 1) + 2.0 * hf(0, 1) + hf(1, 1)
                                    - hf(-1, -1)
                                    - 2.0 * hf(0, -1)
                                    - hf(1, -1))
                                    * (1.0 / 8.0)
                                    * effective_height_scale;
                                let nx = -gx;
                                let nz = -gz;
                                let len =
                                    (nx * nx + lod_scale * lod_scale + nz * nz).sqrt().max(1e-6);
                                normal_bytes
                                        .push(((nx / len).clamp(-1.0, 1.0) * 127.0).round() as i8
                                            as u8);
                                normal_bytes
                                        .push(((nz / len).clamp(-1.0, 1.0) * 127.0).round() as i8
                                            as u8);
                            }
                        }

                        let height_path = height_dir.join(format!("{tx}_{ty}.bin"));
                        std::fs::write(&height_path, &tile_bytes).map_err(|e| {
                            format!(
                                "Failed to write height tile L{lod}/{tx}_{ty}.bin to '{}': {e}",
                                height_path.display()
                            )
                        })?;

                        let normal_path = normal_dir.join(format!("{tx}_{ty}.bin"));
                        std::fs::write(&normal_path, &normal_bytes).map_err(|e| {
                            format!(
                                "Failed to write normal tile L{lod}/{tx}_{ty}.bin to '{}': {e}",
                                normal_path.display()
                            )
                        })?;
                    }
                    Ok(())
                }));
            }

            for handle in handles {
                handle.join().map_err(|_| {
                    format!("Terrain export worker panicked while writing LOD {lod}")
                })??;
            }
            Ok(())
        })?;

        log.send(format!("Level {lod}: {} tiles", tile_count)).ok();
    }

    log.send(format!("Done → '{}'", output_dir.display())).ok();
    Ok(())
}

fn build_height_buf(
    params: &GeneratorParams,
    output_dir: &Path,
    export_res: u32,
    lod: u32,
    tx: i32,
    ty: i32,
) -> Result<Vec<f32>, String> {
    const TILE: usize = 256;
    const BUF: usize = TILE + 2;

    let mut buf = vec![0.0f32; BUF * BUF];

    if lod == 0 {
        for brow in 0..BUF {
            for bcol in 0..BUF {
                let c = tx * TILE as i32 + bcol as i32 - 1;
                let r = ty * TILE as i32 + brow as i32 - 1;
                buf[brow * BUF + bcol] = sample_height_at_grid(params, export_res, c, r, 1);
            }
        }
        return Ok(buf);
    }

    let prev_height_dir = output_dir.join(format!("height/L{}", lod - 1));
    let prev_tile_scale = 1u32 << (lod - 1);
    let mut cache: HashMap<(i32, i32), Option<Vec<f32>>> = HashMap::new();

    for brow in 0..BUF {
        for bcol in 0..BUF {
            let c = tx * TILE as i32 + bcol as i32 - 1;
            let r = ty * TILE as i32 + brow as i32 - 1;
            let gx = c * 2;
            let gz = r * 2;

            let h00 = sample_prev_level_texel(
                params,
                export_res,
                prev_tile_scale,
                &prev_height_dir,
                &mut cache,
                gx,
                gz,
            )?;
            let h10 = sample_prev_level_texel(
                params,
                export_res,
                prev_tile_scale,
                &prev_height_dir,
                &mut cache,
                gx + 1,
                gz,
            )?;
            let h01 = sample_prev_level_texel(
                params,
                export_res,
                prev_tile_scale,
                &prev_height_dir,
                &mut cache,
                gx,
                gz + 1,
            )?;
            let h11 = sample_prev_level_texel(
                params,
                export_res,
                prev_tile_scale,
                &prev_height_dir,
                &mut cache,
                gx + 1,
                gz + 1,
            )?;

            buf[brow * BUF + bcol] = (h00 + h10 + h01 + h11) * 0.25;
        }
    }

    Ok(buf)
}

fn sample_prev_level_texel(
    params: &GeneratorParams,
    export_res: u32,
    prev_tile_scale: u32,
    prev_height_dir: &Path,
    cache: &mut HashMap<(i32, i32), Option<Vec<f32>>>,
    gx: i32,
    gz: i32,
) -> Result<f32, String> {
    const TILE: usize = 256;

    let tx = gx.div_euclid(TILE as i32);
    let ty = gz.div_euclid(TILE as i32);
    let key = (tx, ty);

    if !cache.contains_key(&key) {
        let path = prev_height_dir.join(format!("{tx}_{ty}.bin"));
        let tile = if path.exists() {
            Some(read_r16_tile(&path, TILE)?)
        } else {
            None
        };
        cache.insert(key, tile);
    }

    if let Some(Some(tile)) = cache.get(&key) {
        let local_x = gx.rem_euclid(TILE as i32) as usize;
        let local_y = gz.rem_euclid(TILE as i32) as usize;
        return Ok(tile[local_y * TILE + local_x]);
    }

    Ok(sample_height_at_grid(
        params,
        export_res,
        gx,
        gz,
        prev_tile_scale,
    ))
}

fn read_r16_tile(path: &Path, tile_size: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Failed to read tile '{}': {e}", path.display()))?;
    let expected = tile_size * tile_size * 2;
    if bytes.len() != expected {
        return Err(format!(
            "Unexpected tile size for '{}': expected {} bytes, got {}",
            path.display(),
            expected,
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0)
        .collect())
}

fn sample_height_at_grid(
    params: &GeneratorParams,
    export_res: u32,
    grid_x: i32,
    grid_y: i32,
    tile_scale: u32,
) -> f32 {
    sample_height(
        params,
        sample_coordinate(grid_x, tile_scale, export_res),
        sample_coordinate(grid_y, tile_scale, export_res),
    )
}

fn sample_coordinate(grid_coord: i32, tile_scale: u32, export_res: u32) -> f32 {
    ((grid_coord as f32 + 0.5) * tile_scale as f32 / export_res as f32) + 0.5
}

#[cfg(test)]
mod tests {
    use super::{generate_tiles_direct, read_r16_tile, sample_coordinate};
    use crate::GeneratorParams;
    use std::sync::mpsc;
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("landscape-gen-{label}-{nonce}"))
    }

    fn combine_child_tiles(level_dir: &Path, tx: i32, ty: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; 512 * 512];
        for child_y in 0..2 {
            for child_x in 0..2 {
                let tile = read_r16_tile(
                    &level_dir.join(format!("{}_{}.bin", tx * 2 + child_x, ty * 2 + child_y)),
                    256,
                )
                .unwrap();
                for row in 0..256 {
                    let dst_row = (child_y as usize * 256 + row) * 512 + child_x as usize * 256;
                    let src_row = row * 256;
                    out[dst_row..dst_row + 256].copy_from_slice(&tile[src_row..src_row + 256]);
                }
            }
        }
        out
    }

    fn box_filter_2x(src: &[f32], size: usize) -> Vec<f32> {
        let out_size = size / 2;
        let mut out = vec![0.0f32; out_size * out_size];
        for y in 0..out_size {
            for x in 0..out_size {
                let sx = x * 2;
                let sy = y * 2;
                out[y * out_size + x] = (src[sy * size + sx]
                    + src[sy * size + sx + 1]
                    + src[(sy + 1) * size + sx]
                    + src[(sy + 1) * size + sx + 1])
                    * 0.25;
            }
        }
        out
    }

    #[test]
    fn export_samples_texel_centers() {
        assert_eq!(sample_coordinate(0, 1, 4096), 0.5001220703125);
        assert_eq!(sample_coordinate(0, 2, 4096), 0.500244140625);
        assert_eq!(sample_coordinate(-2048, 1, 4096), 0.0001220703125);
    }

    #[test]
    fn export_builds_filtered_mips() {
        let temp_dir = unique_temp_dir("filtered-mips");
        fs::create_dir_all(&temp_dir).unwrap();

        let params = GeneratorParams {
            export_resolution: 1024,
            ..Default::default()
        };
        let (log_tx, _log_rx) = mpsc::channel();
        generate_tiles_direct(&params, &temp_dir, &log_tx).unwrap();

        let combined = combine_child_tiles(&temp_dir.join("height/L0"), 0, 0);
        let downsampled = box_filter_2x(&combined, 512);
        let l1_tile = read_r16_tile(&temp_dir.join("height/L1/0_0.bin"), 256).unwrap();

        let max_diff = downsampled
            .iter()
            .zip(l1_tile.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff <= 2.0 / 65535.0,
            "expected filtered mip pyramid, max diff was {max_diff}"
        );

        let _ = fs::remove_dir_all(temp_dir);
    }
}

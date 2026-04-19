//! Direct tile-hierarchy generator — no PNG intermediate, supports arbitrary resolution.

use std::path::PathBuf;
use std::sync::{
    mpsc::{self, Receiver},
    Mutex,
};

use bevy::math::{Vec2, Vec3};

use crate::params::GeneratorParams;

/// Receiver is wrapped in Mutex so ExportHandle is Sync (required for Bevy Resource).
pub struct ExportHandle {
    pub log_rx:     Mutex<Receiver<String>>,
    pub done:       std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Set to true only on a clean successful export (not on error).
    pub succeeded:  std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Output directory for the completed export.
    pub output_dir: PathBuf,
}

pub fn start_export(params: GeneratorParams, output_dir: PathBuf) -> ExportHandle {
    let (log_tx, log_rx) = mpsc::channel::<String>();
    let done      = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let succeeded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone      = done.clone();
    let succeeded_clone = succeeded.clone();
    let output_dir_thread = output_dir.clone();

    std::thread::spawn(move || {
        let output_dir = output_dir_thread;
        let run = || -> Result<(), String> {
            log_tx.send(format!(
                "Generating {}×{} tile hierarchy …",
                params.export_resolution, params.export_resolution,
            )).ok();

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

    log.send(format!("  {} LOD levels, tile {}px", levels, TILE)).ok();

    for subdir in ["height", "normal"] {
        let path = output_dir.join(subdir);
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| format!("Failed to clear '{subdir}': {e}"))?;
        }
    }
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;

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

        std::thread::scope(|s| {
            for chunk in tile_coords.chunks(chunk_size) {
                let height_dir = &height_dir;
                let normal_dir = &normal_dir;
                s.spawn(move || {
                    for &(tx, ty) in chunk {
                        // 258×258 buffer: includes 1px border from neighbouring tiles.
                        let mut buf = [0.0f32; BUF * BUF];
                        for brow in 0..BUF {
                            for bcol in 0..BUF {
                                // Pixel coords in mip-level image space (can go 1 past tile edge).
                                let c = tx * TILE as i32 + bcol as i32 - 1;
                                let r = ty * TILE as i32 + brow as i32 - 1;
                                let uv_x = c as f32 * tile_scale as f32 / export_res as f32 + 0.5;
                                let uv_y = r as f32 * tile_scale as f32 / export_res as f32 + 0.5;
                                buf[brow * BUF + bcol] = height_at(params, uv_x, uv_y);
                            }
                        }

                        let mut tile_bytes   = Vec::with_capacity(TILE * TILE * 2);
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
                                    buf[(cy as i32 + dy) as usize * BUF
                                        + (cx as i32 + dx) as usize]
                                };
                                let gx = (hf(1,-1) + 2.0*hf(1,0) + hf(1,1)
                                         - hf(-1,-1) - 2.0*hf(-1,0) - hf(-1,1))
                                         * (1.0 / 8.0) * effective_height_scale;
                                let gz = (hf(-1,1) + 2.0*hf(0,1) + hf(1,1)
                                         - hf(-1,-1) - 2.0*hf(0,-1) - hf(1,-1))
                                         * (1.0 / 8.0) * effective_height_scale;
                                let nx = -gx;
                                let nz = -gz;
                                let len = (nx*nx + lod_scale*lod_scale + nz*nz).sqrt().max(1e-6);
                                normal_bytes.push(((nx / len).clamp(-1.0, 1.0) * 127.0).round() as i8 as u8);
                                normal_bytes.push(((nz / len).clamp(-1.0, 1.0) * 127.0).round() as i8 as u8);
                            }
                        }

                        let _ = std::fs::write(
                            height_dir.join(format!("{tx}_{ty}.bin")),
                            &tile_bytes,
                        );
                        let _ = std::fs::write(
                            normal_dir.join(format!("{tx}_{ty}.bin")),
                            &normal_bytes,
                        );
                    }
                });
            }
        });

        log.send(format!("Level {lod}: {} tiles", tile_count)).ok();
    }

    log.send(format!(
        "Done → '{}'",
        output_dir.display()
    )).ok();
    Ok(())
}

/// Evaluate the FBM heightfield at a UV coordinate [0,1).
fn height_at(params: &GeneratorParams, uv_x: f32, uv_y: f32) -> f32 {
    let seed_off = Vec2::new(
        params.seed as f32 * 0.47316,
        params.seed as f32 * 0.31419,
    );
    let pos = (Vec2::new(uv_x, uv_y) + params.offset + seed_off) * params.frequency;
    let h = fbm_cpu(pos, params.octaves, params.lacunarity, params.gain);
    (h * 0.5 + 0.5).clamp(0.0, 1.0)
}

// --- CPU FBM matching the WGSL shader ---

fn hash22(p: Vec2) -> Vec2 {
    let mut p3 = Vec3::new(p.x * 0.1031, p.y * 0.1030, p.x * 0.0973);
    p3 = p3.fract();
    p3 += p3.dot(Vec3::new(p3.y, p3.z, p3.x) + 33.33);
    Vec2::new(
        ((p3.x + p3.y) * p3.z).fract(),
        ((p3.x + p3.z) * p3.y).fract(),
    )
}

fn gradient_noise(p: Vec2) -> f32 {
    let i = p.floor();
    let f = p.fract();
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let ga = hash22(i + Vec2::new(0.0, 0.0)) * 2.0 - Vec2::ONE;
    let gb = hash22(i + Vec2::new(1.0, 0.0)) * 2.0 - Vec2::ONE;
    let gc = hash22(i + Vec2::new(0.0, 1.0)) * 2.0 - Vec2::ONE;
    let gd = hash22(i + Vec2::new(1.0, 1.0)) * 2.0 - Vec2::ONE;

    let va = ga.dot(f - Vec2::new(0.0, 0.0));
    let vb = gb.dot(f - Vec2::new(1.0, 0.0));
    let vc = gc.dot(f - Vec2::new(0.0, 1.0));
    let vd = gd.dot(f - Vec2::new(1.0, 1.0));

    lerp(lerp(va, vb, u.x), lerp(vc, vd, u.x), u.y)
}

fn fbm_cpu(mut p: Vec2, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 0.5f32;
    for _ in 0..octaves {
        value += amplitude * gradient_noise(p);
        p *= lacunarity;
        amplitude *= gain;
    }
    value
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

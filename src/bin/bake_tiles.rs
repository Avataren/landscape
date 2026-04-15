//! Offline tile baker.
//!
//! Reads a height map (EXR, PNG, or TIFF) and optionally a bump map, builds a
//! mip pyramid, and writes tiles to:
//!   `{output}/height/L{n}/{tx}_{ty}.bin`  — R16Unorm height
//!   `{output}/normal/L{n}/{tx}_{ty}.bin`  — RG8Snorm XZ normals
//!
//! Run from the workspace root:
//!   cargo run --bin bake_tiles --release -- --height <path> [options]
//!
//! Options:
//!   --height <path>       Height map (EXR, PNG, or TIFF)  [required]
//!   --bump <path>         Bump map for normals (PNG or TIFF).
//!                         If omitted, normals are derived from the height map.
//!   --output <dir>        Output directory  [default: assets/tiles]
//!   --height-scale <f32>  World-space height range in units  [default: 2048.0]
//!   --bump-scale <f32>    World-space scale for bump normal derivation
//!                         [default: same as height-scale]
//!   --world-scale <f32>   World-space units per texel at LOD 0  [default: 1.0]
//!   --tile-size <usize>   Tile resolution in pixels  [default: 256]

use exr::prelude::{read_first_flat_layer_from_file, FlatSamples};
use image::ImageReader;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    height_path: PathBuf,
    bump_path: Option<PathBuf>,
    output_dir: PathBuf,
    height_scale: f32,
    bump_scale: Option<f32>,
    world_scale: f32,
    tile_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            height_path: PathBuf::new(),
            bump_path: None,
            output_dir: PathBuf::from("assets/tiles"),
            height_scale: 2048.0,
            bump_scale: None,
            world_scale: 1.0,
            tile_size: 256,
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = Config::default();

    if args.is_empty() {
        // Legacy no-arg invocation: fall back to the original Rocky Terrain path
        // so existing workflows keep working.
        cfg.height_path = PathBuf::from(
            "assets/height_maps/16k Rocky Terrain Heightmap/Height Map 16k Rocky Terrain.exr",
        );
        return Ok(cfg);
    }

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let next = || -> Result<&str, String> {
            args.get(i + 1)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag {
            "--height" => {
                cfg.height_path = PathBuf::from(next()?);
                i += 2;
            }
            "--bump" => {
                cfg.bump_path = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--output" => {
                cfg.output_dir = PathBuf::from(next()?);
                i += 2;
            }
            "--height-scale" => {
                cfg.height_scale = next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?;
                i += 2;
            }
            "--bump-scale" => {
                cfg.bump_scale =
                    Some(next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?);
                i += 2;
            }
            "--world-scale" => {
                cfg.world_scale = next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?;
                i += 2;
            }
            "--tile-size" => {
                cfg.tile_size = next()?.parse::<usize>().map_err(|e| format!("{flag}: {e}"))?;
                i += 2;
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    if cfg.height_path.as_os_str().is_empty() {
        return Err("--height <path> is required".to_string());
    }

    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!();
        eprintln!("Usage: bake_tiles --height <path> [--bump <path>] [--output <dir>]");
        eprintln!("                  [--height-scale <f>] [--bump-scale <f>]");
        eprintln!("                  [--world-scale <f>] [--tile-size <n>]");
        std::process::exit(1);
    });

    let t_start = Instant::now();

    if !cfg.height_path.exists() {
        eprintln!("ERROR: height map not found at {:?}", cfg.height_path);
        eprintln!("Run this tool from the workspace root (next to 'assets/').");
        std::process::exit(1);
    }

    println!("Loading height map: {} ...", cfg.height_path.display());
    let (height_pixels_raw, img_w, img_h) = load_grayscale_image(&cfg.height_path)?;
    println!("Loaded {}×{} in {:.1}s", img_w, img_h, t_start.elapsed().as_secs_f32());

    if img_w != img_h {
        return Err(format!("Expected square heightmap, got {}×{}", img_w, img_h).into());
    }

    // Normalise height to [0, 1].
    let h_min = height_pixels_raw.iter().cloned().fold(f32::INFINITY, f32::min);
    let h_max = height_pixels_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let h_range = h_max - h_min;
    println!("Height range: min={:.6}  max={:.6}  range={:.6}", h_min, h_max, h_range);
    let height_pixels: Vec<f32> = if h_range > 1e-6 {
        height_pixels_raw.iter().map(|&h| (h - h_min) / h_range).collect()
    } else {
        height_pixels_raw
    };

    // Load bump map for normal derivation if provided.
    let bump_pixels: Option<Vec<f32>> = if let Some(ref bump_path) = cfg.bump_path {
        if !bump_path.exists() {
            eprintln!("ERROR: bump map not found at {:?}", bump_path);
            std::process::exit(1);
        }
        println!("Loading bump map: {} ...", bump_path.display());
        let (mut pix, bw, bh) = load_grayscale_image(bump_path)?;
        println!("  {}×{}", bw, bh);
        if bw != img_w || bh != img_h {
            return Err(format!(
                "Bump map {}×{} must match height map {}×{}",
                bw, bh, img_w, img_h
            )
            .into());
        }
        // Normalise to [0, 1].
        let bmin = pix.iter().cloned().fold(f32::INFINITY, f32::min);
        let bmax = pix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let brange = bmax - bmin;
        if brange > 1e-6 {
            pix.iter_mut().for_each(|v| *v = (*v - bmin) / brange);
        }
        Some(pix)
    } else {
        None
    };

    // Use bump_scale for normal derivation if a bump map was loaded; otherwise
    // the height map doubles as the normal source using height_scale.
    let bump_scale = cfg.bump_scale.unwrap_or(cfg.height_scale);

    let tile_size = cfg.tile_size;

    // Auto-compute level count: keep halving until the mip fits in one tile.
    let levels = {
        let mut sz = img_w;
        let mut n = 0u32;
        while sz >= tile_size {
            n += 1;
            sz /= 2;
        }
        n
    };

    println!(
        "Baking {} levels, tile size {}px → '{}'",
        levels,
        tile_size,
        cfg.output_dir.display()
    );

    let mut current_height = height_pixels;
    let mut current_bump = bump_pixels;
    let mut current_size = img_w;
    let mut total_tiles = 0usize;

    for lod in 0..levels {
        let mip_size = current_size;
        let mip_half = (mip_size / 2) as i32;
        let tiles_per_side = (mip_size / tile_size) as i32;
        let tile_half = tiles_per_side / 2;

        if tiles_per_side == 0 {
            eprintln!(
                "WARNING: mip_size {} < tile_size {} at LOD {}; stopping.",
                mip_size, tile_size, lod
            );
            break;
        }

        let height_level_dir = cfg.output_dir.join(format!("height/L{}", lod));
        let normal_level_dir = cfg.output_dir.join(format!("normal/L{}", lod));
        std::fs::create_dir_all(&height_level_dir)?;
        std::fs::create_dir_all(&normal_level_dir)?;

        let level_tile_count = (tiles_per_side * tiles_per_side) as usize;
        let mut level_done = 0usize;

        let lod_scale = cfg.world_scale * (1u32 << lod) as f32;

        // Normal source: use the bump map if available, otherwise the height map.
        let normal_src = current_bump.as_deref().unwrap_or(&current_height);
        let normal_scale = if current_bump.is_some() {
            bump_scale
        } else {
            cfg.height_scale
        };

        for ty in -tile_half..tile_half {
            for tx in -tile_half..tile_half {
                let px_start = (tx * tile_size as i32 + mip_half) as usize;
                let py_start = (ty * tile_size as i32 + mip_half) as usize;

                let mut tile_bytes = Vec::with_capacity(tile_size * tile_size * 2);
                let mut normal_bytes = Vec::with_capacity(tile_size * tile_size * 2);

                for row in 0..tile_size {
                    for col in 0..tile_size {
                        let px = px_start + col;
                        let py = py_start + row;

                        let h = if px < mip_size && py < mip_size {
                            current_height[py * mip_size + px]
                        } else {
                            0.0
                        };
                        let v = (h.clamp(0.0, 1.0) * 65535.0) as u16;
                        tile_bytes.extend_from_slice(&v.to_le_bytes());

                        let enc = if px < mip_size && py < mip_size {
                            encode_normal_xz(compute_normal(
                                normal_src,
                                mip_size,
                                px,
                                py,
                                lod_scale,
                                normal_scale,
                            ))
                        } else {
                            [0u8, 0u8]
                        };
                        normal_bytes.extend_from_slice(&enc);
                    }
                }

                let hpath = height_level_dir.join(format!("{}_{}.bin", tx, ty));
                let npath = normal_level_dir.join(format!("{}_{}.bin", tx, ty));
                std::fs::write(&hpath, &tile_bytes)?;
                std::fs::write(&npath, &normal_bytes)?;

                level_done += 1;
                total_tiles += 1;
            }

            if lod == 0 {
                let pct = level_done * 100 / level_tile_count;
                print!("\r  Level 0: {}%  ({}/{})", pct, level_done, level_tile_count);
                let _ = std::io::stdout().flush();
            }
        }
        if lod == 0 {
            println!();
        }

        println!(
            "Level {}: {}×{} = {} tiles  [{:.1}s]",
            lod,
            tiles_per_side,
            tiles_per_side,
            level_tile_count,
            t_start.elapsed().as_secs_f32(),
        );

        if lod + 1 < levels {
            let next_h = box_filter_2x(&current_height, current_size, current_size);
            current_height = next_h;
            if let Some(ref b) = current_bump {
                let next_b = box_filter_2x(b, current_size, current_size);
                current_bump = Some(next_b);
            }
            current_size /= 2;
        }
    }

    println!(
        "\nDone. {} tiles → '{}'  ({:.1}s total)",
        total_tiles,
        cfg.output_dir.display(),
        t_start.elapsed().as_secs_f32(),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

/// Load any supported format (EXR, PNG, TIFF, …) as normalised f32 greyscale.
fn load_grayscale_image(
    path: &Path,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "exr" {
        load_height_exr(path)
    } else {
        load_grayscale_raster(path)
    }
}

/// PNG / TIFF / … via the `image` crate — converts to 16-bit luma.
fn load_grayscale_raster(
    path: &Path,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?;
    let w = img.width() as usize;
    let h = img.height() as usize;
    let luma = img.into_luma16();
    let pixels = luma.pixels().map(|p| p[0] as f32 / 65535.0).collect();
    Ok((pixels, w, h))
}

/// EXR via the `exr` crate — reads the first channel regardless of name.
fn load_height_exr(
    path: &Path,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let image = read_first_flat_layer_from_file(path)?;
    let layer = &image.layer_data;
    let w = layer.size.x();
    let h = layer.size.y();

    println!("EXR channels ({}):", layer.channel_data.list.len());
    for ch in &layer.channel_data.list {
        println!("  {:?}", ch.name);
    }

    let channel = layer
        .channel_data
        .list
        .first()
        .ok_or("EXR file has no channels")?;
    println!("Using channel {:?} as height", channel.name);

    let pixels: Vec<f32> = match &channel.sample_data {
        FlatSamples::F32(v) => v.clone(),
        FlatSamples::F16(v) => v.iter().map(|h| h.to_f32()).collect(),
        FlatSamples::U32(v) => v.iter().map(|&u| u as f32 / u32::MAX as f32).collect(),
    };

    Ok((pixels, w, h))
}

// ---------------------------------------------------------------------------
// Mip / normal helpers  (unchanged from original)
// ---------------------------------------------------------------------------

fn box_filter_2x(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let dw = w / 2;
    let dh = h / 2;
    let mut dst = vec![0.0f32; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx * 2;
            let sy = dy * 2;
            dst[dy * dw + dx] = (src[sy * w + sx]
                + src[sy * w + sx + 1]
                + src[(sy + 1) * w + sx]
                + src[(sy + 1) * w + sx + 1])
                * 0.25;
        }
    }
    dst
}

fn sample_height_clamped(src: &[f32], size: usize, x: usize, y: usize) -> f32 {
    src[y.min(size - 1) * size + x.min(size - 1)]
}

fn compute_normal(
    src: &[f32],
    size: usize,
    x: usize,
    y: usize,
    level_scale_ws: f32,
    height_scale: f32,
) -> [f32; 3] {
    let h = sample_height_clamped(src, size, x, y) * height_scale;
    let h_r = sample_height_clamped(src, size, x.saturating_add(1), y) * height_scale;
    let h_u = sample_height_clamped(src, size, x, y.saturating_add(1)) * height_scale;
    let dx = h - h_r;
    let dz = h - h_u;
    let len = (dx * dx + level_scale_ws * level_scale_ws + dz * dz)
        .sqrt()
        .max(1e-6);
    [dx / len, level_scale_ws / len, dz / len]
}

fn encode_normal_xz(normal: [f32; 3]) -> [u8; 2] {
    [
        (normal[0].clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
        (normal[2].clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
    ]
}

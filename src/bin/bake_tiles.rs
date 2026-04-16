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
//!   --height-scale <f32>  World-space height range in units.
//!                         If omitted, read from landscape.toml
//!                         [terrain_config] height_scale, then default 1024.0.
//!                         MUST match the runtime height_scale used by the
//!                         renderer — affects baked normal steepness.
//!   --bump-scale <f32>    World-space scale for bump normal derivation
//!                         [default: same as height-scale]
//!   --world-scale <f32>   World-space units per texel at LOD 0  [default: 1.0]
//!   --tile-size <usize>   Tile resolution in pixels  [default: 256]
//!   --smooth-sigma <f32>  Gaussian blur sigma (in source texels) applied to
//!                         the heightmap before mip pyramid + normal
//!                         derivation.  0 = off (default).  ~1.0 removes
//!                         single-texel outliers without softening real
//!                         topography.

use exr::prelude::{read_first_flat_layer_from_file, FlatSamples};
use image::ImageReader;
use serde::Deserialize;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// landscape.toml reader (height_scale only)
// ---------------------------------------------------------------------------

/// Reads `landscape.toml` from the current working directory and returns the
/// `[terrain_config] height_scale` value if present.
fn height_scale_from_toml() -> Option<f32> {
    #[derive(Deserialize)]
    struct Root {
        terrain_config: Option<TerrainCfg>,
    }
    #[derive(Deserialize)]
    struct TerrainCfg {
        height_scale: Option<f32>,
    }

    let text = std::fs::read_to_string("landscape.toml").ok()?;
    let root: Root = toml::from_str(&text).ok()?;
    root.terrain_config?.height_scale
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    height_path: PathBuf,
    bump_path: Option<PathBuf>,
    output_dir: PathBuf,
    /// None = not set via CLI; resolved in main() from landscape.toml or default.
    height_scale: Option<f32>,
    bump_scale: Option<f32>,
    world_scale: f32,
    tile_size: usize,
    /// Negate the G (bitangent) channel of an RGB normal map.
    /// Use for OpenGL-convention maps (G points toward UV top = world -Z).
    /// Default false = DirectX convention (G toward UV bottom = world +Z).
    flip_green: bool,
    /// Gaussian blur sigma (in source texels) applied to the heightmap *before*
    /// the mip pyramid is built and *before* baked normals are derived.  0 =
    /// off.  Removes single-texel outliers that turn into pyramid spikes at
    /// LOD 0; values around 1.0 work well for natural-looking terrain.
    smooth_sigma: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            height_path: PathBuf::new(),
            bump_path: None,
            output_dir: PathBuf::from("assets/tiles"),
            height_scale: None,
            bump_scale: None,
            world_scale: 1.0,
            tile_size: 256,
            flip_green: false,
            smooth_sigma: 0.0,
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
                cfg.height_scale = Some(next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?);
                i += 2;
            }
            "--bump-scale" => {
                cfg.bump_scale = Some(next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?);
                i += 2;
            }
            "--world-scale" => {
                cfg.world_scale = next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?;
                i += 2;
            }
            "--tile-size" => {
                cfg.tile_size = next()?
                    .parse::<usize>()
                    .map_err(|e| format!("{flag}: {e}"))?;
                i += 2;
            }
            "--flip-green" => {
                // Negate the G channel when the RGB normal map uses OpenGL
                // convention (G = toward UV top = world -Z).
                cfg.flip_green = true;
                i += 1;
            }
            "--smooth-sigma" => {
                cfg.smooth_sigma =
                    next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?;
                if cfg.smooth_sigma < 0.0 {
                    return Err(format!("--smooth-sigma must be >= 0, got {}", cfg.smooth_sigma));
                }
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
    let mut cfg = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!();
        eprintln!("Usage: bake_tiles --height <path> [--bump <path>] [--output <dir>]");
        eprintln!("                  [--height-scale <f>] [--bump-scale <f>]");
        eprintln!("                  [--world-scale <f>] [--tile-size <n>]");
        eprintln!("                  [--smooth-sigma <f>] [--flip-green]");
        std::process::exit(1);
    });

    // Resolve height_scale: CLI flag → landscape.toml → hardcoded default.
    // The default matches TerrainConfig::height_scale in config.rs so that
    // baking without an explicit flag produces normals consistent with the
    // renderer's default.
    let height_scale: f32 = cfg.height_scale.unwrap_or_else(|| {
        if let Some(v) = height_scale_from_toml() {
            println!("height_scale: using {v} from landscape.toml");
            v
        } else {
            let v = 1024.0_f32;
            println!("height_scale: using default {v} (set via --height-scale or landscape.toml to override)");
            v
        }
    });
    cfg.height_scale = Some(height_scale);

    let t_start = Instant::now();

    if !cfg.height_path.exists() {
        eprintln!("ERROR: height map not found at {:?}", cfg.height_path);
        eprintln!("Run this tool from the workspace root (next to 'assets/').");
        std::process::exit(1);
    }

    println!("Loading height map: {} ...", cfg.height_path.display());
    let (height_pixels_raw, img_w, img_h) = load_grayscale_image(&cfg.height_path)?;
    println!(
        "Loaded {}×{} in {:.1}s",
        img_w,
        img_h,
        t_start.elapsed().as_secs_f32()
    );

    if img_w != img_h {
        return Err(format!("Expected square heightmap, got {}×{}", img_w, img_h).into());
    }

    // Normalise height to [0, 1].
    let h_min = height_pixels_raw
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let h_max = height_pixels_raw
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let h_range = h_max - h_min;
    println!(
        "Height range: min={:.6}  max={:.6}  range={:.6}",
        h_min, h_max, h_range
    );
    let mut height_pixels: Vec<f32> = if h_range > 1e-6 {
        height_pixels_raw
            .iter()
            .map(|&h| (h - h_min) / h_range)
            .collect()
    } else {
        height_pixels_raw
    };

    // Optional Gaussian pre-smoothing at native resolution.
    //
    // Applied *before* the mip pyramid and normal derivation so coarse LODs
    // and baked normals all describe the same smoothed surface the renderer
    // will display.  Single-texel outliers in the source (very common in
    // authored "rocky terrain" 16k heightmaps) show up as pyramid spikes at
    // LOD 0 because the clipmap's vertex spacing is 1:1 with the source;
    // low-passing here removes them at the cost of sub-texel detail that
    // the renderer cannot resolve anyway.
    if cfg.smooth_sigma > 0.0 {
        let t = Instant::now();
        print!("Smoothing heightmap (sigma={:.2})... ", cfg.smooth_sigma);
        let _ = std::io::stdout().flush();
        height_pixels = gaussian_blur_f32(&height_pixels, img_w, img_h, cfg.smooth_sigma);
        println!("done in {:.1}s", t.elapsed().as_secs_f32());
    }

    // Load bump map for normal derivation if provided.
    // Auto-detects type: single-channel images → displacement (finite-difference
    // normals); RGB images → tangent-space normal map (TBN world-space transform).
    let (bump_height, bump_normal): (Option<Vec<f32>>, Option<Vec<[f32; 2]>>) =
        if let Some(ref bump_path) = cfg.bump_path {
            if !bump_path.exists() {
                eprintln!("ERROR: bump map not found at {:?}", bump_path);
                std::process::exit(1);
            }
            println!("Loading bump map: {} ...", bump_path.display());
            let (mut bheight, bnormal, bw, bh) = load_bump_map(bump_path)?;
            println!("  {}×{}", bw, bh);
            if bw != img_w || bh != img_h {
                return Err(format!(
                    "Bump map {}×{} must match height map {}×{}",
                    bw, bh, img_w, img_h
                )
                .into());
            }
            // Normalise grayscale displacement to [0, 1].
            if let Some(ref mut pix) = bheight {
                let bmin = pix.iter().cloned().fold(f32::INFINITY, f32::min);
                let bmax = pix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let brange = bmax - bmin;
                if brange > 1e-6 {
                    pix.iter_mut().for_each(|v| *v = (*v - bmin) / brange);
                }
            }
            (bheight, bnormal)
        } else {
            (None, None)
        };

    // bump_scale is only used for grayscale displacement bump maps.
    let bump_scale = cfg.bump_scale.unwrap_or(height_scale);

    let tile_size = cfg.tile_size;

    // Auto-compute level count.
    //
    // Stop before tiles_per_side drops below 2.  When tiles_per_side==1 the
    // single tile's grid range doesn't align with the streamer's tile-coordinate
    // system: the terrain centred at the world origin always straddles the
    // tile-0 / tile-(-1) boundary, so you'd need 4 tiles (2×2) even though the
    // mip image is only 1×1 tiles wide.  Stopping at tiles_per_side==2 keeps the
    // coarsest level at a clean 2×2 tile grid that matches the streamer.
    let levels = {
        let mut sz = img_w;
        let mut n = 0u32;
        while sz / tile_size >= 2 {
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
    let mut current_bump_height = bump_height;
    let mut current_bump_normal = bump_normal;
    let mut current_size = img_w;
    let mut total_tiles = 0usize;

    for lod in 0..levels {
        let mip_size = current_size;
        let mip_half = (mip_size / 2) as i32;
        let tiles_per_side = (mip_size / tile_size) as i32;
        // tile_start/end keeps the grid centred on the world origin.
        // Using `tiles_per_side / 2` for both edges handles the odd case
        // (tiles_per_side == 1) correctly: start=0, end=1 → one tile at (0,0).
        // The old `for t in -half..half` form produced an empty range when
        // tiles_per_side was 1 because -0..0 == 0..0.
        let tile_start = -(tiles_per_side / 2);
        let tile_end = tile_start + tiles_per_side;

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

        for ty in tile_start..tile_end {
            for tx in tile_start..tile_end {
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
                            if let Some(ref bts) = current_bump_normal {
                                // RGB tangent-space normal map: TBN transform.
                                encode_normal_xz(compute_normal_from_ts(
                                    bts,
                                    &current_height,
                                    cfg.flip_green,
                                    mip_size,
                                    px,
                                    py,
                                    lod_scale,
                                    height_scale,
                                ))
                            } else {
                                // Grayscale displacement or height-derived normals.
                                let normal_src =
                                    current_bump_height.as_deref().unwrap_or(&current_height);
                                let normal_scale = if current_bump_height.is_some() {
                                    bump_scale
                                } else {
                                    height_scale
                                };
                                encode_normal_xz(compute_normal(
                                    normal_src,
                                    mip_size,
                                    px,
                                    py,
                                    lod_scale,
                                    normal_scale,
                                ))
                            }
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
                print!(
                    "\r  Level 0: {}%  ({}/{})",
                    pct, level_done, level_tile_count
                );
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
            if let Some(ref b) = current_bump_height {
                let next_b = box_filter_2x(b, current_size, current_size);
                current_bump_height = Some(next_b);
            }
            if let Some(ref b) = current_bump_normal {
                let next_b = box_filter_2x_normal_ts(b, current_size, current_size);
                current_bump_normal = Some(next_b);
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
fn load_height_exr(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let image = read_first_flat_layer_from_file(path)?;
    let layer = &image.layer_data;
    let w = layer.size.x();
    let h = layer.size.y();

    println!("EXR channels ({}):", layer.channel_data.list.len());
    for ch in &layer.channel_data.list {
        println!("  {:?}", ch.name);
    }

    // EXR channels are stored alphabetically (A, B, G, R for an RGBA image),
    // so .first() returns the alpha channel (always 1.0) on multi-channel EXRs,
    // producing all-maximum-height terrain.
    // Priority: Y (standard grayscale EXR) → R → first non-alpha → fallback to first.
    let channel = layer
        .channel_data
        .list
        .iter()
        .find(|ch| ch.name.to_string().to_lowercase() == "y")
        .or_else(|| {
            layer
                .channel_data
                .list
                .iter()
                .find(|ch| ch.name.to_string().to_lowercase() == "r")
        })
        .or_else(|| {
            layer
                .channel_data
                .list
                .iter()
                .find(|ch| ch.name.to_string().to_lowercase() != "a")
        })
        .or_else(|| layer.channel_data.list.first())
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
// Mip / normal helpers
// ---------------------------------------------------------------------------

/// Loads a bump map and auto-detects its type:
///   - Single-channel (L8/L16) → grayscale displacement; use finite-difference normals.
///   - RGB/RGBA             → tangent-space normal map; decode R,G as XY [-1,1].
///
/// Returns `(height_pixels, normal_pixels_xy, width, height)`.
fn load_bump_map(
    path: &Path,
) -> Result<(Option<Vec<f32>>, Option<Vec<[f32; 2]>>, usize, usize), Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?;
    let w = img.width() as usize;
    let h = img.height() as usize;

    let is_gray = matches!(
        img.color(),
        image::ColorType::L8
            | image::ColorType::L16
            | image::ColorType::La8
            | image::ColorType::La16
    );

    if is_gray {
        println!("  Detected single-channel — displacement map, using finite-difference normals");
        let luma = img.into_luma16();
        let pixels: Vec<f32> = luma.pixels().map(|p| p[0] as f32 / 65535.0).collect();
        Ok((Some(pixels), None, w, h))
    } else {
        println!("  Detected RGB — tangent-space normal map, applying TBN world-space transform");
        let rgb = img.into_rgb8();
        // Decode R → tangent X, G → bitangent Y, both in [-1, 1].
        // B encodes the surface-normal component; it is NOT stored because the
        // shader already reconstructs ny_world = sqrt(1 - nx^2 - nz^2).
        let pixels: Vec<[f32; 2]> = rgb
            .pixels()
            .map(|p| [(p[0] as f32 - 128.0) / 128.0, (p[1] as f32 - 128.0) / 128.0])
            .collect();
        Ok((None, Some(pixels), w, h))
    }
}

/// Box-filters a tangent-space XY normal map down by 2× in each dimension.
/// Components are averaged; the Z component is not stored and is reconstructed
/// from XY at point of use.
fn box_filter_2x_normal_ts(src: &[[f32; 2]], w: usize, h: usize) -> Vec<[f32; 2]> {
    let dw = w / 2;
    let dh = h / 2;
    let mut dst = vec![[0.0f32; 2]; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx * 2;
            let sy = dy * 2;
            let [ax, ay] = src[sy * w + sx];
            let [bx, by] = src[sy * w + sx + 1];
            let [cx, cy] = src[(sy + 1) * w + sx];
            let [ddx, ddy] = src[(sy + 1) * w + sx + 1];
            dst[dy * dw + dx] = [(ax + bx + cx + ddx) * 0.25, (ay + by + cy + ddy) * 0.25];
        }
    }
    dst
}

/// Converts a tangent-space normal map sample into a world-space normal by
/// building the TBN frame from the height field at the same surface point.
///
/// `bump_ts`      — tangent-space XY components in \[-1, 1\]; Z is reconstructed.
/// `height_src`   — normalised \[0, 1\] height values at the same mip level.
/// `flip_green`   — negate Y for OpenGL-convention maps (G toward UV-top = world -Z).
///                  Leave false for DirectX convention (G toward UV-bottom = world +Z).
/// `lod_scale`    — world units per texel at this mip level.
/// `height_scale` — world-space units for a full \[0→1\] height range.
fn compute_normal_from_ts(
    bump_ts: &[[f32; 2]],
    height_src: &[f32],
    flip_green: bool,
    size: usize,
    px: usize,
    py: usize,
    lod_scale: f32,
    height_scale: f32,
) -> [f32; 3] {
    // Decode tangent-space XY; reconstruct Z = sqrt(1 - x² - y²).
    let [ts_x, ts_y_raw] = bump_ts[py * size + px];
    let ts_y = if flip_green { -ts_y_raw } else { ts_y_raw };
    let ts_z = (1.0_f32 - ts_x * ts_x - ts_y * ts_y).max(0.0).sqrt();

    // Compute the terrain surface TBN from a Sobel-smoothed height gradient
    // rather than a raw one-sided forward difference — a noisy heightfield
    // would otherwise jitter the TBN frame from texel to texel and poison the
    // world-space normal it transforms into.
    let (gx, gz) = sobel_height_gradient(height_src, size, px, py, height_scale);

    // Tangent T = ∂P/∂u = direction of increasing world X.
    let tl = (lod_scale * lod_scale + gx * gx).sqrt().max(1e-6);
    let t = [lod_scale / tl, gx / tl, 0.0_f32];

    // Bitangent B = ∂P/∂v = direction of increasing world Z.
    let bl = (lod_scale * lod_scale + gz * gz).sqrt().max(1e-6);
    let b = [0.0_f32, gz / bl, lod_scale / bl];

    // Surface normal N = B × T (right-hand rule, Y-up: gives (0,1,0) for flat terrain).
    let nx = b[1] * t[2] - b[2] * t[1];
    let ny = b[2] * t[0] - b[0] * t[2];
    let nz = b[0] * t[1] - b[1] * t[0];
    let nl = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
    let n = [nx / nl, ny / nl, nz / nl];

    // World-space normal = T * ts_x + B * ts_y + N * ts_z.
    let wx = t[0] * ts_x + b[0] * ts_y + n[0] * ts_z;
    let wy = t[1] * ts_x + b[1] * ts_y + n[1] * ts_z;
    let wz = t[2] * ts_x + b[2] * ts_y + n[2] * ts_z;

    let wl = (wx * wx + wy * wy + wz * wz).sqrt().max(1e-6);
    [wx / wl, wy / wl, wz / wl]
}

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

/// Sample one offset of a 3×3 stencil with edge clamping.
#[inline]
fn sample_offset(src: &[f32], size: usize, x: usize, y: usize, dx: i32, dy: i32) -> f32 {
    let sx = (x as i32 + dx).clamp(0, size as i32 - 1) as usize;
    let sy = (y as i32 + dy).clamp(0, size as i32 - 1) as usize;
    src[sy * size + sx]
}

/// Sobel-weighted height gradient at (x, y).
///
/// Returns `(dh_dx, dh_dz)` in world-height units **per texel** (i.e. already
/// multiplied by `height_scale` but not divided by `level_scale_ws`).  Caller
/// builds the unnormalized normal as `(-dh_dx, level_scale_ws, -dh_dz)`.
///
/// The 1-2-1 perpendicular weighting averages out single-texel noise that a
/// plain central difference (or a one-sided forward diff) would otherwise
/// inject straight into the normal — eliminating the per-texel jaggies that
/// noisy heightmaps produce on otherwise smooth slopes.
fn sobel_height_gradient(
    src: &[f32],
    size: usize,
    x: usize,
    y: usize,
    height_scale: f32,
) -> (f32, f32) {
    // Sobel X kernel (gradient along +X, i.e. across columns):
    //   -1  0  +1
    //   -2  0  +2
    //   -1  0  +1
    //
    // Sobel Z kernel (gradient along +Z, i.e. across rows):
    //   -1 -2 -1
    //    0  0  0
    //   +1 +2 +1
    //
    // Both kernels sum to 8 in absolute weight on each side, so we divide by 8
    // to recover the per-texel gradient (matches a central difference on a
    // ramp h(x,z) = a·x, which gives dh/dx = a per texel).
    let h = |dx: i32, dy: i32| sample_offset(src, size, x, y, dx, dy);

    let gx = (h(1, -1) + 2.0 * h(1, 0) + h(1, 1)
        - h(-1, -1) - 2.0 * h(-1, 0) - h(-1, 1))
        * (1.0 / 8.0);
    let gz = (h(-1, 1) + 2.0 * h(0, 1) + h(1, 1)
        - h(-1, -1) - 2.0 * h(0, -1) - h(1, -1))
        * (1.0 / 8.0);

    (gx * height_scale, gz * height_scale)
}

fn compute_normal(
    src: &[f32],
    size: usize,
    x: usize,
    y: usize,
    level_scale_ws: f32,
    height_scale: f32,
) -> [f32; 3] {
    // Unnormalized normal:  N = T_z × T_x = (-dh/dx, level_scale, -dh/dz),
    // expressed with dh in world height units per texel and the spacing
    // factored out so the X/Z components compare directly against
    // `level_scale_ws` (1 texel of world spacing).
    let (gx, gz) = sobel_height_gradient(src, size, x, y, height_scale);
    let dx = -gx;
    let dz = -gz;
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

// ---------------------------------------------------------------------------
// Separable Gaussian blur (f32 single-channel)
//
// Two 1-D passes — horizontal then vertical — with clamp-to-edge borders.
// Rows are partitioned across worker threads via `std::thread::scope` so a
// 16k² bake completes in a few seconds instead of tens.  No new crate
// dependency required.
// ---------------------------------------------------------------------------

fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    // 3σ captures >99 % of the Gaussian; rounding up keeps the kernel odd.
    let radius = (3.0 * sigma).ceil() as i32;
    let width = (2 * radius + 1) as usize;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut k = Vec::with_capacity(width);
    let mut sum = 0.0f32;
    for i in 0..width {
        let x = i as i32 - radius;
        let v = (-(x as f32 * x as f32) / two_sigma_sq).exp();
        k.push(v);
        sum += v;
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

fn gaussian_blur_f32(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let kernel = gaussian_kernel_1d(sigma);
    let radius = (kernel.len() / 2) as i32;

    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);

    // --- Horizontal pass: read `src`, write `tmp`. ---
    let mut tmp = vec![0.0f32; src.len()];
    {
        let chunk = (h + thread_count - 1) / thread_count;
        let tmp_chunks: Vec<&mut [f32]> = tmp.chunks_mut(chunk * w).collect();
        std::thread::scope(|s| {
            for (i, out) in tmp_chunks.into_iter().enumerate() {
                let kernel = &kernel;
                let src = &src;
                let y0 = i * chunk;
                let rows = out.len() / w;
                s.spawn(move || {
                    for dy in 0..rows {
                        let y = y0 + dy;
                        for x in 0..w {
                            let mut acc = 0.0f32;
                            for (ki, &kv) in kernel.iter().enumerate() {
                                let sx =
                                    (x as i32 + ki as i32 - radius).clamp(0, w as i32 - 1) as usize;
                                acc += src[y * w + sx] * kv;
                            }
                            out[dy * w + x] = acc;
                        }
                    }
                });
            }
        });
    }

    // --- Vertical pass: read `tmp`, write `dst`. ---
    let mut dst = vec![0.0f32; src.len()];
    {
        let chunk = (h + thread_count - 1) / thread_count;
        let dst_chunks: Vec<&mut [f32]> = dst.chunks_mut(chunk * w).collect();
        std::thread::scope(|s| {
            for (i, out) in dst_chunks.into_iter().enumerate() {
                let kernel = &kernel;
                let tmp = &tmp;
                let y0 = i * chunk;
                let rows = out.len() / w;
                s.spawn(move || {
                    for dy in 0..rows {
                        let y = y0 + dy;
                        for x in 0..w {
                            let mut acc = 0.0f32;
                            for (ki, &kv) in kernel.iter().enumerate() {
                                let sy =
                                    (y as i32 + ki as i32 - radius).clamp(0, h as i32 - 1) as usize;
                                acc += tmp[sy * w + x] * kv;
                            }
                            out[dy * w + x] = acc;
                        }
                    }
                });
            }
        });
    }

    dst
}

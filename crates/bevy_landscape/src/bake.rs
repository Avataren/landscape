//! Heightmap-to-tile baker — usable both as a library and via the `bake_tiles` binary.
//!
//! Call [`bake_heightmap`] with a [`BakeConfig`] and a log callback.  All
//! progress and error messages are delivered through the callback so callers
//! can forward them to stdout, a file, or an egui log panel.

use exr::prelude::{read_first_flat_layer_from_file, FlatSamples};
use image::ImageReader;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// All inputs needed for a single bake run.
#[derive(Clone, Debug)]
pub struct BakeConfig {
    /// Path to the heightmap image (EXR, PNG, TIFF).
    pub height_path: PathBuf,
    /// Optional bump/normal map.  Single-channel → displacement, RGB → tangent-space.
    pub bump_path: Option<PathBuf>,
    /// Where to write the tile hierarchy (`height/L{n}/` and `normal/L{n}/`).
    pub output_dir: PathBuf,
    /// Base Y world-space range: R16Unorm = 1.0 maps to this many world units
    /// *before* `world_scale` is applied.
    pub height_scale: f32,
    /// Override the height scale used for bump-derived normals.  Defaults to
    /// `height_scale * world_scale` when `None`.
    pub bump_scale: Option<f32>,
    /// Uniform world scale multiplier applied on top of `height_scale`.
    pub world_scale: f32,
    /// Tile resolution in pixels (must be a power of two; default 256).
    pub tile_size: usize,
    /// Negate the G channel of an RGB normal map.  Set for OpenGL-convention
    /// maps (G toward UV top = world -Z).  Leave false for DirectX convention.
    pub flip_green: bool,
    /// Gaussian sigma (source texels) applied before building the mip pyramid.
    /// 0 = off.  ~1.0 removes single-texel outliers without softening real detail.
    pub smooth_sigma: f32,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            height_path: PathBuf::new(),
            bump_path: None,
            output_dir: PathBuf::from("assets/tiles"),
            height_scale: 1024.0,
            bump_scale: None,
            world_scale: 1.0,
            tile_size: 256,
            flip_green: false,
            smooth_sigma: 0.0,
        }
    }
}

/// Run the full mip-pyramid bake pipeline.
///
/// All progress messages are delivered via `log`.  Returns `Ok(())` on
/// success or an `Err(message)` string on failure.
pub fn bake_heightmap(config: BakeConfig, log: impl Fn(String)) -> Result<(), String> {
    let t_start = Instant::now();
    let elapsed = || t_start.elapsed().as_secs_f32();

    if config.height_path.as_os_str().is_empty() {
        return Err("height_path is empty".into());
    }
    if !config.height_path.exists() {
        return Err(format!(
            "height map not found: {}",
            config.height_path.display()
        ));
    }

    log(format!(
        "Loading height map: {} …",
        config.height_path.display()
    ));
    let (height_pixels_raw, img_w, img_h) =
        load_grayscale_image(&config.height_path).map_err(|e| e.to_string())?;
    log(format!("Loaded {}×{} in {:.1}s", img_w, img_h, elapsed()));

    if img_w != img_h {
        return Err(format!(
            "Expected square heightmap, got {}×{}",
            img_w, img_h
        ));
    }

    let bake_size = derive_bake_extent(img_w, config.tile_size)?;
    if bake_size != img_w {
        log(format!(
            "Trimming source border sample: baking {}×{} as {}×{}.",
            img_w, img_h, bake_size, bake_size
        ));
    }

    let effective_height_scale = config.height_scale * config.world_scale;
    let height_pixels_raw = crop_square_border(&height_pixels_raw, img_w, bake_size);

    // Normalise to [0, 1].
    let h_min = height_pixels_raw
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let h_max = height_pixels_raw
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let h_range = h_max - h_min;
    log(format!(
        "Height range: min={:.6}  max={:.6}  range={:.6}",
        h_min, h_max, h_range
    ));
    let mut height_pixels: Vec<f32> = if h_range > 1e-6 {
        height_pixels_raw
            .iter()
            .map(|&h| (h - h_min) / h_range)
            .collect()
    } else {
        height_pixels_raw
    };

    if config.smooth_sigma > 0.0 {
        log(format!(
            "Smoothing heightmap (sigma={:.2}) …",
            config.smooth_sigma
        ));
        height_pixels =
            gaussian_blur_f32(&height_pixels, bake_size, bake_size, config.smooth_sigma);
        log(format!("Smoothing done in {:.1}s", elapsed()));
    }

    let (bump_height, bump_normal): (Option<Vec<f32>>, Option<Vec<[f32; 2]>>) =
        if let Some(ref bump_path) = config.bump_path {
            if !bump_path.exists() {
                return Err(format!("bump map not found: {}", bump_path.display()));
            }
            log(format!("Loading bump map: {} …", bump_path.display()));
            let (mut bheight, mut bnormal, bw, bh) =
                load_bump_map(bump_path).map_err(|e| e.to_string())?;
            log(format!("  {}×{}", bw, bh));
            if bw != img_w || bh != img_h {
                return Err(format!(
                    "Bump map {}×{} must match height map {}×{}",
                    bw, bh, img_w, img_h
                ));
            }
            if bake_size != img_w {
                if let Some(ref mut pix) = bheight {
                    *pix = crop_square_border(pix, bw, bake_size);
                }
                if let Some(ref mut pix) = bnormal {
                    *pix = crop_square_border(pix, bw, bake_size);
                }
            }
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
            log("No bump map provided — deriving normals from heightmap.".into());
            (None, None)
        };

    let bump_scale = config.bump_scale.unwrap_or(config.height_scale) * config.world_scale;
    let tile_size = config.tile_size;

    let levels = {
        let mut sz = bake_size;
        let mut n = 0u32;
        while sz / tile_size >= 2 {
            n += 1;
            sz /= 2;
        }
        n
    };

    if levels == 0 {
        return Err(format!(
            "Heightmap resolution {} is too small for {}px tiles. Use at least {} or {} samples.",
            img_w,
            tile_size,
            tile_size * 2,
            tile_size * 2 + 1
        ));
    }

    prepare_output_dir(&config.output_dir, &log)?;
    log(format!(
        "Baking {} levels, tile size {}px → '{}'",
        levels,
        tile_size,
        config.output_dir.display()
    ));

    let mut current_height = height_pixels;
    let mut current_bump_height = bump_height;
    let mut current_bump_normal = bump_normal;
    let mut current_size = bake_size;
    let mut total_tiles = 0usize;

    for lod in 0..levels {
        let mip_size = current_size;
        let mip_half = (mip_size / 2) as i32;
        let tiles_per_side = (mip_size / tile_size) as i32;
        let tile_start = -(tiles_per_side / 2);
        let tile_end = tile_start + tiles_per_side;

        if tiles_per_side == 0 {
            log(format!(
                "WARNING: mip_size {} < tile_size {} at LOD {}; stopping.",
                mip_size, tile_size, lod
            ));
            break;
        }

        let height_level_dir = config.output_dir.join(format!("height/L{}", lod));
        let normal_level_dir = config.output_dir.join(format!("normal/L{}", lod));
        std::fs::create_dir_all(&height_level_dir).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&normal_level_dir).map_err(|e| e.to_string())?;

        let level_tile_count = (tiles_per_side * tiles_per_side) as usize;
        let mut level_done = 0usize;
        let lod_scale = config.world_scale * (1u32 << lod) as f32;

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
                                encode_normal_xz(compute_normal_from_ts(
                                    bts,
                                    &current_height,
                                    config.flip_green,
                                    mip_size,
                                    px,
                                    py,
                                    lod_scale,
                                    effective_height_scale,
                                ))
                            } else {
                                let normal_src =
                                    current_bump_height.as_deref().unwrap_or(&current_height);
                                let normal_scale = if current_bump_height.is_some() {
                                    bump_scale
                                } else {
                                    effective_height_scale
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
                std::fs::write(&hpath, &tile_bytes).map_err(|e| e.to_string())?;
                std::fs::write(&npath, &normal_bytes).map_err(|e| e.to_string())?;

                level_done += 1;
                total_tiles += 1;
            }

            if lod == 0 {
                let pct = level_done * 100 / level_tile_count;
                log(format!(
                    "Level 0: {}%  ({}/{})",
                    pct, level_done, level_tile_count
                ));
            }
        }

        log(format!(
            "Level {}: {}×{} = {} tiles  [{:.1}s]",
            lod,
            tiles_per_side,
            tiles_per_side,
            level_tile_count,
            elapsed(),
        ));

        if lod + 1 < levels {
            let next_h = box_filter_2x(&current_height, current_size, current_size);
            current_height = next_h;
            if let Some(ref b) = current_bump_height {
                current_bump_height = Some(box_filter_2x(b, current_size, current_size));
            }
            if let Some(ref b) = current_bump_normal {
                current_bump_normal = Some(box_filter_2x_normal_ts(b, current_size, current_size));
            }
            current_size /= 2;
        }
    }

    log(format!(
        "Done. {} tiles → '{}'  ({:.1}s total)",
        total_tiles,
        config.output_dir.display(),
        elapsed(),
    ));
    Ok(())
}

fn prepare_output_dir(output_dir: &Path, log: &impl Fn(String)) -> Result<(), String> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "failed to create output directory '{}': {e}",
            output_dir.display()
        )
    })?;

    for subdir in ["height", "normal"] {
        let path = output_dir.join(subdir);
        if !path.exists() {
            continue;
        }
        std::fs::remove_dir_all(&path)
            .map_err(|e| format!("failed to clear stale '{}': {e}", path.display()))?;
        log(format!("Cleared stale '{}'.", path.display()));
    }

    Ok(())
}

fn derive_bake_extent(source_size: usize, tile_size: usize) -> Result<usize, String> {
    if tile_size == 0 || !tile_size.is_power_of_two() {
        return Err(format!(
            "tile_size must be a non-zero power of two, got {}",
            tile_size
        ));
    }

    let bake_size = if source_size % tile_size == 0 {
        source_size
    } else if source_size > 1 && (source_size - 1) % tile_size == 0 {
        source_size - 1
    } else {
        return Err(format!(
            "Unsupported heightmap resolution {}. Expected a square image whose width is either a multiple of {} or that value plus one (for example 4096 or 4097).",
            source_size, tile_size
        ));
    };

    let tiles_per_side = bake_size / tile_size;
    if tiles_per_side < 2 {
        return Err(format!(
            "Heightmap resolution {} is too small for {}px tiles. Use at least {} or {} samples.",
            source_size,
            tile_size,
            tile_size * 2,
            tile_size * 2 + 1
        ));
    }
    if !tiles_per_side.is_power_of_two() {
        return Err(format!(
            "Unsupported heightmap resolution {}. It produces {} tiles per side, but the baker requires a power-of-two tile count (for example 512, 1024, 2048, 4096, or 4097).",
            source_size, tiles_per_side
        ));
    }

    Ok(bake_size)
}

fn crop_square_border<T: Copy>(src: &[T], src_size: usize, dst_size: usize) -> Vec<T> {
    if src_size == dst_size {
        return src.to_vec();
    }

    let mut out = Vec::with_capacity(dst_size * dst_size);
    for row in 0..dst_size {
        let start = row * src_size;
        out.extend_from_slice(&src[start..start + dst_size]);
    }
    out
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

pub fn load_grayscale_image(
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

fn load_grayscale_raster(
    path: &Path,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    // Try the tiff crate first so we can handle signed INT16 TIFFs (sample
    // format 2), which the `image` crate rejects.
    if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff")).unwrap_or(false) {
        if let Ok(pixels) = load_tiff_as_f32(path) {
            return Ok(pixels);
        }
    }

    let img = ImageReader::open(path)?.decode()?;
    let w = img.width() as usize;
    let h = img.height() as usize;
    let luma = img.into_luma16();
    let pixels = luma.pixels().map(|p| p[0] as f32 / 65535.0).collect();
    Ok((pixels, w, h))
}

/// Read a single-band TIFF as normalised f32, handling UINT16, INT16, UINT32,
/// INT32, and FLOAT32 sample formats.
fn load_tiff_as_f32(
    path: &Path,
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::ColorType;

    let file = BufReader::new(File::open(path)?);
    let mut dec = Decoder::new(file)?;

    let (w, h) = dec.dimensions()?;
    let color = dec.colortype()?;

    // Only handle single-band (grayscale) images here.
    match color {
        ColorType::Gray(_) => {}
        _ => {
            return Err(format!(
                "Expected a single-band TIFF, got {color:?}"
            )
            .into());
        }
    }

    let result = dec.read_image()?;
    let pixels: Vec<f32> = match result {
        DecodingResult::U8(v)  => v.iter().map(|&x| x as f32 / u8::MAX as f32).collect(),
        DecodingResult::U16(v) => v.iter().map(|&x| x as f32 / u16::MAX as f32).collect(),
        DecodingResult::U32(v) => v.iter().map(|&x| x as f32 / u32::MAX as f32).collect(),
        DecodingResult::U64(v) => v.iter().map(|&x| x as f32 / u64::MAX as f32).collect(),
        DecodingResult::I8(v)  => v.iter().map(|&x| (x as f32 - i8::MIN as f32)  / u8::MAX as f32).collect(),
        DecodingResult::I16(v) => v.iter().map(|&x| (x as f32 - i16::MIN as f32) / u16::MAX as f32).collect(),
        DecodingResult::I32(v) => v.iter().map(|&x| (x as f32 - i32::MIN as f32) / u32::MAX as f32).collect(),
        DecodingResult::I64(v) => v.iter().map(|&x| (x as f64 - i64::MIN as f64) as f32 / u64::MAX as f32).collect(),
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.iter().map(|&x| x as f32).collect(),
        DecodingResult::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
    };

    Ok((pixels, w as usize, h as usize))
}

fn load_height_exr(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let image = read_first_flat_layer_from_file(path)?;
    let layer = &image.layer_data;
    let w = layer.size.x();
    let h = layer.size.y();

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

    let pixels: Vec<f32> = match &channel.sample_data {
        FlatSamples::F32(v) => v.clone(),
        FlatSamples::F16(v) => v.iter().map(|h| h.to_f32()).collect(),
        FlatSamples::U32(v) => v.iter().map(|&u| u as f32 / u32::MAX as f32).collect(),
    };

    Ok((pixels, w, h))
}

pub fn load_bump_map(
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
        let luma = img.into_luma16();
        let pixels: Vec<f32> = luma.pixels().map(|p| p[0] as f32 / 65535.0).collect();
        Ok((Some(pixels), None, w, h))
    } else {
        let rgb = img.into_rgb8();
        let pixels: Vec<[f32; 2]> = rgb
            .pixels()
            .map(|p| [(p[0] as f32 - 128.0) / 128.0, (p[1] as f32 - 128.0) / 128.0])
            .collect();
        Ok((None, Some(pixels), w, h))
    }
}

// ---------------------------------------------------------------------------
// Normal + mip helpers
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

#[inline]
fn sample_offset(src: &[f32], size: usize, x: usize, y: usize, dx: i32, dy: i32) -> f32 {
    let sx = (x as i32 + dx).clamp(0, size as i32 - 1) as usize;
    let sy = (y as i32 + dy).clamp(0, size as i32 - 1) as usize;
    src[sy * size + sx]
}

fn sobel_height_gradient(
    src: &[f32],
    size: usize,
    x: usize,
    y: usize,
    height_scale: f32,
) -> (f32, f32) {
    let h = |dx: i32, dy: i32| sample_offset(src, size, x, y, dx, dy);
    let gx =
        (h(1, -1) + 2.0 * h(1, 0) + h(1, 1) - h(-1, -1) - 2.0 * h(-1, 0) - h(-1, 1)) * (1.0 / 8.0);
    let gz =
        (h(-1, 1) + 2.0 * h(0, 1) + h(1, 1) - h(-1, -1) - 2.0 * h(0, -1) - h(1, -1)) * (1.0 / 8.0);
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
    let (gx, gz) = sobel_height_gradient(src, size, x, y, height_scale);
    let dx = -gx;
    let dz = -gz;
    let len = (dx * dx + level_scale_ws * level_scale_ws + dz * dz)
        .sqrt()
        .max(1e-6);
    [dx / len, level_scale_ws / len, dz / len]
}

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
    let [ts_x, ts_y_raw] = bump_ts[py * size + px];
    let ts_y = if flip_green { -ts_y_raw } else { ts_y_raw };
    let ts_z = (1.0_f32 - ts_x * ts_x - ts_y * ts_y).max(0.0).sqrt();

    let (gx, gz) = sobel_height_gradient(height_src, size, px, py, height_scale);

    let tl = (lod_scale * lod_scale + gx * gx).sqrt().max(1e-6);
    let t = [lod_scale / tl, gx / tl, 0.0_f32];

    let bl = (lod_scale * lod_scale + gz * gz).sqrt().max(1e-6);
    let b = [0.0_f32, gz / bl, lod_scale / bl];

    let nx = b[1] * t[2] - b[2] * t[1];
    let ny = b[2] * t[0] - b[0] * t[2];
    let nz = b[0] * t[1] - b[1] * t[0];
    let nl = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
    let n = [nx / nl, ny / nl, nz / nl];

    let wx = t[0] * ts_x + b[0] * ts_y + n[0] * ts_z;
    let wy = t[1] * ts_x + b[1] * ts_y + n[1] * ts_z;
    let wz = t[2] * ts_x + b[2] * ts_y + n[2] * ts_z;

    let wl = (wx * wx + wy * wy + wz * wz).sqrt().max(1e-6);
    [wx / wl, wy / wl, wz / wl]
}

fn encode_normal_xz(normal: [f32; 3]) -> [u8; 2] {
    [
        (normal[0].clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
        (normal[2].clamp(-1.0, 1.0) * 127.0).round() as i8 as u8,
    ]
}

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("landscape-{label}-{nonce}"))
    }

    fn write_heightmap(path: &Path, size: u32) {
        let img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_fn(size, size, |x, y| {
            Luma([((x + y) % u16::MAX as u32) as u16])
        });
        img.save(path).unwrap();
    }

    #[test]
    fn bake_clears_stale_output_hierarchy() {
        let temp_dir = unique_temp_dir("bake-clean");
        fs::create_dir_all(temp_dir.join("height/L5")).unwrap();
        fs::create_dir_all(temp_dir.join("normal/L5")).unwrap();
        fs::write(temp_dir.join("height/L5/stale.bin"), [1u8, 2, 3, 4]).unwrap();
        fs::write(temp_dir.join("normal/L5/stale.bin"), [5u8, 6, 7, 8]).unwrap();

        let height_path = temp_dir.join("height.png");
        write_heightmap(&height_path, 512);

        let config = BakeConfig {
            height_path,
            output_dir: temp_dir.clone(),
            tile_size: 256,
            ..Default::default()
        };

        bake_heightmap(config, |_| {}).unwrap();

        assert!(temp_dir.join("height/L0").exists());
        assert!(temp_dir.join("normal/L0").exists());
        assert!(!temp_dir.join("height/L5").exists());
        assert!(!temp_dir.join("normal/L5").exists());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn bake_supports_power_of_two_plus_one_heightmaps() {
        let temp_dir = unique_temp_dir("bake-plus-one");
        let height_path = temp_dir.join("height.png");
        fs::create_dir_all(&temp_dir).unwrap();
        write_heightmap(&height_path, 513);

        let config = BakeConfig {
            height_path,
            output_dir: temp_dir.clone(),
            tile_size: 256,
            ..Default::default()
        };

        bake_heightmap(config, |_| {}).unwrap();

        let tile_count = fs::read_dir(temp_dir.join("height/L0")).unwrap().count();
        assert_eq!(tile_count, 4);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn bake_rejects_single_tile_heightmaps() {
        let temp_dir = unique_temp_dir("bake-too-small");
        let height_path = temp_dir.join("height.png");
        fs::create_dir_all(&temp_dir).unwrap();
        write_heightmap(&height_path, 257);

        let config = BakeConfig {
            height_path,
            output_dir: temp_dir.clone(),
            tile_size: 256,
            ..Default::default()
        };

        let err = bake_heightmap(config, |_| {}).unwrap_err();
        assert!(err.contains("too small"));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn bake_rejects_non_power_of_two_tile_counts() {
        let temp_dir = unique_temp_dir("bake-bad-tiles");
        let height_path = temp_dir.join("height.png");
        fs::create_dir_all(&temp_dir).unwrap();
        write_heightmap(&height_path, 769);

        let config = BakeConfig {
            height_path,
            output_dir: temp_dir.clone(),
            tile_size: 256,
            ..Default::default()
        };

        let err = bake_heightmap(config, |_| {}).unwrap_err();
        assert!(err.contains("power-of-two tile count"));

        let _ = fs::remove_dir_all(temp_dir);
    }
}

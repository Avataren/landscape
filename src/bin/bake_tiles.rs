/// Offline tile baker.
///
/// Reads the 16k EXR height map, builds a 6-level mip pyramid (2×2 box filter),
/// and writes 256×256 R16Unorm tiles to `assets/tiles/height/L{n}/{tx}_{ty}.bin`.
///
/// Run from the workspace root:
///   cargo run --bin bake_tiles --release
///
/// Tiles are written only once; re-running regenerates them unconditionally.
/// Peak memory: ~1.25 GB (level-0 pixels + one mip level at a time).

// Import only the names we actually use from exr, so the exr `Result` alias
// does not shadow `std::result::Result`.
use exr::prelude::{read_first_rgba_layer_from_file, Vec2};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TILE_SIZE: usize = 256;
const LEVELS: u32 = 6;
const HEIGHT_EXR: &str =
    "assets/height_maps/16k Rocky Terrain Heightmap/Height Map 16k Rocky Terrain.exr";
const TILE_ROOT: &str = "assets/tiles";

// ---------------------------------------------------------------------------
// Pixel storage passed through the exr read closure.
// We embed the image width so the pixel setter can compute flat indices.
// ---------------------------------------------------------------------------

struct HeightBuf {
    pixels: Vec<f32>,
    width: usize,
    height: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_start = Instant::now();
    let height_path = Path::new(HEIGHT_EXR);
    let out_root = PathBuf::from(TILE_ROOT);

    if !height_path.exists() {
        eprintln!("ERROR: height map not found at {:?}", height_path);
        eprintln!("Run this tool from the workspace root (next to 'assets/').");
        std::process::exit(1);
    }

    println!("Loading {} ...", HEIGHT_EXR);
    println!("(A 16k EXR takes ~30–60 s to decompress on first read)");

    let image = read_first_rgba_layer_from_file(
        height_path,
        // constructor: allocates per-image pixel storage and records the width
        |size: Vec2<usize>, _| HeightBuf {
            pixels: vec![0.0_f32; size.x() * size.y()],
            width: size.x(),
            height: size.y(),
        },
        // pixel setter: called once per pixel; we take only the R channel
        // (for a greyscale EXR, exr maps Y → R, G, B so R contains the height)
        |buf: &mut HeightBuf, pos: Vec2<usize>, (r, _g, _b, _a): (f32, f32, f32, f32)| {
            buf.pixels[pos.y() * buf.width + pos.x()] = r;
        },
    )?;

    let buf = image.layer_data.channel_data.pixels;
    let img_w = buf.width;
    let img_h = buf.height;

    println!("Loaded {}×{} in {:.1}s", img_w, img_h, t_start.elapsed().as_secs_f32());

    if img_w != img_h {
        return Err(format!("Expected square heightmap, got {}×{}", img_w, img_h).into());
    }

    let mut current_pixels = buf.pixels;
    let mut current_size = img_w;
    let mut total_tiles = 0usize;

    for lod in 0..LEVELS {
        let mip_size = current_size;
        let mip_half = (mip_size / 2) as i32;
        let tiles_per_side = (mip_size / TILE_SIZE) as i32;
        let tile_half = tiles_per_side / 2;

        if tiles_per_side == 0 {
            eprintln!("WARNING: mip_size {} < tile_size {} at LOD {}; stopping.", mip_size, TILE_SIZE, lod);
            break;
        }

        let level_dir = out_root.join(format!("height/L{}", lod));
        std::fs::create_dir_all(&level_dir)?;

        let level_tile_count = (tiles_per_side * tiles_per_side) as usize;
        let mut level_done = 0usize;

        for ty in -tile_half..tile_half {
            for tx in -tile_half..tile_half {
                // Origin of this tile in the mip image (pixel coordinates).
                let px_start = (tx * TILE_SIZE as i32 + mip_half) as usize;
                let py_start = (ty * TILE_SIZE as i32 + mip_half) as usize;

                let mut tile_bytes = Vec::with_capacity(TILE_SIZE * TILE_SIZE * 2);

                for row in 0..TILE_SIZE {
                    for col in 0..TILE_SIZE {
                        let px = px_start + col;
                        let py = py_start + row;
                        let h = if px < mip_size && py < mip_size {
                            current_pixels[py * mip_size + px]
                        } else {
                            0.0
                        };
                        let v = (h.clamp(0.0, 1.0) * 65535.0) as u16;
                        tile_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }

                let tile_path = level_dir.join(format!("{}_{}.bin", tx, ty));
                std::fs::write(&tile_path, &tile_bytes)?;
                level_done += 1;
                total_tiles += 1;
            }

            // Progress indicator for the slowest level.
            if lod == 0 && ty % 8 == (-(tile_half % 8)) {
                let pct = level_done * 100 / level_tile_count;
                print!("\r  Level 0: {}%  ({}/{})", pct, level_done, level_tile_count);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }
        if lod == 0 { println!(); }

        println!(
            "Level {}: {}×{} = {} tiles  [{:.1}s]",
            lod,
            tiles_per_side,
            tiles_per_side,
            level_tile_count,
            t_start.elapsed().as_secs_f32(),
        );

        // Build next mip level (2×2 box filter), releasing the current one.
        if lod + 1 < LEVELS {
            let next = box_filter_2x(&current_pixels, current_size, current_size);
            current_pixels = next;
            current_size /= 2;
        }
    }

    println!(
        "\nDone. {} tiles → '{}'  ({:.1}s total)",
        total_tiles,
        out_root.display(),
        t_start.elapsed().as_secs_f32(),
    );
    Ok(())
}

/// 2×2 box-filter downsample. Each output pixel is the average of four input pixels.
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

//! Offline tile baker — thin CLI wrapper around [`bevy_landscape::bake`].
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
//!   --bump-scale <f32>    World-space scale for bump normal derivation.
//!   --world-scale <f32>   World-space units per texel at LOD 0.
//!   --tile-size <usize>   Tile resolution in pixels  [default: 256]
//!   --smooth-sigma <f32>  Gaussian blur sigma (in source texels).  0 = off.
//!   --flip-green          Negate G channel (OpenGL-convention normal maps).

use bevy_landscape::bake::BakeConfig;
use serde::Deserialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// landscape.toml reader
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RootToml {
    terrain_config: Option<TerrainConfigToml>,
}

#[derive(Deserialize)]
struct TerrainConfigToml {
    height_scale: Option<f32>,
    world_scale: Option<f32>,
}

fn terrain_config_from_toml() -> Option<TerrainConfigToml> {
    let text = std::fs::read_to_string("landscape.toml").ok()?;
    let root: RootToml = toml::from_str(&text).ok()?;
    root.terrain_config
}

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------

struct CliArgs {
    height_path: Option<PathBuf>,
    bump_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    height_scale: Option<f32>,
    bump_scale: Option<f32>,
    world_scale: Option<f32>,
    tile_size: Option<usize>,
    flip_green: bool,
    smooth_sigma: Option<f32>,
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cli = CliArgs {
        height_path: None,
        bump_path: None,
        output_dir: None,
        height_scale: None,
        bump_scale: None,
        world_scale: None,
        tile_size: None,
        flip_green: false,
        smooth_sigma: None,
    };

    if args.is_empty() {
        // Legacy no-arg invocation.
        cli.height_path = Some(PathBuf::from(
            "assets/height_maps/16k Rocky Terrain Heightmap/Height Map 16k Rocky Terrain.exr",
        ));
        return Ok(cli);
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
                cli.height_path = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--bump" => {
                cli.bump_path = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--output" => {
                cli.output_dir = Some(PathBuf::from(next()?));
                i += 2;
            }
            "--height-scale" => {
                cli.height_scale =
                    Some(next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?);
                i += 2;
            }
            "--bump-scale" => {
                cli.bump_scale = Some(next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?);
                i += 2;
            }
            "--world-scale" => {
                cli.world_scale = Some(next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?);
                i += 2;
            }
            "--tile-size" => {
                cli.tile_size = Some(
                    next()?
                        .parse::<usize>()
                        .map_err(|e| format!("{flag}: {e}"))?,
                );
                i += 2;
            }
            "--flip-green" => {
                cli.flip_green = true;
                i += 1;
            }
            "--smooth-sigma" => {
                let v = next()?.parse::<f32>().map_err(|e| format!("{flag}: {e}"))?;
                if v < 0.0 {
                    return Err(format!("--smooth-sigma must be >= 0, got {v}"));
                }
                cli.smooth_sigma = Some(v);
                i += 2;
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
    }

    if cli.height_path.is_none() {
        return Err("--height <path> is required".to_string());
    }
    Ok(cli)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!();
        eprintln!("Usage: bake_tiles --height <path> [--bump <path>] [--output <dir>]");
        eprintln!("                  [--height-scale <f>] [--bump-scale <f>]");
        eprintln!("                  [--world-scale <f>] [--tile-size <n>]");
        eprintln!("                  [--smooth-sigma <f>] [--flip-green]");
        std::process::exit(1);
    });

    let toml_cfg = terrain_config_from_toml();

    let height_scale: f32 = cli.height_scale.unwrap_or_else(|| {
        if let Some(v) = toml_cfg.as_ref().and_then(|c| c.height_scale) {
            println!("height_scale: using {v} from landscape.toml");
            v
        } else {
            let v = 1024.0_f32;
            println!("height_scale: using default {v}");
            v
        }
    });

    let world_scale: f32 = cli.world_scale.unwrap_or_else(|| {
        if let Some(v) = toml_cfg.as_ref().and_then(|c| c.world_scale) {
            println!("world_scale: using {v} from landscape.toml");
            v
        } else {
            let v = 1.0_f32;
            println!("world_scale: using default {v}");
            v
        }
    });
    if world_scale <= 0.0 {
        return Err(format!("world_scale must be > 0, got {world_scale}").into());
    }

    let config = BakeConfig {
        height_path: cli.height_path.unwrap(),
        bump_path: cli.bump_path,
        output_dir: cli
            .output_dir
            .unwrap_or_else(|| PathBuf::from("assets/tiles")),
        height_scale,
        bump_scale: cli.bump_scale,
        world_scale,
        tile_size: cli.tile_size.unwrap_or(256),
        flip_green: cli.flip_green,
        smooth_sigma: cli.smooth_sigma.unwrap_or(0.0),
        sea_level_decoded: None,
    };

    bevy_landscape::bake::bake_heightmap(config, |msg| println!("{msg}")).map_err(|e| e.into())
}

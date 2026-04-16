use serde::Deserialize;

#[derive(Deserialize)]
struct ConfigFile {
    terrain: TerrainSourceToml,
    terrain_config: Option<TerrainConfigToml>,
}

#[derive(Deserialize)]
struct TerrainSourceToml {
    tile_root: Option<String>,
    normal_root: Option<String>,
    diffuse_exr: Option<String>,
    world_min: f32,
    world_max: f32,
    max_mip_level: u8,
}

#[derive(Deserialize)]
struct TerrainConfigToml {
    clipmap_levels: Option<u32>,
    height_scale: Option<f32>,
    macro_color_flip_v: Option<bool>,
}

pub struct TerrainSourceCfg {
    pub tile_root: Option<std::path::PathBuf>,
    pub normal_root: Option<String>,
    pub macro_color_root: Option<String>,
    pub world_min: bevy::math::Vec2,
    pub world_max: bevy::math::Vec2,
    pub max_mip_level: u8,
}

pub struct TerrainRenderCfg {
    pub clipmap_levels: Option<u32>,
    pub height_scale: Option<f32>,
    pub macro_color_flip_v: Option<bool>,
}

pub struct AppConfig {
    pub source: TerrainSourceCfg,
    pub render: TerrainRenderCfg,
}

/// Reads `landscape.toml` next to the executable / workspace root.
/// Panics with a clear message if the file is malformed.
pub fn load() -> AppConfig {
    let toml_str = std::fs::read_to_string("landscape.toml")
        .expect("Could not read landscape.toml — make sure you run from the workspace root");

    let cfg: ConfigFile = toml::from_str(&toml_str).expect("Failed to parse landscape.toml");

    let t = cfg.terrain;
    let rc = cfg.terrain_config.unwrap_or(TerrainConfigToml {
        clipmap_levels: None,
        height_scale: None,
        macro_color_flip_v: None,
    });

    AppConfig {
        source: TerrainSourceCfg {
            tile_root: t.tile_root.map(std::path::PathBuf::from),
            normal_root: t.normal_root,
            macro_color_root: t.diffuse_exr,
            world_min: bevy::math::Vec2::splat(t.world_min),
            world_max: bevy::math::Vec2::splat(t.world_max),
            max_mip_level: t.max_mip_level,
        },
        render: TerrainRenderCfg {
            clipmap_levels: rc.clipmap_levels,
            height_scale: rc.height_scale,
            macro_color_flip_v: rc.macro_color_flip_v,
        },
    }
}

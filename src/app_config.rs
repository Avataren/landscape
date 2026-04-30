use bevy_landscape::{scan_world_bounds, TerrainConfig, TerrainSourceDesc};
use serde::Deserialize;

#[derive(Deserialize)]
struct ConfigFile {
    terrain: TerrainSourceToml,
    terrain_config: Option<TerrainConfigToml>,
    foliage: Option<FoliageSourceToml>,
}

#[derive(Deserialize)]
struct TerrainSourceToml {
    tile_root: Option<String>,
    normal_root: Option<String>,
    diffuse_exr: Option<String>,
    max_mip_level: u8,
    /// Mip level for the collision heightfield (default 2 = 4 m/cell).
    collision_mip_level: Option<u8>,
}

#[derive(Deserialize, Default)]
struct FoliageSourceToml {
    foliage_root: Option<String>,
}

#[derive(Deserialize, Default)]
struct TerrainConfigToml {
    clipmap_levels: Option<u32>,
    /// Source-of-truth X/Z terrain scale; copied into TerrainConfig.world_scale at startup.
    world_scale: Option<f32>,
    /// Finest clipmap mesh vertex spacing in world-space metres (independent of world_scale).
    lod0_mesh_spacing: Option<f32>,
    height_scale: Option<f32>,
    macro_color_flip_v: Option<bool>,
}

pub struct TerrainSourceCfg {
    pub tile_root: Option<std::path::PathBuf>,
    pub normal_root: Option<String>,
    pub macro_color_root: Option<String>,
    pub foliage_root: Option<String>,
    pub world_min: bevy::math::Vec2,
    pub world_max: bevy::math::Vec2,
    pub max_mip_level: u8,
    pub collision_mip_level: u8,
}

pub struct TerrainRenderCfg {
    pub clipmap_levels: Option<u32>,
    /// Value loaded from `landscape.toml`; the app copies this into the runtime
    /// `TerrainConfig.world_scale` field during startup.
    pub world_scale: Option<f32>,
    /// Finest clipmap mesh vertex spacing in world-space metres.
    pub lod0_mesh_spacing: Option<f32>,
    pub height_scale: Option<f32>,
    pub macro_color_flip_v: Option<bool>,
}

pub struct AppConfig {
    pub source: TerrainSourceCfg,
    pub render: TerrainRenderCfg,
}

impl AppConfig {
    pub fn into_runtime(self) -> (TerrainConfig, TerrainSourceDesc) {
        let mut config = TerrainConfig::default();
        if let Some(v) = self.render.clipmap_levels {
            config.clipmap_levels = v;
        }
        if let Some(v) = self.render.world_scale {
            config.world_scale = v;
        }
        config.lod0_mesh_spacing = self.render.lod0_mesh_spacing.unwrap_or(config.world_scale);
        let base_height_scale = self.render.height_scale.unwrap_or(config.height_scale);
        config.height_scale = base_height_scale * config.world_scale;
        if let Some(v) = self.render.macro_color_flip_v {
            config.macro_color_flip_v = v;
        }

        let source = TerrainSourceDesc {
            tile_root: self.source.tile_root,
            normal_root: self.source.normal_root,
            material_root: None,
            macro_color_root: self.source.macro_color_root,
            foliage_root: self.source.foliage_root,
            world_min: self.source.world_min,
            world_max: self.source.world_max,
            max_mip_level: self.source.max_mip_level,
            collision_mip_level: self.source.collision_mip_level,
        };

        (config, source)
    }
}

/// Reads `landscape.toml` next to the executable / workspace root.
/// Panics with a clear message if the file is malformed.
pub fn load() -> AppConfig {
    let toml_str = std::fs::read_to_string("landscape.toml")
        .expect("Could not read landscape.toml — make sure you run from the workspace root");

    let cfg: ConfigFile = toml::from_str(&toml_str).expect("Failed to parse landscape.toml");

    let t = cfg.terrain;
    let rc = cfg.terrain_config.unwrap_or_default();
    let fc = cfg.foliage.unwrap_or_default();

    let world_scale = rc.world_scale.unwrap_or(1.0);
    // tile_size matches TerrainConfig::default() — always 256 for this project.
    const TILE_SIZE: u32 = 256;
    let tile_root_path = t.tile_root.as_deref().map(std::path::Path::new);
    let (world_min, world_max) = tile_root_path
        .map(|p| scan_world_bounds(p, TILE_SIZE, world_scale))
        .unwrap_or_else(|| {
            let h = 8192.0 * world_scale;
            (bevy::math::Vec2::splat(-h), bevy::math::Vec2::splat(h))
        });

    AppConfig {
        source: TerrainSourceCfg {
            tile_root: t.tile_root.map(std::path::PathBuf::from),
            normal_root: t.normal_root,
            macro_color_root: t.diffuse_exr,
            foliage_root: fc.foliage_root,
            world_min,
            world_max,
            max_mip_level: t.max_mip_level,
            collision_mip_level: t.collision_mip_level.unwrap_or(2),
        },
        render: TerrainRenderCfg {
            clipmap_levels: rc.clipmap_levels,
            world_scale: rc.world_scale,
            lod0_mesh_spacing: rc.lod0_mesh_spacing,
            height_scale: rc.height_scale,
            macro_color_flip_v: rc.macro_color_flip_v,
        },
    }
}


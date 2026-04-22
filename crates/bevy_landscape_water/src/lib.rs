use bevy::prelude::*;
use bevy_landscape::{TerrainConfig, TerrainMetadata, TerrainSourceDesc};
use bevy_water::{
    water::{setup_water, WaterTile},
    WaterPlugin, WaterSettings, WaterTiles,
};

#[derive(Resource, Clone, Copy)]
pub struct WaterEnabled(pub bool);

#[derive(Resource, Clone, Copy)]
struct WaterCenter(Vec2);

pub struct LandscapeWaterPlugin {
    /// Pre-computed world-space Y of the water surface.  `None` = no water for this level.
    pub water_height: Option<f32>,
    /// World XZ extents used to size the tile grid to cover the terrain footprint.
    pub world_min: Vec2,
    pub world_max: Vec2,
}

impl Default for LandscapeWaterPlugin {
    fn default() -> Self {
        Self {
            water_height: None,
            world_min: Vec2::splat(-4096.0),
            world_max: Vec2::splat(4096.0),
        }
    }
}

impl Plugin for LandscapeWaterPlugin {
    fn build(&self, app: &mut App) {
        const TILE_SIZE: f32 = bevy_water::water::WATER_SIZE as f32;

        let extent = self.world_max - self.world_min;
        let center = (self.world_min + self.world_max) * 0.5;

        let tiles_x = (extent.x / TILE_SIZE).ceil() as u32 + 2;
        let tiles_y = (extent.y / TILE_SIZE).ceil() as u32 + 2;

        let height = self.water_height.unwrap_or(0.0);
        let spawn_tiles = self.water_height.map(|_| UVec2::new(tiles_x, tiles_y));

        app.insert_resource(WaterSettings {
            height,
            spawn_tiles,
            ..WaterSettings::default()
        });
        app.insert_resource(WaterEnabled(self.water_height.is_some()));
        app.insert_resource(WaterCenter(center));
        app.add_plugins(WaterPlugin);
        app.add_systems(Startup, center_water_tiles.after(setup_water));
        app.add_systems(Update, (sync_water_to_terrain, toggle_water));
    }
}

/// Move the WaterTiles root to the terrain's XZ center after tiles are spawned.
fn center_water_tiles(
    center: Res<WaterCenter>,
    mut water_tiles: Query<&mut Transform, With<WaterTiles>>,
) {
    let Ok(mut t) = water_tiles.single_mut() else {
        return;
    };
    t.translation.x = center.0.x;
    t.translation.z = center.0.y;
}

/// Runs every frame; when TerrainSourceDesc changes (terrain reload or initial load)
/// re-reads metadata.toml to pick up the new water_level and repositions water tiles.
fn sync_water_to_terrain(
    source_desc: Res<TerrainSourceDesc>,
    config: Res<TerrainConfig>,
    mut settings: ResMut<WaterSettings>,
    mut enabled: ResMut<WaterEnabled>,
    mut water_root: Query<(&mut Transform, &mut Visibility), With<WaterTiles>>,
    mut tile_transforms: Query<&mut Transform, (With<WaterTile>, Without<WaterTiles>)>,
) {
    if !source_desc.is_changed() {
        return;
    }

    let meta = source_desc
        .tile_root
        .as_deref()
        .map(|p| TerrainMetadata::load(p))
        .unwrap_or_default();

    let new_height = meta
        .water_level
        .map(|wl| wl * config.height_scale)
        .unwrap_or(0.0);
    let has_water = meta.water_level.is_some();

    settings.height = new_height;
    enabled.0 = has_water;

    let new_cx = (source_desc.world_min.x + source_desc.world_max.x) * 0.5;
    let new_cz = (source_desc.world_min.y + source_desc.world_max.y) * 0.5;
    let vis = if has_water {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    if let Ok((mut root_t, mut root_v)) = water_root.single_mut() {
        root_t.translation.x = new_cx;
        root_t.translation.z = new_cz;
        *root_v = vis;
    }

    for mut t in &mut tile_transforms {
        t.translation.y = new_height;
    }

    info!(
        "Water level: {} (F2 to toggle)",
        if has_water {
            format!("{:.1}m", new_height)
        } else {
            "none".into()
        }
    );
}

fn toggle_water(
    keys: Res<ButtonInput<KeyCode>>,
    mut enabled: ResMut<WaterEnabled>,
    mut water_tiles: Query<&mut Visibility, With<WaterTiles>>,
) {
    if !keys.just_pressed(KeyCode::F2) {
        return;
    }
    enabled.0 = !enabled.0;
    let vis = if enabled.0 {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut v in &mut water_tiles {
        *v = vis;
    }
    info!("Water {} (F2)", if enabled.0 { "ON" } else { "OFF" });
}

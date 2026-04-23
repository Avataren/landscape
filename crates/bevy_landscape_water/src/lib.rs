use bevy::{
    light::{NotShadowCaster, NotShadowReceiver},
    mesh::PlaneMeshBuilder,
    prelude::*,
};
use bevy_landscape::{TerrainConfig, TerrainMetadata, TerrainSourceDesc};
use bevy_water::{
    water::material::{StandardWaterMaterial, WaterMaterial},
    water::{setup_water, WaterTile},
    WaterPlugin, WaterSettings, WaterTiles, WaveDirection, WATER_SIZE,
};

#[derive(Resource, Clone, Copy)]
pub struct WaterEnabled(pub bool);

#[derive(Resource, Clone, Copy)]
struct InitialWaterLayout(WaterLayout);

#[derive(Clone, Copy, Debug, PartialEq)]
struct WaterLayout {
    height: f32,
    center: Vec2,
    grid: Option<UVec2>,
    /// World-space size of each tile plane.  Equals `WATER_SIZE` for small
    /// worlds; auto-scales up for large worlds so the grid stays ≤ 256 tiles
    /// per axis and still covers the full terrain footprint.
    tile_size: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaterChunkLayout {
    grid: UVec2,
}

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
        let initial_layout = water_layout(
            self.world_min,
            self.world_max,
            self.water_height.filter(|height| *height > 0.0),
        );

        app.insert_resource(WaterSettings {
            height: initial_layout.height,
            amplitude: 8.0,
            clarity: 0.9,
            foam_threshold: 1.5,
            // LandscapeWaterPlugin owns tile spawning so reloaded terrains can
            // rebuild the grid instead of staying stuck with the startup size.
            spawn_tiles: None,
            ..WaterSettings::default()
        });
        app.insert_resource(WaterEnabled(initial_layout.grid.is_some()));
        app.insert_resource(InitialWaterLayout(initial_layout));
        app.add_plugins(WaterPlugin);
        app.add_systems(Startup, spawn_initial_water_tiles.after(setup_water));
        app.add_systems(
            Update,
            (
                sync_water_to_terrain,
                toggle_water,
                apply_water_enabled.after(toggle_water),
            ),
        );
    }
}

fn effective_water_level(level: Option<f32>) -> Option<f32> {
    level.filter(|level| *level > 0.0)
}

fn water_layout(world_min: Vec2, world_max: Vec2, water_height: Option<f32>) -> WaterLayout {
    let extent = world_max - world_min;
    let center = (world_min + world_max) * 0.5;

    // Keep the tile grid ≤ MAX_TILES_PER_AXIS on each axis.  When the world is
    // wider than that many 256 m tiles, the tile footprint scales up by the
    // smallest integer that keeps the count in budget while still covering the
    // full terrain footprint.
    const MAX_TILES_PER_AXIS: u32 = 256;

    let (grid, tile_size) = match water_height {
        None => (None, WATER_SIZE as f32),
        Some(_) => {
            let raw_x = (extent.x / WATER_SIZE as f32).ceil() as u32 + 2;
            let raw_y = (extent.y / WATER_SIZE as f32).ceil() as u32 + 2;
            let scale =
                ((raw_x.max(raw_y) as f32 / MAX_TILES_PER_AXIS as f32).ceil() as u32).max(1);
            let tile_size = WATER_SIZE as f32 * scale as f32;
            // +1 so the scaled grid still reaches the edge of the world extent.
            let tiles_x = (raw_x / scale + 1).min(MAX_TILES_PER_AXIS).max(1);
            let tiles_y = (raw_y / scale + 1).min(MAX_TILES_PER_AXIS).max(1);
            if scale > 1 {
                info!(
                    "[Water] World is {:.0}×{:.0} m — scaling water tiles {}× to {:.0} m ({tiles_x}×{tiles_y} grid).",
                    extent.x, extent.y, scale, tile_size
                );
            }
            (Some(UVec2::new(tiles_x, tiles_y)), tile_size)
        }
    };

    WaterLayout {
        height: water_height.unwrap_or(0.0),
        center,
        grid,
        tile_size,
    }
}

fn ocean_subdivisions(world_size: Vec2) -> u32 {
    const TARGET_VERTEX_SPACING_M: f32 = 8.0;
    const MIN_SUBDIVISIONS: u32 = 96;
    const MAX_SUBDIVISIONS: u32 = 384;

    ((world_size.max_element() / TARGET_VERTEX_SPACING_M).ceil() as u32)
        .saturating_sub(2)
        .clamp(MIN_SUBDIVISIONS, MAX_SUBDIVISIONS)
}

fn water_chunk_layout(world_size: Vec2) -> WaterChunkLayout {
    const MAX_CHUNK_SIZE_M: f32 = 2_048.0;

    WaterChunkLayout {
        grid: UVec2::new(
            (world_size.x / MAX_CHUNK_SIZE_M).ceil().max(1.0) as u32,
            (world_size.y / MAX_CHUNK_SIZE_M).ceil().max(1.0) as u32,
        ),
    }
}

fn water_chunk_size(world_size: Vec2, chunk_layout: WaterChunkLayout) -> Vec2 {
    Vec2::new(
        world_size.x / chunk_layout.grid.x as f32,
        world_size.y / chunk_layout.grid.y as f32,
    )
}

fn rebuild_water_tiles(
    commands: &mut Commands,
    settings: &WaterSettings,
    layout: WaterLayout,
    existing_roots: &[Entity],
    root_children: &Query<&Children, With<WaterTiles>>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardWaterMaterial>,
) {
    for root in existing_roots {
        if let Ok(children) = root_children.get(*root) {
            for child in children.iter() {
                commands.entity(child).despawn();
            }
        }
        commands.entity(*root).despawn();
    }

    let Some(grid) = layout.grid else {
        return;
    };

    let world_size = Vec2::new(
        layout.tile_size * grid.x as f32,
        layout.tile_size * grid.y as f32,
    );
    let chunk_layout = water_chunk_layout(world_size);
    let chunk_size = water_chunk_size(world_size, chunk_layout);
    let mesh =
        Mesh3d(meshes.add(
            PlaneMeshBuilder::from_size(chunk_size).subdivisions(ocean_subdivisions(chunk_size)),
        ));
    let water_height = layout.height;

    let material = MeshMaterial3d(materials.add(StandardWaterMaterial {
        base: StandardMaterial {
            base_color: settings.base_color,
            alpha_mode: settings.alpha_mode,
            perceptual_roughness: 0.22,
            ..default()
        },
        extension: WaterMaterial {
            amplitude: settings.amplitude,
            clarity: settings.clarity,
            deep_color: settings.deep_color,
            shallow_color: settings.shallow_color,
            edge_color: settings.edge_color,
            edge_scale: settings.edge_scale,
            wave_speed: settings.wave_speed,
            quality: settings.water_quality.into(),
            refraction_strength: settings.refraction_strength,
            foam_threshold: settings.foam_threshold,
            foam_color: settings.foam_color,
        },
    }));

    commands
        .spawn((
            WaterTiles,
            Name::new("Water"),
            Transform::from_xyz(layout.center.x, 0.0, layout.center.y),
            Visibility::Inherited,
        ))
        .with_children(|parent| {
            for tile_y in 0..chunk_layout.grid.y {
                for tile_x in 0..chunk_layout.grid.x {
                    let offset = Vec2::new(
                        (tile_x as f32 + 0.5) * chunk_size.x - world_size.x * 0.5,
                        (tile_y as f32 + 0.5) * chunk_size.y - world_size.y * 0.5,
                    );

                    parent.spawn((
                        WaterTile { offset },
                        Name::new(format!("Water Tile {tile_x}x{tile_y}")),
                        Transform::from_xyz(offset.x, water_height, offset.y),
                        mesh.clone(),
                        material.clone(),
                        WaveDirection::with_duration(
                            settings.wave_direction,
                            settings.wave_direction_blend_duration,
                        ),
                        NotShadowCaster,
                        NotShadowReceiver,
                    ));
                }
            }
        });
}

fn apply_water_layout(
    commands: &mut Commands,
    settings: &mut WaterSettings,
    enabled: &mut WaterEnabled,
    layout: WaterLayout,
    water_roots: &Query<Entity, With<WaterTiles>>,
    root_children: &Query<&Children, With<WaterTiles>>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardWaterMaterial>,
) {
    settings.height = layout.height;
    settings.spawn_tiles = layout.grid;
    enabled.0 = layout.grid.is_some();

    let roots: Vec<Entity> = water_roots.iter().collect();
    rebuild_water_tiles(
        commands,
        settings,
        layout,
        &roots,
        root_children,
        meshes,
        materials,
    );
}

fn spawn_initial_water_tiles(
    initial_layout: Res<InitialWaterLayout>,
    mut settings: ResMut<WaterSettings>,
    mut enabled: ResMut<WaterEnabled>,
    water_roots: Query<Entity, With<WaterTiles>>,
    root_children: Query<&Children, With<WaterTiles>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardWaterMaterial>>,
    mut commands: Commands,
) {
    apply_water_layout(
        &mut commands,
        &mut settings,
        &mut enabled,
        initial_layout.0,
        &water_roots,
        &root_children,
        &mut meshes,
        &mut materials,
    );
}

/// Runs every frame; when TerrainSourceDesc changes (terrain reload or initial load)
/// re-reads metadata.toml to pick up the new water_level and rebuilds the water
/// tile grid to match the terrain footprint.
fn sync_water_to_terrain(
    source_desc: Res<TerrainSourceDesc>,
    config: Res<TerrainConfig>,
    mut settings: ResMut<WaterSettings>,
    mut enabled: ResMut<WaterEnabled>,
    water_roots: Query<Entity, With<WaterTiles>>,
    root_children: Query<&Children, With<WaterTiles>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardWaterMaterial>>,
    mut commands: Commands,
) {
    if !source_desc.is_changed() {
        return;
    }

    let meta = source_desc
        .tile_root
        .as_deref()
        .map(|p| TerrainMetadata::load(p))
        .unwrap_or_default();

    let water_height = effective_water_level(meta.water_level).map(|wl| wl * config.height_scale);
    let layout = water_layout(source_desc.world_min, source_desc.world_max, water_height);

    apply_water_layout(
        &mut commands,
        &mut settings,
        &mut enabled,
        layout,
        &water_roots,
        &root_children,
        &mut meshes,
        &mut materials,
    );

    info!(
        "Water level: {} (F2 to toggle)",
        if let Some(height) = water_height {
            format!("{:.1}m", height)
        } else {
            "none".into()
        }
    );
}

fn toggle_water(keys: Res<ButtonInput<KeyCode>>, mut enabled: ResMut<WaterEnabled>) {
    if keys.just_pressed(KeyCode::F2) {
        enabled.0 = !enabled.0;
        info!("Water {} (F2)", if enabled.0 { "ON" } else { "OFF" });
    }
}

/// Reactively syncs water tile visibility whenever `WaterEnabled` changes.
/// This allows external code (e.g. the editor during diffusion inference) to
/// hide/show water by simply setting `WaterEnabled` without needing to hold a
/// `WaterTiles` query themselves.
fn apply_water_enabled(
    enabled: Res<WaterEnabled>,
    mut water_tiles: Query<&mut Visibility, With<WaterTiles>>,
) {
    if !enabled.is_changed() {
        return;
    }
    let vis = if enabled.0 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut water_tiles {
        *v = vis;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        effective_water_level, ocean_subdivisions, water_chunk_layout, water_chunk_size,
        water_layout,
    };
    use bevy::prelude::*;

    #[test]
    fn zero_water_level_is_treated_as_no_water() {
        assert_eq!(effective_water_level(Some(0.0)), None);
        assert_eq!(effective_water_level(Some(0.25)), Some(0.25));
    }

    #[test]
    fn layout_scales_grid_with_terrain_extent() {
        let layout = water_layout(
            Vec2::new(-15_360.0, -7_680.0),
            Vec2::new(15_360.0, 7_680.0),
            Some(120.0),
        );

        assert_eq!(layout.center, Vec2::ZERO);
        assert_eq!(layout.grid, Some(UVec2::new(123, 63)));
        assert_eq!(layout.height, 120.0);
    }

    #[test]
    fn ocean_subdivision_budget_is_clamped() {
        assert_eq!(ocean_subdivisions(Vec2::splat(512.0)), 96);
        assert_eq!(ocean_subdivisions(Vec2::splat(1_024.0)), 126);
        assert_eq!(ocean_subdivisions(Vec2::splat(4_096.0)), 384);
    }

    #[test]
    fn chunk_layout_caps_tile_extent() {
        let world_size = Vec2::new(15_360.0, 7_680.0);
        let layout = water_chunk_layout(world_size);

        assert_eq!(layout.grid, UVec2::new(8, 4));
        assert_eq!(
            water_chunk_size(world_size, layout),
            Vec2::new(1_920.0, 1_920.0)
        );
    }
}

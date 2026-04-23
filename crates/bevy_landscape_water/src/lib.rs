use bevy::{
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
};
use bevy_landscape::{TerrainCamera, TerrainConfig, TerrainMetadata, TerrainSourceDesc};
use bevy_water::{
    water::material::{StandardWaterMaterial, WaterMaterial},
    water::{setup_water, WaterTile},
    WaterPlugin, WaterSettings, WaterTiles, WaveDirection,
};

#[derive(Resource, Clone, Copy)]
pub struct WaterEnabled(pub bool);

#[derive(Resource, Clone, Copy)]
struct InitialWaterLayout(WaterLayout);

#[derive(Clone, Copy, Debug, PartialEq)]
struct WaterLayout {
    height: f32,
    center: Vec2,
    clipmap: Option<WaterClipmapLayout>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaterClipmapLayout {
    levels: u32,
}

#[derive(Clone, Debug)]
struct PatchInstanceCpu {
    lod_level: u32,
    origin_ws: Vec2,
    level_scale_ws: f32,
}

#[derive(Clone, Debug)]
struct TrimInstanceCpu {
    lod_level: u32,
    origin_ws: Vec2,
    level_scale_ws: f32,
    is_horizontal: bool,
}

#[derive(Resource, Default)]
struct WaterPatchEntities {
    entities: Vec<Entity>,
    block_count: usize,
    last_clip_centers: Vec<IVec2>,
}

pub struct LandscapeWaterPlugin {
    /// Pre-computed world-space Y of the water surface.  `None` = no water for this level.
    pub water_height: Option<f32>,
    /// World XZ extents used to choose clipmap coverage for the terrain footprint.
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
            // LandscapeWaterPlugin owns patch spawning so reloaded terrains can
            // rebuild the clipmap instead of staying stuck with the startup layout.
            spawn_tiles: None,
            ..WaterSettings::default()
        });
        app.insert_resource(WaterEnabled(initial_layout.clipmap.is_some()));
        app.insert_resource(InitialWaterLayout(initial_layout));
        app.init_resource::<WaterPatchEntities>();
        app.add_plugins(WaterPlugin);
        app.add_systems(Startup, spawn_initial_water_tiles.after(setup_water));
        app.add_systems(
            Update,
            (
                (sync_water_to_terrain, update_water_patch_transforms).chain(),
                toggle_water,
                apply_water_enabled.after(toggle_water),
            ),
        );
    }
}

fn effective_water_level(level: Option<f32>) -> Option<f32> {
    level.filter(|level| *level > 0.0)
}

const WATER_CLIPMAP_BLOCK_SIZE: u32 = 64;
const WATER_CLIPMAP_BASE_SCALE: f32 = 4.0;
const WATER_CLIPMAP_MIN_LEVELS: u32 = 6;
const WATER_CLIPMAP_MAX_LEVELS: u32 = 8;
const WATER_CLIPMAP_MIN_OUTER_HALF_EXTENT: f32 = 16_384.0;

fn water_layout(world_min: Vec2, world_max: Vec2, water_height: Option<f32>) -> WaterLayout {
    let extent = world_max - world_min;
    let center = (world_min + world_max) * 0.5;

    WaterLayout {
        height: water_height.unwrap_or(0.0),
        center,
        clipmap: water_height.map(|_| water_clipmap_layout(extent)),
    }
}

fn water_level_scale(level: u32) -> f32 {
    WATER_CLIPMAP_BASE_SCALE * (1u32 << level) as f32
}

fn water_ring_half_extent(level: u32) -> f32 {
    2.0 * WATER_CLIPMAP_BLOCK_SIZE as f32 * water_level_scale(level)
}

fn water_clipmap_layout(extent: Vec2) -> WaterClipmapLayout {
    let target_half_extent = (extent.max_element() * 0.75).max(WATER_CLIPMAP_MIN_OUTER_HALF_EXTENT);
    let mut levels = 1;
    let mut outer_half_extent = water_ring_half_extent(0);

    while outer_half_extent < target_half_extent && levels < WATER_CLIPMAP_MAX_LEVELS {
        outer_half_extent *= 2.0;
        levels += 1;
    }

    WaterClipmapLayout {
        levels: levels.max(WATER_CLIPMAP_MIN_LEVELS),
    }
}

fn snap_camera_to_level_grid(camera_xz: Vec2, level_scale: f32) -> IVec2 {
    IVec2::new(
        (camera_xz.x / level_scale).floor() as i32,
        (camera_xz.y / level_scale).floor() as i32,
    )
}

fn build_water_clip_centers(camera_xz: Vec2, levels: u32) -> Vec<IVec2> {
    let fine_center = snap_camera_to_level_grid(camera_xz, water_level_scale(0));
    (0..levels)
        .map(|level| {
            let shift = level as i32;
            IVec2::new(fine_center.x >> shift, fine_center.y >> shift)
        })
        .collect()
}

fn build_block_origins(center: IVec2, scale: f32, m: u32, has_inner_hole: bool) -> Vec<Vec2> {
    let m = m as i32;
    let cols = [-2 * m, -m, 0, m];
    let mut origins = Vec::with_capacity(if has_inner_hole { 12 } else { 16 });

    for &bz in &cols {
        for &bx in &cols {
            if has_inner_hole && bx >= -m && bx < m && bz >= -m && bz < m {
                continue;
            }
            origins.push(Vec2::new(
                (center.x + bx) as f32 * scale,
                (center.y + bz) as f32 * scale,
            ));
        }
    }

    origins
}

fn build_patch_instances(clip_centers: &[IVec2]) -> Vec<PatchInstanceCpu> {
    let mut out = Vec::new();

    for (level, center) in clip_centers.iter().copied().enumerate() {
        let level = level as u32;
        let scale = water_level_scale(level);
        for origin in build_block_origins(center, scale, WATER_CLIPMAP_BLOCK_SIZE, level > 0) {
            out.push(PatchInstanceCpu {
                lod_level: level,
                origin_ws: origin,
                level_scale_ws: scale,
            });
        }
    }

    out
}

fn build_trim_instances(clip_centers: &[IVec2]) -> Vec<TrimInstanceCpu> {
    let m = WATER_CLIPMAP_BLOCK_SIZE as i32;
    let mut instances = Vec::new();

    for level in 1..clip_centers.len() {
        let scale = water_level_scale(level as u32);
        let center = clip_centers[level];

        let min_x = (center.x - m) as f32 * scale;
        let min_z = (center.y - m) as f32 * scale;
        let max_x = (center.x + m) as f32 * scale;
        let max_z = (center.y + m) as f32 * scale;

        instances.push(TrimInstanceCpu {
            lod_level: level as u32,
            origin_ws: Vec2::new(min_x, min_z),
            level_scale_ws: scale,
            is_horizontal: false,
        });
        instances.push(TrimInstanceCpu {
            lod_level: level as u32,
            origin_ws: Vec2::new(max_x - scale, min_z),
            level_scale_ws: scale,
            is_horizontal: false,
        });
        instances.push(TrimInstanceCpu {
            lod_level: level as u32,
            origin_ws: Vec2::new(min_x, min_z),
            level_scale_ws: scale,
            is_horizontal: true,
        });
        instances.push(TrimInstanceCpu {
            lod_level: level as u32,
            origin_ws: Vec2::new(min_x, max_z - scale),
            level_scale_ws: scale,
            is_horizontal: true,
        });
    }

    instances
}

fn build_rect_mesh(nx: u32, nz: u32) -> Mesh {
    let verts_per_x = nx + 1;
    let verts_per_z = nz + 1;
    let total_verts = (verts_per_x * verts_per_z) as usize;
    let inv_nx = 1.0 / nx as f32;
    let inv_nz = 1.0 / nz as f32;

    let mut positions = Vec::with_capacity(total_verts);
    let mut normals = Vec::with_capacity(total_verts);
    let mut uvs = Vec::with_capacity(total_verts);

    for z in 0..verts_per_z {
        for x in 0..verts_per_x {
            positions.push([x as f32, 0.0, z as f32]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 * inv_nx, z as f32 * inv_nz]);
        }
    }

    let mut indices = Vec::with_capacity((nx * nz * 6) as usize);
    for z in 0..nz {
        for x in 0..nx {
            let i00 = z * verts_per_x + x;
            let i10 = i00 + 1;
            let i01 = i00 + verts_per_x;
            let i11 = i01 + 1;

            indices.extend_from_slice(&[i00, i01, i10, i10, i01, i11]);
        }
    }

    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
    mesh
}

fn rebuild_water_tiles(
    commands: &mut Commands,
    settings: &WaterSettings,
    layout: WaterLayout,
    camera_xz: Vec2,
    patch_entities: &mut WaterPatchEntities,
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

    patch_entities.entities.clear();
    patch_entities.block_count = 0;
    patch_entities.last_clip_centers.clear();

    let Some(clipmap) = layout.clipmap else {
        return;
    };

    let clip_centers = build_water_clip_centers(camera_xz, clipmap.levels);
    let root_center_ws = clip_centers[0].as_vec2() * water_level_scale(0);
    let patches = build_patch_instances(&clip_centers);
    let trims = build_trim_instances(&clip_centers);

    let block_mesh = Mesh3d(meshes.add(build_rect_mesh(
        WATER_CLIPMAP_BLOCK_SIZE,
        WATER_CLIPMAP_BLOCK_SIZE,
    )));
    let trim_quads = 2 * WATER_CLIPMAP_BLOCK_SIZE;
    let trim_v_mesh = Mesh3d(meshes.add(build_rect_mesh(1, trim_quads)));
    let trim_h_mesh = Mesh3d(meshes.add(build_rect_mesh(trim_quads, 1)));
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

    let wave_direction = WaveDirection::with_duration(
        settings.wave_direction,
        settings.wave_direction_blend_duration,
    );

    let mut spawned_entities = Vec::with_capacity(patches.len() + trims.len());
    let root = commands
        .spawn((
            WaterTiles,
            Name::new("Water"),
            Transform::from_xyz(root_center_ws.x, 0.0, root_center_ws.y),
            Visibility::Inherited,
        ))
        .id();

    commands.entity(root).with_children(|parent| {
        for patch in &patches {
            let offset = patch.origin_ws - root_center_ws;
            let entity = parent
                .spawn((
                    WaterTile { offset },
                    Name::new(format!("Water Patch L{}", patch.lod_level)),
                    Transform {
                        translation: Vec3::new(offset.x, water_height, offset.y),
                        scale: Vec3::new(patch.level_scale_ws, 1.0, patch.level_scale_ws),
                        ..default()
                    },
                    block_mesh.clone(),
                    material.clone(),
                    wave_direction,
                    NotShadowCaster,
                    NotShadowReceiver,
                ))
                .id();
            spawned_entities.push(entity);
        }

        for trim in &trims {
            let offset = trim.origin_ws - root_center_ws;
            let mesh = if trim.is_horizontal {
                trim_h_mesh.clone()
            } else {
                trim_v_mesh.clone()
            };
            let entity = parent
                .spawn((
                    WaterTile { offset },
                    Name::new(format!("Water Trim L{}", trim.lod_level)),
                    Transform {
                        translation: Vec3::new(offset.x, water_height, offset.y),
                        scale: Vec3::new(trim.level_scale_ws, 1.0, trim.level_scale_ws),
                        ..default()
                    },
                    mesh,
                    material.clone(),
                    wave_direction,
                    NotShadowCaster,
                    NotShadowReceiver,
                ))
                .id();
            spawned_entities.push(entity);
        }
    });

    patch_entities.entities = spawned_entities;
    patch_entities.block_count = patches.len();
    patch_entities.last_clip_centers = clip_centers;
}

fn current_water_camera_xz(
    camera_q: &Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
    fallback: Vec2,
) -> Vec2 {
    if let Ok(camera) = camera_q.single() {
        camera.translation.xz()
    } else {
        fallback
    }
}

fn update_water_patch_transforms(
    mut patch_entities: ResMut<WaterPatchEntities>,
    camera_q: Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
    mut water_root_q: Query<
        &mut Transform,
        (With<WaterTiles>, Without<TerrainCamera>, Without<WaterTile>),
    >,
    mut water_tile_q: Query<
        (&mut Transform, &mut WaterTile),
        (Without<WaterTiles>, Without<TerrainCamera>),
    >,
) {
    if patch_entities.entities.is_empty() {
        return;
    }

    let Ok(mut root_transform) = water_root_q.single_mut() else {
        return;
    };

    let clip_centers = build_water_clip_centers(
        current_water_camera_xz(&camera_q, root_transform.translation.xz()),
        patch_entities.last_clip_centers.len() as u32,
    );
    if clip_centers == patch_entities.last_clip_centers {
        return;
    }

    let root_center_ws = clip_centers[0].as_vec2() * water_level_scale(0);
    root_transform.translation.x = root_center_ws.x;
    root_transform.translation.z = root_center_ws.y;

    let patches = build_patch_instances(&clip_centers);
    let trims = build_trim_instances(&clip_centers);
    let block_count = patches.len();

    for (entity, patch) in patch_entities.entities[..block_count]
        .iter()
        .zip(patches.iter())
    {
        if let Ok((mut transform, mut water_tile)) = water_tile_q.get_mut(*entity) {
            let offset = patch.origin_ws - root_center_ws;
            transform.translation.x = offset.x;
            transform.translation.z = offset.y;
            transform.scale = Vec3::new(patch.level_scale_ws, 1.0, patch.level_scale_ws);
            water_tile.offset = offset;
        }
    }

    for (entity, trim) in patch_entities.entities[block_count..]
        .iter()
        .zip(trims.iter())
    {
        if let Ok((mut transform, mut water_tile)) = water_tile_q.get_mut(*entity) {
            let offset = trim.origin_ws - root_center_ws;
            transform.translation.x = offset.x;
            transform.translation.z = offset.y;
            transform.scale = Vec3::new(trim.level_scale_ws, 1.0, trim.level_scale_ws);
            water_tile.offset = offset;
        }
    }

    patch_entities.last_clip_centers = clip_centers;
}

fn apply_water_layout(
    commands: &mut Commands,
    settings: &mut WaterSettings,
    enabled: &mut WaterEnabled,
    layout: WaterLayout,
    camera_xz: Vec2,
    patch_entities: &mut WaterPatchEntities,
    water_roots: &Query<Entity, With<WaterTiles>>,
    root_children: &Query<&Children, With<WaterTiles>>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardWaterMaterial>,
) {
    settings.height = layout.height;
    settings.spawn_tiles = None;
    enabled.0 = layout.clipmap.is_some();

    let roots: Vec<Entity> = water_roots.iter().collect();
    rebuild_water_tiles(
        commands,
        settings,
        layout,
        camera_xz,
        patch_entities,
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
    mut patch_entities: ResMut<WaterPatchEntities>,
    camera_q: Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
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
        current_water_camera_xz(&camera_q, initial_layout.0.center),
        &mut patch_entities,
        &water_roots,
        &root_children,
        &mut meshes,
        &mut materials,
    );
}

/// Runs every frame; when TerrainSourceDesc changes (terrain reload or initial load)
/// re-reads metadata.toml to pick up the new water_level and rebuilds the water
/// clipmap to match the terrain footprint.
fn sync_water_to_terrain(
    source_desc: Res<TerrainSourceDesc>,
    config: Res<TerrainConfig>,
    mut settings: ResMut<WaterSettings>,
    mut enabled: ResMut<WaterEnabled>,
    mut patch_entities: ResMut<WaterPatchEntities>,
    camera_q: Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
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
        current_water_camera_xz(&camera_q, layout.center),
        &mut patch_entities,
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
        build_patch_instances, build_trim_instances, build_water_clip_centers,
        effective_water_level, water_clipmap_layout, water_layout,
    };
    use bevy::prelude::*;

    #[test]
    fn zero_water_level_is_treated_as_no_water() {
        assert_eq!(effective_water_level(Some(0.0)), None);
        assert_eq!(effective_water_level(Some(0.25)), Some(0.25));
    }

    #[test]
    fn layout_scales_clipmap_with_terrain_extent() {
        let layout = water_layout(
            Vec2::new(-15_360.0, -7_680.0),
            Vec2::new(15_360.0, 7_680.0),
            Some(120.0),
        );

        assert_eq!(layout.center, Vec2::ZERO);
        assert_eq!(
            layout.clipmap,
            Some(super::WaterClipmapLayout { levels: 7 })
        );
        assert_eq!(layout.height, 120.0);
    }

    #[test]
    fn clipmap_layout_keeps_minimum_levels() {
        assert_eq!(water_clipmap_layout(Vec2::splat(2_048.0)).levels, 6);
    }

    #[test]
    fn clipmap_centers_are_nested_by_shift() {
        let centers = build_water_clip_centers(Vec2::new(13.9, 29.9), 4);

        assert_eq!(centers[0], IVec2::new(3, 7));
        assert_eq!(centers[1], IVec2::new(1, 3));
        assert_eq!(centers[2], IVec2::new(0, 1));
        assert_eq!(centers[3], IVec2::ZERO);
    }

    #[test]
    fn clipmap_patch_and_trim_counts_match_gpu_gems_layout() {
        let centers = build_water_clip_centers(Vec2::ZERO, 6);

        assert_eq!(build_patch_instances(&centers).len(), 16 + 12 * 5);
        assert_eq!(build_trim_instances(&centers).len(), 4 * 5);
    }
}

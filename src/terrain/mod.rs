pub mod clipmap;
pub mod clipmap_texture;
pub mod collision;
pub mod components;
pub mod config;
pub mod debug;
pub mod material;
pub mod macro_color;
pub mod math;
pub mod patch_mesh;
pub mod render;
pub mod residency;
pub mod resources;
pub mod streamer;
pub mod world_desc;

pub use debug::TerrainDebugPlugin;
pub use world_desc::TerrainSourceDesc;

use bevy::{camera::visibility::NoFrustumCulling, prelude::*};
use clipmap::{build_patch_instances_for_view, PatchInstanceCpu};
use clipmap_texture::{
    apply_tiles_to_clipmap, compute_clip_levels, compute_initial_clip_levels,
    create_initial_clipmap_texture, create_initial_normal_clipmap_texture,
    update_clipmap_textures, TerrainClipmapState,
};
use collision::{update_collision_tiles, TerrainCollisionCache};
use components::TerrainCamera;
use config::TerrainConfig;
use material::{TerrainMaterial, TerrainMaterialUniforms};
use macro_color::load_macro_color_texture;
use math::{compute_needed_tiles_for_level, level_scale, snap_camera_to_level_grid};
use render::{extract::extract_terrain_frame, TerrainRenderPlugin};
use residency::update_required_tiles;
use resources::{TerrainResidency, TerrainStreamQueue, TerrainViewState, TileKey, TileState};
use streamer::{load_tile_data, poll_tile_stream_jobs, request_tile_loads, setup_tile_channel};

// ---------------------------------------------------------------------------
// Patch entity tracker
// ---------------------------------------------------------------------------

/// Stores entity handles for all spawned terrain patch entities in the same
/// order that `build_patch_instances_for_view` returns descriptors.
/// Allows update_patch_transforms to update positions cheaply.
#[derive(Resource, Default)]
pub struct PatchEntities {
    pub entities: Vec<Entity>,
    /// Cached clip centers from last update (used to skip no-op frames).
    pub last_clip_centers: Vec<IVec2>,
}

// ---------------------------------------------------------------------------
// Main terrain plugin
// ---------------------------------------------------------------------------

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app
            // Custom terrain material
            .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            // Resources
            .init_resource::<TerrainConfig>()
            .init_resource::<TerrainSourceDesc>()
            .init_resource::<TerrainViewState>()
            .init_resource::<TerrainResidency>()
            .init_resource::<TerrainStreamQueue>()
            .init_resource::<TerrainCollisionCache>()
            .init_resource::<PatchEntities>()
            // Startup
            .add_systems(Startup, (setup_tile_channel, setup_terrain).chain())
            .add_systems(PostStartup, preload_terrain_startup)
            // Update: ordered as per handoff spec
            .add_systems(Update, update_terrain_view_state)
            .add_systems(
                Update,
                (
                    update_required_tiles,
                    request_tile_loads,
                    poll_tile_stream_jobs,
                    update_collision_tiles,
                    extract_terrain_frame,
                    update_patch_transforms, // Phase 1: live patch repositioning
                    update_clipmap_textures, // Phase 5: procedural clipmap refresh
                )
                    .after(update_terrain_view_state),
            )
            .add_systems(
                Update,
                apply_tiles_to_clipmap // Phase 5: tile-based GPU upload
                    .after(poll_tile_stream_jobs)
                    .after(update_clipmap_textures)
                    .after(update_terrain_view_state),
            )
            // Render sub-plugin
            .add_plugins(TerrainRenderPlugin);
    }
}

// ---------------------------------------------------------------------------
// Startup systems
// ---------------------------------------------------------------------------

/// Synchronously pre-loads all terrain tiles visible from the starting camera
/// position and writes them into the clipmap texture before the first Update
/// frame.  Without this, tiles only begin loading on the first Update tick and
/// the terrain stays flat until background threads finish (many frames later).
///
/// Runs in PostStartup (after setup_terrain) so TerrainClipmapState exists.
fn preload_terrain_startup(
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    camera_q: Query<&Transform, With<TerrainCamera>>,
    mut state: ResMut<TerrainClipmapState>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut residency: ResMut<TerrainResidency>,
) {
    let levels = config.active_clipmap_levels();

    // --- Compute clip centers from the starting camera position ---------------
    let cam_pos = match camera_q.single() {
        Ok(t) => t.translation,
        Err(_) => {
            warn!("[Terrain] preload: no camera found");
            return;
        }
    };

    let scale_0 = level_scale(config.world_scale, 0);
    let fine_center = snap_camera_to_level_grid(cam_pos.xz(), scale_0);

    let clip_centers: Vec<IVec2> = (0..levels)
        .map(|l| {
            let s = l as i32;
            IVec2::new(fine_center.x >> s, fine_center.y >> s)
        })
        .collect();
    let level_scales: Vec<f32> = (0..levels)
        .map(|l| level_scale(config.world_scale, l))
        .collect();

    // --- Collect all needed tile keys, coarser levels first ------------------
    let mut all_keys: Vec<TileKey> = Vec::new();
    for level in 0..levels {
        let keys = compute_needed_tiles_for_level(
            clip_centers[level as usize],
            level_scales[level as usize],
            config.patch_resolution,
            config.ring_patches,
            config.tile_size,
            level as u8,
        );
        all_keys.extend(keys);
    }
    all_keys.sort_by(|a, b| b.level.cmp(&a.level));
    all_keys.dedup();

    // --- Load all tiles in parallel, block until done -------------------------
    let tile_size = config.tile_size;
    let world_scale = config.world_scale;
    let height_scale = config.height_scale;
    let max_mip_level = desc.max_mip_level;
    let use_procedural = config.procedural_fallback;
    let tile_root = desc.tile_root.clone();
    let normal_root = desc.normal_root.as_ref().map(std::path::PathBuf::from);
    let world_bounds = Some((desc.world_min, desc.world_max));

    let results: Vec<crate::terrain::resources::HeightTileCpu> = std::thread::scope(|s| {
        let handles: Vec<_> = all_keys
            .iter()
            .map(|&key| {
                let tile_root = tile_root.clone();
                let normal_root = normal_root.clone();
                s.spawn(move || {
                    load_tile_data(
                        key,
                        tile_size,
                        world_scale,
                        height_scale,
                        max_mip_level,
                        tile_root.as_deref(),
                        normal_root.as_deref(),
                        world_bounds,
                        use_procedural,
                    )
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // --- Write tile data into the clipmap texture and residency ---------------
    let res = config.clipmap_resolution();
    let height_bpl = (res * res * 2) as usize; // R16Unorm
    let normal_bpl = (res * res * 2) as usize; // RG8Snorm
    let half = (res / 2) as i32;
    let ts = config.tile_size;
    let levels = levels as usize;
    let mut written = 0u32;

    {
        let Some(image) = images.get_mut(&state.height_texture_handle) else {
            warn!("[Terrain] preload: height clipmap image not found");
            return;
        };
        let Some(ref mut img_data) = image.data else {
            warn!("[Terrain] preload: height clipmap image has no pixel data");
            return;
        };

        for tile in &results {
            let key = tile.key;
            let level = key.level as usize;
            if level >= levels {
                continue;
            }

            let clip_center = clip_centers[level];
            let height_layer_base = level * height_bpl;

            for row in 0..ts {
                for col in 0..ts {
                    let gx = key.x * ts as i32 + col as i32;
                    let gz = key.y * ts as i32 + row as i32;
                    let dx = gx - clip_center.x;
                    let dz = gz - clip_center.y;
                    if dx < -half || dx >= half || dz < -half || dz >= half {
                        continue;
                    }

                    let tx = gx.rem_euclid(res as i32) as usize;
                    let tz = gz.rem_euclid(res as i32) as usize;
                    let height_dst = height_layer_base + (tz * res as usize + tx) * 2;
                    let h = tile.data[(row * ts + col) as usize];
                    let v = (h * 65535.0) as u16;

                    if height_dst + 2 <= img_data.len() {
                        img_data[height_dst..height_dst + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }

    {
        let Some(image) = images.get_mut(&state.normal_texture_handle) else {
            warn!("[Terrain] preload: normal clipmap image not found");
            return;
        };
        let Some(ref mut img_data) = image.data else {
            warn!("[Terrain] preload: normal clipmap image has no pixel data");
            return;
        };

        for tile in &results {
            let key = tile.key;
            let level = key.level as usize;
            if level >= levels {
                continue;
            }

            let clip_center = clip_centers[level];
            let normal_layer_base = level * normal_bpl;

            for row in 0..ts {
                for col in 0..ts {
                    let gx = key.x * ts as i32 + col as i32;
                    let gz = key.y * ts as i32 + row as i32;
                    let dx = gx - clip_center.x;
                    let dz = gz - clip_center.y;
                    if dx < -half || dx >= half || dz < -half || dz >= half {
                        continue;
                    }

                    let tx = gx.rem_euclid(res as i32) as usize;
                    let tz = gz.rem_euclid(res as i32) as usize;
                    let normal_dst = normal_layer_base + (tz * res as usize + tx) * 2;
                    let enc = tile.normal_data[(row * ts + col) as usize];

                    if normal_dst + 2 <= img_data.len() {
                        img_data[normal_dst..normal_dst + 2].copy_from_slice(&enc);
                    }
                }
            }
        }
    }

    for tile in &results {
        let key = tile.key;
        residency.resident_cpu.insert(key, crate::terrain::resources::HeightTileCpu {
            key,
            data: tile.data.clone(),
            normal_data: tile.normal_data.clone(),
            tile_size: tile.tile_size,
        });
        residency.tiles.insert(key, TileState::ResidentGpu { slot: 0 });
        residency.touch(key);
        written += 1;
    }

    // --- Update material clip_levels so UVs map to the actual camera pos -----
    if let Some(mat) = materials.get_mut(&state.material_handle) {
        mat.params.clip_levels = compute_clip_levels(&config, &clip_centers, &level_scales);
    }

    // --- Set sentinels so Update-frame systems see no change and skip --------
    // update_clipmap_textures compares view.clip_centers vs last_clip_centers.
    // apply_tiles_to_clipmap compares view.clip_centers vs tile_apply_centers.
    // Both will be equal on the first Update frame (camera hasn't moved).
    state.last_clip_centers = clip_centers.clone();
    state.tile_apply_centers = clip_centers;

    info!(
        "[Terrain] Startup preload: {} tiles loaded synchronously.",
        written
    );
}

/// Spawns patch entities and creates the initial clipmap texture array.
fn setup_terrain(
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut patch_entities: ResMut<PatchEntities>,
    mut commands: Commands,
) {
    let levels = config.active_clipmap_levels();

    // --- Clipmap texture array (Phase 5) ---
    // One R8Unorm layer per LOD level, each 512×512 texels.
    // Layers are regenerated live by `update_clipmap_textures` as the camera moves.
    let height_image = create_initial_clipmap_texture(&config);
    let height_handle = images.add(height_image);
    let normal_image = create_initial_normal_clipmap_texture(&config);
    let normal_handle = images.add(normal_image);
    let macro_color = load_macro_color_texture(&config, &desc);
    let macro_color_handle = images.add(macro_color.image);

    let base_patch_size = config.patch_resolution as f32 * config.world_scale;
    let bounds_fade_distance = config.tile_size as f32 * config.world_scale * 4.0;

    let mat_handle = terrain_materials.add(TerrainMaterial {
        height_texture: height_handle.clone(),
        macro_color_texture: macro_color_handle,
        normal_texture: normal_handle.clone(),
        params: TerrainMaterialUniforms {
            height_scale: config.height_scale,
            base_patch_size,
            morph_start_ratio: config.morph_start_ratio,
            ring_patches: config.ring_patches as f32,
            num_lod_levels: levels as f32,
            patch_resolution: config.patch_resolution as f32,
            world_bounds: Vec4::new(desc.world_min.x, desc.world_min.y, desc.world_max.x, desc.world_max.y),
            bounds_fade: Vec4::new(
                bounds_fade_distance,
                if macro_color.enabled { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ),
            clip_levels: compute_initial_clip_levels(&config),
        },
    });

    // Insert the runtime state resource so `update_clipmap_textures` can find it.
    // Initialise last_clip_centers to ZERO (matching the initial texture), so the
    // first real camera position triggers a full regeneration on the first frame.
    commands.insert_resource(TerrainClipmapState {
        height_texture_handle: height_handle,
        normal_texture_handle: normal_handle,
        material_handle: mat_handle.clone(),
        last_clip_centers: vec![IVec2::ZERO; levels as usize],
        // Sentinel forces a full tile re-apply on the first frame.
        tile_apply_centers: vec![IVec2::new(i32::MAX, i32::MAX); levels as usize],
    });

    // --- Patch mesh (shared by all entities) ---
    let patch_mesh = patch_mesh::build_patch_mesh(config.patch_resolution);
    let mesh_handle = meshes.add(patch_mesh);

    // --- Spawn one entity per patch from the default (zero) view state ---
    let view = TerrainViewState::default();
    let patches: Vec<PatchInstanceCpu> = build_patch_instances_for_view(&config, &view);

    patch_entities.entities.clear();

    for patch in &patches {
        let entity = commands
            .spawn((
                Mesh3d(mesh_handle.clone()),
                MeshMaterial3d(mat_handle.clone()),
                Transform {
                    translation: Vec3::new(patch.origin_ws.x, 0.0, patch.origin_ws.y),
                    scale: Vec3::new(patch.patch_size_ws, 1.0, patch.patch_size_ws),
                    ..default()
                },
                components::TerrainPatchInstance {
                    lod_level: patch.lod_level,
                    patch_kind: patch.patch_kind,
                    patch_origin_ws: patch.origin_ws,
                    patch_scale_ws: patch.patch_size_ws,
                },
                NoFrustumCulling,
            ))
            .id();

        patch_entities.entities.push(entity);
    }

    info!(
        "[Terrain] Setup complete: {} patches across {} LOD levels. \
         Clipmap {}×{}×{} (R16Unorm array, derived from ring_patches × patch_resolution).",
        patches.len(),
        levels,
        config.clipmap_resolution(),
        config.clipmap_resolution(),
        levels,
    );
}

// ---------------------------------------------------------------------------
// Update systems
// ---------------------------------------------------------------------------

/// Rebuilds the terrain view (camera position, clip centers, level scales).
/// Runs first so all subsequent systems see fresh data.
pub fn update_terrain_view_state(
    config: Res<TerrainConfig>,
    camera_q: Query<&Transform, With<TerrainCamera>>,
    mut view: ResMut<TerrainViewState>,
) {
    let Ok(cam) = camera_q.single() else {
        return;
    };

    let cam_pos: Vec3 = cam.translation;
    view.camera_pos_ws = cam_pos;
    view.clip_centers.clear();
    view.level_scales.clear();

    // Build a strictly nested center chain from finest to coarsest.
    //
    // Important: derive every coarser center from the finest center via
    // integer right shifts. This preserves exact parent/child alignment while
    // still letting each level move at its natural cadence:
    // L0 every 1 texel, L1 every 2 texels, L2 every 4 texels, etc.
    //
    // This removes large multi-cell jumps on mid/far levels that create
    // temporal shimmer and visible instability when moving the camera.
    let scale_0 = level_scale(config.world_scale, 0);
    let fine_center = snap_camera_to_level_grid(cam_pos.xz(), scale_0);

    for level in 0..config.active_clipmap_levels() {
        let scale = level_scale(config.world_scale, level);
        // Right-shift the finest center to get each coarser center.
        let shift = level as i32;
        let center = IVec2::new(fine_center.x >> shift, fine_center.y >> shift);
        view.level_scales.push(scale);
        view.clip_centers.push(center);
    }
}

/// Repositions patch entities to match the current clipmap ring layout.
/// Only does work when the snapped clip centers have actually changed.
fn update_patch_transforms(
    config: Res<TerrainConfig>,
    view: Res<TerrainViewState>,
    mut patch_entities: ResMut<PatchEntities>,
    mut query: Query<(&mut Transform, &mut components::TerrainPatchInstance)>,
) {
    // Skip if nothing has moved to a new grid cell.
    if view.clip_centers == patch_entities.last_clip_centers {
        return;
    }
    if view.clip_centers.is_empty() || patch_entities.entities.is_empty() {
        return;
    }

    let patches = build_patch_instances_for_view(&config, &view);

    if patches.len() != patch_entities.entities.len() {
        // Config changed — would need full respawn. Skip for now.
        warn!(
            "[Terrain] Patch count mismatch ({} vs {})",
            patches.len(),
            patch_entities.entities.len()
        );
        return;
    }

    for (entity, patch) in patch_entities.entities.iter().zip(patches.iter()) {
        if let Ok((mut transform, mut instance)) = query.get_mut(*entity) {
            transform.translation = Vec3::new(patch.origin_ws.x, 0.0, patch.origin_ws.y);
            transform.scale = Vec3::new(patch.patch_size_ws, 1.0, patch.patch_size_ws);
            instance.patch_origin_ws = patch.origin_ws;
            instance.patch_scale_ws = patch.patch_size_ws;
        }
    }

    patch_entities.last_clip_centers = view.clip_centers.clone();
}

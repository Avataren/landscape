pub mod clipmap;
pub mod clipmap_texture;
pub mod collision;
pub mod components;
pub mod config;
pub mod debug;
pub mod macro_color;
pub mod material;
pub mod material_slots;
pub mod math;
pub mod patch_mesh;
pub mod physics_colliders;
pub mod render;
pub mod residency;
pub mod resources;
pub mod streamer;
pub mod world_desc;

pub use debug::TerrainDebugPlugin;
pub use world_desc::TerrainSourceDesc;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    pbr::wireframe::{NoWireframe, WireframePlugin},
    prelude::*,
    render::storage::ShaderStorageBuffer,
};
use clipmap::{build_patch_instances_for_view_in_bounds, PatchInstanceCpu};
use clipmap_texture::{
    apply_tiles_to_clipmap, begin_terrain_upload_frame, compute_clip_levels,
    compute_initial_clip_levels, create_initial_clipmap_texture,
    create_initial_normal_clipmap_texture, update_clipmap_textures, TerrainClipmapState,
    TerrainClipmapUploads,
};
use collision::{update_collision_tiles, TerrainCollisionCache};
use components::TerrainCamera;
use config::TerrainConfig;
use macro_color::load_macro_color_texture;
use material::{TerrainMaterial, TerrainMaterialUniforms};
use material_slots::{sync_material_library_to_terrain_material, MaterialLibrary};
use math::{compute_needed_tiles_for_level, level_scale, snap_camera_to_level_grid};
use physics_colliders::spawn_global_heightfield;
use render::{gpu_types::PatchDescriptorGpu, TerrainRenderPlugin};
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
    /// Handle to the `ShaderStorageBuffer` containing one `PatchDescriptorGpu`
    /// per patch, in the same order as `entities`.  Kept in sync whenever the
    /// clip centers change (same condition that triggers Transform updates).
    pub patch_buffer_handle: Handle<ShaderStorageBuffer>,
    /// Shared mesh handle used by all terrain patches.
    pub mesh_handle: Handle<Mesh>,
    /// Shared material handle used by all terrain patches.
    pub material_handle: Handle<TerrainMaterial>,
}

// ---------------------------------------------------------------------------
// Main terrain plugin
// ---------------------------------------------------------------------------

pub struct TerrainPlugin {
    pub config: TerrainConfig,
    pub source: TerrainSourceDesc,
}

impl Default for TerrainPlugin {
    fn default() -> Self {
        Self {
            config: TerrainConfig::default(),
            source: TerrainSourceDesc::default(),
        }
    }
}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<WireframePlugin>() {
            app.add_plugins(WireframePlugin::default());
        }
        app
            // Custom terrain material
            .add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            // Resources
            .insert_resource(self.config.clone())
            .insert_resource(self.source.clone())
            .init_resource::<TerrainViewState>()
            .init_resource::<TerrainResidency>()
            .init_resource::<TerrainStreamQueue>()
            .init_resource::<TerrainCollisionCache>()
            .init_resource::<PatchEntities>()
            .init_resource::<TerrainClipmapUploads>()
            .init_resource::<MaterialLibrary>()
            // Startup
            .add_systems(Startup, (setup_tile_channel, setup_terrain).chain())
            .add_systems(PostStartup, preload_terrain_startup)
            .add_systems(
                PostStartup,
                spawn_global_heightfield.after(preload_terrain_startup),
            )
            // Update: ordered as per handoff spec
            .add_systems(First, begin_terrain_upload_frame)
            .add_systems(Update, update_terrain_view_state)
            .add_systems(
                Update,
                (
                    update_required_tiles,
                    request_tile_loads,
                    poll_tile_stream_jobs,
                    update_collision_tiles,
                    update_patch_transforms,
                    update_clipmap_textures,
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
            .add_systems(Update, sync_material_library_to_terrain_material)
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
    camera_q: Query<(&Transform, &TerrainCamera)>,
    mut state: ResMut<TerrainClipmapState>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut residency: ResMut<TerrainResidency>,
    mut collision: ResMut<TerrainCollisionCache>,
) {
    let levels = config.active_clipmap_levels();

    // --- Compute clip centers from the starting camera position ---------------
    let (cam_transform, terrain_cam) = match camera_q.single() {
        Ok(pair) => pair,
        Err(_) => {
            warn!("[Terrain] preload: no camera found");
            return;
        }
    };
    let cam_pos = cam_transform.translation;

    let scale_0 = level_scale(config.world_scale, 0);
    let bias_xz = forward_bias_xz(&config, cam_transform, terrain_cam.forward_bias_ratio);
    let fine_center = snap_camera_to_level_grid(cam_pos.xz() + bias_xz, scale_0);

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

    if let Some(image) = images.get(&state.height_texture_handle) {
        if let Some(data) = &image.data {
            state.height_cpu_data = data.clone();
        }
    }
    if let Some(image) = images.get(&state.normal_texture_handle) {
        if let Some(data) = &image.data {
            state.normal_cpu_data = data.clone();
        }
    }

    // Initialise collision cache params before the tile loop so upload_tile
    // can use them immediately (same initialisation update_collision_tiles does
    // every frame, but we can't wait until the first Update tick).
    collision.tile_size = config.tile_size;
    collision.world_scale = config.world_scale;
    collision.height_scale = config.height_scale;

    for tile in &results {
        let key = tile.key;
        residency.resident_cpu.insert(
            key,
            crate::terrain::resources::HeightTileCpu {
                key,
                data: tile.data.clone(),
                normal_data: tile.normal_data.clone(),
                tile_size: tile.tile_size,
            },
        );
        residency
            .tiles
            .insert(key, TileState::ResidentGpu { slot: 0 });
        residency.touch(key);
        // Populate the collision cache directly — preloaded tiles bypass
        // pending_upload (which apply_tiles_to_clipmap drains), so
        // update_collision_tiles would never see them otherwise.
        collision.upload_tile(tile);
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
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut patch_entities: ResMut<PatchEntities>,
    mut material_library: ResMut<MaterialLibrary>,
    mut commands: Commands,
) {
    let levels = config.active_clipmap_levels();

    // --- Clipmap texture array (Phase 5) ---
    // One R8Unorm layer per LOD level, each 512×512 texels.
    // Layers are regenerated live by `update_clipmap_textures` as the camera moves.
    let height_image = create_initial_clipmap_texture(&config);
    let height_cpu_data = height_image.data.clone().unwrap_or_default();
    let height_handle = images.add(height_image);
    let normal_image = create_initial_normal_clipmap_texture(&config);
    let normal_cpu_data = normal_image.data.clone().unwrap_or_default();
    let normal_handle = images.add(normal_image);
    let macro_color = load_macro_color_texture(&config, &desc);
    let macro_color_handle = images.add(macro_color.image);
    // Record whether the macro color texture was actually loaded so the
    // Materials panel can gate its "macro color override" toggle on it.
    material_library.macro_color_loaded = macro_color.enabled;

    let base_patch_size = config.patch_resolution as f32 * config.world_scale;
    let bounds_fade_distance = config.tile_size as f32 * config.world_scale * 4.0;

    // --- Patch mesh (shared by all entities) ---
    let patch_mesh = patch_mesh::build_patch_mesh(config.patch_resolution);
    let mesh_handle = meshes.add(patch_mesh);

    // --- Build initial patch list before creating the material so we can pass
    //     the real storage-buffer handle into the material from the start ---
    let view = TerrainViewState::default();
    let patches: Vec<PatchInstanceCpu> =
        build_patch_instances_for_view_in_bounds(&config, &view, desc.world_min, desc.world_max);

    // Build the initial patch storage buffer so the vertex shader can read
    // per-patch data (origin, size, lod) without decoding the Transform matrix.
    let patch_buffer_handle = {
        let descs: Vec<PatchDescriptorGpu> = patches
            .iter()
            .map(|p| PatchDescriptorGpu {
                origin_ws: [p.origin_ws.x, p.origin_ws.y],
                patch_size_ws: p.patch_size_ws,
                lod_level: p.lod_level,
                morph_start: p.morph_start,
                morph_end: p.morph_end,
                patch_kind: p.patch_kind,
                _pad0: 0,
            })
            .collect();
        let ssb = ShaderStorageBuffer::new(
            bytemuck::cast_slice(&descs),
            RenderAssetUsages::RENDER_WORLD,
        );
        storage_buffers.add(ssb)
    };

    patch_entities.entities.clear();
    patch_entities.patch_buffer_handle = patch_buffer_handle.clone();
    patch_entities.mesh_handle = mesh_handle.clone();

    let mat_handle = terrain_materials.add(TerrainMaterial {
        height_texture: height_handle.clone(),
        macro_color_texture: macro_color_handle,
        normal_texture: normal_handle.clone(),
        patch_buffer: patch_buffer_handle,
        params: TerrainMaterialUniforms {
            height_scale: config.height_scale,
            base_patch_size,
            morph_start_ratio: config.morph_start_ratio,
            ring_patches: config.ring_patches as f32,
            num_lod_levels: levels as f32,
            patch_resolution: config.patch_resolution as f32,
            world_bounds: Vec4::new(
                desc.world_min.x,
                desc.world_min.y,
                desc.world_max.x,
                desc.world_max.y,
            ),
            bounds_fade: Vec4::new(
                bounds_fade_distance,
                // Macro color override starts off — the procedural library
                // drives albedo by default.  The Materials panel toggles this
                // via `MaterialLibrary::use_macro_color_override`; the sync
                // system writes the final value each time the library changes.
                0.0,
                if config.macro_color_flip_v { 1.0 } else { 0.0 },
                0.0,
            ),
            // debug_flags.y = 1.0 → fragment shader samples baked normals.
            // Enabled whenever a normal tile root is configured; falls back to
            // finite-difference normals on the height clipmap otherwise.
            debug_flags: Vec4::new(
                0.0,
                if desc.normal_root.is_some() { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ),
            clip_levels: compute_initial_clip_levels(&config),
            // Slot data is filled in on the first Update tick by
            // sync_material_library_to_terrain_material.  Until then, the
            // shader falls back to the built-in procedural palette (count == 0).
            slot_header: Vec4::ZERO,
            slots: [material::MaterialSlotGpu::default(); material::MAX_SHADER_MATERIAL_SLOTS],
        },
    });
    patch_entities.material_handle = mat_handle.clone();

    // Insert the runtime state resource so `update_clipmap_textures` can find it.
    // Initialise last_clip_centers to ZERO (matching the initial texture), so the
    // first real camera position triggers a full regeneration on the first frame.
    commands.insert_resource(TerrainClipmapState {
        height_texture_handle: height_handle,
        normal_texture_handle: normal_handle,
        material_handle: mat_handle.clone(),
        height_cpu_data,
        normal_cpu_data,
        last_clip_centers: vec![IVec2::ZERO; levels as usize],
        // Sentinel forces a full tile re-apply on the first frame.
        tile_apply_centers: vec![IVec2::new(i32::MAX, i32::MAX); levels as usize],
    });

    respawn_patch_entities(&mut commands, &mut patch_entities, &patches);

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
    camera_q: Query<(&Transform, &TerrainCamera)>,
    mut view: ResMut<TerrainViewState>,
) {
    let Ok((cam, terrain_cam)) = camera_q.single() else {
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
    //
    // Optionally bias the finest center along the camera's forward XZ
    // direction before snapping.  Since every coarser center is derived from
    // the biased fine center via right shift, the entire ring stack moves
    // coherently and inner-hole alignment is preserved.
    let scale_0 = level_scale(config.world_scale, 0);
    let bias_xz = forward_bias_xz(&config, cam, terrain_cam.forward_bias_ratio);
    let fine_center = snap_camera_to_level_grid(cam_pos.xz() + bias_xz, scale_0);

    for level in 0..config.active_clipmap_levels() {
        let scale = level_scale(config.world_scale, level);
        // Right-shift the finest center to get each coarser center.
        let shift = level as i32;
        let center = IVec2::new(fine_center.x >> shift, fine_center.y >> shift);
        view.level_scales.push(scale);
        view.clip_centers.push(center);
    }
}

/// Offset the finest clipmap center along the camera's forward XZ direction.
///
/// Scaled by `ratio * half_ring_L0_ws`: at ratio = 0.5 the LOD-0 ring sits
/// ~half its radius ahead of the camera, which is a good default for near-
/// horizontal views where the visible ground is forward of the player rather
/// than directly below.  Returns zero if the ratio or the projected forward
/// vector is negligible.
fn forward_bias_xz(config: &TerrainConfig, cam: &Transform, ratio: f32) -> Vec2 {
    if ratio.abs() <= 1e-6 {
        return Vec2::ZERO;
    }
    let forward: Vec3 = cam.forward().into();
    let forward_xz = Vec2::new(forward.x, forward.z);
    let len = forward_xz.length();
    if len <= 1e-3 {
        return Vec2::ZERO;
    }
    let half_ring_l0 = config.ring_patches as f32
        * config.patch_resolution as f32
        * level_scale(config.world_scale, 0)
        * 0.5;
    (forward_xz / len) * (half_ring_l0 * ratio)
}

/// Repositions patch entities and refreshes the patch storage buffer to match
/// the current clipmap ring layout.  Only does work when the snapped clip
/// centers have actually changed.
fn update_patch_transforms(
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    view: Res<TerrainViewState>,
    mut patch_entities: ResMut<PatchEntities>,
    mut query: Query<(&mut Transform, &mut components::TerrainPatchInstance)>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut commands: Commands,
) {
    if view.clip_centers.is_empty() || patch_entities.entities.is_empty() {
        return;
    }

    let positions_changed = view.clip_centers != patch_entities.last_clip_centers;

    // Only rebuild the patch list and upload the storage buffer when the clip
    // grid changes.  Entities carry NoFrustumCulling so Bevy never hides them;
    // the GPU rasteriser discards out-of-view triangles at near-zero cost, so
    // storage-buffer frustum culling (which had incorrect results for non-ground-
    // level cameras in Bevy 0.18) is not needed.
    if !positions_changed {
        return;
    }

    let patches =
        build_patch_instances_for_view_in_bounds(&config, &view, desc.world_min, desc.world_max);

    let mut respawned = false;
    if patches.len() != patch_entities.entities.len() {
        respawn_patch_entities(&mut commands, &mut patch_entities, &patches);
        respawned = true;
    }

    if let Some(ssb) = storage_buffers.get_mut(&patch_entities.patch_buffer_handle) {
        let descs: Vec<PatchDescriptorGpu> = patches
            .iter()
            .map(|p| PatchDescriptorGpu {
                origin_ws: [p.origin_ws.x, p.origin_ws.y],
                patch_size_ws: p.patch_size_ws,
                lod_level: p.lod_level,
                morph_start: p.morph_start,
                morph_end: p.morph_end,
                patch_kind: p.patch_kind,
                _pad0: 0,
            })
            .collect();
        ssb.data = Some(bytemuck::cast_slice(&descs).to_vec());
    }

    if respawned {
        patch_entities.last_clip_centers = view.clip_centers.clone();
        return;
    }

    // Only update ECS Transforms when the clip grid cell changes — these drive
    // Bevy's own visibility / AABB checks for the entity, not the vertex path.
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

fn respawn_patch_entities(
    commands: &mut Commands,
    patch_entities: &mut PatchEntities,
    patches: &[PatchInstanceCpu],
) {
    for entity in patch_entities.entities.drain(..) {
        commands.entity(entity).despawn();
    }

    for patch in patches {
        let entity = commands
            .spawn((
                Mesh3d(patch_entities.mesh_handle.clone()),
                MeshMaterial3d(patch_entities.material_handle.clone()),
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
                NoWireframe,
                NoFrustumCulling,
            ))
            .id();

        patch_entities.entities.push(entity);
    }
}

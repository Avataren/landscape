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
pub mod pbr_textures;
pub mod physics_colliders;
pub mod render;
pub mod residency;
pub mod resources;
pub mod streamer;
pub mod world_desc;

pub use debug::TerrainDebugPlugin;
pub use world_desc::TerrainSourceDesc;

use bevy::{
    camera::primitives::Aabb,
    camera::visibility::NoAutoAabb,
    pbr::wireframe::{NoWireframe, WireframePlugin},
    prelude::*,
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
use math::{compute_needed_tiles_for_level, level_scale, snap_camera_to_nested_clipmap_grid};
use pbr_textures::{
    rebuild_pbr_textures_system, PbrRebuildProgress, PbrRebuildState, PbrTexturesDirty,
};
use physics_colliders::spawn_global_heightfield;
use physics_colliders::{spawn_global_heightfield_for_desc, GlobalTerrainHeightfield};
use render::TerrainRenderPlugin;
use residency::update_required_tiles;
use resources::{
    HeightTileCpu, TerrainResidency, TerrainStreamQueue, TerrainViewState, TileKey, TileState,
};
use streamer::{load_tile_data, poll_tile_stream_jobs, request_tile_loads, setup_tile_channel};

// ---------------------------------------------------------------------------
// ReloadTerrainRequest — hot-swap the active terrain without restarting
// ---------------------------------------------------------------------------

/// Send this message to replace the active terrain at runtime.
///
/// The `reload_terrain_system` will update all live resources on the next
/// frame: streamer, clipmap textures, material uniforms, and macro color.
#[derive(Message)]
pub struct ReloadTerrainRequest {
    pub config: TerrainConfig,
    pub source: TerrainSourceDesc,
    pub material_library: MaterialLibrary,
}

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
            // Messages
            .add_message::<ReloadTerrainRequest>()
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
            .init_resource::<PbrTexturesDirty>()
            .init_resource::<PbrRebuildProgress>()
            .init_resource::<PbrRebuildState>()
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
            .add_systems(Update, rebuild_pbr_textures_system)
            .add_systems(
                Update,
                reload_terrain_system.before(update_terrain_view_state),
            )
            // Render sub-plugin
            .add_plugins(TerrainRenderPlugin);
    }
}

// ---------------------------------------------------------------------------
// Startup systems
// ---------------------------------------------------------------------------

/// Computes the terrain edge fade distance.
///
/// The fade is applied with `smoothstep(-fade_dist, 0, edge_dist)`, so it only
/// affects vertices whose world position is *outside* the terrain bounds.  Any
/// vertex inside (edge_dist >= 0) always receives full height — meaning the
/// fade_dist value no longer influences visible terrain and can safely be a
/// generous multiple of the coarsest patch size.
fn compute_bounds_fade_dist(
    tile_size: u32,
    world_scale: f32,
    _world_min: Vec2,
    _world_max: Vec2,
) -> f32 {
    // Cover the maximum overshoot: patches that pass the bounds intersection
    // test can extend at most one patch_size_ws past the terrain boundary.
    // tile_size * 4 covers even coarse rings comfortably.
    tile_size as f32 * world_scale * 4.0
}

fn has_baked_normal_tiles(desc: &TerrainSourceDesc) -> bool {
    if let Some(root) = desc.normal_root.as_deref() {
        return std::path::Path::new(root).exists();
    }

    desc.tile_root
        .as_deref()
        .map(|root| root.join("normal").exists())
        .unwrap_or(false)
}

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
    let Ok((cam_transform, terrain_cam)) = camera_q.single() else {
        warn!("[Terrain] preload: no camera found");
        return;
    };

    let (clip_centers, level_scales) =
        compute_preload_centers(&config, cam_transform, terrain_cam.forward_bias_ratio);

    sync_preload_tiles(
        &config,
        &desc,
        &clip_centers,
        &level_scales,
        0, // startup generation
        &mut state,
        &mut images,
        &mut materials,
        &mut residency,
        &mut collision,
    );
}

/// Computes per-LOD clip centers and level scales from the current camera pose.
fn compute_preload_centers(
    config: &TerrainConfig,
    cam: &Transform,
    forward_bias_ratio: f32,
) -> (Vec<IVec2>, Vec<f32>) {
    let scale_0 = level_scale(config.world_scale, 0);
    let bias_xz = forward_bias_xz(config, cam, forward_bias_ratio);
    let fine_center = snap_camera_to_nested_clipmap_grid(
        cam.translation.xz() + bias_xz,
        scale_0,
        config.active_clipmap_levels(),
    );
    let levels = config.active_clipmap_levels();
    let clip_centers: Vec<IVec2> = (0..levels)
        .map(|l| IVec2::new(fine_center.x >> l as i32, fine_center.y >> l as i32))
        .collect();
    let level_scales: Vec<f32> = (0..levels)
        .map(|l| level_scale(config.world_scale, l))
        .collect();
    (clip_centers, level_scales)
}

/// Synchronously loads all terrain tiles visible from the given clip centers,
/// writes them into the GPU clipmap images (via `img.data`), syncs CPU mirrors,
/// populates residency + collision, and updates the material `clip_levels`
/// uniform. `last_clip_centers` is set to the preloaded values, but
/// `tile_apply_centers` is left at a sentinel and `clipmap_needs_rebuild` is
/// raised so the first Update tick re-applies every resident tile and performs
/// explicit GPU uploads through the runtime clipmap upload path.
///
/// Used at startup (`generation = 0`) and after a hot-reload (`generation =
/// freshly-bumped reload_generation`).
#[allow(clippy::too_many_arguments)]
fn sync_preload_tiles(
    config: &TerrainConfig,
    desc: &TerrainSourceDesc,
    clip_centers: &[IVec2],
    level_scales: &[f32],
    current_gen: u64,
    state: &mut TerrainClipmapState,
    images: &mut Assets<Image>,
    materials: &mut Assets<TerrainMaterial>,
    residency: &mut TerrainResidency,
    collision: &mut TerrainCollisionCache,
) {
    let levels = clip_centers.len();

    // Collect all tile keys for every LOD ring.
    let mut all_keys: Vec<TileKey> = Vec::new();
    for level in 0..levels {
        let keys = compute_needed_tiles_for_level(
            clip_centers[level],
            level_scales[level],
            config.patch_resolution,
            config.ring_patches,
            config.tile_size,
            level as u8,
        );
        all_keys.extend(keys);
    }
    all_keys.sort_by(|a, b| b.level.cmp(&a.level));
    all_keys.dedup();

    // Load all tiles in parallel, blocking until all complete.
    let tile_size = config.tile_size;
    let world_scale = config.world_scale;
    let height_scale = config.height_scale;
    let max_mip_level = desc.max_mip_level;
    let use_procedural = config.procedural_fallback;
    let tile_root = desc.tile_root.clone();
    let normal_root = desc.normal_root.as_ref().map(std::path::PathBuf::from);
    let world_bounds = Some((desc.world_min, desc.world_max));

    let results: Vec<HeightTileCpu> = std::thread::scope(|s| {
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

    let res = config.clipmap_resolution();
    let height_bpl = (res * res * 2) as usize; // R16Unorm
    let normal_bpl = (res * res * 2) as usize; // RG8Snorm
    let half = (res / 2) as i32;
    let ts = config.tile_size;

    // Write height tiles into the GPU image CPU buffer.
    if let Some(image) = images.get_mut(&state.height_texture_handle) {
        if let Some(ref mut img_data) = image.data {
            for tile in &results {
                let level = tile.key.level as usize;
                if level >= levels {
                    continue;
                }
                let clip_center = clip_centers[level];
                let layer_base = level * height_bpl;
                for row in 0..ts {
                    for col in 0..ts {
                        let gx = tile.key.x * ts as i32 + col as i32;
                        let gz = tile.key.y * ts as i32 + row as i32;
                        if (gx - clip_center.x) < -half
                            || (gx - clip_center.x) >= half
                            || (gz - clip_center.y) < -half
                            || (gz - clip_center.y) >= half
                        {
                            continue;
                        }
                        let tx = gx.rem_euclid(res as i32) as usize;
                        let tz = gz.rem_euclid(res as i32) as usize;
                        let dst = layer_base + (tz * res as usize + tx) * 2;
                        let v = (tile.data[(row * ts + col) as usize] * 65535.0) as u16;
                        if dst + 2 <= img_data.len() {
                            img_data[dst..dst + 2].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
    } else {
        warn!("[Terrain] preload: height clipmap image not found");
        return;
    }

    // Write normal tiles into the GPU image CPU buffer.
    if let Some(image) = images.get_mut(&state.normal_texture_handle) {
        if let Some(ref mut img_data) = image.data {
            for tile in &results {
                let level = tile.key.level as usize;
                if level >= levels {
                    continue;
                }
                let clip_center = clip_centers[level];
                let layer_base = level * normal_bpl;
                for row in 0..ts {
                    for col in 0..ts {
                        let gx = tile.key.x * ts as i32 + col as i32;
                        let gz = tile.key.y * ts as i32 + row as i32;
                        if (gx - clip_center.x) < -half
                            || (gx - clip_center.x) >= half
                            || (gz - clip_center.y) < -half
                            || (gz - clip_center.y) >= half
                        {
                            continue;
                        }
                        let tx = gx.rem_euclid(res as i32) as usize;
                        let tz = gz.rem_euclid(res as i32) as usize;
                        let dst = layer_base + (tz * res as usize + tx) * 2;
                        let enc = tile.normal_data[(row * ts + col) as usize];
                        if dst + 2 <= img_data.len() {
                            img_data[dst..dst + 2].copy_from_slice(&enc);
                        }
                    }
                }
            }
        }
    } else {
        warn!("[Terrain] preload: normal clipmap image not found");
        return;
    }

    // Sync CPU mirrors from the freshly-written image data.
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

    // Initialise collision cache parameters.
    collision.tile_size = config.tile_size;
    collision.world_scale = config.world_scale;
    collision.height_scale = config.height_scale;

    // Populate residency and collision caches.
    for tile in &results {
        let key = tile.key;
        residency.resident_cpu.insert(
            key,
            HeightTileCpu {
                key,
                data: tile.data.clone(),
                normal_data: tile.normal_data.clone(),
                tile_size: tile.tile_size,
                generation: current_gen,
            },
        );
        residency
            .tiles
            .insert(key, TileState::ResidentGpu { slot: 0 });
        residency.touch(key);
        collision.upload_tile(tile);
    }

    // Update clip_levels uniform so UVs map to the actual preloaded centers.
    if let Some(mat) = materials.get_mut(&state.material_handle) {
        mat.params.clip_levels = compute_clip_levels(config, clip_centers, level_scales);
    }

    // Keep the procedural update path in sync with the preloaded centers so it
    // doesn't overwrite the freshly populated CPU mirrors on the first frame.
    state.last_clip_centers = clip_centers.to_vec();
    // Force the tile-apply path to rebuild and upload every resident tile on the
    // first Update tick. Relying only on the asset full-upload path can leave
    // the near-field L0 window stale if the texture's initial contents never
    // make it to the GPU before incremental strip uploads begin.
    state.tile_apply_centers = vec![IVec2::new(i32::MAX, i32::MAX); levels];
    residency.clipmap_needs_rebuild = true;

    info!(
        "[Terrain] Preload: {} tiles loaded synchronously.",
        results.len()
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
    let bounds_fade_distance = compute_bounds_fade_dist(
        config.tile_size,
        config.world_scale,
        desc.world_min,
        desc.world_max,
    );

    // --- Patch mesh (shared by all entities) ---
    let patch_mesh = patch_mesh::build_patch_mesh(config.patch_resolution);
    let mesh_handle = meshes.add(patch_mesh);

    // --- Build initial patch list before creating the material ---
    let view = TerrainViewState::default();
    let patches: Vec<PatchInstanceCpu> =
        build_patch_instances_for_view_in_bounds(&config, &view, desc.world_min, desc.world_max);

    patch_entities.entities.clear();
    patch_entities.mesh_handle = mesh_handle.clone();

    let assets_dir = std::path::Path::new("assets");
    let pbr_albedo_handle = images.add(pbr_textures::build_albedo_array(
        &material_library.slots,
        assets_dir,
    ));
    let pbr_normal_handle = images.add(pbr_textures::build_normal_array(
        &material_library.slots,
        assets_dir,
    ));
    let pbr_orm_handle = images.add(pbr_textures::build_orm_array(
        &material_library.slots,
        assets_dir,
    ));

    let mat_handle = terrain_materials.add(TerrainMaterial {
        height_texture: height_handle.clone(),
        macro_color_texture: macro_color_handle,
        normal_texture: normal_handle.clone(),
        pbr_albedo_array: pbr_albedo_handle,
        pbr_normal_array: pbr_normal_handle,
        pbr_orm_array: pbr_orm_handle,
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
            // Enabled whenever explicit normal tiles are configured or an
            // implicit `tile_root/normal` hierarchy exists; falls back to
            // finite-difference normals on the height clipmap otherwise.
            debug_flags: Vec4::new(
                0.0,
                if has_baked_normal_tiles(&desc) {
                    1.0
                } else {
                    0.0
                },
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

    respawn_patch_entities(
        &mut commands,
        &mut patch_entities,
        &patches,
        config.height_scale,
        config.patch_resolution,
    );

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
    let fine_center = snap_camera_to_nested_clipmap_grid(
        cam_pos.xz() + bias_xz,
        scale_0,
        config.active_clipmap_levels(),
    );

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
    mut commands: Commands,
) {
    if view.clip_centers.is_empty() {
        return;
    }

    let positions_changed = view.clip_centers != patch_entities.last_clip_centers;

    // Only rebuild the patch list when the clip grid changes. Patch entities
    // carry a conservative local-space AABB covering the displaced terrain, so
    // Bevy can frustum-cull them and feed the built-in GPU preprocessing path.
    if !positions_changed {
        return;
    }

    let patches =
        build_patch_instances_for_view_in_bounds(&config, &view, desc.world_min, desc.world_max);

    let mut respawned = false;
    if patches.len() != patch_entities.entities.len() {
        respawn_patch_entities(
            &mut commands,
            &mut patch_entities,
            &patches,
            config.height_scale,
            config.patch_resolution,
        );
        respawned = true;
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

// ---------------------------------------------------------------------------
// Hot-reload system
// ---------------------------------------------------------------------------

/// Replaces the active terrain by tearing down all CPU and GPU resources and
/// rebuilding them fresh from the new config/source.
///
/// Strategy: create new GPU clipmap textures → update the material to reference
/// them (Bevy frees the old textures automatically when their refcount drops) →
/// despawn all old patch entities → synchronously preload tiles into the new
/// textures → let `update_patch_transforms` respawn patches on the same frame.
#[allow(clippy::too_many_arguments)]
fn reload_terrain_system(
    mut reload_rx: MessageReader<ReloadTerrainRequest>,
    mut config: ResMut<TerrainConfig>,
    mut desc: ResMut<TerrainSourceDesc>,
    mut material_library: ResMut<MaterialLibrary>,
    mut residency: ResMut<TerrainResidency>,
    mut stream_queue: ResMut<TerrainStreamQueue>,
    mut view: ResMut<TerrainViewState>,
    mut clipmap_state: ResMut<TerrainClipmapState>,
    mut collision: ResMut<TerrainCollisionCache>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut patch_entities: ResMut<PatchEntities>,
    mut pbr_dirty: ResMut<PbrTexturesDirty>,
    camera_q: Query<(&Transform, &TerrainCamera)>,
    global_heightfield_q: Query<Entity, With<GlobalTerrainHeightfield>>,
) {
    for req in reload_rx.read() {
        let new_config = req.config.clone();
        let new_desc = req.source.clone();
        let new_library = req.material_library.clone();

        info!(
            "[Terrain] Hot-reload: tile_root={:?}  world_scale={:.2}  \
             height_scale={:.2}  clipmap_levels={}",
            new_desc.tile_root,
            new_config.world_scale,
            new_config.height_scale,
            new_config.clipmap_levels,
        );

        // --- 1. Update the live config/source/library resources ---------------
        *config = new_config.clone();
        *desc = new_desc.clone();
        *material_library = new_library;
        pbr_dirty.0 = true; // rebuild PBR texture arrays from the newly loaded library

        // --- 2. Despawn ALL old patch entities immediately --------------------
        // This prevents stale geometry (wrong world_scale transforms) from
        // appearing in the render — especially important for shadow cascades.
        for entity in patch_entities.entities.drain(..) {
            commands.entity(entity).despawn();
        }
        for entity in &global_heightfield_q {
            commands.entity(entity).despawn();
        }
        patch_entities.last_clip_centers.clear();

        // --- 3. Create brand-new GPU clipmap textures -------------------------
        // Always MAX_SUPPORTED_CLIPMAP_LEVELS layers deep so we never need to
        // recreate textures when switching between terrains that have different
        // mip-level counts.  The old texture handles lose their last reference
        // here (material update below + clipmap_state update further down) and
        // Bevy will free the GPU allocations automatically.
        let new_height_image = create_initial_clipmap_texture(&new_config);
        let new_normal_image = create_initial_normal_clipmap_texture(&new_config);
        let new_height_handle = images.add(new_height_image);
        let new_normal_handle = images.add(new_normal_image);

        // --- 4. Update the material: new texture handles + updated uniforms ---
        if let Some(mat) = materials.get_mut(&clipmap_state.material_handle) {
            mat.height_texture = new_height_handle.clone();
            mat.normal_texture = new_normal_handle.clone();
            mat.params.height_scale = new_config.height_scale;
            mat.params.base_patch_size =
                new_config.patch_resolution as f32 * new_config.world_scale;
            mat.params.morph_start_ratio = new_config.morph_start_ratio;
            mat.params.ring_patches = new_config.ring_patches as f32;
            mat.params.num_lod_levels = new_config.active_clipmap_levels() as f32;
            mat.params.patch_resolution = new_config.patch_resolution as f32;
            mat.params.world_bounds = Vec4::new(
                new_desc.world_min.x,
                new_desc.world_min.y,
                new_desc.world_max.x,
                new_desc.world_max.y,
            );
            mat.params.bounds_fade.x = compute_bounds_fade_dist(
                new_config.tile_size,
                new_config.world_scale,
                new_desc.world_min,
                new_desc.world_max,
            );
            mat.params.bounds_fade.z = if new_config.macro_color_flip_v {
                1.0
            } else {
                0.0
            };
            mat.params.debug_flags.y = if has_baked_normal_tiles(&new_desc) {
                1.0
            } else {
                0.0
            };

            let new_macro = load_macro_color_texture(&new_config, &new_desc);
            material_library.macro_color_loaded = new_macro.enabled;
            if let Some(img) = images.get_mut(&mat.macro_color_texture) {
                *img = new_macro.image;
            }
        }

        // --- 5. Point clipmap state at the new textures -----------------------
        // CPU mirrors will be populated by sync_preload_tiles below.
        clipmap_state.height_texture_handle = new_height_handle;
        clipmap_state.normal_texture_handle = new_normal_handle;
        clipmap_state.last_clip_centers.clear();
        clipmap_state.tile_apply_centers.clear();

        // --- 6. Reset all streaming / collision / view state ------------------
        let new_gen = stream_queue.reload_generation + 1;
        *residency = TerrainResidency::default();
        *stream_queue = TerrainStreamQueue::default();
        stream_queue.reload_generation = new_gen;
        *collision = TerrainCollisionCache::default();
        *view = TerrainViewState::default();

        // --- 7. Synchronous preload -------------------------------------------
        // Reuses the same logic as preload_terrain_startup: loads all visible
        // tiles in parallel on the calling thread, writes them directly into
        // img.data (triggering a full GPU re-upload via prepare_assets on the
        // next render frame), and sets last_clip_centers so streaming systems
        // see no delta on the first Update tick.
        let Ok((cam_transform, terrain_cam)) = camera_q.single() else {
            // No camera found — fall back to sentinel-driven streaming.
            let levels = new_config.active_clipmap_levels() as usize;
            let sentinel = IVec2::new(i32::MAX, i32::MAX);
            clipmap_state.last_clip_centers = vec![sentinel; levels];
            clipmap_state.tile_apply_centers = vec![sentinel; levels];
            residency.clipmap_needs_rebuild = true;
            warn!("[Terrain] Hot-reload: no camera found, skipping synchronous preload.");
            continue;
        };

        let (clip_centers, level_scales) =
            compute_preload_centers(&new_config, cam_transform, terrain_cam.forward_bias_ratio);

        sync_preload_tiles(
            &new_config,
            &new_desc,
            &clip_centers,
            &level_scales,
            new_gen,
            &mut clipmap_state,
            &mut images,
            &mut materials,
            &mut residency,
            &mut collision,
        );

        spawn_global_heightfield_for_desc(&new_desc, &new_config, &mut commands);
    }
}

fn respawn_patch_entities(
    commands: &mut Commands,
    patch_entities: &mut PatchEntities,
    patches: &[PatchInstanceCpu],
    height_scale: f32,
    patch_resolution: u32,
) {
    let local_aabb = terrain_patch_local_aabb(height_scale, patch_resolution);

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
                local_aabb,
                // Keep Bevy's built-in frustum culling enabled, but stop its
                // automatic bounds pass from replacing the conservative
                // displaced-terrain AABB with the flat source mesh bounds.
                NoAutoAabb,
                NoWireframe,
            ))
            .id();

        patch_entities.entities.push(entity);
    }
}

fn terrain_patch_local_aabb(height_scale: f32, patch_resolution: u32) -> Aabb {
    // Geomorphing can snap vertices onto the next coarser lattice, which moves
    // seam vertices up to one coarse texel (2 / patch_resolution in local X/Z)
    // outside the nominal [0, 1] patch footprint. Pad the local bounds so
    // Bevy's frustum culler stays conservative for near seam patches.
    let morph_pad = 2.0 / patch_resolution.max(1) as f32;
    Aabb::from_min_max(
        Vec3::new(-morph_pad, 0.0, -morph_pad),
        Vec3::new(1.0 + morph_pad, height_scale.max(1.0), 1.0 + morph_pad),
    )
}

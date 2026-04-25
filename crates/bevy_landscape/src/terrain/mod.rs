pub mod clipmap;
pub mod clipmap_texture;
pub mod collision;
pub mod components;
pub mod config;
pub mod debug;
pub mod detail_synthesis;
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
pub mod source_heightmap;
pub mod streamer;
pub mod world_desc;

pub use debug::TerrainDebugPlugin;
pub use world_desc::TerrainSourceDesc;

use bevy::{
    camera::primitives::Aabb,
    camera::visibility::{NoAutoAabb, NoFrustumCulling},
    pbr::wireframe::{NoWireframe, WireframePlugin},
    prelude::*,
};
use clipmap::build_patch_instances_for_view_in_bounds;
use clipmap::PatchInstanceCpu;
use clipmap::{build_trim_instances_for_view_in_bounds, TrimInstanceCpu};
use clipmap_texture::{
    apply_tiles_to_clipmap, begin_terrain_upload_frame, compute_clip_levels,
    compute_initial_clip_levels, create_initial_clipmap_texture,
    create_initial_normal_clipmap_texture, update_clipmap_textures, TerrainClipmapState,
    TerrainClipmapUploads,
};
use collision::{update_collision_tiles, TerrainCollisionCache};
use components::TerrainCamera;
use config::TerrainConfig;
use detail_synthesis::{update_synthesis_state, DetailSynthesisPlugin};
use macro_color::load_macro_color_texture;
use source_heightmap::{load_source_heightmap, SourceHeightmapState};
use material::{TerrainMaterial, TerrainMaterialUniforms};
use material_slots::{sync_material_library_to_terrain_material, MaterialLibrary};
use math::{level_scale, snap_camera_to_nested_clipmap_grid};
use pbr_textures::{
    rebuild_pbr_textures_system, PbrRebuildProgress, PbrRebuildState, PbrTexturesDirty,
};
use physics_colliders::{
    apply_terrain_collider_result, cancel_collider_task_on_reload, start_terrain_collider_build,
    LocalColliderState, LocalColliderTask, ShowTerrainCollision,
};
use render::TerrainRenderPlugin;
use residency::update_required_tiles;
use resources::{
    TerrainResidency, TerrainStreamQueue, TerrainViewState,
};
use streamer::{poll_tile_stream_jobs, request_tile_loads, setup_tile_channel};

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
    /// All spawned terrain entities: blocks first, then trim strips.
    pub entities: Vec<Entity>,
    /// Number of block entities at the start of `entities`.
    pub block_count: usize,
    /// Cached clip centers from last update (used to skip no-op frames).
    pub last_clip_centers: Vec<IVec2>,
    /// Shared mesh handle used by all terrain block patches.
    pub mesh_handle: Handle<Mesh>,
    /// Shared material handle used by all terrain patches.
    pub material_handle: Handle<TerrainMaterial>,
    /// Trim mesh: 1 quad wide × 2m quads tall (left / vertical strip).
    pub trim_v_mesh_handle: Handle<Mesh>,
    /// Trim mesh: 2m quads wide × 1 quad tall (bottom / horizontal strip).
    pub trim_h_mesh_handle: Handle<Mesh>,
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
            .init_resource::<LocalColliderState>()
            .init_resource::<LocalColliderTask>()
            .init_resource::<ShowTerrainCollision>()
            .init_resource::<PatchEntities>()
            .init_resource::<TerrainClipmapUploads>()
            .init_resource::<MaterialLibrary>()
            .init_resource::<PbrTexturesDirty>()
            .init_resource::<PbrRebuildProgress>()
            .init_resource::<PbrRebuildState>()
            // Startup
            // Note: SourceHeightmapState is inserted by setup_terrain, not init'd here,
            // because it needs config/desc to load tile data.
            .add_systems(Startup, (setup_tile_channel, setup_terrain).chain())
            .add_systems(PostStartup, preload_terrain_startup)
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
                (
                    start_terrain_collider_build
                        .after(update_collision_tiles)
                        .after(update_terrain_view_state),
                    apply_terrain_collider_result,
                    cancel_collider_task_on_reload,
                ),
            )
            .add_systems(
                Update,
                apply_tiles_to_clipmap // Phase 5: tile-based GPU upload
                    .after(poll_tile_stream_jobs)
                    .after(update_clipmap_textures)
                    .after(update_terrain_view_state),
            )
            .add_systems(
                Update,
                update_patch_aabbs
                    .after(apply_tiles_to_clipmap)
                    .after(update_clipmap_textures),
            )
            .add_systems(Update, sync_material_library_to_terrain_material)
            .add_systems(Update, rebuild_pbr_textures_system)
            .add_systems(
                Update,
                update_synthesis_state.after(update_terrain_view_state),
            )
            .add_systems(
                Update,
                reload_terrain_system.before(update_terrain_view_state),
            )
            // Render sub-plugins
            .add_plugins(TerrainRenderPlugin)
            .add_plugins(DetailSynthesisPlugin);
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
    let scale_0 = level_scale(config.lod0_mesh_spacing, 0);
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
        .map(|l| level_scale(config.lod0_mesh_spacing, l))
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
    // The detail-synthesis compute pass writes every clipmap layer from the
    // global source-heightmap texture (with optional fBM detail for fine LODs),
    // so no per-ring CPU tile uploads are needed for rendering.  This preload
    // just primes the clipmap state and material uniforms; the first synthesis
    // dispatch fills the texture array with correct heights.
    let _ = images;
    let _ = residency;
    let levels = clip_centers.len();

    collision.tile_size = config.tile_size;
    collision.world_scale = config.world_scale;
    collision.height_scale = config.height_scale;

    if let Some(mat) = materials.get_mut(&state.material_handle) {
        mat.params.clip_levels = compute_clip_levels(config, clip_centers, level_scales);
    }

    state.last_clip_centers = clip_centers.to_vec();
    state.tile_apply_centers = vec![IVec2::new(i32::MAX, i32::MAX); levels];

    let _ = current_gen;
    let _ = desc;

    info!("[Terrain] Preload: synthesis-only path (no tile uploads).");
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
    material_library.macro_color_loaded = macro_color.enabled;

    // --- Source heightmap (single large GPU texture at max_mip_level) ---
    let source_hmap = load_source_heightmap(&config, &desc);
    let source_hmap_handle = images.add(source_hmap.image);
    commands.insert_resource(SourceHeightmapState {
        handle: source_hmap_handle.clone(),
        world_origin: source_hmap.world_origin,
        world_extent: source_hmap.world_extent,
        texel_size: source_hmap.texel_size,
    });

    // base_patch_size = world-space size of one grid unit at LOD 0 = lod0_mesh_spacing.
    // The vertex shader derives LOD from: round(log2(level_scale_ws / base_patch_size)).
    let base_patch_size = config.lod0_mesh_spacing;
    let bounds_fade_distance = compute_bounds_fade_dist(
        config.tile_size,
        config.world_scale,
        desc.world_min,
        desc.world_max,
    );

    // --- Canonical block mesh (shared by all block instances) ---
    let patch_mesh = patch_mesh::build_block_mesh(config.block_size());
    let mesh_handle = meshes.add(patch_mesh);

    // --- Trim strip meshes (shared across all trim strip instances) ---
    // The long side spans 2m quads at the coarse scale, matching the inner-hole
    // edge length of each ring level.
    let trim_quads = 2 * config.block_size();
    let trim_v_mesh = patch_mesh::build_rect_mesh(1, trim_quads); // left/vertical strip
    let trim_h_mesh = patch_mesh::build_rect_mesh(trim_quads, 1); // bottom/horizontal strip
    let trim_v_mesh_handle = meshes.add(trim_v_mesh);
    let trim_h_mesh_handle = meshes.add(trim_h_mesh);

    // --- Build initial patch list before creating the material ---
    let view = TerrainViewState::default();
    let patches: Vec<PatchInstanceCpu> =
        build_patch_instances_for_view_in_bounds(&config, &view, desc.world_min, desc.world_max);

    patch_entities.entities.clear();
    patch_entities.mesh_handle = mesh_handle.clone();
    patch_entities.trim_v_mesh_handle = trim_v_mesh_handle;
    patch_entities.trim_h_mesh_handle = trim_h_mesh_handle;

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
        source_heightmap: source_hmap_handle,
        params: TerrainMaterialUniforms {
            height_scale: config.height_scale,
            base_patch_size,
            morph_start_ratio: config.morph_start_ratio,
            ring_patches: 4.0, // always 4 canonical columns in GPU Gems 2 layout
            num_lod_levels: levels as f32,
            patch_resolution: config.block_size() as f32,
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
        height_generation: 0,
    });

    respawn_patch_entities(
        &mut commands,
        &mut patch_entities,
        &patches,
        &[], // no trim instances at startup — camera not yet positioned
        config.height_scale,
        config.block_size(),
    );

    info!(
        "[Terrain] Setup complete: {} blocks across {} LOD levels (GPU Gems 2 layout, m={}). \
         Clipmap {}×{}×{} (R16Unorm array).",
        patches.len(),
        levels,
        config.block_size(),
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
    let scale_0 = level_scale(config.lod0_mesh_spacing, 0);
    let bias_xz = forward_bias_xz(&config, cam, terrain_cam.forward_bias_ratio);
    let fine_center = snap_camera_to_nested_clipmap_grid(
        cam_pos.xz() + bias_xz,
        scale_0,
        config.active_clipmap_levels(),
    );

    for level in 0..config.active_clipmap_levels() {
        let scale = level_scale(config.lod0_mesh_spacing, level);
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
    let half_ring_l0 = 2.0 * config.block_size() as f32 * level_scale(config.lod0_mesh_spacing, 0);
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
    let trims =
        build_trim_instances_for_view_in_bounds(&config, &view, desc.world_min, desc.world_max);
    let total = patches.len() + trims.len();

    let mut respawned = false;
    if patches.len() != patch_entities.block_count || total != patch_entities.entities.len() {
        respawn_patch_entities(
            &mut commands,
            &mut patch_entities,
            &patches,
            &trims,
            config.height_scale,
            config.block_size(),
        );
        respawned = true;
    }

    if respawned {
        patch_entities.last_clip_centers = view.clip_centers.clone();
        return;
    }

    // Only update ECS Transforms when the clip grid cell changes — these drive
    // Bevy's own visibility / AABB checks for the entity, not the vertex path.
    let n_blocks = patches.len();
    for (entity, patch) in patch_entities.entities[..n_blocks]
        .iter()
        .zip(patches.iter())
    {
        if let Ok((mut transform, mut instance)) = query.get_mut(*entity) {
            transform.translation = Vec3::new(patch.origin_ws.x, 0.0, patch.origin_ws.y);
            transform.scale = Vec3::new(patch.level_scale_ws, 1.0, patch.level_scale_ws);
            instance.patch_origin_ws = patch.origin_ws;
            instance.patch_scale_ws = patch.level_scale_ws;
        }
    }
    for (entity, trim) in patch_entities.entities[n_blocks..].iter().zip(trims.iter()) {
        if let Ok((mut transform, mut instance)) = query.get_mut(*entity) {
            transform.translation = Vec3::new(trim.origin_ws.x, 0.0, trim.origin_ws.y);
            transform.scale = Vec3::new(trim.level_scale_ws, 1.0, trim.level_scale_ws);
            instance.patch_origin_ws = trim.origin_ws;
            instance.patch_scale_ws = trim.level_scale_ws;
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
    mut local_collider_state: ResMut<LocalColliderState>,
    camera_q: Query<(&Transform, &TerrainCamera)>,
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
        for entity in patch_entities.entities.drain(..) {
            commands.entity(entity).despawn();
        }
        patch_entities.block_count = 0;
        // Reset local collider state so update_local_terrain_collider rebuilds
        // with fresh tile data on the same frame (it runs after this system).
        *local_collider_state = LocalColliderState::default();
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
            mat.params.base_patch_size = new_config.lod0_mesh_spacing;
            mat.params.morph_start_ratio = new_config.morph_start_ratio;
            mat.params.ring_patches = 4.0;
            mat.params.num_lod_levels = new_config.active_clipmap_levels() as f32;
            mat.params.patch_resolution = new_config.block_size() as f32;
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

        // --- 4b. Rebuild source heightmap, update image in-place, overwrite resource ---
        let new_source = load_source_heightmap(&new_config, &new_desc);
        let source_handle = materials
            .get(&clipmap_state.material_handle)
            .map(|m| m.source_heightmap.clone())
            .unwrap_or_default();
        if let Some(img) = images.get_mut(&source_handle) {
            *img = new_source.image;
        }
        // insert_resource is deferred — takes effect at end of this command batch.
        commands.insert_resource(SourceHeightmapState {
            handle: source_handle,
            world_origin: new_source.world_origin,
            world_extent: new_source.world_extent,
            texel_size: new_source.texel_size,
        });

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
    }
}

fn respawn_patch_entities(
    commands: &mut Commands,
    patch_entities: &mut PatchEntities,
    patches: &[PatchInstanceCpu],
    trims: &[TrimInstanceCpu],
    height_scale: f32,
    block_size: u32,
) {
    let local_aabb = terrain_block_local_aabb(height_scale, block_size);

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
                    // Scale = one grid unit in world space; the block mesh has
                    // vertices at integer grid positions so this maps each
                    // local unit to level_scale_ws world units.
                    scale: Vec3::new(patch.level_scale_ws, 1.0, patch.level_scale_ws),
                    ..default()
                },
                components::TerrainPatchInstance {
                    lod_level: patch.lod_level,
                    patch_kind: 0,
                    patch_origin_ws: patch.origin_ws,
                    patch_scale_ws: patch.level_scale_ws,
                },
                local_aabb,
                NoAutoAabb,
                NoWireframe,
            ))
            .id();

        patch_entities.entities.push(entity);
    }

    patch_entities.block_count = patch_entities.entities.len();

    let m = block_size as f32;
    for trim in trims {
        // Conservative AABB in local mesh space: mesh quads go from 0 to nx/nz.
        // V strip: nx=1, nz=2m  |  H strip: nx=2m, nz=1  (coarse-scale quads)
        // Pad by 1 unit on each geomorphed edge.
        let (nx, nz) = if trim.is_horizontal {
            (2.0 * m, 1.0)
        } else {
            (1.0, 2.0 * m)
        };
        let trim_aabb = Aabb::from_min_max(
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(nx + 1.0, height_scale.max(1.0), nz + 1.0),
        );
        let mesh = if trim.is_horizontal {
            patch_entities.trim_h_mesh_handle.clone()
        } else {
            patch_entities.trim_v_mesh_handle.clone()
        };
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(patch_entities.material_handle.clone()),
                Transform {
                    translation: Vec3::new(trim.origin_ws.x, 0.0, trim.origin_ws.y),
                    scale: Vec3::new(trim.level_scale_ws, 1.0, trim.level_scale_ws),
                    ..default()
                },
                components::TerrainPatchInstance {
                    lod_level: trim.lod_level,
                    patch_kind: 1,
                    patch_origin_ws: trim.origin_ws,
                    patch_scale_ws: trim.level_scale_ws,
                },
                trim_aabb,
                NoAutoAabb,
                NoFrustumCulling,
                NoWireframe,
            ))
            .id();

        patch_entities.entities.push(entity);
    }
}

fn terrain_block_local_aabb(height_scale: f32, block_size: u32) -> Aabb {
    // Block mesh vertices are at integer grid positions [0, m] × [0, m].
    // Geomorphing can snap outer-edge vertices to the next coarser grid,
    // displacing them by at most 1 grid unit in X or Z. Pad by 1 unit.
    let m = block_size as f32;
    Aabb::from_min_max(
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(m + 1.0, height_scale.max(1.0), m + 1.0),
    )
}

/// Samples the toroidal height CPU buffer for a block's footprint and returns
/// the normalized [0,1] min/max heights (multiply by height_scale for world Y).
///
/// Important: the rendered seam geometry is morphed against the next coarser
/// clipmap level, so a fine-only scan can under-bound the outer edge and let
/// frustum culling drop a seam patch entirely. Scan both the fine layer and the
/// morphed coarse footprint at full resolution to keep the AABB conservative.
fn compute_block_height_range(
    height_data: &[u8],
    lod: u32,
    coarse_lod: u32,
    gx_start: i32,
    gz_start: i32,
    block_size: u32,
    res: u32,
) -> (f32, f32) {
    let res = res as usize;
    let bpl = res * res * 2; // R16Unorm: 2 bytes per texel
    let fine_layer_base = lod as usize * bpl;
    let coarse_layer_base = coarse_lod as usize * bpl;

    if fine_layer_base + bpl > height_data.len() {
        return (0.0, 1.0);
    }

    let m = block_size as i32;
    let mut h_min = f32::MAX;
    let mut h_max = 0.0_f32;
    let coarse_factor = 1i32 << coarse_lod.saturating_sub(lod);
    let has_distinct_coarse =
        coarse_lod != lod && coarse_layer_base + bpl <= height_data.len() && coarse_factor > 1;

    let mut update_range = |raw: u16| {
        let h = raw as f32 * (1.0 / 65535.0);
        if h < h_min {
            h_min = h;
        }
        if h > h_max {
            h_max = h;
        }
    };

    for gz in (gz_start - 1)..=(gz_start + m + 1) {
        let tz = gz.rem_euclid(res as i32) as usize;
        let fine_row_base = fine_layer_base + tz * res * 2;

        let coarse_row_base = if has_distinct_coarse {
            let coarse_tz = gz.div_euclid(coarse_factor).rem_euclid(res as i32) as usize;
            Some(coarse_layer_base + coarse_tz * res * 2)
        } else {
            None
        };

        for gx in (gx_start - 1)..=(gx_start + m + 1) {
            let tx = gx.rem_euclid(res as i32) as usize;
            let fine_off = fine_row_base + tx * 2;
            if fine_off + 2 <= height_data.len() {
                let raw = u16::from_le_bytes([height_data[fine_off], height_data[fine_off + 1]]);
                update_range(raw);
            }

            if let Some(coarse_row_base) = coarse_row_base {
                let coarse_tx = gx.div_euclid(coarse_factor).rem_euclid(res as i32) as usize;
                let coarse_off = coarse_row_base + coarse_tx * 2;
                if coarse_off + 2 <= height_data.len() {
                    let raw =
                        u16::from_le_bytes([height_data[coarse_off], height_data[coarse_off + 1]]);
                    update_range(raw);
                }
            }
        }
    }

    if h_min > h_max {
        (0.0, 1.0)
    } else {
        (h_min, h_max)
    }
}

/// Tightens per-block AABBs each frame by sampling actual height data.
///
/// Replaces the conservative full-height-range AABB with per-block [y_min, y_max]
/// derived from the CPU-side clipmap texture mirror.  Runs after both clipmap
/// update systems so the data is always current.
fn update_patch_aabbs(
    config: Res<TerrainConfig>,
    view: Res<TerrainViewState>,
    clipmap_state: Res<TerrainClipmapState>,
    patch_entities: Res<PatchEntities>,
    mut last_gen: Local<u64>,
    mut query: Query<(&components::TerrainPatchInstance, &mut Aabb)>,
) {
    if clipmap_state.height_cpu_data.is_empty() || view.level_scales.is_empty() {
        return;
    }

    // Skip if height data hasn't changed since our last run.
    if clipmap_state.height_generation == *last_gen {
        return;
    }
    *last_gen = clipmap_state.height_generation;

    let m = config.block_size();
    let res = config.clipmap_resolution();
    let height_scale = config.height_scale;
    const PAD: f32 = 1.0;

    for entity in &patch_entities.entities {
        if let Ok((instance, mut aabb)) = query.get_mut(*entity) {
            let level_scale_ws = instance.patch_scale_ws;
            let origin_ws = instance.patch_origin_ws;
            let coarse_lod =
                (instance.lod_level + 1).min(config.active_clipmap_levels().saturating_sub(1));

            let gx_start = (origin_ws.x / level_scale_ws).round() as i32;
            let gz_start = (origin_ws.y / level_scale_ws).round() as i32;

            let (h_min, h_max) = compute_block_height_range(
                &clipmap_state.height_cpu_data,
                instance.lod_level,
                coarse_lod,
                gx_start,
                gz_start,
                m,
                res,
            );

            // Use conservative full-height AABB for blocks with no loaded data
            // (all zeros → h_max = 0) to avoid culling blocks whose tiles haven't
            // arrived yet.
            let (y_min, y_max) = if h_max < 1e-6 {
                (0.0_f32, height_scale)
            } else {
                (h_min * height_scale, (h_max * height_scale).max(1.0))
            };

            let new_aabb = Aabb::from_min_max(
                Vec3::new(-PAD, y_min, -PAD),
                Vec3::new(m as f32 + PAD, y_max, m as f32 + PAD),
            );

            // Only write when the value changes — avoids triggering Bevy's
            // change-detection which would re-extract all patches every frame.
            if *aabb != new_aabb {
                *aabb = new_aabb;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::compute_block_height_range;

    fn make_height_data(layers: usize, res: u32) -> Vec<u8> {
        vec![0u8; layers * (res * res * 2) as usize]
    }

    fn write_height(data: &mut [u8], res: u32, lod: u32, gx: i32, gz: i32, value: u16) {
        let res = res as usize;
        let layer_bytes = res * res * 2;
        let layer_base = lod as usize * layer_bytes;
        let tx = gx.rem_euclid(res as i32) as usize;
        let tz = gz.rem_euclid(res as i32) as usize;
        let off = layer_base + (tz * res + tx) * 2;
        data[off..off + 2].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn block_height_range_sees_peaks_between_old_stride4_samples() {
        let res = 8;
        let mut height_data = make_height_data(1, res);

        // The old stride-4 scan visited gx,gz = -1,3,7,... and would miss this.
        write_height(&mut height_data, res, 0, 2, 2, u16::MAX);

        let (_, h_max) = compute_block_height_range(&height_data, 0, 0, 0, 0, 4, res);
        assert!(h_max > 0.99, "exact AABB scan must include unsampled peaks");
    }

    #[test]
    fn block_height_range_includes_morphed_coarse_layer() {
        let res = 8;
        let mut height_data = make_height_data(2, res);

        // Fine LOD is flat, but the seam morph can fully snap to the next
        // coarser level on the outer edge. The AABB must include that height.
        write_height(&mut height_data, res, 1, 1, 1, u16::MAX);

        let (_, h_max) = compute_block_height_range(&height_data, 0, 1, 0, 0, 4, res);
        assert!(h_max > 0.99, "AABB must include coarse seam heights");
    }
}

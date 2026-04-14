pub mod clipmap;
pub mod clipmap_texture;
pub mod collision;
pub mod components;
pub mod config;
pub mod debug;
pub mod material;
pub mod math;
pub mod patch_mesh;
pub mod render;
pub mod residency;
pub mod resources;
pub mod streamer;
pub mod world_desc;

pub use debug::TerrainDebugPlugin;

use bevy::{camera::visibility::NoFrustumCulling, prelude::*};
use clipmap_texture::{
    TerrainClipmapState, compute_initial_clip_levels,
    create_initial_clipmap_texture, update_clipmap_textures,
};
use components::TerrainCamera;
use config::TerrainConfig;
use math::{level_scale, snap_camera_to_level_grid};
use material::{TerrainMaterial, TerrainMaterialUniforms};
use render::{TerrainRenderPlugin, extract::extract_terrain_frame};
use residency::update_required_tiles;
use resources::{TerrainResidency, TerrainStreamQueue, TerrainViewState};
use streamer::{poll_tile_stream_jobs, request_tile_loads, setup_tile_channel};
use world_desc::TerrainSourceDesc;
use collision::{update_collision_tiles, TerrainCollisionCache};
use clipmap::{build_patch_instances_for_view, PatchInstanceCpu};

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
                    update_patch_transforms,     // Phase 1: live patch repositioning
                    update_clipmap_textures,     // Phase 5: live clipmap texture update
                )
                    .after(update_terrain_view_state),
            )
            // Render sub-plugin
            .add_plugins(TerrainRenderPlugin);
    }
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

/// Spawns patch entities and creates the initial clipmap texture array.
fn setup_terrain(
    config: Res<TerrainConfig>,
    mut meshes:            ResMut<Assets<Mesh>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut images:            ResMut<Assets<Image>>,
    mut patch_entities:    ResMut<PatchEntities>,
    mut commands:          Commands,
) {
    // --- Clipmap texture array (Phase 5) ---
    // One R8Unorm layer per LOD level, each 512×512 texels.
    // Layers are regenerated live by `update_clipmap_textures` as the camera moves.
    let height_image  = create_initial_clipmap_texture(&config);
    let height_handle = images.add(height_image);

    let base_patch_size = config.patch_resolution as f32 * config.world_scale;

    let mat_handle = terrain_materials.add(TerrainMaterial {
        height_texture: height_handle.clone(),
        params: TerrainMaterialUniforms {
            height_scale:      config.height_scale,
            base_patch_size,
            morph_start_ratio: config.morph_start_ratio,
            ring_patches:      config.ring_patches as f32,
            pad0: 0.0, pad1: 0.0, pad2: 0.0, pad3: 0.0,
            clip_levels: compute_initial_clip_levels(&config),
        },
    });

    // Insert the runtime state resource so `update_clipmap_textures` can find it.
    // Initialise last_clip_centers to ZERO (matching the initial texture), so the
    // first real camera position triggers a full regeneration on the first frame.
    commands.insert_resource(TerrainClipmapState {
        texture_handle:    height_handle,
        material_handle:   mat_handle.clone(),
        last_clip_centers: vec![IVec2::ZERO; config.clipmap_levels as usize],
    });

    // --- Patch mesh (shared by all entities) ---
    let patch_mesh  = patch_mesh::build_patch_mesh(config.patch_resolution);
    let mesh_handle = meshes.add(patch_mesh);

    // --- Spawn one entity per patch from the default (zero) view state ---
    let view    = TerrainViewState::default();
    let patches: Vec<PatchInstanceCpu> = build_patch_instances_for_view(&config, &view);

    patch_entities.entities.clear();

    for patch in &patches {
        let entity = commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(mat_handle.clone()),
            Transform {
                translation: Vec3::new(patch.origin_ws.x, 0.0, patch.origin_ws.y),
                scale:       Vec3::new(patch.patch_size_ws, 1.0, patch.patch_size_ws),
                ..default()
            },
            components::TerrainPatchInstance {
                lod_level:      patch.lod_level,
                patch_kind:     patch.patch_kind,
                patch_origin_ws: patch.origin_ws,
                patch_scale_ws:  patch.patch_size_ws,
            },
            NoFrustumCulling,
        )).id();

        patch_entities.entities.push(entity);
    }

    info!(
        "[Terrain] Setup complete: {} patches across {} LOD levels. \
         Clipmap {}×{}×{} (R8Unorm array).",
        patches.len(),
        config.clipmap_levels,
        config.clipmap_resolution,
        config.clipmap_resolution,
        config.clipmap_levels,
    );
}

// ---------------------------------------------------------------------------
// Update systems
// ---------------------------------------------------------------------------

/// Rebuilds the terrain view (camera position, clip centers, level scales).
/// Runs first so all subsequent systems see fresh data.
pub fn update_terrain_view_state(
    config:   Res<TerrainConfig>,
    camera_q: Query<&GlobalTransform, With<TerrainCamera>>,
    mut view: ResMut<TerrainViewState>,
) {
    let Ok(cam) = camera_q.single() else {
        return;
    };

    let cam_pos: Vec3 = cam.translation();
    view.camera_pos_ws = cam_pos;
    view.clip_centers.clear();
    view.level_scales.clear();

    // Compute an aligned level-0 center.
    //
    // If each level snaps independently, level-0's center can be odd while
    // level-1's center (in level-0 units) is even, creating a 1-grid-unit
    // gap on one side and a 1-unit overlap on the other.  The gap appears as
    // a flickering seam that shows up in half the compass directions and
    // flickers as the camera crosses grid cells.
    //
    // Fix: round level-0's grid coordinate DOWN to the nearest multiple of
    // 2^(clipmap_levels-1), then derive every coarser center by right-shifting.
    // This guarantees  center_L * scale_L == center_{L+1} * scale_{L+1}  for
    // all L, so ring boundaries are always flush with no gaps or overlaps.
    let scale_0   = level_scale(config.world_scale, 0);
    let raw_0     = snap_camera_to_level_grid(cam_pos.xz(), scale_0);
    let align_shift = (config.clipmap_levels - 1) as i32;
    let aligned_0 = IVec2::new(
        (raw_0.x >> align_shift) << align_shift,
        (raw_0.y >> align_shift) << align_shift,
    );

    for level in 0..config.clipmap_levels {
        let scale = level_scale(config.world_scale, level);
        // Right-shift the aligned level-0 center to get each coarser center.
        let shift  = level as i32;
        let center = IVec2::new(aligned_0.x >> shift, aligned_0.y >> shift);
        view.level_scales.push(scale);
        view.clip_centers.push(center);
    }
}

/// Repositions patch entities to match the current clipmap ring layout.
/// Only does work when the snapped clip centers have actually changed.
fn update_patch_transforms(
    config:  Res<TerrainConfig>,
    view:    Res<TerrainViewState>,
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
            patches.len(), patch_entities.entities.len()
        );
        return;
    }

    for (entity, patch) in patch_entities.entities.iter().zip(patches.iter()) {
        if let Ok((mut transform, mut instance)) = query.get_mut(*entity) {
            transform.translation = Vec3::new(patch.origin_ws.x, 0.0, patch.origin_ws.y);
            transform.scale       = Vec3::new(patch.patch_size_ws, 1.0, patch.patch_size_ws);
            instance.patch_origin_ws = patch.origin_ws;
            instance.patch_scale_ws  = patch.patch_size_ws;
        }
    }

    patch_entities.last_clip_centers = view.clip_centers.clone();
}

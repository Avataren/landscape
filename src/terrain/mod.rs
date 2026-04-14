pub mod clipmap;
pub mod collision;
pub mod components;
pub mod config;
pub mod debug;
pub mod math;
pub mod patch_mesh;
pub mod render;
pub mod residency;
pub mod resources;
pub mod streamer;
pub mod world_desc;

pub use debug::TerrainDebugPlugin;

use bevy::prelude::*;
use components::TerrainCamera;
use config::TerrainConfig;
use math::{level_scale, snap_camera_to_level_grid};
use render::{TerrainRenderPlugin, extract::extract_terrain_frame};
use residency::update_required_tiles;
use resources::{TerrainResidency, TerrainStreamQueue, TerrainViewState};
use streamer::{poll_tile_stream_jobs, request_tile_loads, setup_tile_channel};
use world_desc::TerrainSourceDesc;
use collision::{update_collision_tiles, TerrainCollisionCache};

// ---------------------------------------------------------------------------
// Main terrain plugin
// ---------------------------------------------------------------------------

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app
            // Resources
            .init_resource::<TerrainConfig>()
            .init_resource::<TerrainSourceDesc>()
            .init_resource::<TerrainViewState>()
            .init_resource::<TerrainResidency>()
            .init_resource::<TerrainStreamQueue>()
            .init_resource::<TerrainCollisionCache>()
            // Startup
            .add_systems(Startup, (setup_tile_channel, setup_terrain).chain())
            // Update: ordered as specified in the handoff
            .add_systems(Update, update_terrain_view_state)
            .add_systems(Update, update_required_tiles.after(update_terrain_view_state))
            .add_systems(Update, request_tile_loads.after(update_required_tiles))
            .add_systems(Update, poll_tile_stream_jobs.after(request_tile_loads))
            .add_systems(Update, update_collision_tiles.after(poll_tile_stream_jobs))
            .add_systems(Update, extract_terrain_frame.after(update_terrain_view_state))
            // Render sub-plugin
            .add_plugins(TerrainRenderPlugin);
    }
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

/// Spawns the initial terrain state and registers the patch mesh.
fn setup_terrain(
    config: Res<TerrainConfig>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    let patch_mesh = patch_mesh::build_patch_mesh(config.patch_resolution);
    let mesh_handle = meshes.add(patch_mesh);

    // Phase 1: flat CPU-instanced patches using Bevy's standard PBR path.
    // Each patch is a separate entity. In Phase 8 this will be replaced with
    // indirect GPU draws.
    let mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.6, 0.3),
        perceptual_roughness: 0.9,
        ..default()
    });

    // Build an initial dummy view state for patch spawning.
    let view = TerrainViewState::default();
    let patches = clipmap::build_patch_instances_for_view(&config, &view);

    for patch in &patches {
        let scale = Vec3::new(patch.patch_size_ws, 1.0, patch.patch_size_ws);
        let translation = Vec3::new(patch.origin_ws.x, 0.0, patch.origin_ws.y);
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(mat.clone()),
            Transform::from_translation(translation).with_scale(scale),
            components::TerrainPatchInstance {
                lod_level: patch.lod_level,
                patch_kind: patch.patch_kind,
                patch_origin_ws: patch.origin_ws,
                patch_scale_ws: patch.patch_size_ws,
            },
        ));
    }

    info!(
        "[Terrain] Setup complete: {} patches, {} LOD levels",
        patches.len(),
        config.clipmap_levels,
    );
}

// ---------------------------------------------------------------------------
// Update systems
// ---------------------------------------------------------------------------

/// Updates the terrain view state from the terrain camera.
pub fn update_terrain_view_state(
    config: Res<TerrainConfig>,
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

    let base_spacing = config.world_scale;

    for level in 0..config.clipmap_levels {
        let scale = level_scale(base_spacing, level);
        let center = snap_camera_to_level_grid(cam_pos.xz(), scale);
        view.level_scales.push(scale);
        view.clip_centers.push(center);
    }
}

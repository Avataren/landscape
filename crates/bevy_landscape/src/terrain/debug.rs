use crate::terrain::{
    clipmap::build_patch_instances_for_view_in_bounds,
    clipmap_texture::TerrainClipmapState,
    config::TerrainConfig,
    material::TerrainMaterial,
    resources::{TerrainResidency, TerrainViewState},
    world_desc::TerrainSourceDesc,
};
use bevy::{pbr::wireframe::WireframeConfig, prelude::*};

// ---------------------------------------------------------------------------
// Debug config resource
// ---------------------------------------------------------------------------

/// Toggle individual debug overlays at runtime.
#[derive(Resource, Debug)]
pub struct TerrainDebugConfig {
    pub show_patch_bounds: bool,
    pub show_lod_colors: bool,
    pub show_stats: bool,
    pub show_wireframe: bool,
    /// Render the per-pixel surface normal as colour (`n * 0.5 + 0.5`) instead
    /// of the full material — useful for inspecting normal-map staircasing or
    /// LOD seam discontinuities without lighting noise getting in the way.
    pub show_normals_only: bool,
}

impl Default for TerrainDebugConfig {
    fn default() -> Self {
        Self {
            show_patch_bounds: false,
            show_lod_colors: false,
            show_stats: false,
            show_wireframe: false,
            show_normals_only: false,
        }
    }
}

// ---------------------------------------------------------------------------
// LOD colors (one per level, cycles if more than 8 levels)
// ---------------------------------------------------------------------------

const LOD_COLORS: [Color; 8] = [
    Color::srgb(0.0, 1.0, 0.0),
    Color::srgb(0.0, 0.8, 1.0),
    Color::srgb(1.0, 1.0, 0.0),
    Color::srgb(1.0, 0.5, 0.0),
    Color::srgb(1.0, 0.0, 0.0),
    Color::srgb(0.8, 0.0, 1.0),
    Color::srgb(0.4, 0.4, 1.0),
    Color::srgb(1.0, 1.0, 1.0),
];

/// Runtime debug hotkeys:
/// - F8  = render terrain normals as colour (no lighting/material)
/// - F9  = stats logging
/// - F10 = patch bounds
/// - F11 = LOD center markers
/// - F12 = global wireframe + terrain wireframe overlay
pub fn toggle_terrain_debug_hotkeys(
    keys: Res<ButtonInput<KeyCode>>,
    mut debug_cfg: ResMut<TerrainDebugConfig>,
) {
    if keys.just_pressed(KeyCode::F8) {
        debug_cfg.show_normals_only = !debug_cfg.show_normals_only;
        info!(
            "[Terrain] Normals-only debug {} (F8)",
            if debug_cfg.show_normals_only {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    if keys.just_pressed(KeyCode::F9) {
        debug_cfg.show_stats = !debug_cfg.show_stats;
        info!(
            "[Terrain] Debug stats {} (F9)",
            if debug_cfg.show_stats {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    if keys.just_pressed(KeyCode::F10) {
        debug_cfg.show_patch_bounds = !debug_cfg.show_patch_bounds;
        info!(
            "[Terrain] Patch bounds {} (F10)",
            if debug_cfg.show_patch_bounds {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    if keys.just_pressed(KeyCode::F11) {
        debug_cfg.show_lod_colors = !debug_cfg.show_lod_colors;
        info!(
            "[Terrain] LOD markers {} (F11)",
            if debug_cfg.show_lod_colors {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    if keys.just_pressed(KeyCode::F12) {
        debug_cfg.show_wireframe = !debug_cfg.show_wireframe;
        info!(
            "[Terrain] Global wireframe {} (F12)",
            if debug_cfg.show_wireframe {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

/// Keep Bevy's global wireframe mode and the terrain material wireframe flag in sync.
pub fn sync_wireframe_modes(
    debug_cfg: Res<TerrainDebugConfig>,
    clipmap_state: Res<TerrainClipmapState>,
    mut wireframe_cfg: ResMut<WireframeConfig>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
) {
    if wireframe_cfg.global != debug_cfg.show_wireframe {
        wireframe_cfg.global = debug_cfg.show_wireframe;
    }

    let Some(material) = terrain_materials.get_mut(&clipmap_state.material_handle) else {
        return;
    };

    let desired = if debug_cfg.show_wireframe { 1.0 } else { 0.0 };
    if material.params.bounds_fade.w != desired {
        material.params.bounds_fade.w = desired;
    }

    let desired_normals = if debug_cfg.show_normals_only { 1.0 } else { 0.0 };
    if material.params.debug_flags.x != desired_normals {
        material.params.debug_flags.x = desired_normals;
    }
}

// ---------------------------------------------------------------------------
// Debug systems
// ---------------------------------------------------------------------------

/// Draw wireframe patch bounds using Bevy's built-in gizmo API.
pub fn draw_terrain_debug(
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    view: Res<TerrainViewState>,
    debug_cfg: Res<TerrainDebugConfig>,
    _residency: Res<TerrainResidency>,
    mut gizmos: Gizmos,
) {
    if debug_cfg.show_patch_bounds {
        let patches = build_patch_instances_for_view_in_bounds(
            &config,
            &view,
            desc.world_min,
            desc.world_max,
        );
        for patch in &patches {
            let color = LOD_COLORS[patch.lod_level as usize % LOD_COLORS.len()];
            let cx = patch.origin_ws.x + patch.patch_size_ws * 0.5;
            let cz = patch.origin_ws.y + patch.patch_size_ws * 0.5;
            let size = patch.patch_size_ws;
            // Draw the 4 bottom edges of the patch bounding box.
            let corners = [
                Vec3::new(cx - size * 0.5, 0.0, cz - size * 0.5),
                Vec3::new(cx + size * 0.5, 0.0, cz - size * 0.5),
                Vec3::new(cx + size * 0.5, 0.0, cz + size * 0.5),
                Vec3::new(cx - size * 0.5, 0.0, cz + size * 0.5),
            ];
            for i in 0..4 {
                gizmos.line(corners[i], corners[(i + 1) % 4], color);
            }
        }
    }

    if debug_cfg.show_lod_colors {
        for (i, center) in view.clip_centers.iter().enumerate() {
            let scale = view.level_scales.get(i).copied().unwrap_or(1.0);
            let wx = center.x as f32 * scale;
            let wz = center.y as f32 * scale;
            let color = LOD_COLORS[i % LOD_COLORS.len()];
            let y = 5.0 + i as f32 * 2.0;
            let r = 3.0 + i as f32;
            let p = Vec3::new(wx, y, wz);
            // Draw a simple cross marker for each clip center.
            gizmos.line(p - Vec3::X * r, p + Vec3::X * r, color);
            gizmos.line(p - Vec3::Z * r, p + Vec3::Z * r, color);
        }
    }
}

/// Log terrain stats to the console (throttled).
pub fn log_terrain_stats(
    _config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    view: Res<TerrainViewState>,
    residency: Res<TerrainResidency>,
    debug_cfg: Res<TerrainDebugConfig>,
    time: Res<Time>,
) {
    if !debug_cfg.show_stats {
        return;
    }
    // Log once per second.
    if (time.elapsed_secs() % 1.0) > time.delta_secs() {
        return;
    }

    let patch_count =
        build_patch_instances_for_view_in_bounds(&_config, &view, desc.world_min, desc.world_max)
            .len();

    info!(
        "[Terrain] cam={:.1},{:.1},{:.1}  patches={}  resident_tiles={}  pending_upload={}  required_now={}",
        view.camera_pos_ws.x,
        view.camera_pos_ws.y,
        view.camera_pos_ws.z,
        patch_count,
        residency.tiles.len(),
        residency.pending_upload.len(),
        residency.required_now.len(),
    );
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct TerrainDebugPlugin;

impl Plugin for TerrainDebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainDebugConfig>().add_systems(
            Update,
            (
                toggle_terrain_debug_hotkeys,
                sync_wireframe_modes,
                draw_terrain_debug,
                log_terrain_stats,
            ),
        );
    }
}

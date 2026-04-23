use crate::terrain::{
    clipmap::build_patch_instances_for_view_in_bounds,
    clipmap_texture::TerrainClipmapState,
    components::TerrainPatchInstance,
    config::TerrainConfig,
    material::TerrainMaterial,
    resources::{TerrainResidency, TerrainViewState},
    world_desc::TerrainSourceDesc,
};
use bevy::{camera::primitives::Aabb, pbr::wireframe::WireframeConfig, prelude::*};

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
    pub show_normals_only: bool,
    pub show_ruler: bool,
    pub show_pbr_debug: u8,
}

impl Default for TerrainDebugConfig {
    fn default() -> Self {
        Self {
            show_patch_bounds: false,
            show_lod_colors: false,
            show_stats: false,
            show_wireframe: false,
            show_normals_only: false,
            show_ruler: false,
            show_pbr_debug: 0,
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
/// - F5  = cycle PBR texture debug (off / raw normal-map / ORM roughness)
/// - F6  = ruler grid
/// - F8  = render terrain normals as colour (no lighting/material)
/// - F9  = stats logging
/// - F10 = patch bounds
/// - F11 = LOD center markers
/// - F12 = global wireframe + terrain wireframe overlay
pub fn toggle_terrain_debug_hotkeys(
    keys: Res<ButtonInput<KeyCode>>,
    mut debug_cfg: ResMut<TerrainDebugConfig>,
) {
    if keys.just_pressed(KeyCode::F5) {
        debug_cfg.show_pbr_debug = (debug_cfg.show_pbr_debug + 1) % 3;
        let label =
            ["off", "raw normal-map tex", "ORM roughness"][debug_cfg.show_pbr_debug as usize];
        info!("[Terrain] PBR debug: {label} (F5)");
    }

    if keys.just_pressed(KeyCode::F6) {
        debug_cfg.show_ruler = !debug_cfg.show_ruler;
        info!(
            "[Terrain] Ruler grid {} (F6)",
            if debug_cfg.show_ruler {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

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

    let desired_normals = if debug_cfg.show_normals_only {
        1.0
    } else {
        0.0
    };
    if material.params.debug_flags.x != desired_normals {
        material.params.debug_flags.x = desired_normals;
    }

    let desired_pbr = debug_cfg.show_pbr_debug as f32;
    if material.params.debug_flags.z != desired_pbr {
        material.params.debug_flags.z = desired_pbr;
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
            let cx = patch.origin_ws.x + patch.block_world_size * 0.5;
            let cz = patch.origin_ws.y + patch.block_world_size * 0.5;
            let size = patch.block_world_size;
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
    patch_query: Query<(&Aabb, &TerrainPatchInstance)>,
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

    // Print the AABB of the first terrain patch to diagnose culling issues.
    // If Y half_extents ≈ 0, Bevy's calculate_bounds overwrote the manual AABB.
    if let Some((aabb, patch)) = patch_query.iter().next() {
        let min = Vec3::from(aabb.center) - Vec3::from(aabb.half_extents);
        let max = Vec3::from(aabb.center) + Vec3::from(aabb.half_extents);
        info!(
            "[Terrain] L{} patch AABB local: min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2})",
            patch.lod_level, min.x, min.y, min.z, max.x, max.y, max.z,
        );
    }
}

// ---------------------------------------------------------------------------
// Ruler gizmo
// ---------------------------------------------------------------------------

/// Draws a world-space scale ruler at Y=0:
///   - Sky-blue lines every 100 m within ±500 m of the camera
///   - Amber lines every 1 000 m within ±6 km of the camera
///   - White vertical tick markers at every 1 km intersection,
///     scaled to half the camera altitude so they stay readable
pub fn draw_ruler(
    view: Res<TerrainViewState>,
    debug_cfg: Res<TerrainDebugConfig>,
    mut gizmos: Gizmos,
) {
    if !debug_cfg.show_ruler {
        return;
    }

    let cam = view.camera_pos_ws;
    let y = 0.0_f32; // draw at sea-level / terrain base

    let color_100m = Color::srgba(0.3, 0.75, 1.0, 0.55);
    let color_1km = Color::srgba(1.0, 0.75, 0.1, 0.90);
    let color_tick = Color::srgba(1.0, 1.0, 1.0, 0.85);

    // ---- 100 m grid  (±500 m, skip multiples of 1 000 m — drawn by 1km pass) ----
    let cx100 = (cam.x / 100.0).floor() as i32;
    let cz100 = (cam.z / 100.0).floor() as i32;
    let r100 = 5_i32;
    let xlo100 = (cx100 - r100) as f32 * 100.0;
    let xhi100 = (cx100 + r100) as f32 * 100.0;
    let zlo100 = (cz100 - r100) as f32 * 100.0;
    let zhi100 = (cz100 + r100) as f32 * 100.0;

    for i in (cx100 - r100)..=(cx100 + r100) {
        if i % 10 == 0 {
            continue;
        } // 1 km line — skip
        let x = i as f32 * 100.0;
        gizmos.line(Vec3::new(x, y, zlo100), Vec3::new(x, y, zhi100), color_100m);
    }
    for i in (cz100 - r100)..=(cz100 + r100) {
        if i % 10 == 0 {
            continue;
        }
        let z = i as f32 * 100.0;
        gizmos.line(Vec3::new(xlo100, y, z), Vec3::new(xhi100, y, z), color_100m);
    }

    // ---- 1 km grid  (±6 km) ----
    let cx1k = (cam.x / 1000.0).floor() as i32;
    let cz1k = (cam.z / 1000.0).floor() as i32;
    let r1k = 6_i32;
    let xlo1k = (cx1k - r1k) as f32 * 1000.0;
    let xhi1k = (cx1k + r1k) as f32 * 1000.0;
    let zlo1k = (cz1k - r1k) as f32 * 1000.0;
    let zhi1k = (cz1k + r1k) as f32 * 1000.0;

    for i in (cx1k - r1k)..=(cx1k + r1k) {
        let x = i as f32 * 1000.0;
        gizmos.line(Vec3::new(x, y, zlo1k), Vec3::new(x, y, zhi1k), color_1km);
    }
    for i in (cz1k - r1k)..=(cz1k + r1k) {
        let z = i as f32 * 1000.0;
        gizmos.line(Vec3::new(xlo1k, y, z), Vec3::new(xhi1k, y, z), color_1km);
    }

    // ---- Vertical tick markers at every 1 km intersection ----
    // Height scales with camera altitude so ticks are readable both low and high.
    let tick_h = (cam.y * 0.5).clamp(100.0, 2000.0);
    for ix in (cx1k - r1k)..=(cx1k + r1k) {
        for iz in (cz1k - r1k)..=(cz1k + r1k) {
            let x = ix as f32 * 1000.0;
            let z = iz as f32 * 1000.0;
            gizmos.line(Vec3::new(x, y, z), Vec3::new(x, y + tick_h, z), color_tick);
        }
    }
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
                draw_ruler,
                log_terrain_stats,
            ),
        );
    }
}

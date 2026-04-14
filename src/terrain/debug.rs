use bevy::prelude::*;
use crate::terrain::{
    clipmap::build_patch_instances_for_view,
    config::TerrainConfig,
    resources::{TerrainResidency, TerrainViewState},
};

// ---------------------------------------------------------------------------
// Debug config resource
// ---------------------------------------------------------------------------

/// Toggle individual debug overlays at runtime.
#[derive(Resource, Debug)]
pub struct TerrainDebugConfig {
    pub show_patch_bounds: bool,
    pub show_lod_colors: bool,
    pub show_stats: bool,
}

impl Default for TerrainDebugConfig {
    fn default() -> Self {
        Self {
            show_patch_bounds: false,
            show_lod_colors: false,
            show_stats: true,
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

// ---------------------------------------------------------------------------
// Debug systems
// ---------------------------------------------------------------------------

/// Draw wireframe patch bounds using Bevy's built-in gizmo API.
pub fn draw_terrain_debug(
    config: Res<TerrainConfig>,
    view: Res<TerrainViewState>,
    debug_cfg: Res<TerrainDebugConfig>,
    _residency: Res<TerrainResidency>,
    mut gizmos: Gizmos,
) {
    if debug_cfg.show_patch_bounds {
        let patches = build_patch_instances_for_view(&config, &view);
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
    config: Res<TerrainConfig>,
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

    let patch_count: usize = (0..config.clipmap_levels as usize)
        .map(|level| {
            let has_hole = level > 0;
            let full = config.ring_patches * config.ring_patches;
            let hole = if has_hole {
                let inner = config.ring_patches / 2;
                inner * inner
            } else {
                0
            };
            (full - hole) as usize
        })
        .sum();

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
        app.init_resource::<TerrainDebugConfig>()
            .add_systems(
                Update,
                (draw_terrain_debug, log_terrain_stats),
            );
    }
}

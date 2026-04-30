//! Debug panel — editor UI for all runtime debug overlays.
//!
//! Replaces / mirrors the function-key hotkeys:
//!   F2 = water, F3 = collision mesh, F5 = PBR debug, F6 = ruler,
//!   F8 = fragment debug, F9 = stats, F10 = patch bounds,
//!   F11 = LOD markers, F12 = wireframe.
//! Also exposes the new SSAO visualizer (debug_flags.y).
//! F1 (camera walk/fly) and F7 (shadows) are binary-level hotkeys and are
//! shown as read-only reminders only.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{LocalColliderState, ShowTerrainCollision, TerrainDebugConfig};
use bevy_landscape_water::WaterEnabled;

use crate::rendering_panel::SsaoSettings;
use crate::toolbar::ToolbarState;

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct DebugPanelPlugin;

impl Plugin for DebugPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(EguiPrimaryContextPass, debug_panel_system);
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

fn debug_panel_system(
    mut contexts: EguiContexts,
    mut toolbar: ResMut<ToolbarState>,
    mut terrain_dbg: ResMut<TerrainDebugConfig>,
    mut show_collision: ResMut<ShowTerrainCollision>,
    mut collider_state: ResMut<LocalColliderState>,
    water_enabled: Option<ResMut<WaterEnabled>>,
    ssao: Res<SsaoSettings>,
) -> Result {
    if !toolbar.debug_open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;

    let mut open = toolbar.debug_open;
    egui::Window::new("Debug Overlays")
        .open(&mut open)
        .resizable(false)
        .min_width(300.0)
        .show(ctx, |ui| {
            // ----------------------------------------------------------------
            // Terrain fragment debug (F8)
            // ----------------------------------------------------------------
            ui.heading("Terrain Fragment Debug  (F8)");
            ui.separator();

            let frag_labels = ["Off", "Normals", "Height (greyscale)", "LOD / Morph alpha"];
            for (i, label) in frag_labels.iter().enumerate() {
                let cur = terrain_dbg.fragment_debug_mode as usize == i;
                if ui.radio(cur, *label).clicked() {
                    terrain_dbg.fragment_debug_mode = i as u8;
                }
            }

            ui.add_space(8.0);

            // ----------------------------------------------------------------
            // SSAO visualizer (new)
            // ----------------------------------------------------------------
            ui.heading("SSAO Visualizer");
            ui.separator();

            let ssao_active = ssao.enabled;
            ui.add_enabled_ui(ssao_active, |ui| {
                let mut v = terrain_dbg.ssao_debug;
                if ui.checkbox(&mut v, "Show AO buffer (greyscale)").changed() {
                    terrain_dbg.ssao_debug = v;
                }
            });
            if !ssao_active {
                ui.label("⚠ Enable SSAO in the Rendering panel first.");
            }

            ui.add_space(8.0);

            // ----------------------------------------------------------------
            // PBR texture debug (F5)
            // ----------------------------------------------------------------
            ui.heading("PBR Texture Debug  (F5)");
            ui.separator();

            let pbr_labels = ["Off", "Raw normal map", "ORM roughness"];
            for (i, label) in pbr_labels.iter().enumerate() {
                let cur = terrain_dbg.show_pbr_debug as usize == i;
                if ui.radio(cur, *label).clicked() {
                    terrain_dbg.show_pbr_debug = i as u8;
                }
            }

            ui.add_space(8.0);

            // ----------------------------------------------------------------
            // Simple toggles
            // ----------------------------------------------------------------
            ui.heading("Overlays");
            ui.separator();

            let mut ruler = terrain_dbg.show_ruler;
            if ui.checkbox(&mut ruler, "Ruler grid  (F6)").changed() {
                terrain_dbg.show_ruler = ruler;
            }

            let mut patch_bounds = terrain_dbg.show_patch_bounds;
            if ui.checkbox(&mut patch_bounds, "Patch bounds  (F10)").changed() {
                terrain_dbg.show_patch_bounds = patch_bounds;
            }

            let mut lod_markers = terrain_dbg.show_lod_colors;
            if ui.checkbox(&mut lod_markers, "LOD center markers  (F11)").changed() {
                terrain_dbg.show_lod_colors = lod_markers;
            }

            let mut wireframe = terrain_dbg.show_wireframe;
            if ui.checkbox(&mut wireframe, "Global wireframe  (F12)").changed() {
                terrain_dbg.show_wireframe = wireframe;
            }

            let mut stats = terrain_dbg.show_stats;
            if ui.checkbox(&mut stats, "Stats logging  (F9)").changed() {
                terrain_dbg.show_stats = stats;
            }

            ui.add_space(8.0);

            // ----------------------------------------------------------------
            // Physics collision mesh (F3)
            // ----------------------------------------------------------------
            ui.heading("Physics");
            ui.separator();

            let mut col = show_collision.0;
            if ui.checkbox(&mut col, "Collision mesh  (F3)").changed() {
                show_collision.0 = col;
                // Force a collider rebuild so the debug mesh is spawned or
                // removed immediately (mirrors the F3 hotkey behaviour).
                collider_state.force_rebuild();
            }

            ui.add_space(8.0);

            // ----------------------------------------------------------------
            // Water (F2)
            // ----------------------------------------------------------------
            if let Some(mut we) = water_enabled {
                ui.heading("Water");
                ui.separator();

                let mut enabled = we.0;
                if ui.checkbox(&mut enabled, "Water enabled  (F2)").changed() {
                    we.0 = enabled;
                }
                ui.add_space(8.0);
            }

            // ----------------------------------------------------------------
            // Binary-level hotkeys (editor can't control these directly)
            // ----------------------------------------------------------------
            ui.separator();
            ui.weak("F1 = toggle walk / fly camera");
            ui.weak("F7 = toggle directional shadows");
        });

    toolbar.debug_open = open;
    Ok(())
}

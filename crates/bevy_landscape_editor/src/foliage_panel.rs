//! Foliage editor panel — UI for foliage instance generation and control.
//!
//! Provides:
//! - Generate/Regenerate foliage button
//! - Procedural parameter inspection
//! - Preview toggle
//! - Status display (instance count, memory usage)

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{FoliageSourceDesc, FoliageConfigResource, FoliageConfig};
use crate::toolbar::ToolbarState;

/// Editor-local UI state for the foliage panel (preview settings only).
#[derive(Resource, Default)]
pub struct FoliagePanelState {
    pub preview_enabled: bool,
    pub generation_in_progress: bool,
}

pub struct FoliagePanelPlugin;

impl Plugin for FoliagePanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FoliagePanelState>()
            .add_systems(EguiPrimaryContextPass, foliage_panel_system);
    }
}

fn foliage_panel_system(
    mut contexts: EguiContexts,
    mut panel: ResMut<FoliagePanelState>,
    mut toolbar: ResMut<ToolbarState>,
    foliage_config: Option<Res<FoliageConfigResource>>,
    foliage_source: Option<Res<FoliageSourceDesc>>,
) -> Result {
    if !toolbar.foliage_open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;
    let mut open = toolbar.foliage_open;

    egui::Window::new("Foliage")
        .open(&mut open)
        .default_width(360.0)
        .default_height(300.0)
        .resizable(true)
        .show(ctx, |ui| {
            draw_foliage_panel(
                ui,
                &mut panel,
                foliage_config.as_deref().and_then(|fc| fc.0.as_ref()),
                foliage_source.as_deref(),
            );
        });

    toolbar.foliage_open = open;
    Ok(())
}

fn draw_foliage_panel(
    ui: &mut egui::Ui,
    panel: &mut FoliagePanelState,
    config: Option<&FoliageConfig>,
    source: Option<&FoliageSourceDesc>,
) {
    if config.is_none() || source.is_none() {
        ui.label("ℹ Foliage not loaded.");
        ui.label("Import a heightmap to enable foliage generation.");
        return;
    }

    let config = config.unwrap();
    let source = source.unwrap();

    ui.group(|ui| {
        ui.label("📦 Foliage Generation");
        ui.separator();

        // Status
        ui.horizontal(|ui| {
            ui.label("Root:");
            if let Some(root) = &source.foliage_root {
                ui.monospace(root.to_string_lossy().as_ref());
            } else {
                ui.label("(none)");
            }
        });

        ui.horizontal(|ui| {
            ui.label("Instances/Cell:");
            ui.label(format!("{}", config.instances_per_cell));
        });

        ui.separator();

        // Generation button
        if ui
            .button("🔄 Generate / Regenerate Foliage")
            .clicked()
        {
            panel.generation_in_progress = true;
            // TODO: Trigger instance generation pipeline in Phase 9
        }
        ui.label("").on_hover_text("Bake instance buffers from procedural rules and painted splatmap");

        if panel.generation_in_progress {
            ui.label("⏳ Generation in progress...");
        }
    });

    ui.group(|ui| {
        ui.label("👁 Preview");
        ui.separator();

        ui.checkbox(&mut panel.preview_enabled, "Show Instances in Viewport");
        ui.label("").on_hover_text("Toggle runtime rendering of foliage instances");
    });

    ui.group(|ui| {
        ui.label("⚙ Procedural Parameters");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Slope Threshold:");
            ui.label(format!("{:.2}", config.slope_threshold));
        });

        ui.horizontal(|ui| {
            ui.label("Altitude Min:");
            ui.label(format!("{:.1} m", config.altitude_min));
        });

        ui.horizontal(|ui| {
            ui.label("Altitude Max:");
            ui.label(format!("{:.1} m", config.altitude_max));
        });

        ui.horizontal(|ui| {
            ui.label("LOD0 Distance:");
            ui.label(format!("{:.1} m", config.lod0_distance));
        });

        ui.horizontal(|ui| {
            ui.label("LOD1 Distance:");
            ui.label(format!("{:.1} m", config.lod1_distance));
        });

        ui.horizontal(|ui| {
            ui.label("LOD0 Density:");
            ui.label(format!("{:.0}%", config.lod0_density * 100.0));
        });

        ui.horizontal(|ui| {
            ui.label("LOD1 Density:");
            ui.label(format!("{:.0}%", config.lod1_density * 100.0));
        });

        ui.horizontal(|ui| {
            ui.label("LOD2 Density:");
            ui.label(format!("{:.0}%", config.lod2_density * 100.0));
        });
    });
}




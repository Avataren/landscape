//! Foliage editor panel — UI for foliage instance generation and control.
//!
//! Provides:
//! - Generate/Regenerate foliage button
//! - Procedural parameter inspection
//! - Preview toggle
//! - Status display (instance count, memory usage)

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{FoliageSourceDesc, FoliageConfigResource, FoliageConfig, FoliageGenerateRequest};
use bevy_landscape::foliage_backend::FoliageGenerationState;
use crate::toolbar::ToolbarState;

/// Editor-local UI state for the foliage panel.
#[derive(Resource, Default)]
pub struct FoliagePanelState {
    pub preview_enabled: bool,
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
    mut generate_events: MessageWriter<FoliageGenerateRequest>,
    gen_state: Option<Res<FoliageGenerationState>>,
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
                gen_state.as_deref(),
                &mut generate_events,
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
    gen_state: Option<&FoliageGenerationState>,
    generate_events: &mut MessageWriter<FoliageGenerateRequest>,
) {
    let default_config = FoliageConfig::default();
    let config = config.unwrap_or(&default_config);
    let foliage_root = source.and_then(|s| s.foliage_root.as_ref());

    ui.group(|ui| {
        ui.label("📦 Foliage Generation");
        ui.separator();

        // Status
        ui.horizontal(|ui| {
            ui.label("Root:");
            if let Some(root) = foliage_root {
                ui.monospace(root.to_string_lossy().as_ref());
            } else {
                ui.colored_label(egui::Color32::YELLOW, "Not set — load a level with foliage_root");
            }
        });

        ui.horizontal(|ui| {
            ui.label("Instances/Cell:");
            ui.label(format!("{}", config.instances_per_cell));
        });

        ui.separator();

        // Generation button
        let is_running = gen_state.map(|s| s.is_running).unwrap_or(false);
        let can_generate = foliage_root.is_some() && !is_running;
        ui.add_enabled_ui(can_generate, |ui| {
            if ui
                .button("🔄 Generate / Regenerate Foliage")
                .on_hover_text("Bake instance buffers from procedural rules and painted splatmap")
                .clicked()
            {
                generate_events.write(FoliageGenerateRequest);
            }
        });

        if let Some(state) = gen_state {
            if state.is_running {
                ui.horizontal(|ui| {
                    ui.spinner();
                    if state.tiles_total > 0 {
                        ui.label(format!("⏳ {}/{} tiles...", state.tiles_done, state.tiles_total));
                    } else {
                        ui.label("⏳ Starting...");
                    }
                });
            } else if !state.progress_message.is_empty() {
                ui.label(&state.progress_message);
            }
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




//! Rendering quality panel — runtime controls for SSAO and SSR.

use bevy::{
    core_pipeline::prepass::DepthPrepass,
    pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel},
    prelude::*,
};
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape_water::WaterSettings;
use serde::{Deserialize, Serialize};

use crate::toolbar::ToolbarState;

// ---------------------------------------------------------------------------
// SSAO quality — a serialisable mirror of Bevy's quality level enum.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SsaoQuality {
    Low,
    Medium,
    #[default]
    High,
    Ultra,
}

impl SsaoQuality {
    fn to_bevy(self) -> ScreenSpaceAmbientOcclusionQualityLevel {
        match self {
            Self::Low    => ScreenSpaceAmbientOcclusionQualityLevel::Low,
            Self::Medium => ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            Self::High   => ScreenSpaceAmbientOcclusionQualityLevel::High,
            Self::Ultra  => ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Low    => "Low",
            Self::Medium => "Medium",
            Self::High   => "High",
            Self::Ultra  => "Ultra",
        }
    }
}

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Authoritative SSAO settings.  The `apply_ssao_system` syncs these to the
/// camera entity every time the resource changes.
#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct SsaoSettings {
    pub enabled: bool,
    pub quality: SsaoQuality,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: SsaoQuality::High,
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct RenderingPanelPlugin;

impl Plugin for RenderingPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SsaoSettings>()
            .add_systems(EguiPrimaryContextPass, rendering_panel_system)
            .add_systems(PostUpdate, apply_ssao_system);
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Syncs `SsaoSettings` → camera `ScreenSpaceAmbientOcclusion` component.
fn apply_ssao_system(
    settings: Res<SsaoSettings>,
    camera_q: Query<Entity, (With<Camera3d>, With<DepthPrepass>)>,
    mut commands: Commands,
) {
    if !settings.is_changed() {
        return;
    }
    let Ok(camera) = camera_q.single() else {
        return;
    };
    if settings.enabled {
        commands.entity(camera).insert(ScreenSpaceAmbientOcclusion {
            quality_level: settings.quality.to_bevy(),
            ..default()
        });
    } else {
        commands.entity(camera).remove::<ScreenSpaceAmbientOcclusion>();
    }
}

fn rendering_panel_system(
    mut contexts: EguiContexts,
    mut toolbar: ResMut<ToolbarState>,
    mut ssao: ResMut<SsaoSettings>,
    mut water: Option<ResMut<WaterSettings>>,
) -> Result {
    if !toolbar.rendering_open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;

    let mut open = toolbar.rendering_open;
    egui::Window::new("Rendering")
        .open(&mut open)
        .resizable(true)
        .min_width(280.0)
        .show(ctx, |ui| {
            // ----------------------------------------------------------------
            // SSAO
            // ----------------------------------------------------------------
            ui.heading("Ambient Occlusion (SSAO)");
            ui.separator();

            let mut enabled = ssao.enabled;
            if ui.checkbox(&mut enabled, "Enabled").changed() {
                ssao.enabled = enabled;
            }

            ui.add_enabled_ui(ssao.enabled, |ui| {
                ui.label("Quality");
                for q in [
                    SsaoQuality::Low,
                    SsaoQuality::Medium,
                    SsaoQuality::High,
                    SsaoQuality::Ultra,
                ] {
                    let selected = ssao.quality == q;
                    if ui.radio(selected, q.label()).clicked() && selected {
                        ssao.quality = q;
                    }
                }
            });

            ui.add_space(8.0);

            // ----------------------------------------------------------------
            // SSR
            // ----------------------------------------------------------------
            ui.heading("Screen-Space Reflections (SSR)");
            ui.separator();

            if let Some(ref mut ws) = water {
                let mut ssr_enabled = ws.ssr_enabled;
                if ui.checkbox(&mut ssr_enabled, "Enabled").changed() {
                    ws.ssr_enabled = ssr_enabled;
                }

                ui.add_enabled_ui(ws.ssr_enabled, |ui| {
                    let mut steps = ws.ssr_steps;
                    if ui
                        .add(
                            egui::Slider::new(&mut steps, 4..=128)
                                .text("Steps")
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed()
                    {
                        ws.ssr_steps = steps;
                    }
                    ui.label("  Higher = fewer missed hits, more GPU cost.");

                    let mut dist = ws.ssr_max_distance;
                    if ui
                        .add(
                            egui::Slider::new(&mut dist, 50.0..=1000.0)
                                .text("Max distance (m)")
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed()
                    {
                        ws.ssr_max_distance = dist;
                    }

                    let mut thickness = ws.ssr_thickness;
                    if ui
                        .add(
                            egui::Slider::new(&mut thickness, 1.0..=30.0)
                                .text("Thickness (m)")
                                .clamping(egui::SliderClamping::Always),
                        )
                        .changed()
                    {
                        ws.ssr_thickness = thickness;
                    }
                    ui.label("  Lower = less banding, may miss thin geometry.");
                });
            } else {
                ui.label("(Water not active — SSR has no effect)");
            }
        });

    toolbar.rendering_open = open;
    Ok(())
}

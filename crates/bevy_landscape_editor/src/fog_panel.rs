use bevy::light::{FogVolume, VolumetricFog};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

#[derive(Resource)]
pub struct FogPanelState {
    pub open: bool,
    pub enabled: bool,
    /// Saved values restored when re-enabling fog.
    saved_density: f32,
    saved_ambient_intensity: f32,
}

impl Default for FogPanelState {
    fn default() -> Self {
        Self {
            open: false,
            enabled: false,
            saved_density: 0.0003,
            saved_ambient_intensity: 0.6,
        }
    }
}

pub struct FogPanelPlugin;

impl Plugin for FogPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FogPanelState>()
            .add_systems(EguiPrimaryContextPass, fog_panel_system)
            .add_systems(Update, apply_fog_enabled);
    }
}

fn color_to_rgb(c: Color) -> [f32; 3] {
    let l = c.to_linear();
    [l.red, l.green, l.blue]
}

fn rgb_to_color(rgb: [f32; 3]) -> Color {
    Color::linear_rgb(rgb[0], rgb[1], rgb[2])
}

pub(crate) fn fog_panel_system(
    mut contexts: EguiContexts,
    mut state: ResMut<FogPanelState>,
    mut fog_volumes: Query<&mut FogVolume>,
    mut volumetric_fogs: Query<&mut VolumetricFog>,
) -> Result {
    if !state.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    let mut open = state.open;
    egui::Window::new("Fog Settings")
        .open(&mut open)
        .resizable(true)
        .min_width(320.0)
        .show(ctx, |ui| {
            // --- Enable toggle ---
            let mut enabled = state.enabled;
            if ui.checkbox(&mut enabled, "Enabled").changed() {
                state.enabled = enabled;
            }

            ui.separator();

            let fog_ok = fog_volumes.single_mut();
            if let Ok(mut fog) = fog_ok {
                ui.heading("Fog Volume");
                egui::Grid::new("fog_volume_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Density");
                        if ui.add(
                            egui::Slider::new(&mut state.saved_density, 0.0..=0.002)
                                .logarithmic(true)
                                .fixed_decimals(6),
                        ).changed() && state.enabled {
                            fog.density_factor = state.saved_density;
                        }
                        ui.end_row();

                        ui.label("Absorption");
                        ui.add(egui::Slider::new(&mut fog.absorption, 0.0..=1.0).fixed_decimals(3));
                        ui.end_row();

                        ui.label("Scattering");
                        ui.add(egui::Slider::new(&mut fog.scattering, 0.0..=1.0).fixed_decimals(3));
                        ui.end_row();

                        ui.label("Asymmetry");
                        ui.add(
                            egui::Slider::new(&mut fog.scattering_asymmetry, -1.0..=1.0)
                                .fixed_decimals(3),
                        );
                        ui.end_row();

                        ui.label("Fog color");
                        let mut rgb = color_to_rgb(fog.fog_color);
                        if ui.color_edit_button_rgb(&mut rgb).changed() {
                            fog.fog_color = rgb_to_color(rgb);
                        }
                        ui.end_row();

                        ui.label("Light tint");
                        let mut rgb = color_to_rgb(fog.light_tint);
                        if ui.color_edit_button_rgb(&mut rgb).changed() {
                            fog.light_tint = rgb_to_color(rgb);
                        }
                        ui.end_row();
                    });
            }

            ui.separator();

            if let Ok(mut vfog) = volumetric_fogs.single_mut() {
                ui.heading("Volumetric (camera)");
                egui::Grid::new("vfog_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Ambient intensity");
                        if ui.add(
                            egui::Slider::new(&mut state.saved_ambient_intensity, 0.0..=1.0)
                                .fixed_decimals(3),
                        ).changed() && state.enabled {
                            vfog.ambient_intensity = state.saved_ambient_intensity;
                        }
                        ui.end_row();

                        ui.label("Ambient color");
                        let mut rgb = color_to_rgb(vfog.ambient_color);
                        if ui.color_edit_button_rgb(&mut rgb).changed() {
                            vfog.ambient_color = rgb_to_color(rgb);
                        }
                        ui.end_row();

                        ui.label("Step count");
                        let mut steps = vfog.step_count;
                        if ui.add(egui::Slider::new(&mut steps, 16..=256)).changed() {
                            vfog.step_count = steps;
                        }
                        ui.end_row();

                        ui.label("Jitter");
                        ui.add(egui::Slider::new(&mut vfog.jitter, 0.0..=1.0).fixed_decimals(3));
                        ui.end_row();
                    });
            }
        });
    state.open = open;
    Ok(())
}

fn apply_fog_enabled(
    state: Res<FogPanelState>,
    mut fog_volumes: Query<&mut FogVolume>,
    mut volumetric_fogs: Query<&mut VolumetricFog>,
) {
    if !state.is_changed() {
        return;
    }
    if let Ok(mut fog) = fog_volumes.single_mut() {
        fog.density_factor = if state.enabled { state.saved_density } else { 0.0 };
    }
    // Zero ambient_intensity when disabled — the volumetric renderer accumulates
    // ambient on every ray-march step inside the FogVolume AABB, so even with
    // density=0 a non-zero ambient_intensity creates a visible seam at the slab top.
    if let Ok(mut vfog) = volumetric_fogs.single_mut() {
        vfog.ambient_intensity = if state.enabled { state.saved_ambient_intensity } else { 0.0 };
    }
}

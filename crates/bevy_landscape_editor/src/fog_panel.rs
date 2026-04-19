use bevy::light::{FogVolume, VolumetricFog};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

#[derive(Resource)]
pub struct FogPanelState {
    pub open: bool,
    pub enabled: bool,
    // FogVolume (entity always present; density zeroed when disabled)
    pub density: f32,
    // VolumetricFog (camera component removed entirely when disabled to stop ray-marching)
    pub ambient_intensity: f32,
    pub ambient_color: [f32; 3],
    pub step_count: u32,
    pub jitter: f32,
}

impl Default for FogPanelState {
    fn default() -> Self {
        Self {
            open: false,
            enabled: false,
            density: 0.0003,
            ambient_intensity: 0.6,
            ambient_color: [0.0, 0.0, 0.0],
            step_count: 64,
            jitter: 0.5,
        }
    }
}

pub struct FogPanelPlugin;

impl Plugin for FogPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FogPanelState>()
            .add_systems(EguiPrimaryContextPass, fog_panel_system)
            .add_systems(Update, apply_fog_state);
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
            let mut enabled = state.enabled;
            if ui.checkbox(&mut enabled, "Enabled").changed() {
                state.enabled = enabled;
            }

            ui.separator();

            if let Ok(mut fog) = fog_volumes.single_mut() {
                ui.heading("Fog Volume");
                egui::Grid::new("fog_volume_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Density");
                        if ui
                            .add(
                                egui::Slider::new(&mut state.density, 0.0..=0.002)
                                    .logarithmic(true)
                                    .fixed_decimals(6),
                            )
                            .changed()
                            && state.enabled
                        {
                            fog.density_factor = state.density;
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

            // VolumetricFog settings are stored in state regardless of enabled/disabled
            // so values are preserved across toggles.
            ui.heading("Volumetric (camera)");
            ui.add_enabled_ui(state.enabled, |ui| {
                egui::Grid::new("vfog_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Ambient intensity");
                        ui.add(
                            egui::Slider::new(&mut state.ambient_intensity, 0.0..=1.0)
                                .fixed_decimals(3),
                        );
                        ui.end_row();

                        ui.label("Ambient color");
                        ui.color_edit_button_rgb(&mut state.ambient_color);
                        ui.end_row();

                        ui.label("Step count");
                        ui.add(egui::Slider::new(&mut state.step_count, 16..=256));
                        ui.end_row();

                        ui.label("Jitter");
                        ui.add(egui::Slider::new(&mut state.jitter, 0.0..=1.0).fixed_decimals(3));
                        ui.end_row();
                    });
            });
        });
    state.open = open;
    Ok(())
}

/// Applies fog panel state to the scene.
///
/// When enabled: ensures the camera has a `VolumetricFog` component with the
/// panel's current settings, and sets the FogVolume density.
/// When disabled: removes `VolumetricFog` from the camera entirely so Bevy
/// does not dispatch a volumetric render pass at all.
fn apply_fog_state(
    mut commands: Commands,
    state: Res<FogPanelState>,
    mut fog_volumes: Query<&mut FogVolume>,
    cameras: Query<Entity, With<Camera3d>>,
) {
    if !state.is_changed() {
        return;
    }
    if let Ok(mut fog) = fog_volumes.single_mut() {
        fog.density_factor = if state.enabled { state.density } else { 0.0 };
    }
    let Ok(camera) = cameras.single() else { return };
    if state.enabled {
        commands.entity(camera).insert(VolumetricFog {
            ambient_intensity: state.ambient_intensity,
            ambient_color: rgb_to_color(state.ambient_color),
            step_count: state.step_count,
            jitter: state.jitter,
            ..default()
        });
    } else {
        commands.entity(camera).remove::<VolumetricFog>();
    }
}

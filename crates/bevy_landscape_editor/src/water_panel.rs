use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape_water::WaterSettings;

#[derive(Resource, Default)]
pub struct WaterPanelState {
    pub open: bool,
}

pub struct WaterPanelPlugin;

impl Plugin for WaterPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaterPanelState>()
            .add_systems(EguiPrimaryContextPass, water_panel_system);
    }
}

fn color_to_rgb(c: Color) -> [f32; 3] {
    let l = c.to_linear();
    [l.red, l.green, l.blue]
}

fn rgb_to_color(rgb: [f32; 3]) -> Color {
    Color::linear_rgb(rgb[0], rgb[1], rgb[2])
}

pub(crate) fn water_panel_system(
    mut contexts: EguiContexts,
    mut state: ResMut<WaterPanelState>,
    settings: Option<ResMut<WaterSettings>>,
) -> Result {
    if !state.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    let mut open = state.open;
    let Some(mut settings) = settings else {
        state.open = open;
        return Ok(());
    };

    egui::Window::new("Water Settings")
        .open(&mut open)
        .resizable(true)
        .min_width(360.0)
        .show(ctx, |ui| {
            // Mut::as_mut() is the standard egui-pattern entry point — it
            // marks the resource changed for the frame, which is fine since
            // the uniform upload is cheap and the panel is editor-only.
            let s = settings.as_mut();

            ui.heading("Waves");
            egui::Grid::new("water_waves")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Amplitude").on_hover_text(
                        "Global Gerstner amplitude (m).  Drives wave height.",
                    );
                    ui.add(egui::Slider::new(&mut s.amplitude, 0.0..=20.0).fixed_decimals(2));
                    ui.end_row();

                    ui.label("Wave speed").on_hover_text(
                        "1.0 = physical dispersion. Lower = sluggish, higher = stormier.",
                    );
                    ui.add(egui::Slider::new(&mut s.wave_speed, 0.0..=4.0).fixed_decimals(2));
                    ui.end_row();

                    ui.label("Direction X");
                    ui.add(
                        egui::Slider::new(&mut s.wave_direction.x, -1.0..=1.0).fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Direction Z");
                    ui.add(
                        egui::Slider::new(&mut s.wave_direction.y, -1.0..=1.0).fixed_decimals(2),
                    );
                    ui.end_row();
                });

            ui.separator();
            ui.heading("Macro Noise (anti-repetition)");
            ui.label("Long-wavelength stochastic FBM that survives at the horizon.")
                .on_hover_text(
                    "0 amplitude = pure Gerstner. >0 breaks the visible periodicity.",
                );
            egui::Grid::new("water_macro")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Amplitude (m)");
                    ui.add(
                        egui::Slider::new(&mut s.macro_noise_amplitude, 0.0..=8.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Wavelength (m)");
                    ui.add(
                        egui::Slider::new(&mut s.macro_noise_scale, 20.0..=400.0)
                            .fixed_decimals(0),
                    );
                    ui.end_row();
                });

            ui.separator();
            ui.heading("Surface Detail");
            egui::Grid::new("water_detail")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Capillary strength").on_hover_text(
                        "High-frequency stochastic normal noise (sun glitter).",
                    );
                    ui.add(
                        egui::Slider::new(&mut s.capillary_strength, 0.0..=3.0).fixed_decimals(2),
                    );
                    ui.end_row();
                });

            ui.separator();
            ui.heading("Foam");
            egui::Grid::new("water_foam")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Crest threshold").on_hover_text(
                        "0 = foam everywhere, 1 = foam only on the very tallest crests.",
                    );
                    ui.add(
                        egui::Slider::new(&mut s.foam_threshold, 0.0..=1.0).fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Jacobian (foldover)").on_hover_text(
                        "Foam from wave-folding. Streaky/organic. 0 = off.",
                    );
                    ui.add(
                        egui::Slider::new(&mut s.jacobian_foam_strength, 0.0..=3.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Shore foam depth (m)");
                    ui.add(
                        egui::Slider::new(&mut s.shoreline_foam_depth, 0.0..=10.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Shore wave damp (m)").on_hover_text(
                        "Depth range over which wave displacement fades flat near shore.",
                    );
                    ui.add(
                        egui::Slider::new(&mut s.shore_wave_damp_width, 0.0..=20.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Foam color");
                    let mut rgb = color_to_rgb(s.foam_color);
                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                        s.foam_color = rgb_to_color(rgb);
                    }
                    ui.end_row();
                });

            ui.separator();
            ui.heading("Optical");
            egui::Grid::new("water_optical")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Clarity");
                    ui.add(egui::Slider::new(&mut s.clarity, 0.0..=1.0).fixed_decimals(2));
                    ui.end_row();

                    ui.label("Refraction strength");
                    ui.add(
                        egui::Slider::new(&mut s.refraction_strength, 0.0..=64.0)
                            .fixed_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Edge scale");
                    ui.add(egui::Slider::new(&mut s.edge_scale, 0.001..=2.0).fixed_decimals(3));
                    ui.end_row();

                    ui.label("Deep color");
                    let mut rgb = color_to_rgb(s.deep_color);
                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                        s.deep_color = rgb_to_color(rgb);
                    }
                    ui.end_row();

                    ui.label("Shallow color");
                    let mut rgb = color_to_rgb(s.shallow_color);
                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                        s.shallow_color = rgb_to_color(rgb);
                    }
                    ui.end_row();

                    ui.label("Edge color");
                    let mut rgb = color_to_rgb(s.edge_color);
                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                        s.edge_color = rgb_to_color(rgb);
                    }
                    ui.end_row();
                });

            ui.separator();
            if ui.button("Reset to defaults").clicked() {
                let prev_height = s.height;
                let defaults = WaterSettings::default();
                *s = WaterSettings {
                    height: prev_height,
                    ..defaults
                };
            }
        });
    state.open = open;
    Ok(())
}

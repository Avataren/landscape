use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::DetailSynthesisConfig;

#[derive(Resource, Default)]
pub struct SynthesisPanelState {
    pub open: bool,
}

pub struct SynthesisPanelPlugin;

impl Plugin for SynthesisPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SynthesisPanelState>()
            .add_systems(EguiPrimaryContextPass, synthesis_panel_system);
    }
}

pub(crate) fn synthesis_panel_system(
    mut contexts: EguiContexts,
    mut state: ResMut<SynthesisPanelState>,
    synthesis: Option<ResMut<DetailSynthesisConfig>>,
) -> Result {
    if !state.open {
        return Ok(());
    }
    let Some(mut syn) = synthesis else {
        return Ok(());
    };

    let ctx = contexts.ctx_mut()?;
    let mut open = state.open;
    egui::Window::new("Landscape Synthesis")
        .open(&mut open)
        .resizable(true)
        .min_width(320.0)
        .show(ctx, |ui| {
            ui.checkbox(&mut syn.enabled, "Enabled")
                .on_hover_text("Toggle GPU fBM detail synthesis. When off, terrain shows the raw source heightmap.");

            ui.add_enabled_ui(syn.enabled, |ui| {
                ui.separator();
                ui.heading("Detail fBM");
                egui::Grid::new("synth_detail_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Amplitude");
                        ui.add(
                            egui::Slider::new(&mut syn.max_amplitude, 0.0_f32..=500.0)
                                .suffix(" m")
                                .fixed_decimals(1),
                        )
                        .on_hover_text("Maximum fBM height residual added on top of the source.");
                        ui.end_row();

                        ui.label("Lacunarity");
                        ui.add(
                            egui::Slider::new(&mut syn.lacunarity, 1.5_f32..=3.0)
                                .fixed_decimals(2),
                        )
                        .on_hover_text("Frequency multiplier per fBM octave (typical 2.0–2.2).");
                        ui.end_row();

                        ui.label("Gain");
                        ui.add(
                            egui::Slider::new(&mut syn.gain, 0.25_f32..=0.75)
                                .fixed_decimals(2),
                        )
                        .on_hover_text(
                            "Amplitude multiplier per octave (persistence). Useful \
                             range ~0.3–0.7; lower values collapse the fBM to a \
                             single visible octave and the noise lattice shows.",
                        );
                        ui.end_row();

                        ui.label("Erosion");
                        ui.add(
                            egui::Slider::new(&mut syn.erosion_strength, 0.0_f32..=1.0)
                                .fixed_decimals(2),
                        )
                        .on_hover_text(
                            "0 = isotropic fBM bumps; 1 = gradient-attenuated, ridge/valley shaped.",
                        );
                        ui.end_row();
                    });

                ui.separator();
                ui.heading("Per-fragment normal");
                egui::Grid::new("synth_normal_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Detail strength");
                        ui.add(
                            egui::Slider::new(&mut syn.normal_detail_strength, 0.0_f32..=200.0)
                                .suffix(" m")
                                .fixed_decimals(1),
                        )
                        .on_hover_text(
                            "Virtual fBM displacement (in metres) used only for the \
                             per-fragment normal perturbation.  Independent of \
                             Amplitude — set Amplitude=0 with this >0 to keep noise \
                             lighting on a geometrically smooth surface.",
                        );
                        ui.end_row();
                    });

                ui.separator();
                ui.heading("Slope mask");
                egui::Grid::new("synth_slope_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Cutoff");
                        ui.add(
                            egui::Slider::new(&mut syn.slope_mask_threshold_deg, 0.0_f32..=80.0)
                                .suffix("°")
                                .fixed_decimals(1),
                        )
                        .on_hover_text(
                            "Slope angle (from source heightmap) above which fBM starts fading.",
                        );
                        ui.end_row();

                        ui.label("Falloff");
                        ui.add(
                            egui::Slider::new(&mut syn.slope_mask_falloff_deg, 0.1_f32..=45.0)
                                .suffix("°")
                                .fixed_decimals(1),
                        )
                        .on_hover_text("Width of the fade band above the cutoff.");
                        ui.end_row();
                    });

                ui.separator();
                ui.heading("Noise seed");
                egui::Grid::new("synth_seed_grid")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Seed X");
                        ui.add(
                            egui::DragValue::new(&mut syn.seed.x)
                                .speed(7.31)
                                .fixed_decimals(2),
                        );
                        ui.end_row();

                        ui.label("Seed Z");
                        ui.add(
                            egui::DragValue::new(&mut syn.seed.y)
                                .speed(7.31)
                                .fixed_decimals(2),
                        );
                        ui.end_row();
                    });

                if ui.button("Reset to defaults").clicked() {
                    *syn = DetailSynthesisConfig::default();
                }
            });
        });
    state.open = open;
    Ok(())
}

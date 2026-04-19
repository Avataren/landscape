use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape_clouds::CloudsConfig;

#[derive(Resource, Default)]
pub struct CloudPanelState {
    pub open: bool,
}

pub struct CloudPanelPlugin;

impl Plugin for CloudPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CloudPanelState>()
            .add_systems(EguiPrimaryContextPass, cloud_panel_system);
    }
}

fn vec4_to_rgba(color: Vec4) -> [f32; 4] {
    [color.x, color.y, color.z, color.w]
}

fn rgba_to_vec4(color: [f32; 4]) -> Vec4 {
    Vec4::from_array(color)
}

pub(crate) fn cloud_panel_system(
    mut contexts: EguiContexts,
    mut state: ResMut<CloudPanelState>,
    mut config: ResMut<CloudsConfig>,
) -> Result {
    if !state.open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;
    let mut open = state.open;
    egui::Window::new("Cloud Settings")
        .open(&mut open)
        .resizable(true)
        .min_width(340.0)
        .show(ctx, |ui| {
            if ui.button("Reset to Defaults").clicked() {
                *config = CloudsConfig::default();
            }

            ui.separator();
            egui::Grid::new("cloud_settings_grid")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Bottom height");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_bottom_height,
                        200.0..=4000.0,
                    ));
                    ui.end_row();

                    ui.label("Top height");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_top_height,
                        400.0..=6000.0,
                    ));
                    ui.end_row();

                    ui.label("Planet radius");
                    ui.add(
                        egui::Slider::new(&mut config.planet_radius, 1_000_000.0..=10_000_000.0)
                            .logarithmic(true),
                    );
                    ui.end_row();

                    ui.label("Coverage");
                    ui.add(
                        egui::Slider::new(&mut config.cloud_coverage, 0.0..=1.0).fixed_decimals(3),
                    );
                    ui.end_row();

                    ui.label("Density");
                    ui.add(
                        egui::Slider::new(&mut config.cloud_density, 0.001..=0.12)
                            .logarithmic(true)
                            .fixed_decimals(4),
                    );
                    ui.end_row();

                    ui.label("Base scale");
                    ui.add(
                        egui::Slider::new(&mut config.cloud_base_scale, 0.25..=6.0)
                            .logarithmic(true),
                    );
                    ui.end_row();

                    ui.label("Detail scale");
                    ui.add(
                        egui::Slider::new(&mut config.cloud_detail_scale, 1.0..=96.0)
                            .logarithmic(true),
                    );
                    ui.end_row();

                    ui.label("Detail strength");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_detail_strength,
                        0.0..=1.0,
                    ));
                    ui.end_row();

                    ui.label("Edge softness");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_base_edge_softness,
                        0.01..=0.4,
                    ));
                    ui.end_row();

                    ui.label("Bottom softness");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_bottom_softness,
                        0.01..=0.5,
                    ));
                    ui.end_row();

                    ui.label("View steps");
                    ui.add(egui::Slider::new(&mut config.cloud_view_steps, 4..=64));
                    ui.end_row();

                    ui.label("Shadow steps");
                    ui.add(egui::Slider::new(&mut config.cloud_shadow_steps, 0..=12));
                    ui.end_row();

                    ui.label("Shadow step size");
                    ui.add(
                        egui::Slider::new(&mut config.cloud_shadow_step_size, 1.0..=40.0)
                            .logarithmic(true),
                    );
                    ui.end_row();

                    ui.label("Shadow step x");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_shadow_step_multiply,
                        1.0..=2.5,
                    ));
                    ui.end_row();

                    ui.label("History blend");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_history_blend,
                        0.0..=0.98,
                    ));
                    ui.end_row();

                    ui.label("Forward phase");
                    ui.add(egui::Slider::new(
                        &mut config.forward_scattering_g,
                        0.0..=0.95,
                    ));
                    ui.end_row();

                    ui.label("Backward phase");
                    ui.add(egui::Slider::new(
                        &mut config.backward_scattering_g,
                        -0.95..=0.0,
                    ));
                    ui.end_row();

                    ui.label("Phase mix");
                    ui.add(egui::Slider::new(&mut config.scattering_lerp, 0.0..=1.0));
                    ui.end_row();

                    ui.label("Min transmittance");
                    ui.add(egui::Slider::new(
                        &mut config.cloud_min_transmittance,
                        0.0..=0.5,
                    ));
                    ui.end_row();

                    ui.label("Wind X");
                    ui.add(egui::Slider::new(&mut config.wind_velocity.x, -20.0..=20.0));
                    ui.end_row();

                    ui.label("Wind Z");
                    ui.add(egui::Slider::new(&mut config.wind_velocity.z, -20.0..=20.0));
                    ui.end_row();
                });

            ui.separator();
            ui.label("Ambient top");
            let mut ambient_top = vec4_to_rgba(config.cloud_ambient_color_top);
            if ui
                .color_edit_button_rgba_unmultiplied(&mut ambient_top)
                .changed()
            {
                config.cloud_ambient_color_top = rgba_to_vec4(ambient_top);
            }
            ui.label("Ambient bottom");
            let mut ambient_bottom = vec4_to_rgba(config.cloud_ambient_color_bottom);
            if ui
                .color_edit_button_rgba_unmultiplied(&mut ambient_bottom)
                .changed()
            {
                config.cloud_ambient_color_bottom = rgba_to_vec4(ambient_bottom);
            }
        });
    state.open = open;
    if config.cloud_top_height <= config.cloud_bottom_height + 50.0 {
        config.cloud_top_height = config.cloud_bottom_height + 50.0;
    }
    Ok(())
}

//! Foliage editor panel — controls for the two-LOD GPU grass system.

use crate::toolbar::ToolbarState;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{GpuGrassConfig, GRASS_MAX_GRID};

pub struct FoliagePanelPlugin;

impl Plugin for FoliagePanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(EguiPrimaryContextPass, foliage_panel_system);
    }
}

fn foliage_panel_system(
    mut contexts: EguiContexts,
    mut toolbar: ResMut<ToolbarState>,
    mut grass: Option<ResMut<GpuGrassConfig>>,
) -> Result {
    if !toolbar.foliage_open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;
    let mut open = toolbar.foliage_open;

    egui::Window::new("Grass")
        .open(&mut open)
        .default_width(310.0)
        .resizable(true)
        .show(ctx, |ui| {
            let Some(ref mut grass_res) = grass else {
                ui.label("GpuGrassConfig not available.");
                return;
            };
            // Explicit deref so field borrows are plain struct projections,
            // not split borrows across DerefMut (which the borrow checker rejects).
            let cfg: &mut GpuGrassConfig = &mut **grass_res;

            ui.checkbox(&mut cfg.enabled, "Enabled");
            ui.separator();

            // ── Near LOD ───────────────────────────────────────────────────
            ui.strong("Near grass  (dense, close range)");
            {
                let range_before = cfg.near_range;
                let max_near = (GRASS_MAX_GRID as f32 / 2.0) * cfg.near_spacing;
                ui.horizontal(|ui| {
                    ui.label("Range  ");
                    let mut r = cfg.near_range;
                    if ui
                        .add(
                            egui::Slider::new(&mut r, 20.0..=max_near.max(200.0))
                                .suffix(" m")
                                .integer(),
                        )
                        .changed()
                    {
                        cfg.near_range = r.min(max_near);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Spacing");
                    let mut s = cfg.near_spacing;
                    if ui
                        .add(
                            egui::Slider::new(&mut s, 0.3..=3.0)
                                .suffix(" m")
                                .logarithmic(true),
                        )
                        .changed()
                    {
                        cfg.near_spacing = s;
                        // Keep range constant: adjust grid_size to compensate.
                        let new_g = ((range_before * 2.0) / s).round() as u32;
                        let new_g = new_g.clamp(4, GRASS_MAX_GRID);
                        cfg.near_range = (new_g as f32 / 2.0) * s;
                    }
                });
                let ng = cfg.near_grid_size();
                ui.label(
                    egui::RichText::new(format!(
                        "{}×{} = {}k blades  |  range {:.0} m",
                        ng,
                        ng,
                        ng * ng / 1000,
                        cfg.near_range
                    ))
                    .small()
                    .color(egui::Color32::GRAY),
                );
            }

            ui.add_space(6.0);

            // ── Far LOD ────────────────────────────────────────────────────
            ui.strong("Far grass  (sparse, extension past near)");
            {
                // Far spacing must be coarser than near.
                let min_far_sp = (cfg.near_spacing * 2.0).max(1.0);
                if cfg.far_spacing < min_far_sp {
                    cfg.far_spacing = min_far_sp;
                }

                let range_before = cfg.far_range;
                // Max far extension: limited by how many cells fit at current spacing
                // after accounting for the near radius already consumed.
                let max_far_ext =
                    ((GRASS_MAX_GRID as f32 / 2.0) * cfg.far_spacing - cfg.near_range).max(100.0);
                ui.horizontal(|ui| {
                    ui.label("Extension");
                    let mut r = cfg.far_range;
                    if ui
                        .add(
                            egui::Slider::new(&mut r, 50.0..=max_far_ext.max(2000.0))
                                .suffix(" m past near")
                                .integer(),
                        )
                        .changed()
                    {
                        cfg.far_range = r.min(max_far_ext);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Spacing  ");
                    let mut s = cfg.far_spacing;
                    if ui
                        .add(
                            egui::Slider::new(&mut s, min_far_sp..=5.0)
                                .suffix(" m")
                                .logarithmic(true),
                        )
                        .changed()
                    {
                        cfg.far_spacing = s;
                        // Keep total radius constant when spacing changes.
                        let total_r = cfg.near_range + range_before;
                        let new_g = ((total_r * 2.0) / s).round() as u32;
                        let new_g = new_g.clamp(4, GRASS_MAX_GRID);
                        let new_total = (new_g as f32 / 2.0) * s;
                        cfg.far_range = (new_total - cfg.near_range).max(50.0);
                    }
                });
                let fg = cfg.far_grid_size();
                let total_r = cfg.near_range + cfg.far_range;
                ui.label(
                    egui::RichText::new(format!(
                        "{}×{} = {}k blades  |  {:.0}–{:.0} m from camera",
                        fg,
                        fg,
                        fg * fg / 1000,
                        cfg.near_range,
                        total_r
                    ))
                    .small()
                    .color(egui::Color32::GRAY),
                );
            }

            // Total.
            let ng = cfg.near_grid_size();
            let fg = cfg.far_grid_size();
            ui.label(
                egui::RichText::new(format!(
                    "Total: {}k + {}k = {}k blades",
                    ng * ng / 1000,
                    fg * fg / 1000,
                    (ng * ng + fg * fg) / 1000
                ))
                .small()
                .color(egui::Color32::DARK_GRAY),
            );

            ui.separator();

            // ── Blade appearance ───────────────────────────────────────────
            ui.label("Blade appearance");
            ui.horizontal(|ui| {
                ui.label("Height");
                ui.add(egui::Slider::new(&mut cfg.blade_height, 0.1..=5.0).suffix(" m"));
            });
            ui.horizontal(|ui| {
                ui.label("Width ");
                ui.add(egui::Slider::new(&mut cfg.blade_width, 0.1..=3.0).suffix(" m"));
            });
            ui.horizontal(|ui| {
                ui.label("Color ");
                let mut rgb = [
                    cfg.base_color.red,
                    cfg.base_color.green,
                    cfg.base_color.blue,
                ];
                if ui.color_edit_button_rgb(&mut rgb).changed() {
                    cfg.base_color.red = rgb[0];
                    cfg.base_color.green = rgb[1];
                    cfg.base_color.blue = rgb[2];
                }
            });

            ui.separator();

            // ── Wind ───────────────────────────────────────────────────────
            ui.label("Wind");
            ui.horizontal(|ui| {
                ui.label("Strength");
                ui.add(egui::Slider::new(&mut cfg.wind_strength, 0.0..=2.0));
            });
            ui.horizontal(|ui| {
                ui.label("Scale   ");
                ui.add(egui::Slider::new(&mut cfg.wind_scale, 0.005..=0.2).logarithmic(true));
            });

            ui.separator();

            // ── Material slot 0 binding ────────────────────────────────────
            ui.separator();
            ui.checkbox(
                &mut cfg.link_to_slot0,
                "Follow material slot 0  (altitude & slope from ground texture)",
            );

            // ── Slope ──────────────────────────────────────────────────────
            ui.label("Slope limit");
            ui.horizontal(|ui| {
                let mut slope_enabled = cfg.slope_max < 90.0;
                if ui.checkbox(&mut slope_enabled, "").changed() {
                    cfg.slope_max = if slope_enabled { 0.7 } else { 999.0 };
                }
                ui.add_enabled(
                    slope_enabled,
                    egui::Slider::new(&mut cfg.slope_max, 0.1..=3.0).custom_formatter(|v, _| {
                        format!("{:.1}  (~{:.0}°)", v, v.atan() * 57.2958)
                    }),
                );
            });

            // ── Altitude ───────────────────────────────────────────────────
            ui.collapsing("Altitude filter", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Min");
                    ui.add(
                        egui::DragValue::new(&mut cfg.altitude_min)
                            .suffix(" m")
                            .speed(10.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Max");
                    ui.add(
                        egui::DragValue::new(&mut cfg.altitude_max)
                            .suffix(" m")
                            .speed(10.0),
                    );
                });
            });
        });

    toolbar.foliage_open = open;
    Ok(())
}

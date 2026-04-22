use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, TextureId},
    EguiContexts, EguiPrimaryContextPass,
};
use bevy_landscape::{MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainSourceDesc};
use bevy_landscape_generator::{
    export::{GeneratorExportState, StartGeneratorExport},
    GeneratorErosionBuffers, GeneratorErosionControlState, GeneratorErosionParams, GeneratorParams,
    HeightfieldImage,
};

use crate::diffusion_panel::{draw_diffusion_tab, poll_diffusion_state, DiffusionPanelState};
use crate::preferences::AppPreferences;

#[derive(Default, Clone, Copy, PartialEq)]
enum PreviewMode {
    #[default]
    Height,
    Water,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum GeneratorTab {
    #[default]
    Legacy,
    Diffusion,
}

#[derive(Resource)]
pub struct GeneratorPanelState {
    pub open: bool,
    preview_id: Option<TextureId>,
    preview_handle: Option<Handle<Image>>,
    water_preview_id: Option<TextureId>,
    water_preview_handle: Option<Handle<Image>>,
    preview_mode: PreviewMode,
    output_dir: String,
    last_completed_generation: u64,
    active_tab: GeneratorTab,
    diffusion: DiffusionPanelState,
}

impl Default for GeneratorPanelState {
    fn default() -> Self {
        Self {
            open: false,
            preview_id: None,
            preview_handle: None,
            water_preview_id: None,
            water_preview_handle: None,
            preview_mode: PreviewMode::Height,
            output_dir: "assets/tiles_generated".into(),
            last_completed_generation: 0,
            active_tab: GeneratorTab::Legacy,
            diffusion: DiffusionPanelState::new(),
        }
    }
}

pub(crate) struct GeneratorPanelPlugin;

impl Plugin for GeneratorPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeneratorPanelState>()
            .add_systems(EguiPrimaryContextPass, generator_panel_system);
    }
}

fn generator_panel_system(
    mut contexts: EguiContexts,
    mut panel: ResMut<GeneratorPanelState>,
    mut params: ResMut<GeneratorParams>,
    mut erosion: ResMut<GeneratorErosionParams>,
    erosion_ctrl: Res<GeneratorErosionControlState>,
    export_state: Res<GeneratorExportState>,
    gen_image: Option<Res<HeightfieldImage>>,
    erosion_buffers: Option<Res<GeneratorErosionBuffers>>,
    active_config: Res<TerrainConfig>,
    active_library: Res<MaterialLibrary>,
    prefs: Res<AppPreferences>,
    mut reload_tx: MessageWriter<ReloadTerrainRequest>,
    mut export_tx: MessageWriter<StartGeneratorExport>,
) -> Result {
    panel.diffusion.apply_startup_preferences(&prefs);

    if export_state.completed_generation != panel.last_completed_generation {
        panel.last_completed_generation = export_state.completed_generation;
        if export_state.succeeded {
            if let Some(out_dir) = export_state.output_dir.as_deref() {
                trigger_reload(
                    out_dir,
                    params.bypass_change_detection(),
                    &active_config,
                    &active_library,
                    &mut reload_tx,
                );
            }
        }
    }

    // Register the heightfield texture with egui once it's available.
    if let Some(ref img) = gen_image {
        let needs_refresh = panel.preview_handle.as_ref() != Some(&img.heightfield);
        if needs_refresh {
            panel.preview_handle = Some(img.heightfield.clone());
            panel.preview_id = Some(contexts.add_image(bevy_egui::EguiTextureHandle::Strong(
                img.heightfield.clone(),
            )));
        }
    }

    // Register the water depth texture with egui once erosion buffers are available.
    if let Some(ref eb) = erosion_buffers {
        let needs_refresh = panel.water_preview_handle.as_ref() != Some(&eb.water);
        if needs_refresh {
            panel.water_preview_handle = Some(eb.water.clone());
            panel.water_preview_id =
                Some(contexts.add_image(bevy_egui::EguiTextureHandle::Strong(eb.water.clone())));
        }
    }

    poll_diffusion_state(
        &mut panel.diffusion,
        &active_config,
        &active_library,
        &mut reload_tx,
    );

    if !panel.open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;

    // Use bypass_change_detection so that merely rendering the UI doesn't mark
    // GeneratorParams as changed. We call params.set_changed() below only if a
    // widget actually reports a modification.
    let p = params.bypass_change_detection();
    let mut params_changed = false;

    let mut open = panel.open;
    egui::Window::new("Terrain Generator")
        .resizable(true)
        .default_size([860.0, 700.0])
        .open(&mut open)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut panel.active_tab, GeneratorTab::Legacy, "Legacy");
                ui.selectable_value(
                    &mut panel.active_tab,
                    GeneratorTab::Diffusion,
                    "Diffusion",
                );
            });
            ui.separator();

            match panel.active_tab {
                GeneratorTab::Legacy => {
                    ui.horizontal_top(|ui| {
                    // ── Left column: all parameter sections ──────────────────────
                    ui.vertical(|ui| {
                        ui.set_min_width(420.0);
                        ui.set_max_width(420.0);
                        egui::ScrollArea::vertical()
                            .id_salt("params_scroll")
                            .show(ui, |ui| {
                        ui.heading("Noise Parameters");
                        ui.separator();

                        egui::Grid::new("gen_params")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Resolution");
                                let resolutions = [512u32, 1024, 2048, 4096, 8192, 16384];
                                let mut res_idx = resolutions
                                    .iter()
                                    .position(|&r| r == p.resolution)
                                    .unwrap_or(1);
                                egui::ComboBox::from_id_salt("gen_res")
                                    .selected_text(format!("{}×{}", p.resolution, p.resolution))
                                    .show_ui(ui, |ui| {
                                        for (i, &r) in resolutions.iter().enumerate() {
                                            if ui
                                                .selectable_label(res_idx == i, format!("{}×{}", r, r))
                                                .clicked()
                                            {
                                                res_idx = i;
                                                p.resolution = r;
                                                params_changed = true;
                                            }
                                        }
                                    });
                                ui.end_row();

                                ui.label("Seed");
                                let mut seed_i = p.seed as i32;
                                if ui.add(egui::DragValue::new(&mut seed_i).speed(1)).changed() {
                                    p.seed = seed_i.max(0) as u32;
                                    params_changed = true;
                                }
                                ui.end_row();

                                ui.label("Octaves");
                                let mut oct = p.octaves as i32;
                                if ui.add(egui::Slider::new(&mut oct, 1..=10)).changed() {
                                    p.octaves = oct as u32;
                                    params_changed = true;
                                }
                                ui.end_row();

                                ui.label("Frequency");
                                params_changed |= ui.add(
                                    egui::Slider::new(&mut p.frequency, 0.1..=16.0).logarithmic(true),
                                ).changed();
                                ui.end_row();

                                ui.label("Lacunarity");
                                params_changed |= ui.add(egui::Slider::new(&mut p.lacunarity, 1.1..=4.0)).changed();
                                ui.end_row();

                                ui.label("Gain");
                                params_changed |= ui.add(egui::Slider::new(&mut p.gain, 0.1..=0.9)).changed();
                                ui.end_row();

                                ui.label("Offset X");
                                params_changed |= ui.add(egui::DragValue::new(&mut p.offset.x).speed(0.01)).changed();
                                ui.end_row();

                                ui.label("Offset Z");
                                params_changed |= ui.add(egui::DragValue::new(&mut p.offset.y).speed(0.01)).changed();
                                ui.end_row();
                            });

                        ui.add_space(6.0);
                        ui.heading("Landform Shaping");
                        ui.separator();

                        egui::Grid::new("gen_landform")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Continent Frequency");
                                params_changed |= ui.add(
                                    egui::Slider::new(&mut p.continent_frequency, 0.1..=4.0)
                                        .logarithmic(true),
                                ).changed();
                                ui.end_row();

                                ui.label("Continent Strength");
                                params_changed |= ui.add(egui::Slider::new(&mut p.continent_strength, 0.0..=1.0)).changed();
                                ui.end_row();

                                ui.label("Ridge Strength");
                                params_changed |= ui.add(egui::Slider::new(&mut p.ridge_strength, 0.0..=1.0)).changed();
                                ui.end_row();

                                ui.label("Warp Frequency");
                                params_changed |= ui.add(
                                    egui::Slider::new(&mut p.warp_frequency, 0.1..=8.0)
                                        .logarithmic(true),
                                ).changed();
                                ui.end_row();

                                ui.label("Warp Strength");
                                params_changed |= ui.add(egui::Slider::new(&mut p.warp_strength, 0.0..=2.0)).changed();
                                ui.end_row();

                                ui.label("Noise Erosion (legacy)");
                                params_changed |= ui.add(egui::Slider::new(&mut p.erosion_strength, 0.0..=1.0)).changed();
                                ui.end_row();
                            });

                        ui.add_space(6.0);
                        ui.heading("World Scale");
                        ui.separator();

                        egui::Grid::new("gen_world")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Height Scale (base m)");
                                params_changed |= ui.add(
                                    egui::DragValue::new(&mut p.height_scale)
                                        .speed(16.0)
                                        .range(64.0..=8192.0),
                                )
                                .on_hover_text("Base world-space Y range before world_scale is applied.")
                                .changed();
                                ui.end_row();

                                ui.label("World Scale (m/px)");
                                params_changed |= ui.add(
                                    egui::DragValue::new(&mut p.world_scale)
                                        .speed(0.1)
                                        .range(0.25..=32.0),
                                )
                                .on_hover_text(
                                    "Uniform X/Y/Z scale. Final terrain height span is height_scale × world_scale.",
                                )
                                .changed();
                                ui.end_row();

                                ui.label("Final Height Span");
                                ui.label(format!("{:.1} m", p.height_scale * p.world_scale));
                                ui.end_row();

                                ui.label("Baked Rings");
                                ui.label(format!("{}", baked_clipmap_levels(p.resolution)));
                                ui.end_row();

                                ui.label("Water Level")
                                    .on_hover_text("Normalised sea level [0–1]. The preview shows water with depth-based transparency below this height. 0 = no water.");
                                params_changed |= ui.add(
                                    egui::Slider::new(&mut p.water_level, 0.0..=1.0)
                                        .step_by(0.01),
                                ).changed();
                                ui.end_row();

                                ui.label("Export Smooth σ")
                                    .on_hover_text("Gaussian pre-smooth sigma (texels) applied to L0 before downsampling. 0 = off. ~1.0 removes spike outliers. Matches the import wizard's Smooth σ.");
                                if ui.add(egui::DragValue::new(&mut p.smooth_sigma).speed(0.05).range(0.0..=10.0)).changed() {
                                    params_changed = true;
                                }
                                ui.end_row();
                            });

                        // Erosion
                        ui.add_space(6.0);
                        ui.heading("Hydraulic Erosion");
                        ui.separator();

                        let ticks_done = erosion_ctrl.ticks_done();
                        let running = erosion_ctrl.is_dirty() && erosion.enabled;
                        let done    = !erosion_ctrl.is_dirty() && erosion.enabled;

                        ui.horizontal(|ui| {
                            if running {
                                ui.spinner();
                                ui.label(format!("Running: {}/{}", ticks_done, erosion.iterations));
                                if ui.button("Stop").clicked() {
                                    erosion.enabled = false;
                                    erosion_ctrl.dirty.store(
                                        false,
                                        std::sync::atomic::Ordering::Release,
                                    );
                                }
                            } else if done {
                                ui.label(format!("Done ({} iters)", erosion.iterations));
                                if ui.button("Re-run").clicked() {
                                    erosion_ctrl.mark_dirty();
                                }
                                if ui.button("Clear").clicked() {
                                    erosion.enabled = false;
                                    params_changed = true; // re-run base noise to restore pre-erosion terrain
                                }
                            } else if ui.button("Run Erosion").clicked() {
                                erosion.enabled = true;
                                erosion_ctrl.mark_dirty();
                            }
                        });

                        ui.add_space(4.0);
                        egui::Grid::new("erosion_main")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Iterations");
                                let mut iter = erosion.iterations as i32;
                                if ui.add(egui::Slider::new(&mut iter, 10..=2000)).changed() {
                                    erosion.iterations = iter as u32;
                                }
                                ui.end_row();

                                ui.label("dt");
                                ui.add(egui::Slider::new(&mut erosion.dt, 0.001..=0.1));
                                ui.end_row();

                                ui.label("Pipe Area (A)");
                                ui.add(egui::Slider::new(&mut erosion.pipe_area, 0.1..=50.0).logarithmic(true));
                                ui.end_row();

                                ui.label("Gravity");
                                ui.add(egui::Slider::new(&mut erosion.gravity, 0.1..=20.0));
                                ui.end_row();

                                ui.label("Rain Rate");
                                ui.add(egui::Slider::new(&mut erosion.rain_rate, 0.001..=0.1));
                                ui.end_row();

                                ui.label("Evaporation");
                                ui.add(egui::Slider::new(&mut erosion.evaporation_rate, 0.001..=0.1));
                                ui.end_row();

                                ui.label("Sediment Capacity");
                                ui.add(egui::Slider::new(&mut erosion.sediment_capacity, 0.1..=10.0));
                                ui.end_row();

                                ui.label("Erosion Rate");
                                ui.add(egui::Slider::new(&mut erosion.erosion_rate, 0.01..=5.0));
                                ui.end_row();

                                ui.label("Deposition Rate");
                                ui.add(egui::Slider::new(&mut erosion.deposition_rate, 0.01..=5.0));
                                ui.end_row();

                                ui.label("Erosion Depth Max");
                                ui.add(egui::Slider::new(&mut erosion.erosion_depth_max, 0.001..=1.0).logarithmic(true));
                                ui.end_row();

                                ui.label("Min Slope");
                                ui.add(egui::Slider::new(&mut erosion.min_slope, 0.0001..=0.1).logarithmic(true));
                                ui.end_row();

                                ui.label("Hardness Influence");
                                ui.add(egui::Slider::new(&mut erosion.hardness_influence, 0.0..=1.0));
                                ui.end_row();
                            });

                        ui.add_space(4.0);
                        ui.label("Thermal Erosion");
                        egui::Grid::new("erosion_thermal")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Enable Thermal");
                                ui.checkbox(&mut erosion.thermal_enabled, "");
                                ui.end_row();

                                ui.label("Repose Angle (deg)");
                                ui.add(egui::Slider::new(&mut erosion.repose_angle, 5.0..=60.0));
                                ui.end_row();

                                ui.label("Talus Rate");
                                ui.add(egui::Slider::new(&mut erosion.talus_rate, 0.01..=1.0));
                                ui.end_row();

                                ui.label("Thermal Iters/Tick");
                                let mut ti = erosion.thermal_iterations as i32;
                                if ui.add(egui::Slider::new(&mut ti, 1..=20)).changed() {
                                    erosion.thermal_iterations = ti as u32;
                                }
                                ui.end_row();
                            });

                        ui.add_space(4.0);
                        ui.label("Particle Erosion");
                        egui::Grid::new("erosion_particle")
                            .num_columns(2)
                            .spacing([8.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Enable Particles");
                                ui.checkbox(&mut erosion.particle_enabled, "");
                                ui.end_row();

                                ui.label("Num Particles");
                                let mut np = erosion.num_particles as i32;
                                if ui.add(egui::Slider::new(&mut np, 1000..=2_000_000).logarithmic(true)).changed() {
                                    erosion.num_particles = np as u32;
                                }
                                ui.end_row();

                                ui.label("Max Steps");
                                let mut ms = erosion.particle_max_steps as i32;
                                if ui.add(egui::Slider::new(&mut ms, 8..=256)).changed() {
                                    erosion.particle_max_steps = ms as u32;
                                }
                                ui.end_row();

                                ui.label("Inertia");
                                ui.add(egui::Slider::new(&mut erosion.particle_inertia, 0.0..=0.99));
                                ui.end_row();
                            });
                            }); // end ScrollArea
                    }); // end left vertical

                    ui.separator();

                    // ── Right column: preview + export ───────────────────────────
                    ui.vertical(|ui| {
                        ui.heading("Preview");
                        ui.separator();

                    // Mode selector — swaps which texture is shown; does NOT
                    // touch params_changed, so no shader re-dispatch fires.
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut panel.preview_mode, PreviewMode::Height, "Height");
                        ui.selectable_value(&mut panel.preview_mode, PreviewMode::Water,  "Water depth");
                    });
                    ui.add_space(2.0);

                    match panel.preview_mode {
                        PreviewMode::Height => {
                            if let Some(tex_id) = panel.preview_id {
                                let mut grayscale = p.grayscale != 0;
                                if ui.toggle_value(&mut grayscale, "Grayscale").on_hover_text(
                                    "Switch between pure heightmap (grayscale) and colour hillshade.",
                                ).changed() {
                                    p.grayscale = grayscale as u32;
                                    params_changed = true;
                                }
                                let avail = ui.available_width();
                                ui.add(egui::Image::new((tex_id, egui::Vec2::splat(avail))));
                                if p.grayscale != 0 {
                                    ui.small("Raw heightmap — linear [0, 1]");
                                } else {
                                    ui.small("Contrast-enhanced hillshade preview");
                                }
                            } else {
                                ui.label("(waiting for GPU texture…)");
                            }
                        }
                        PreviewMode::Water => {
                            if let Some(tex_id) = panel.water_preview_id {
                                let avail = ui.available_width();
                                ui.add(egui::Image::new((tex_id, egui::Vec2::splat(avail))));
                                let ticks = erosion_ctrl.ticks_done();
                                if erosion_ctrl.is_dirty() && erosion.enabled {
                                    ui.small(format!("Water depth (live) — tick {}/{}", ticks, erosion.iterations));
                                } else if erosion.enabled {
                                    ui.small(format!("Water depth — final ({} iterations)", erosion.iterations));
                                } else {
                                    ui.small("Water depth — run erosion to populate");
                                }
                            } else {
                                ui.label("(waiting for erosion buffers…)");
                            }
                        }
                    }

                    ui.add_space(8.0);
                    ui.heading("Export & Load");
                    ui.separator();

                    ui.label("Output directory:");
                    ui.text_edit_singleline(&mut panel.output_dir);

                    let exporting = export_state.active;
                    ui.add_space(4.0);
                    ui.add_enabled_ui(!exporting, |ui| {
                        if ui.button("Generate & Load").clicked() {
                            export_tx.write(StartGeneratorExport {
                                params: p.clone(),
                                output_dir: PathBuf::from(&panel.output_dir),
                            });
                        }
                    });

                    if exporting {
                        ui.spinner();
                    }

                    if !export_state.log.is_empty() {
                        ui.add_space(4.0);
                        egui::ScrollArea::vertical()
                            .id_salt("gen_log")
                            .max_height(120.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                for line in &export_state.log {
                                    ui.label(line);
                                }
                            });
                    }
                    }); // end right vertical
                    });
                }
                GeneratorTab::Diffusion => {
                    egui::ScrollArea::vertical()
                        .id_salt("diffusion_tab_scroll")
                        .show(ui, |ui| {
                            draw_diffusion_tab(ui, &mut panel.diffusion);
                        });
                }
            }
        });
    panel.open = open;

    if params_changed {
        params.set_changed();
    }

    Ok(())
}

fn trigger_reload(
    output_dir: &Path,
    params: &GeneratorParams,
    active_config: &TerrainConfig,
    active_library: &MaterialLibrary,
    reload_tx: &mut MessageWriter<ReloadTerrainRequest>,
) {
    let (world_min, world_max) = scan_world_bounds(output_dir, params.world_scale);
    let max_mip = scan_max_mip(output_dir.join("height").as_path());

    let mut new_config = active_config.clone();
    new_config.world_scale = params.world_scale;
    new_config.height_scale = params.height_scale * params.world_scale;
    new_config.clipmap_levels = derive_clipmap_levels(active_config.clipmap_levels, max_mip);

    let new_source = TerrainSourceDesc {
        tile_root: Some(output_dir.to_path_buf()),
        normal_root: None,
        macro_color_root: None,
        world_min,
        world_max,
        max_mip_level: max_mip,
        collision_mip_level: 2,
        ..Default::default()
    };

    reload_tx.write(ReloadTerrainRequest {
        config: new_config,
        source: new_source,
        material_library: active_library.clone(),
    });
}

fn scan_max_mip(height_dir: &Path) -> u8 {
    let mut max = 0u8;
    if let Ok(entries) = std::fs::read_dir(height_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(rest) = name.strip_prefix('L') {
                    if let Ok(n) = rest.parse::<u8>() {
                        max = max.max(n);
                    }
                }
            }
        }
    }
    max
}

fn scan_world_bounds(output_dir: &Path, world_scale: f32) -> (Vec2, Vec2) {
    const TILE_SIZE: u32 = 256;
    let fallback_h = 8192.0 * world_scale;
    let fallback = (Vec2::splat(-fallback_h), Vec2::splat(fallback_h));
    let l0 = output_dir.join("height").join("L0");
    let Ok(dir) = std::fs::read_dir(&l0) else {
        return fallback;
    };
    let mut min_tx = i32::MAX;
    let mut min_ty = i32::MAX;
    let mut max_tx = i32::MIN;
    let mut max_ty = i32::MIN;
    let mut found = false;
    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(stem) = name.to_str().and_then(|s| s.strip_suffix(".bin")) else {
            continue;
        };
        let Some((xs, ys)) = stem.split_once('_') else {
            continue;
        };
        let (Ok(tx), Ok(ty)) = (xs.parse::<i32>(), ys.parse::<i32>()) else {
            continue;
        };
        min_tx = min_tx.min(tx);
        min_ty = min_ty.min(ty);
        max_tx = max_tx.max(tx);
        max_ty = max_ty.max(ty);
        found = true;
    }
    if !found {
        return fallback;
    }
    let ts = TILE_SIZE as f32 * world_scale;
    (
        Vec2::new(min_tx as f32 * ts, min_ty as f32 * ts),
        Vec2::new((max_tx + 1) as f32 * ts, (max_ty + 1) as f32 * ts),
    )
}

fn derive_clipmap_levels(requested: u32, max_mip: u8) -> u32 {
    // Preserve the user's view distance, but ensure we expose at least the
    // baked hierarchy depth. Farther runtime rings reuse the coarsest baked mip.
    requested.max(max_mip as u32 + 1).max(1)
}

fn baked_clipmap_levels(resolution: u32) -> u32 {
    const TILE_SIZE: u32 = 256;
    let mut sz = resolution;
    let mut levels = 0u32;
    while sz / TILE_SIZE >= 2 {
        levels += 1;
        sz /= 2;
    }
    levels.max(1)
}

#[cfg(test)]
mod tests {
    use super::{baked_clipmap_levels, derive_clipmap_levels};

    #[test]
    fn generator_clipmap_levels_preserve_view_distance() {
        assert_eq!(derive_clipmap_levels(8, 3), 8);
        assert_eq!(derive_clipmap_levels(4, 3), 4);
        assert_eq!(derive_clipmap_levels(2, 5), 6);
    }

    #[test]
    fn export_resolution_reports_baked_ring_count() {
        assert_eq!(baked_clipmap_levels(512), 1);
        assert_eq!(baked_clipmap_levels(4096), 4);
        assert_eq!(baked_clipmap_levels(16384), 6);
    }
}

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, TextureId},
    EguiContexts, EguiPrimaryContextPass,
};
use bevy_landscape::{MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainSourceDesc};
use bevy_landscape_generator::{export::ExportHandle, GeneratorParams, HeightfieldImage};

#[derive(Resource)]
pub struct GeneratorPanelState {
    pub open: bool,
    preview_id: Option<TextureId>,
    preview_handle: Option<Handle<Image>>,
    output_dir: String,
    export: Option<ExportHandle>,
    log: Vec<String>,
}

impl Default for GeneratorPanelState {
    fn default() -> Self {
        Self {
            open: false,
            preview_id: None,
            preview_handle: None,
            output_dir: "assets/tiles_generated".into(),
            export: None,
            log: Vec::new(),
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
    gen_image: Option<Res<HeightfieldImage>>,
    active_config: Res<TerrainConfig>,
    active_library: Res<MaterialLibrary>,
    mut reload_tx: MessageWriter<ReloadTerrainRequest>,
) -> Result {
    // Drain export log messages; detect completion.
    {
        let mut new_msgs: Vec<String> = Vec::new();
        let mut finished = false;
        let mut succeeded = false;
        let mut out_dir = PathBuf::new();

        if let Some(ref handle) = panel.export {
            while let Ok(msg) = handle.log_rx.lock().unwrap().try_recv() {
                new_msgs.push(msg);
            }
            finished = handle.done.load(std::sync::atomic::Ordering::Acquire);
            succeeded = handle.succeeded.load(std::sync::atomic::Ordering::Acquire);
            out_dir = handle.output_dir.clone();
        }

        panel.log.extend(new_msgs);

        if finished {
            if succeeded {
                panel
                    .log
                    .push("Reloading terrain from generated tiles…".into());
                trigger_reload(
                    &out_dir,
                    &params,
                    &active_config,
                    &active_library,
                    &mut reload_tx,
                );
            }
            panel.export = None;
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

    if !panel.open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Terrain Generator")
        .resizable(true)
        .default_size([520.0, 700.0])
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Noise Parameters");
                ui.separator();

                egui::Grid::new("gen_params")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Resolution");
                        let resolutions = [512u32, 1024, 2048];
                        let mut res_idx = resolutions
                            .iter()
                            .position(|&r| r == params.resolution)
                            .unwrap_or(1);
                        egui::ComboBox::from_id_salt("gen_res")
                            .selected_text(format!("{}×{}", params.resolution, params.resolution))
                            .show_ui(ui, |ui| {
                                for (i, &r) in resolutions.iter().enumerate() {
                                    if ui
                                        .selectable_label(res_idx == i, format!("{}×{}", r, r))
                                        .clicked()
                                    {
                                        res_idx = i;
                                        params.resolution = r;
                                    }
                                }
                            });
                        ui.end_row();

                        ui.label("Seed");
                        let mut seed_i = params.seed as i32;
                        if ui.add(egui::DragValue::new(&mut seed_i).speed(1)).changed() {
                            params.seed = seed_i.max(0) as u32;
                        }
                        ui.end_row();

                        ui.label("Octaves");
                        let mut oct = params.octaves as i32;
                        if ui.add(egui::Slider::new(&mut oct, 1..=10)).changed() {
                            params.octaves = oct as u32;
                        }
                        ui.end_row();

                        ui.label("Frequency");
                        ui.add(
                            egui::Slider::new(&mut params.frequency, 0.1..=16.0).logarithmic(true),
                        );
                        ui.end_row();

                        ui.label("Lacunarity");
                        ui.add(egui::Slider::new(&mut params.lacunarity, 1.1..=4.0));
                        ui.end_row();

                        ui.label("Gain");
                        ui.add(egui::Slider::new(&mut params.gain, 0.1..=0.9));
                        ui.end_row();

                        ui.label("Offset X");
                        ui.add(egui::DragValue::new(&mut params.offset.x).speed(0.01));
                        ui.end_row();

                        ui.label("Offset Z");
                        ui.add(egui::DragValue::new(&mut params.offset.y).speed(0.01));
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
                        ui.add(
                            egui::Slider::new(&mut params.continent_frequency, 0.1..=4.0)
                                .logarithmic(true),
                        );
                        ui.end_row();

                        ui.label("Continent Strength");
                        ui.add(egui::Slider::new(&mut params.continent_strength, 0.0..=1.0));
                        ui.end_row();

                        ui.label("Ridge Strength");
                        ui.add(egui::Slider::new(&mut params.ridge_strength, 0.0..=1.0));
                        ui.end_row();

                        ui.label("Warp Frequency");
                        ui.add(
                            egui::Slider::new(&mut params.warp_frequency, 0.1..=8.0)
                                .logarithmic(true),
                        );
                        ui.end_row();

                        ui.label("Warp Strength");
                        ui.add(egui::Slider::new(&mut params.warp_strength, 0.0..=2.0));
                        ui.end_row();

                        ui.label("Erosion Strength");
                        ui.add(egui::Slider::new(&mut params.erosion_strength, 0.0..=1.0));
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
                        ui.add(
                            egui::DragValue::new(&mut params.height_scale)
                                .speed(16.0)
                                .range(64.0..=8192.0),
                        )
                        .on_hover_text(
                            "Base world-space Y range before world_scale is applied."
                        );
                        ui.end_row();

                        ui.label("World Scale (m/px)");
                        ui.add(
                            egui::DragValue::new(&mut params.world_scale)
                                .speed(0.1)
                                .range(0.25..=32.0),
                        )
                        .on_hover_text(
                            "Uniform X/Y/Z scale. Final terrain height span is height_scale × world_scale."
                        );
                        ui.end_row();

                        ui.label("Final Height Span");
                        ui.label(format!(
                            "{:.1} m",
                            params.height_scale * params.world_scale
                        ));
                        ui.end_row();

                        ui.label("Export Resolution");
                        let export_res_options: &[(u32, &str)] = &[
                            (512, "512"),
                            (1024, "1k"),
                            (2048, "2k"),
                            (4096, "4k"),
                            (8192, "8k"),
                            (16384, "16k"),
                            (32768, "32k"),
                        ];
                        let selected_label = export_res_options
                            .iter()
                            .find(|&&(r, _)| r == params.export_resolution)
                            .map(|&(_, lbl)| lbl)
                            .unwrap_or("?");
                        egui::ComboBox::from_id_salt("gen_export_res")
                            .selected_text(selected_label)
                            .show_ui(ui, |ui| {
                                for &(r, lbl) in export_res_options {
                                    if ui
                                        .selectable_label(params.export_resolution == r, lbl)
                                        .clicked()
                                    {
                                        params.export_resolution = r;
                                    }
                                }
                            });
                        ui.end_row();

                        ui.label("Baked Rings");
                        let baked_levels = baked_clipmap_levels(params.export_resolution);
                        ui.label(format!("{baked_levels}"));
                        ui.end_row();
                    });

                let baked_levels = baked_clipmap_levels(params.export_resolution);
                ui.label(format!(
                    "Current export resolution bakes {baked_levels} mip levels before the runtime starts reusing the coarsest level."
                ));

                // Preview
                ui.add_space(8.0);
                ui.heading("Preview");
                ui.separator();

                if let Some(tex_id) = panel.preview_id {
                    let avail = ui.available_width().min(480.0);
                    ui.add(egui::Image::new((tex_id, egui::Vec2::splat(avail))));
                } else {
                    ui.label("(waiting for GPU texture…)");
                }

                // Export
                ui.add_space(8.0);
                ui.heading("Export & Load");
                ui.separator();

                ui.label("Output directory:");
                ui.text_edit_singleline(&mut panel.output_dir);

                let exporting = panel.export.is_some();
                ui.add_space(4.0);
                ui.add_enabled_ui(!exporting, |ui| {
                    if ui.button("Generate & Load").clicked() {
                        let p = params.clone();
                        let dir = PathBuf::from(&panel.output_dir);
                        panel.log.clear();
                        panel
                            .log
                            .push(format!("Export started → {}", dir.display()));
                        panel.export = Some(bevy_landscape_generator::export::start_export(p, dir));
                    }
                });

                if exporting {
                    ui.spinner();
                }

                if !panel.log.is_empty() {
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical()
                        .id_salt("gen_log")
                        .max_height(120.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in &panel.log {
                                ui.label(line);
                            }
                        });
                }
            });
        });

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

fn baked_clipmap_levels(export_resolution: u32) -> u32 {
    const TILE_SIZE: u32 = 256;

    let mut sz = export_resolution;
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

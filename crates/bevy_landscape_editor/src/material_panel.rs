//! Materials panel — editor UI that reads from and writes to
//! `bevy_landscape::MaterialLibrary`.
//!
//! The panel is a floating egui window opened from the toolbar's Tools menu
//! (see `MaterialPanelState::open`).  As the material system grows — texture
//! assignment, splatmap painting, procedural rule evaluation — the controls
//! here expand in lock-step; the UI always mirrors the current shape of
//! `MaterialLibrary`.

use bevy::{
    light::{
        AtmosphereEnvironmentMapLight, CascadeShadowConfig, CascadeShadowConfigBuilder,
        DirectionalLight, DirectionalLightShadowMap, EnvironmentMapLight,
        GeneratedEnvironmentMapLight, GlobalAmbientLight,
    },
    prelude::*,
};
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{MaterialLibrary, MaterialSlot};

use bevy_landscape::{PbrRebuildProgress, PbrTexturesDirty};

use crate::texture_browser::TextureBrowser;

/// Editor-local UI state for the materials panel.
///
/// Open/close is driven by the toolbar; `selected_slot` is a pure UI concern
/// so it lives here rather than in the engine-side `MaterialLibrary`.
#[derive(Resource, Default)]
pub struct MaterialPanelState {
    pub open: bool,
    pub selected_slot: Option<usize>,
}

pub struct MaterialPanelPlugin;

impl Plugin for MaterialPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaterialPanelState>()
            .add_systems(EguiPrimaryContextPass, material_panel_system);
    }
}

fn material_panel_system(
    mut contexts: EguiContexts,
    mut panel: ResMut<MaterialPanelState>,
    mut library: ResMut<MaterialLibrary>,
    mut browser: ResMut<TextureBrowser>,
    mut pbr_dirty: ResMut<PbrTexturesDirty>,
    rebuild_progress: Res<PbrRebuildProgress>,
    mut sky_lights: Query<
        (
            Option<&mut AtmosphereEnvironmentMapLight>,
            Option<&mut GeneratedEnvironmentMapLight>,
            Option<&mut EnvironmentMapLight>,
        ),
        With<Camera3d>,
    >,
    mut global_ambient: ResMut<GlobalAmbientLight>,
    mut sun_shadows: Query<(&mut DirectionalLight, &mut CascadeShadowConfig)>,
    mut directional_shadow_map: ResMut<DirectionalLightShadowMap>,
) -> Result {
    if !panel.open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;
    let mut open = panel.open;

    egui::Window::new("Materials")
        .open(&mut open)
        .default_width(360.0)
        .default_height(480.0)
        .resizable(true)
        .show(ctx, |ui| {
            draw_material_library(
                ui,
                &mut panel,
                &mut library,
                &mut browser,
                &mut pbr_dirty,
                &rebuild_progress,
                &mut sky_lights,
                global_ambient.as_mut(),
                &mut sun_shadows,
                directional_shadow_map.as_mut(),
            );
        });

    panel.open = open;
    Ok(())
}

fn draw_material_library(
    ui: &mut egui::Ui,
    panel: &mut MaterialPanelState,
    library: &mut MaterialLibrary,
    browser: &mut TextureBrowser,
    pbr_dirty: &mut PbrTexturesDirty,
    rebuild_progress: &PbrRebuildProgress,
    sky_lights: &mut Query<
        (
            Option<&mut AtmosphereEnvironmentMapLight>,
            Option<&mut GeneratedEnvironmentMapLight>,
            Option<&mut EnvironmentMapLight>,
        ),
        With<Camera3d>,
    >,
    global_ambient: &mut GlobalAmbientLight,
    sun_shadows: &mut Query<(&mut DirectionalLight, &mut CascadeShadowConfig)>,
    directional_shadow_map: &mut DirectionalLightShadowMap,
) {
    if rebuild_progress.active {
        let label = match (rebuild_progress.fraction * 3.0) as u32 {
            0 => "Loading albedo…",
            1 => "Loading normals…",
            2 => "Loading ORM…",
            _ => "Uploading…",
        };
        ui.horizontal(|ui| {
            ui.spinner();
            ui.add(
                egui::ProgressBar::new(rebuild_progress.fraction)
                    .text(label)
                    .desired_width(ui.available_width() - 24.0),
            );
        });
        ui.add_space(2.0);
    }

    ui.horizontal(|ui| {
        ui.label(format!(
            "Slots: {} / {}",
            library.slots.len(),
            library.max_slots
        ));
        let can_add = library.slots.len() < library.max_slots;
        if ui
            .add_enabled(can_add, egui::Button::new("+ Add Slot"))
            .clicked()
        {
            let name = format!("Slot {}", library.slots.len());
            library.slots.push(MaterialSlot::new(name));
            panel.selected_slot = Some(library.slots.len() - 1);
            pbr_dirty.0 = true;
        }
    });

    ui.horizontal(|ui| {
        let loaded = library.macro_color_loaded;
        ui.add_enabled(
            loaded,
            egui::Checkbox::new(
                &mut library.use_macro_color_override,
                "Override with macro color map",
            ),
        )
        .on_hover_text(if loaded {
            "When on, the baked diffuse EXR replaces the library-blended albedo."
        } else {
            "No macro color texture is loaded (see landscape.toml `diffuse_exr`)."
        });
    });

    draw_lighting_controls(
        ui,
        sky_lights,
        global_ambient,
        sun_shadows,
        directional_shadow_map,
    );

    ui.separator();

    // Slot list
    let mut to_delete: Option<usize> = None;
    let mut to_duplicate: Option<usize> = None;

    egui::ScrollArea::vertical()
        .id_salt("material_slot_list")
        .max_height(160.0)
        .show(ui, |ui| {
            for (idx, slot) in library.slots.iter_mut().enumerate() {
                let selected = panel.selected_slot == Some(idx);
                ui.horizontal(|ui| {
                    ui.checkbox(&mut slot.visible, "");
                    let preview = texture_preview_button(ui, browser, slot, egui::vec2(28.0, 28.0));
                    if preview.clicked() {
                        panel.selected_slot = Some(idx);
                        browser.open_for(idx);
                    }
                    if ui.selectable_label(selected, &slot.name).clicked() {
                        panel.selected_slot = Some(idx);
                    }
                    if ui.small_button("⧉").on_hover_text("Duplicate").clicked() {
                        to_duplicate = Some(idx);
                    }
                    if ui.small_button("✕").on_hover_text("Delete").clicked() {
                        to_delete = Some(idx);
                    }
                });
            }
        });

    if let Some(idx) = to_duplicate {
        if library.slots.len() < library.max_slots {
            let mut copy = library.slots[idx].clone();
            copy.name = format!("{} Copy", copy.name);
            library.slots.insert(idx + 1, copy);
            panel.selected_slot = Some(idx + 1);
            pbr_dirty.0 = true;
        }
    }
    if let Some(idx) = to_delete {
        library.slots.remove(idx);
        panel.selected_slot = match panel.selected_slot {
            Some(sel) if sel == idx => None,
            Some(sel) if sel > idx => Some(sel - 1),
            other => other,
        };
        pbr_dirty.0 = true;
    }

    ui.separator();

    // Slot detail
    let selected = panel.selected_slot.and_then(|i| {
        if i < library.slots.len() {
            Some(i)
        } else {
            None
        }
    });
    match selected {
        None => {
            ui.label("Select a slot to edit its properties.");
        }
        Some(idx) => {
            draw_slot_detail(ui, idx, &mut library.slots[idx], browser, pbr_dirty);
        }
    }
}

fn draw_lighting_controls(
    ui: &mut egui::Ui,
    sky_lights: &mut Query<
        (
            Option<&mut AtmosphereEnvironmentMapLight>,
            Option<&mut GeneratedEnvironmentMapLight>,
            Option<&mut EnvironmentMapLight>,
        ),
        With<Camera3d>,
    >,
    global_ambient: &mut GlobalAmbientLight,
    sun_shadows: &mut Query<(&mut DirectionalLight, &mut CascadeShadowConfig)>,
    directional_shadow_map: &mut DirectionalLightShadowMap,
) {
    ui.collapsing("Lighting", |ui| {
        let mut sky_intensity = current_sky_ambient_intensity(sky_lights);
        if let Some(intensity) = sky_intensity.as_mut() {
            if ui
                .add(
                    egui::Slider::new(intensity, 0.0..=6.0)
                        .text("Sky ambient")
                        .step_by(0.05),
                )
                .on_hover_text(
                    "Scales atmosphere image-based lighting for terrain and PBR materials.",
                )
                .changed()
            {
                set_sky_ambient_intensity(sky_lights, *intensity);
            }
        } else {
            let mut disabled = 0.0;
            ui.add_enabled(
                false,
                egui::Slider::new(&mut disabled, 0.0..=6.0).text("Sky ambient"),
            )
            .on_hover_text("No camera with AtmosphereEnvironmentMapLight is active.");
        }

        let mut flat_brightness = global_ambient.brightness;
        if ui
            .add(
                egui::Slider::new(&mut flat_brightness, 0.0..=3000.0)
                    .text("Flat fill")
                    .suffix(" cd/m2")
                    .step_by(10.0),
            )
            .on_hover_text("Adds directionless ambient fill. Keep low to preserve terrain relief.")
            .changed()
        {
            global_ambient.brightness = flat_brightness.max(0.0);
        }

        ui.separator();
        draw_cascade_shadow_controls(ui, sun_shadows, directional_shadow_map);
    });
}

fn draw_cascade_shadow_controls(
    ui: &mut egui::Ui,
    sun_shadows: &mut Query<(&mut DirectionalLight, &mut CascadeShadowConfig)>,
    directional_shadow_map: &mut DirectionalLightShadowMap,
) {
    ui.label("Cascaded shadows");

    let mut iter = sun_shadows.iter_mut();
    let Some((mut light, mut config)) = iter.next() else {
        ui.label(
            egui::RichText::new("No directional light with cascade shadows is active.").small(),
        );
        return;
    };

    ui.checkbox(&mut light.shadows_enabled, "Enabled")
        .on_hover_text("Toggles shadow casting on the directional sun light.");

    let mut cascade_count = config.bounds.len().max(1);
    let mut minimum_distance = config.minimum_distance.max(0.0);
    let mut first_bound = config.bounds.first().copied().unwrap_or(400.0).max(1.0);
    let mut maximum_distance = config
        .bounds
        .last()
        .copied()
        .unwrap_or(5_000.0)
        .max(first_bound + 1.0);
    let mut overlap = config.overlap_proportion.clamp(0.0, 0.95);

    let mut changed = false;
    changed |= ui
        .add(
            egui::Slider::new(&mut maximum_distance, 500.0..=40_000.0)
                .text("Range")
                .suffix(" m")
                .step_by(100.0),
        )
        .on_hover_text("Maximum camera distance that can receive directional shadows.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut first_bound, 25.0..=2_000.0)
                .text("First split")
                .suffix(" m")
                .step_by(25.0),
        )
        .on_hover_text("Far bound of the highest-detail cascade near the camera.")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut cascade_count, 1..=6).text("Cascades"))
        .on_hover_text(
            "More cascades preserve quality across longer shadow ranges, at a render cost.",
        )
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut overlap, 0.0..=0.6).text("Blend overlap"))
        .on_hover_text("Softens transitions between cascades.")
        .changed();

    ui.horizontal(|ui| {
        ui.label("Map size");
        for size in [1024, 2048, 4096, 8192] {
            ui.selectable_value(&mut directional_shadow_map.size, size, size.to_string())
                .on_hover_text("Resolution of each directional shadow cascade.");
        }
    });

    ui.horizontal(|ui| {
        ui.label("Depth bias");
        ui.add(
            egui::DragValue::new(&mut light.shadow_depth_bias)
                .speed(0.001)
                .range(0.0..=0.2),
        )
        .on_hover_text("Raise to reduce acne; lower if shadows detach from casters.");
    });
    ui.horizontal(|ui| {
        ui.label("Normal bias");
        ui.add(
            egui::DragValue::new(&mut light.shadow_normal_bias)
                .speed(0.05)
                .range(0.0..=8.0),
        )
        .on_hover_text("Slope-scaled bias. Raise to reduce self-shadowing on terrain slopes.");
    });

    if changed {
        first_bound = first_bound.max(minimum_distance + 1.0);
        maximum_distance = maximum_distance.max(first_bound + 1.0);
        minimum_distance = minimum_distance.min(first_bound - 0.001);
        *config = CascadeShadowConfigBuilder {
            num_cascades: cascade_count,
            minimum_distance,
            maximum_distance,
            first_cascade_far_bound: first_bound,
            overlap_proportion: overlap.clamp(0.0, 0.95),
        }
        .build();
    }
}

fn current_sky_ambient_intensity(
    sky_lights: &mut Query<
        (
            Option<&mut AtmosphereEnvironmentMapLight>,
            Option<&mut GeneratedEnvironmentMapLight>,
            Option<&mut EnvironmentMapLight>,
        ),
        With<Camera3d>,
    >,
) -> Option<f32> {
    for (atmosphere, generated, environment) in sky_lights.iter_mut() {
        let intensity = environment
            .as_ref()
            .map(|light| light.intensity)
            .or_else(|| generated.as_ref().map(|light| light.intensity))
            .or_else(|| atmosphere.as_ref().map(|light| light.intensity));
        if intensity.is_some() {
            return intensity;
        }
    }
    None
}

fn set_sky_ambient_intensity(
    sky_lights: &mut Query<
        (
            Option<&mut AtmosphereEnvironmentMapLight>,
            Option<&mut GeneratedEnvironmentMapLight>,
            Option<&mut EnvironmentMapLight>,
        ),
        With<Camera3d>,
    >,
    intensity: f32,
) {
    let intensity = intensity.max(0.0);
    for (atmosphere, generated, environment) in sky_lights.iter_mut() {
        if let Some(mut light) = atmosphere {
            light.intensity = intensity;
        }
        if let Some(mut light) = generated {
            light.intensity = intensity;
        }
        if let Some(mut light) = environment {
            light.intensity = intensity;
        }
    }
}

fn draw_slot_detail(
    ui: &mut egui::Ui,
    slot_idx: usize,
    slot: &mut MaterialSlot,
    browser: &mut TextureBrowser,
    pbr_dirty: &mut PbrTexturesDirty,
) {
    egui::ScrollArea::vertical()
        .id_salt("material_slot_detail")
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut slot.name);
            });

            ui.horizontal(|ui| {
                let preview = texture_preview_button(ui, browser, slot, egui::vec2(84.0, 84.0))
                    .on_hover_text(
                    "Diffuse preview tinted by the slot colour. Click to open the texture atlas.",
                );
                if preview.clicked() {
                    browser.open_for(slot_idx);
                }
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Tint");
                        ui.color_edit_button_rgb(&mut slot.tint);
                    });
                    ui.checkbox(&mut slot.visible, "Visible");
                    if let Some(path) = slot.albedo_path.as_ref() {
                        ui.label(
                            egui::RichText::new(path.display().to_string())
                                .small()
                                .monospace()
                                .color(egui::Color32::GRAY),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("No diffuse texture assigned.")
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    }
                    if ui.button("Choose From Texture Atlas…").clicked() {
                        browser.open_for(slot_idx);
                    }
                });
            });

            ui.collapsing("Textures", |ui| {
                if ui
                    .add(
                        egui::Button::new("Open Texture Atlas…")
                            .min_size(egui::vec2(ui.available_width(), 0.0)),
                    )
                    .on_hover_text("Open the texture atlas to assign all PBR maps at once.")
                    .clicked()
                {
                    browser.open_for(slot_idx);
                }
                ui.add_space(2.0);
                ui.separator();
                let changed = texture_path_row(ui, "Albedo", &mut slot.albedo_path)
                    | texture_path_row(ui, "Normal", &mut slot.normal_path)
                    | texture_path_row(ui, "ORM", &mut slot.orm_path)
                    | texture_path_row(ui, "Height", &mut slot.height_path);
                if changed {
                    pbr_dirty.0 = true;
                }
            });

            ui.collapsing("Sampling", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Fine scale (m)");
                    ui.add(
                        egui::DragValue::new(&mut slot.fine_scale_m)
                            .speed(0.05)
                            .range(0.05..=32.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Coarse ×");
                    ui.add(
                        egui::DragValue::new(&mut slot.coarse_scale_mul)
                            .speed(0.1)
                            .range(1.0..=16.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Triplanar threshold °");
                    ui.add(
                        egui::DragValue::new(&mut slot.triplanar_threshold_deg)
                            .speed(0.5)
                            .range(0.0..=90.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Height-blend sharpness");
                    ui.add(
                        egui::DragValue::new(&mut slot.height_blend_sharpness)
                            .speed(0.005)
                            .range(0.001..=1.0),
                    );
                });
            });

            ui.collapsing("Procedural rules", |ui| {
                let rules = &mut slot.procedural;
                ui.label("Altitude range (m)");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut rules.altitude_range_m.x).speed(1.0));
                    ui.label("→");
                    ui.add(egui::DragValue::new(&mut rules.altitude_range_m.y).speed(1.0));
                });
                ui.label("Slope range (°)");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut rules.slope_range_deg.x)
                            .speed(0.5)
                            .range(0.0..=90.0),
                    );
                    ui.label("→");
                    ui.add(
                        egui::DragValue::new(&mut rules.slope_range_deg.y)
                            .speed(0.5)
                            .range(0.0..=90.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Curvature bias");
                    ui.add(
                        egui::Slider::new(&mut rules.curvature_bias, -1.0..=1.0)
                            .text("concave ↔ convex"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Noise scale (m)");
                    ui.add(
                        egui::DragValue::new(&mut rules.noise_scale_m)
                            .speed(0.5)
                            .range(0.1..=1000.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Noise strength");
                    ui.add(egui::Slider::new(&mut rules.noise_strength, 0.0..=1.0));
                });
            });
        });
}

/// Returns `true` when the path was committed (focus lost, Enter pressed, or cleared),
/// signalling that a PBR texture rebuild should be triggered.
fn texture_path_row(ui: &mut egui::Ui, label: &str, path: &mut Option<std::path::PathBuf>) -> bool {
    let mut rebuild = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let mut text = path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let resp = ui.add(
            egui::TextEdit::singleline(&mut text)
                .hint_text("(unset)")
                .desired_width(200.0),
        );
        if resp.changed() {
            *path = if text.trim().is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(text))
            };
        }
        // Trigger rebuild when the user leaves the field (lost_focus covers both
        // Tab/click-away and Enter).  We don't gate on changed() because that flag
        // is true only on the frame of the keystroke, not the frame focus is lost.
        if resp.lost_focus() {
            rebuild = true;
        }
        if ui.small_button("✕").on_hover_text("Clear").clicked() {
            *path = None;
            rebuild = true;
        }
    });
    rebuild
}

fn texture_preview_button(
    ui: &mut egui::Ui,
    browser: &mut TextureBrowser,
    slot: &MaterialSlot,
    size: egui::Vec2,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    if !ui.is_rect_visible(rect) {
        return response;
    }

    let tint = tint_color(slot.tint);
    let bg = if response.hovered() {
        egui::Color32::from_gray(56)
    } else {
        egui::Color32::from_gray(40)
    };
    ui.painter().rect_filled(rect, 6.0, bg);

    let image_rect = rect.shrink(2.0);
    if let Some(handle) = browser.preview_for(ui.ctx(), slot.albedo_path.as_deref()) {
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        ui.painter().image(handle.id(), image_rect, uv, tint);
    } else if slot.albedo_path.is_some() {
        ui.painter()
            .rect_filled(image_rect, 4.0, egui::Color32::from_gray(24));
        ui.painter().text(
            image_rect.center(),
            egui::Align2::CENTER_CENTER,
            "...",
            egui::FontId::proportional(13.0),
            egui::Color32::GRAY,
        );
    } else {
        ui.painter().rect_filled(image_rect, 4.0, tint);
        ui.painter().text(
            image_rect.center(),
            egui::Align2::CENTER_CENTER,
            "+",
            egui::FontId::proportional(18.0),
            egui::Color32::from_black_alpha(180),
        );
    }

    response.on_hover_text(if slot.albedo_path.is_some() {
        "Click to choose a different texture set from the atlas."
    } else {
        "Click to assign a diffuse texture from the atlas."
    })
}

fn tint_color(rgb: [f32; 3]) -> egui::Color32 {
    egui::Color32::from_rgb(
        ((rgb[0].clamp(0.0, 1.0) * 255.0) + 0.5) as u8,
        ((rgb[1].clamp(0.0, 1.0) * 255.0) + 0.5) as u8,
        ((rgb[2].clamp(0.0, 1.0) * 255.0) + 0.5) as u8,
    )
}

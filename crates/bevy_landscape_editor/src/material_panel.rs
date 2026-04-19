//! Materials panel — editor UI that reads from and writes to
//! `bevy_landscape::MaterialLibrary`.
//!
//! The panel is a floating egui window opened from the toolbar's Tools menu
//! (see `MaterialPanelState::open`).  As the material system grows — texture
//! assignment, splatmap painting, procedural rule evaluation — the controls
//! here expand in lock-step; the UI always mirrors the current shape of
//! `MaterialLibrary`.

use bevy::prelude::*;
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

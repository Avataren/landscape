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
            draw_material_library(ui, &mut panel, &mut library);
        });

    panel.open = open;
    Ok(())
}

fn draw_material_library(
    ui: &mut egui::Ui,
    panel: &mut MaterialPanelState,
    library: &mut MaterialLibrary,
) {
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
                    // Small tint swatch doubles as a colour picker.
                    ui.color_edit_button_rgb(&mut slot.tint);
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
        }
    }
    if let Some(idx) = to_delete {
        library.slots.remove(idx);
        panel.selected_slot = match panel.selected_slot {
            Some(sel) if sel == idx => None,
            Some(sel) if sel > idx => Some(sel - 1),
            other => other,
        };
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
            draw_slot_detail(ui, &mut library.slots[idx]);
        }
    }
}

fn draw_slot_detail(ui: &mut egui::Ui, slot: &mut MaterialSlot) {
    egui::ScrollArea::vertical()
        .id_salt("material_slot_detail")
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut slot.name);
            });

            ui.horizontal(|ui| {
                ui.label("Tint");
                ui.color_edit_button_rgb(&mut slot.tint);
                ui.checkbox(&mut slot.visible, "Visible");
            });

            ui.collapsing("Textures", |ui| {
                texture_path_row(ui, "Albedo", &mut slot.albedo_path);
                texture_path_row(ui, "Normal", &mut slot.normal_path);
                texture_path_row(ui, "ORM", &mut slot.orm_path);
                texture_path_row(ui, "Height", &mut slot.height_path);
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

fn texture_path_row(ui: &mut egui::Ui, label: &str, path: &mut Option<std::path::PathBuf>) {
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
        if ui.small_button("✕").on_hover_text("Clear").clicked() {
            *path = None;
        }
    });
}

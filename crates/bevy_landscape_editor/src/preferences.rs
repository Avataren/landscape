//! Editor preferences — persisted to `preferences.json` in the workspace root.
//!
//! Preferences are loaded once at startup (before the Bevy app is built) and
//! stored as a `Resource` so systems can read or mutate them.  The user edits
//! them through the **Preferences** dialog (File → Preferences…).

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{mpsc, Mutex};

const PREFS_FILE: &str = "preferences.json";

// ---------------------------------------------------------------------------
// Preferences data
// ---------------------------------------------------------------------------

/// Persisted editor preferences.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AppPreferences {
    /// Level JSON file loaded by default at startup when no `--level` argument
    /// is supplied.  `None` means fall back to `landscape.toml`.
    pub default_level: Option<String>,
}

impl AppPreferences {
    /// Read `preferences.json` from the current working directory.
    /// Returns `Default` when the file is absent or malformed.
    pub fn load() -> Self {
        std::fs::read_to_string(PREFS_FILE)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Write `preferences.json` to the current working directory.
    pub fn save(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(PREFS_FILE, json).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Dialog state
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct PreferencesDialog {
    pub open: bool,
    /// Working copy being edited — only written back to `AppPreferences` on Save.
    draft: AppPreferences,
    /// In-flight file picker for the default level field.
    pick_rx: Option<Mutex<mpsc::Receiver<Option<PathBuf>>>>,
    status: Option<String>,
}

impl PreferencesDialog {
    pub fn open(&mut self, current: &AppPreferences) {
        self.draft = current.clone();
        self.open = true;
        self.status = None;
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct PreferencesPlugin;

impl Plugin for PreferencesPlugin {
    fn build(&self, app: &mut App) {
        let prefs = AppPreferences::load();
        app.insert_resource(prefs)
            .init_resource::<PreferencesDialog>()
            .add_systems(EguiPrimaryContextPass, preferences_system);
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

fn preferences_system(
    mut contexts: EguiContexts,
    mut dialog: ResMut<PreferencesDialog>,
    mut prefs: ResMut<AppPreferences>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // Poll file picker.
    let picked = if let Some(rx) = &dialog.pick_rx {
        rx.lock().ok().and_then(|g| g.try_recv().ok())
    } else {
        None
    };
    if let Some(maybe_path) = picked {
        dialog.pick_rx = None;
        if let Some(p) = maybe_path {
            dialog.draft.default_level = Some(p.display().to_string());
        }
    }

    if !dialog.open {
        return Ok(());
    }

    let mut close = false;
    let mut pick = false;
    let mut save = false;
    let mut clear = false;

    // Work with a local copy of the level string so we can detect empty → None.
    let mut level_str = dialog.draft.default_level.clone().unwrap_or_default();

    egui::Window::new("Preferences")
        .default_width(460.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            egui::Grid::new("prefs_grid")
                .num_columns(3)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Default level file").on_hover_text(
                        "Level JSON loaded at startup. Leave empty to use landscape.toml.",
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut level_str)
                            .hint_text("none — uses landscape.toml")
                            .desired_width(260.0),
                    );
                    ui.horizontal(|ui| {
                        if ui.small_button("Browse…").clicked() {
                            pick = true;
                        }
                        if ui.small_button("✕").on_hover_text("Clear").clicked() {
                            clear = true;
                        }
                    });
                    ui.end_row();
                });

            if let Some(ref s) = dialog.status {
                let color = if s.starts_with("✓") {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::from_rgb(255, 180, 60)
                };
                ui.colored_label(color, s);
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    save = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });

    // Write edits back to draft.
    dialog.draft.default_level = if level_str.trim().is_empty() {
        None
    } else {
        Some(level_str)
    };

    if clear {
        dialog.draft.default_level = None;
    }

    if pick {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = rfd::FileDialog::new()
                .add_filter("Level JSON", &["json"])
                .set_title("Select default level file")
                .pick_file();
            let _ = tx.send(result);
        });
        dialog.pick_rx = Some(Mutex::new(rx));
    }

    if save {
        *prefs = dialog.draft.clone();
        match prefs.save() {
            Ok(()) => {
                dialog.status = Some("✓ Saved.".into());
            }
            Err(e) => {
                dialog.status = Some(format!("✗ {e}"));
            }
        }
    }

    if close {
        dialog.open = false;
    }

    Ok(())
}

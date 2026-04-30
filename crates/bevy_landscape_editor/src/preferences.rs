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
    /// Default terrain-diffusion checkout path for the diffusion generator tab.
    pub diffusion_repo_path: Option<String>,
    /// Python executable used to launch terrain-diffusion.
    pub diffusion_python: Option<String>,
    /// Optional extra environment variables for terrain-diffusion subprocesses.
    /// Format: `KEY=VALUE;KEY2=VALUE2` or newline-separated pairs.
    pub diffusion_env: Option<String>,
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
            let stored = std::env::current_dir()
                .ok()
                .and_then(|cwd| p.strip_prefix(&cwd).ok().map(|r| r.to_path_buf()))
                .unwrap_or(p);
            dialog.draft.default_level = Some(stored.display().to_string());
        }
    }

    if !dialog.open {
        return Ok(());
    }

    let mut close = false;
    let mut pick = false;
    let mut save = false;
    let mut clear = false;
    let mut clear_repo = false;
    let mut clear_python = false;
    let mut clear_env = false;

    // Work with a local copy of the level string so we can detect empty → None.
    let mut level_str = dialog.draft.default_level.clone().unwrap_or_default();
    let mut diffusion_repo_str = dialog.draft.diffusion_repo_path.clone().unwrap_or_default();
    let mut diffusion_python_str = dialog.draft.diffusion_python.clone().unwrap_or_default();
    let mut diffusion_env_str = dialog.draft.diffusion_env.clone().unwrap_or_default();

    egui::Window::new("Preferences")
        .default_width(460.0)
        .resizable(true)
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

                    ui.label("Diffusion repo").on_hover_text(
                        "Default terrain-diffusion checkout path for the diffusion generator tab.",
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut diffusion_repo_str)
                            .hint_text("/path/to/terrain-diffusion")
                            .desired_width(260.0),
                    );
                    if ui.small_button("✕").on_hover_text("Clear").clicked() {
                        clear_repo = true;
                    }
                    ui.end_row();

                    ui.label("Diffusion Python").on_hover_text(
                        "Interpreter path used to run terrain-diffusion. Point this at `.venv/bin/python` for a ROCm-enabled venv.",
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut diffusion_python_str)
                            .hint_text("/path/to/.venv/bin/python")
                            .desired_width(260.0),
                    );
                    if ui.small_button("✕").on_hover_text("Clear").clicked() {
                        clear_python = true;
                    }
                    ui.end_row();
                });

            ui.add_space(6.0);
            ui.label("Diffusion extra env").on_hover_text(
                "Optional env vars passed to terrain-diffusion. Use `KEY=VALUE;KEY2=VALUE2` or newline-separated pairs. Example: `HSA_OVERRIDE_GFX_VERSION=11.0.0`.",
            );
            ui.add(
                egui::TextEdit::multiline(&mut diffusion_env_str)
                    .desired_width(420.0)
                    .desired_rows(3)
                    .hint_text("HSA_OVERRIDE_GFX_VERSION=11.0.0"),
            );
            if ui.small_button("Clear diffusion env").clicked() {
                clear_env = true;
            }

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
    dialog.draft.diffusion_repo_path = if diffusion_repo_str.trim().is_empty() {
        None
    } else {
        Some(diffusion_repo_str)
    };
    dialog.draft.diffusion_python = if diffusion_python_str.trim().is_empty() {
        None
    } else {
        Some(diffusion_python_str)
    };
    dialog.draft.diffusion_env = if diffusion_env_str.trim().is_empty() {
        None
    } else {
        Some(diffusion_env_str)
    };

    if clear {
        dialog.draft.default_level = None;
    }
    if clear_repo {
        dialog.draft.diffusion_repo_path = None;
    }
    if clear_python {
        dialog.draft.diffusion_python = None;
    }
    if clear_env {
        dialog.draft.diffusion_env = None;
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

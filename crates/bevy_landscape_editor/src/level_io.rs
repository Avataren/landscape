//! File → Save Landscape… / Load Landscape… implementation.
//!
//! "Save Landscape…" serialises the current `TerrainSourceDesc`, `TerrainConfig`,
//! and `MaterialLibrary` into a `LevelDesc` JSON file chosen by the user.
//!
//! "Load Landscape…" opens a JSON file, parses it back, and sends a
//! `ReloadTerrainRequest` so the terrain hot-swaps without restarting.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{
    level::{save_level, LevelDesc},
    load_level, MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainMetadata,
    TerrainSourceDesc,
};
use bevy_landscape_clouds::CloudsConfig;
use bevy_landscape_generator::GeneratorParams;
use std::path::PathBuf;
use std::sync::{mpsc, Mutex};

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

enum LevelIoOp {
    Save,
    Load,
}

#[derive(Resource, Default)]
pub struct LevelIoState {
    /// In-flight file-dialog channel.
    pick_rx: Option<Mutex<mpsc::Receiver<Option<PathBuf>>>>,
    op: Option<LevelIoOp>,
    /// Feedback shown in the status toast.
    pub status: Option<(String, bool)>, // (message, is_error)
}

impl LevelIoState {
    pub fn start_save(&mut self) {
        if self.pick_rx.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = rfd::FileDialog::new()
                .add_filter("Landscape JSON", &["json"])
                .set_title("Save Landscape…")
                .set_file_name("level.json")
                .save_file();
            let _ = tx.send(result);
        });
        self.pick_rx = Some(Mutex::new(rx));
        self.op = Some(LevelIoOp::Save);
    }

    pub fn start_load(&mut self) {
        if self.pick_rx.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = rfd::FileDialog::new()
                .add_filter("Landscape JSON", &["json"])
                .set_title("Load Landscape…")
                .pick_file();
            let _ = tx.send(result);
        });
        self.pick_rx = Some(Mutex::new(rx));
        self.op = Some(LevelIoOp::Load);
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct LevelIoPlugin;

impl Plugin for LevelIoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LevelIoState>()
            .add_systems(EguiPrimaryContextPass, level_io_system);
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

fn level_io_system(
    mut contexts: EguiContexts,
    mut state: ResMut<LevelIoState>,
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    library: Res<MaterialLibrary>,
    mut clouds_config: ResMut<CloudsConfig>,
    mut reload_tx: MessageWriter<ReloadTerrainRequest>,
    generator_params: Option<Res<GeneratorParams>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // Poll the file-dialog channel.
    let picked = if let Some(rx) = &state.pick_rx {
        rx.lock().ok().and_then(|g| g.try_recv().ok())
    } else {
        None
    };

    if let Some(maybe_path) = picked {
        let op = state.op.take();
        state.pick_rx = None;

        if let Some(path) = maybe_path {
            match op {
                Some(LevelIoOp::Save) => {
                    let mut level_desc = LevelDesc::from_current(&config, &desc, &library);
                    level_desc.clouds = serde_json::to_value(&*clouds_config).ok();
                    if let Some(gp) = &generator_params {
                        level_desc.metadata = TerrainMetadata {
                            water_level: if gp.water_level > 0.0 {
                                Some(gp.water_level)
                            } else {
                                None
                            },
                        };
                    }
                    match save_level(&path, &level_desc) {
                        Ok(()) => {
                            state.status = Some((format!("✓ Saved → {}", path.display()), false));
                        }
                        Err(e) => {
                            state.status = Some((format!("✗ Save failed: {e}"), true));
                        }
                    }
                }
                Some(LevelIoOp::Load) => match load_level(&path) {
                    Ok(level_desc) => {
                        if let Some(cc) = level_desc
                            .clouds
                            .as_ref()
                            .and_then(|v| serde_json::from_value(v.clone()).ok())
                        {
                            *clouds_config = cc;
                        }
                        let (new_config, new_source, new_library, _, _, _meta) =
                            level_desc.into_runtime();
                        reload_tx.write(ReloadTerrainRequest {
                            config: new_config,
                            source: new_source,
                            material_library: new_library,
                        });
                        state.status = Some((format!("✓ Loaded → {}", path.display()), false));
                    }
                    Err(e) => {
                        state.status = Some((format!("✗ Load failed: {e}"), true));
                    }
                },
                None => {}
            }
        }
    }

    // Draw a brief status toast if we have one.
    if let Some((msg, is_error)) = state.status.clone() {
        let color = if is_error {
            egui::Color32::RED
        } else {
            egui::Color32::GREEN
        };
        let mut dismiss = false;
        egui::Window::new("##level_io_toast")
            .title_bar(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -30.0])
            .show(ctx, |ui| {
                ui.colored_label(color, &msg);
                if ui.small_button("✕").clicked() {
                    dismiss = true;
                }
            });
        if dismiss {
            state.status = None;
        }
    }

    Ok(())
}

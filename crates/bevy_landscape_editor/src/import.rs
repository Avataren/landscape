//! Heightmap import wizard.
//!
//! Flow:
//!   1. User clicks **File → Import Heightmap…** in the toolbar.
//!   2. A native file-picker opens on a background thread (avoids freezing Bevy).
//!   3. After a file is chosen the **Settings** dialog appears: paths, scale
//!      parameters, optional diffuse colour map and bump/normal map.
//!   4. Clicking **Import** launches the bake pipeline on a background thread.
//!      Log output streams into a scrollable egui panel in real-time.
//!   5. On completion the live terrain is hot-reloaded to display the new data.
//!      **Close** returns to normal editing.
//!
//! Clicking **Cancel** at any stage dismisses the wizard without side effects.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::bake::BakeConfig;
use bevy_landscape::{
    MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainSourceDesc,
    MAX_SUPPORTED_CLIPMAP_LEVELS,
};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Mutex};

// ---------------------------------------------------------------------------
// Messages streamed from the background bake thread
// ---------------------------------------------------------------------------

pub enum BakeMsg {
    Log(String),
    Done(Result<(), String>),
}

// ---------------------------------------------------------------------------
// Import state machine
// ---------------------------------------------------------------------------

/// Which field triggered a file-picker.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FilePickTarget {
    Height,
    Diffuse,
    Bump,
    Output,
}

/// Current step of the import wizard.
///
/// Both `Receiver` types are wrapped in `Mutex` so that `ImportWizard`
/// satisfies Bevy's `Resource: Send + Sync` bound.  We only ever call
/// `try_recv` from the main thread, so the mutex is uncontested.
pub enum ImportStep {
    Closed,
    /// A native file-picker is open on a background thread.
    PickingFile {
        rx: Mutex<mpsc::Receiver<Option<PathBuf>>>,
        target: FilePickTarget,
        settings: Box<ImportSettings>,
    },
    /// User is editing import settings.
    Settings(Box<ImportSettings>),
    /// Bake is running on a background thread.
    Baking {
        log_lines: Vec<String>,
        rx: Mutex<mpsc::Receiver<BakeMsg>>,
        finished: bool,
        error: Option<String>,
        /// Original settings — kept so we can send a ReloadTerrainRequest on success.
        settings: Box<ImportSettings>,
        /// Set once the terrain has been hot-reloaded after a successful bake.
        reloaded: bool,
    },
}

/// All user-editable fields for one import run.
#[derive(Clone)]
pub struct ImportSettings {
    pub height_path: String,
    pub diffuse_path: String,
    pub bump_path: String,
    pub output_dir: String,
    pub world_scale: f32,
    pub height_scale: f32,
    pub smooth_sigma: f32,
    pub flip_green: bool,
    /// Number of clipmap LOD rings to render (controls view distance).
    /// More rings = farther visibility but more GPU memory.
    pub clipmap_levels: u32,
}

impl Default for ImportSettings {
    fn default() -> Self {
        Self {
            height_path: String::new(),
            diffuse_path: String::new(),
            bump_path: String::new(),
            output_dir: "assets/tiles".into(),
            world_scale: 1.0,
            height_scale: 2048.0,
            smooth_sigma: 1.0,
            flip_green: false,
            clipmap_levels: bevy_landscape::TerrainConfig::default().clipmap_levels,
        }
    }
}

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct ImportWizard {
    pub step: ImportStep,
}

impl Default for ImportWizard {
    fn default() -> Self {
        Self {
            step: ImportStep::Closed,
        }
    }
}

impl ImportWizard {
    /// Entry point: open the file-picker for the heightmap.
    pub fn open(&mut self) {
        let (tx, rx) = mpsc::channel();
        spawn_file_picker(tx, FilePickTarget::Height);
        self.step = ImportStep::PickingFile {
            rx: Mutex::new(rx),
            target: FilePickTarget::Height,
            settings: Box::new(ImportSettings::default()),
        };
    }

    /// Open a picker for a specific settings field.
    pub fn pick_file(&mut self, target: FilePickTarget) {
        let settings = match &self.step {
            ImportStep::Settings(s) => s.clone(),
            _ => return,
        };
        let (tx, rx) = mpsc::channel();
        spawn_file_picker(tx, target);
        self.step = ImportStep::PickingFile {
            rx: Mutex::new(rx),
            target,
            settings,
        };
    }

    /// Launch the bake from the settings dialog.
    pub fn start_bake(&mut self) {
        let settings = match std::mem::replace(&mut self.step, ImportStep::Closed) {
            ImportStep::Settings(s) => *s,
            other => {
                self.step = other;
                return;
            }
        };

        let config = settings_to_config(&settings);
        let (tx, rx) = mpsc::channel::<BakeMsg>();
        let tx_log = tx.clone();
        std::thread::spawn(move || {
            let result = bevy_landscape::bake::bake_heightmap(config, move |msg| {
                let _ = tx_log.send(BakeMsg::Log(msg));
            });
            let _ = tx.send(BakeMsg::Done(result));
        });

        self.step = ImportStep::Baking {
            log_lines: Vec::new(),
            rx: Mutex::new(rx),
            finished: false,
            error: None,
            settings: Box::new(settings),
            reloaded: false,
        };
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn spawn_file_picker(tx: mpsc::Sender<Option<PathBuf>>, target: FilePickTarget) {
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new();
        dialog = match target {
            FilePickTarget::Height => dialog
                .add_filter("Heightmap", &["exr", "png", "tif", "tiff"])
                .set_title("Select heightmap"),
            FilePickTarget::Diffuse => dialog
                .add_filter("Image", &["exr", "png", "jpg", "jpeg", "tif", "tiff"])
                .set_title("Select diffuse colour map (optional)"),
            FilePickTarget::Bump => dialog
                .add_filter("Image", &["png", "jpg", "jpeg", "tif", "tiff"])
                .set_title("Select bump / normal map (optional)"),
            FilePickTarget::Output => dialog.set_title("Select output directory"),
        };
        let result = if target == FilePickTarget::Output {
            dialog.pick_folder()
        } else {
            dialog.pick_file()
        };
        let _ = tx.send(result);
    });
}

fn settings_to_config(s: &ImportSettings) -> BakeConfig {
    BakeConfig {
        height_path: PathBuf::from(&s.height_path),
        bump_path: if s.bump_path.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(&s.bump_path))
        },
        output_dir: PathBuf::from(&s.output_dir),
        height_scale: s.height_scale,
        bump_scale: None,
        world_scale: s.world_scale,
        tile_size: 256,
        flip_green: s.flip_green,
        smooth_sigma: s.smooth_sigma,
    }
}

fn derive_import_clipmap_levels(requested: u32, max_mip: u8) -> u32 {
    requested
        .max(max_mip as u32 + 1)
        .min(MAX_SUPPORTED_CLIPMAP_LEVELS as u32)
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct ImportPlugin;

impl Plugin for ImportPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImportWizard>()
            .add_systems(EguiPrimaryContextPass, import_wizard_system);
    }
}

// ---------------------------------------------------------------------------
// UI system — polls channels and draws the active step
// ---------------------------------------------------------------------------

fn import_wizard_system(
    mut contexts: EguiContexts,
    mut wizard: ResMut<ImportWizard>,
    active_config: Res<TerrainConfig>,
    active_library: Res<MaterialLibrary>,
    mut reload_tx: MessageWriter<ReloadTerrainRequest>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let w = wizard.as_mut();

    poll_file_picker(w);
    poll_bake(w, &active_config, &active_library, &mut reload_tx);

    match &w.step {
        ImportStep::Closed => {}
        ImportStep::PickingFile { .. } => {
            egui::Window::new("Import Heightmap")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("Waiting for file selection…");
                    ui.add(egui::Spinner::new());
                });
        }
        ImportStep::Settings(_) => draw_settings(ctx, w),
        ImportStep::Baking { .. } => draw_baking(ctx, w),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Channel polling
// ---------------------------------------------------------------------------

fn poll_file_picker(wizard: &mut ImportWizard) {
    // Try to receive from the picker channel without blocking.
    let result = if let ImportStep::PickingFile { rx, .. } = &wizard.step {
        rx.lock().ok().and_then(|guard| guard.try_recv().ok())
    } else {
        return;
    };

    if let Some(maybe_path) = result {
        let (target, settings) = match std::mem::replace(&mut wizard.step, ImportStep::Closed) {
            ImportStep::PickingFile {
                target, settings, ..
            } => (target, *settings),
            other => {
                wizard.step = other;
                return;
            }
        };
        let mut settings = settings;
        if let Some(path) = maybe_path {
            let s = path.display().to_string();
            match target {
                FilePickTarget::Height => settings.height_path = s,
                FilePickTarget::Diffuse => settings.diffuse_path = s,
                FilePickTarget::Bump => settings.bump_path = s,
                FilePickTarget::Output => settings.output_dir = s,
            }
        }
        wizard.step = ImportStep::Settings(Box::new(settings));
    }
}

fn poll_bake(
    wizard: &mut ImportWizard,
    active_config: &TerrainConfig,
    active_library: &MaterialLibrary,
    reload_tx: &mut MessageWriter<ReloadTerrainRequest>,
) {
    loop {
        let msg = if let ImportStep::Baking { rx, .. } = &wizard.step {
            rx.lock().ok().and_then(|guard| guard.try_recv().ok())
        } else {
            return;
        };

        match msg {
            None => break,
            Some(BakeMsg::Log(line)) => {
                if let ImportStep::Baking { log_lines, .. } = &mut wizard.step {
                    log_lines.push(line);
                }
            }
            Some(BakeMsg::Done(result)) => {
                if let ImportStep::Baking {
                    log_lines,
                    finished,
                    error,
                    settings,
                    reloaded,
                    ..
                } = &mut wizard.step
                {
                    match result {
                        Ok(()) => {
                            log_lines.push("✓ Bake complete — reloading terrain…".into());
                            // Build a TerrainSourceDesc pointing at the baked output.
                            let output_dir = PathBuf::from(&settings.output_dir);
                            let mut new_config = active_config.clone();
                            new_config.world_scale = settings.world_scale;
                            new_config.height_scale = settings.height_scale * settings.world_scale;
                            let (world_min, world_max) =
                                scan_world_bounds_from_output(&output_dir, new_config.world_scale);
                            let max_mip = scan_max_mip_level(&output_dir.join("height"));
                            new_config.clipmap_levels =
                                derive_import_clipmap_levels(settings.clipmap_levels, max_mip);
                            let new_source = TerrainSourceDesc {
                                tile_root: Some(output_dir.clone()),
                                normal_root: None, // bake writes into tile_root/normal
                                macro_color_root: if settings.diffuse_path.trim().is_empty() {
                                    None
                                } else {
                                    Some(settings.diffuse_path.clone())
                                },
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
                            *reloaded = true;
                        }
                        Err(ref e) => {
                            log_lines.push(format!("✗ Error: {e}"));
                            *error = Some(e.clone());
                        }
                    }
                    *finished = true;
                }
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Settings dialog
// ---------------------------------------------------------------------------

fn draw_settings(ctx: &egui::Context, wizard: &mut ImportWizard) {
    let mut do_import = false;
    let mut do_cancel = false;
    let mut pick_target: Option<FilePickTarget> = None;

    egui::Window::new("Import Heightmap")
        .default_width(500.0)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            let settings = match &mut wizard.step {
                ImportStep::Settings(s) => s.as_mut(),
                _ => return,
            };

            egui::Grid::new("import_paths")
                .num_columns(3)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    path_row(ui, "Heightmap *", &mut settings.height_path,
                             "required — EXR, PNG, or TIFF", &mut pick_target, FilePickTarget::Height, false);
                    path_row(ui, "Colour map", &mut settings.diffuse_path,
                             "optional EXR / PNG", &mut pick_target, FilePickTarget::Diffuse, false);
                    path_row(ui, "Bump / normal", &mut settings.bump_path,
                             "optional — normals derived from height if absent",
                             &mut pick_target, FilePickTarget::Bump, false);
                    path_row(ui, "Output directory", &mut settings.output_dir,
                             "where to write height/ and normal/ tiles",
                             &mut pick_target, FilePickTarget::Output, true);
                });

            ui.separator();

            egui::Grid::new("import_params")
                .num_columns(2)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    let settings = match &mut wizard.step {
                        ImportStep::Settings(s) => s.as_mut(),
                        _ => return,
                    };
                    ui.label("World scale")
                        .on_hover_text("Uniform multiplier for X, Y, and Z world extents.");
                    ui.add(egui::DragValue::new(&mut settings.world_scale).speed(0.05).range(0.01..=100.0));
                    ui.end_row();

                    ui.label("Height scale (base)")
                        .on_hover_text("World-space Y range for a fully-white texel before world_scale.");
                    ui.add(egui::DragValue::new(&mut settings.height_scale).speed(1.0).range(1.0..=100_000.0));
                    ui.end_row();

                    ui.label("Smooth σ")
                        .on_hover_text("Gaussian pre-smooth sigma (source texels).  0 = off.  ~1.0 removes spike outliers.");
                    ui.add(egui::DragValue::new(&mut settings.smooth_sigma).speed(0.05).range(0.0..=10.0));
                    ui.end_row();

                    ui.label("View distance (rings)")
                        .on_hover_text("Number of clipmap LOD rings (4–16). More rings extend view distance at the cost of GPU memory. Default 12 is suitable for most terrains.");
                    ui.add(egui::DragValue::new(&mut settings.clipmap_levels).speed(1).range(4..=MAX_SUPPORTED_CLIPMAP_LEVELS as u32));
                    ui.end_row();

                    ui.label("Flip green")
                        .on_hover_text("Negate G for OpenGL-convention (Y-up) normal maps.");
                    ui.checkbox(&mut settings.flip_green, "");
                    ui.end_row();
                });

            ui.separator();

            let can_import = match &wizard.step {
                ImportStep::Settings(s) => !s.height_path.trim().is_empty(),
                _ => false,
            };

            ui.horizontal(|ui| {
                if ui.add_enabled(can_import, egui::Button::new("⬆  Import"))
                    .on_disabled_hover_text("Select a heightmap first.")
                    .clicked()
                {
                    do_import = true;
                }
                if ui.button("Cancel").clicked() {
                    do_cancel = true;
                }
            });
            if !can_import {
                ui.colored_label(egui::Color32::YELLOW, "⚠  A heightmap path is required.");
            }
        });

    if let Some(target) = pick_target {
        wizard.pick_file(target);
        return;
    }
    if do_import {
        wizard.start_bake();
        return;
    }
    if do_cancel {
        wizard.step = ImportStep::Closed;
    }
}

fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    hint: &str,
    pick_target: &mut Option<FilePickTarget>,
    target: FilePickTarget,
    is_dir: bool,
) {
    ui.label(label);
    ui.add(
        egui::TextEdit::singleline(value)
            .hint_text(hint)
            .desired_width(280.0),
    );
    let _btn_label = if is_dir { "Browse…" } else { "Browse…" };
    if ui.small_button("Browse…").clicked() {
        *pick_target = Some(target);
    }
    ui.end_row();
}

// ---------------------------------------------------------------------------
// Baking dialog
// ---------------------------------------------------------------------------

fn draw_baking(ctx: &egui::Context, wizard: &mut ImportWizard) {
    let mut close_requested = false;

    let (finished, has_error, reloaded) = match &wizard.step {
        ImportStep::Baking {
            finished,
            error,
            reloaded,
            ..
        } => (*finished, error.is_some(), *reloaded),
        _ => return,
    };

    egui::Window::new("Import Heightmap — Baking")
        .default_width(560.0)
        .default_height(420.0)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            if !finished {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label("Baking tiles…");
                });
            } else if has_error {
                ui.colored_label(egui::Color32::RED, "✗  Bake failed — see log below.");
            } else if reloaded {
                ui.colored_label(egui::Color32::GREEN, "✓  Bake complete — terrain updated!");
                ui.label(
                    egui::RichText::new(
                        "Use File → Save Landscape… to save this level for future sessions.",
                    )
                    .color(egui::Color32::GRAY)
                    .small(),
                );
            } else {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label("Reloading terrain…");
                });
            }

            ui.separator();

            let log_height = (ui.available_height() - 40.0).max(80.0);
            egui::ScrollArea::vertical()
                .id_salt("bake_log")
                .max_height(log_height)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if let ImportStep::Baking { log_lines, .. } = &wizard.step {
                        for line in log_lines {
                            let color = if line.starts_with("✗") || line.contains("WARNING") {
                                egui::Color32::from_rgb(255, 180, 60)
                            } else if line.starts_with("✓") {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::LIGHT_GRAY
                            };
                            ui.colored_label(color, line);
                        }
                    }
                });

            ui.separator();
            if ui
                .add_enabled(finished, egui::Button::new("Close"))
                .clicked()
            {
                close_requested = true;
            }
        });

    if close_requested {
        wizard.step = ImportStep::Closed;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scan `height_dir` for `L*` subdirectories and return the highest index found.
fn scan_max_mip_level(height_dir: &Path) -> u8 {
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

/// Derive world bounds from baked tiles, mirroring the logic in `app_config`.
fn scan_world_bounds_from_output(output_dir: &Path, world_scale: f32) -> (Vec2, Vec2) {
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

#[cfg(test)]
mod tests {
    use super::derive_import_clipmap_levels;

    #[test]
    fn import_preserves_large_view_distance_requests() {
        assert_eq!(derive_import_clipmap_levels(12, 3), 12);
        assert_eq!(derive_import_clipmap_levels(4, 3), 4);
    }
}

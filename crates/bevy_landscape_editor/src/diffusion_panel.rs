use std::{
    collections::HashSet,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{mpsc, Mutex},
};

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_landscape::{
    bake::BakeConfig, MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainSourceDesc,
    MAX_SUPPORTED_CLIPMAP_LEVELS,
};

use crate::preferences::AppPreferences;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DiffusionWorkflow {
    #[default]
    AzgaarJson,
    ConditioningFolder,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DiffusionDevice {
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl DiffusionDevice {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Gpu => "GPU (CUDA/ROCm)",
            Self::Cpu => "CPU",
        }
    }

    fn cli_value(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Gpu => Some("cuda"),
            Self::Cpu => Some("cpu"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DiffusionDtype {
    #[default]
    Fp32,
    Bf16,
    Fp16,
}

impl DiffusionDtype {
    fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
}

const AZGAAR_REGION_CELL_OPTIONS: &[u32] = &[4, 8, 16, 32, 64, 128, 256, 512];
const GENERATED_PREVIEW_SELECTED_MAX_SIDE: u32 = 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffusionPickTarget {
    RepoPath,
    AzgaarJson,
    ConditioningDir,
    WorkingDir,
    TileOutputDir,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffusionRunKind {
    Generate,
    Probe,
    Preview,
}

impl DiffusionRunKind {
    fn title(self) -> &'static str {
        match self {
            Self::Generate => "Run Log",
            Self::Probe => "Probe Log",
            Self::Preview => "Preview Log",
        }
    }

    fn running_label(self) -> &'static str {
        match self {
            Self::Generate => "Terrain diffusion job is running...",
            Self::Probe => "Diffusion runtime probe is running...",
            Self::Preview => "Diffusion preview job is running...",
        }
    }

    fn failure_label(self) -> &'static str {
        match self {
            Self::Generate => "Diffusion pipeline failed",
            Self::Probe => "Diffusion runtime probe failed",
            Self::Preview => "Diffusion preview failed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GeneratedPreviewMode {
    Coarse,
}

impl GeneratedPreviewMode {
    fn label(self) -> &'static str {
        match self {
            Self::Coarse => "Generated selection preview (fast coarse)",
        }
    }
}

#[derive(Clone)]
struct DiffusionPreviewArtifacts {
    overview_image: Option<PathBuf>,
    selected_image: PathBuf,
    selected_mode: GeneratedPreviewMode,
    preview_key: String,
}

enum DiffusionOutcome {
    Generated(PathBuf),
    Probed,
    Previewed(DiffusionPreviewArtifacts),
}

enum DiffusionMsg {
    Log(String),
    Done(Result<DiffusionOutcome, String>),
}

enum DiffusionRunState {
    Idle,
    PickingFile {
        rx: Mutex<mpsc::Receiver<Option<PathBuf>>>,
        target: DiffusionPickTarget,
    },
    Running {
        kind: DiffusionRunKind,
        log_lines: Vec<String>,
        rx: Mutex<mpsc::Receiver<DiffusionMsg>>,
        finished: bool,
        error: Option<String>,
        reloaded: bool,
        /// 0.0 = not started, 1.0 = done.
        progress: f32,
        /// Human-readable label for the current stage.
        stage: String,
    },
}

impl Default for DiffusionRunState {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Clone)]
struct DiffusionJobConfig {
    workflow: DiffusionWorkflow,
    repo_path: String,
    python_bin: String,
    model_path: String,
    azgaar_json_path: String,
    azgaar_region_cells: u32,
    azgaar_center_x: f32,
    azgaar_center_y: f32,
    conditioning_dir: String,
    conditioning_scale_km: f32,
    working_dir: String,
    tile_output_dir: String,
    world_scale: f32,
    final_height_span_m: f32,
    smooth_sigma: f32,
    device: DiffusionDevice,
    dtype: DiffusionDtype,
    snr: String,
    batch_size: String,
    cache_size: String,
    seed: String,
    use_compile: bool,
    extra_env: String,
}

#[derive(Default)]
pub(crate) struct DiffusionPanelState {
    pub workflow: DiffusionWorkflow,
    pub repo_path: String,
    pub python_bin: String,
    pub model_path: String,
    pub azgaar_json_path: String,
    pub azgaar_region_cells: u32,
    pub azgaar_center_x: f32,
    pub azgaar_center_y: f32,
    pub conditioning_dir: String,
    pub conditioning_scale_km: f32,
    pub working_dir: String,
    pub tile_output_dir: String,
    pub world_scale: f32,
    pub final_height_span_m: f32,
    pub smooth_sigma: f32,
    pub clipmap_levels: u32,
    pub device: DiffusionDevice,
    pub dtype: DiffusionDtype,
    pub snr: String,
    pub batch_size: String,
    pub cache_size: String,
    pub seed: String,
    pub use_compile: bool,
    pub extra_env: String,
    run_state: DiffusionRunState,
    prefs_applied: bool,
    /// Dimensions of the full conditioning raster (after azgaar-to-tiff).
    /// Populated lazily from an existing TIFF or from log output.
    pub conditioning_raster_size: Option<(u32, u32)>,
    /// Egui texture handle for the world preview thumbnail.
    preview_texture: Option<egui::TextureHandle>,
    /// Raster size the preview texture was built for (to detect staleness).
    preview_loaded_for: Option<(u32, u32)>,
    /// Generated diffusion overview preview texture.
    generated_overview_texture: Option<egui::TextureHandle>,
    /// Generated selected-area preview texture.
    generated_selected_texture: Option<egui::TextureHandle>,
    /// Last generated overview preview image path.
    generated_overview_image: Option<PathBuf>,
    /// Last generated selected preview image path.
    generated_selected_image: Option<PathBuf>,
    /// Whether the selected preview was rendered from native detail or coarse output.
    generated_selected_mode: Option<GeneratedPreviewMode>,
    /// Tracks whether the loaded previews match the current UI settings.
    generated_preview_key: Option<String>,
    /// Cache-buster for egui textures when preview files are overwritten.
    generated_preview_revision: u64,
    /// High-resolution thumbnail of just the selected crop region.
    crop_preview_texture: Option<egui::TextureHandle>,
    /// Crop params the thumbnail was built for: (x0, y0, side, src_w, src_h).
    crop_preview_key: Option<(u32, u32, u32, u32, u32)>,
}

impl DiffusionPanelState {
    pub(crate) fn new() -> Self {
        let repo_path = detect_repo_path();
        let working_dir = "assets/terrain_diffusion".to_string();
        Self {
            workflow: DiffusionWorkflow::AzgaarJson,
            repo_path: repo_path.display().to_string(),
            python_bin: detect_python_bin(&repo_path).display().to_string(),
            model_path: "xandergos/terrain-diffusion-30m".into(),
            azgaar_json_path: String::new(),
            azgaar_region_cells: 8,
            azgaar_center_x: 0.5,
            azgaar_center_y: 0.5,
            conditioning_dir: format!("{working_dir}/conditioning"),
            conditioning_scale_km: 7.7,
            working_dir,
            tile_output_dir: "assets/tiles_diffusion".into(),
            world_scale: 30.0,
            final_height_span_m: 8192.0,
            smooth_sigma: 1.0,
            clipmap_levels: TerrainConfig::default().clipmap_levels,
            device: DiffusionDevice::Auto,
            dtype: DiffusionDtype::Fp32,
            snr: "0.2,0.2,1.0,0.2,1.0".into(),
            batch_size: "1,2,4,8".into(),
            cache_size: "1G".into(),
            seed: String::new(),
            use_compile: true,
            extra_env: "HSA_OVERRIDE_GFX_VERSION=11.0.0".into(),
            run_state: DiffusionRunState::Idle,
            prefs_applied: false,
            conditioning_raster_size: None,
            preview_texture: None,
            preview_loaded_for: None,
            generated_overview_texture: None,
            generated_selected_texture: None,
            generated_overview_image: None,
            generated_selected_image: None,
            generated_selected_mode: None,
            generated_preview_key: None,
            generated_preview_revision: 0,
            crop_preview_texture: None,
            crop_preview_key: None,
        }
    }

    fn is_running(&self) -> bool {
        matches!(
            self.run_state,
            DiffusionRunState::Running {
                finished: false,
                ..
            }
        )
    }

    fn is_busy(&self) -> bool {
        matches!(
            self.run_state,
            DiffusionRunState::PickingFile { .. }
                | DiffusionRunState::Running {
                    finished: false,
                    ..
                }
        )
    }

    pub(crate) fn should_throttle_rendering(&self) -> bool {
        matches!(
            self.run_state,
            DiffusionRunState::Running {
                kind: DiffusionRunKind::Generate | DiffusionRunKind::Preview,
                finished: false,
                ..
            }
        )
    }

    pub(crate) fn apply_startup_preferences(&mut self, prefs: &AppPreferences) {
        if self.prefs_applied {
            return;
        }
        if let Some(path) = &prefs.diffusion_repo_path {
            self.repo_path = path.clone();
        }
        if let Some(path) = &prefs.diffusion_python {
            self.python_bin = path.clone();
        } else {
            self.python_bin = detect_python_bin(Path::new(&self.repo_path))
                .display()
                .to_string();
        }
        if let Some(env) = &prefs.diffusion_env {
            self.extra_env = env.clone();
        }
        self.prefs_applied = true;
    }
}

#[derive(Clone, Copy)]
struct AzgaarCropWindow {
    source_width: u32,
    source_height: u32,
    side: u32,
    x0: u32,
    y0: u32,
}

impl AzgaarCropWindow {
    fn uv_rect(self) -> egui::Rect {
        let min = egui::pos2(
            self.x0 as f32 / self.source_width as f32,
            self.y0 as f32 / self.source_height as f32,
        );
        let max = egui::pos2(
            (self.x0 + self.side) as f32 / self.source_width as f32,
            (self.y0 + self.side) as f32 / self.source_height as f32,
        );
        egui::Rect::from_min_max(min, max)
    }

    fn preview_rect(self, rect: egui::Rect) -> egui::Rect {
        let uv = self.uv_rect();
        egui::Rect::from_min_max(
            egui::pos2(
                rect.left() + uv.left() * rect.width(),
                rect.top() + uv.top() * rect.height(),
            ),
            egui::pos2(
                rect.left() + uv.right() * rect.width(),
                rect.top() + uv.bottom() * rect.height(),
            ),
        )
    }

    fn area_fraction(self) -> f32 {
        (self.side as f32 * self.side as f32)
            / (self.source_width as f32 * self.source_height as f32)
    }
}

fn max_azgaar_crop_side(source_width: u32, source_height: u32) -> Option<u32> {
    let min_dim = source_width.min(source_height);
    if min_dim == 0 {
        return None;
    }
    Some(1u32 << (u32::BITS - 1 - min_dim.leading_zeros()))
}

fn compute_azgaar_crop_window(
    source_width: u32,
    source_height: u32,
    requested_side: u32,
    center_x: f32,
    center_y: f32,
) -> Option<AzgaarCropWindow> {
    let max_side = max_azgaar_crop_side(source_width, source_height)?;
    let side = requested_side.min(max_side);
    if side == 0 {
        return None;
    }

    let px = source_width.saturating_sub(1) as f32 * center_x.clamp(0.0, 1.0);
    let py = source_height.saturating_sub(1) as f32 * center_y.clamp(0.0, 1.0);
    let max_x0 = source_width.saturating_sub(side) as i32;
    let max_y0 = source_height.saturating_sub(side) as i32;
    let x0 = ((px - side as f32 / 2.0).round() as i32).clamp(0, max_x0) as u32;
    let y0 = ((py - side as f32 / 2.0).round() as i32).clamp(0, max_y0) as u32;

    Some(AzgaarCropWindow {
        source_width,
        source_height,
        side,
        x0,
        y0,
    })
}

fn generated_preview_mode_for_crop(crop: AzgaarCropWindow) -> GeneratedPreviewMode {
    let _ = crop;
    GeneratedPreviewMode::Coarse
}

fn azgaar_conditioning_cache_key(config: &DiffusionJobConfig) -> Result<String, String> {
    let source_path = PathBuf::from(&config.azgaar_json_path);
    let canonical_source = source_path
        .canonicalize()
        .unwrap_or_else(|_| source_path.clone());
    let metadata = std::fs::metadata(&source_path).map_err(|e| {
        format!(
            "Failed to read Azgaar JSON metadata from {}: {e}",
            source_path.display()
        )
    })?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|ts| ts.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|dur| dur.as_secs_f64())
        .unwrap_or(0.0);
    Ok(format!(
        "source={}\nsize={}\nmodified={modified:.3}\nscale={:.6}\n",
        canonical_source.display(),
        metadata.len(),
        config.conditioning_scale_km,
    ))
}

fn ensure_preview_azgaar_conditioning(
    config: &DiffusionJobConfig,
    repo_path: &Path,
    conditioning_dir: &Path,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<(), String> {
    std::fs::create_dir_all(conditioning_dir).map_err(|e| {
        format!(
            "Failed to create conditioning directory {}: {e}",
            conditioning_dir.display()
        )
    })?;

    let cache_key = azgaar_conditioning_cache_key(config)?;
    let cache_marker = conditioning_dir.join(".landscape_azgaar_cache");
    let heightmap_path = conditioning_dir.join("heightmap.tif");
    if heightmap_path.exists()
        && std::fs::read_to_string(&cache_marker).ok().as_deref() == Some(cache_key.as_str())
    {
        send_log(
            tx,
            format!(
                "Reusing cached Azgaar conditioning TIFFs in {}",
                conditioning_dir.display()
            ),
        );
        return Ok(());
    }

    send_log(
        tx,
        format!(
            "Converting Azgaar export to conditioning TIFFs in {}",
            conditioning_dir.display()
        ),
    );
    let scale = format!("{:.3}", config.conditioning_scale_km);
    let args = vec![
        "-m".to_string(),
        "terrain_diffusion".to_string(),
        "azgaar-to-tiff".to_string(),
        config.azgaar_json_path.clone(),
        conditioning_dir.display().to_string(),
        "--scale".to_string(),
        scale,
    ];
    run_logged_command(&config.python_bin, &args, repo_path, &config.extra_env, tx)?;
    std::fs::write(&cache_marker, cache_key).map_err(|e| {
        format!(
            "Failed to write conditioning cache metadata {}: {e}",
            cache_marker.display()
        )
    })?;
    Ok(())
}

fn build_preview_key(
    config: &DiffusionJobConfig,
    source_size: Option<(u32, u32)>,
    crop: Option<AzgaarCropWindow>,
    selected_mode: Option<GeneratedPreviewMode>,
) -> String {
    format!(
        concat!(
            "workflow={:?}|repo={}|python={}|model={}|azgaar={}|cells={}|cx={:.6}|cy={:.6}|",
            "cond_dir={}|cond_scale={:.3}|seed={}|snr={}|batch={}|cache={}|device={:?}|dtype={:?}|",
            "compile={}|source={:?}|crop={:?}|selected_mode={:?}"
        ),
        config.workflow,
        config.repo_path,
        config.python_bin,
        config.model_path,
        config.azgaar_json_path,
        config.azgaar_region_cells,
        config.azgaar_center_x,
        config.azgaar_center_y,
        config.conditioning_dir,
        config.conditioning_scale_km,
        config.seed,
        config.snr,
        config.batch_size,
        config.cache_size,
        config.device,
        config.dtype,
        config.use_compile,
        source_size,
        crop.map(|c| (c.x0, c.y0, c.side, c.source_width, c.source_height)),
        selected_mode,
    )
}

fn load_preview_texture(
    ctx: &egui::Context,
    path: &Path,
    cache_key: impl Into<String>,
) -> Option<egui::TextureHandle> {
    let img = image::open(path).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    let ci = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img);
    Some(ctx.load_texture(cache_key.into(), ci, egui::TextureOptions::LINEAR))
}

pub(crate) fn poll_diffusion_state(
    state: &mut DiffusionPanelState,
    active_config: &TerrainConfig,
    active_library: &MaterialLibrary,
    reload_tx: &mut MessageWriter<ReloadTerrainRequest>,
) {
    poll_file_picker(state);
    poll_run(state, active_config, active_library, reload_tx);
}

pub(crate) fn draw_diffusion_tab(ui: &mut egui::Ui, state: &mut DiffusionPanelState) {
    // Lazily probe conditioning TIFF for world dimensions if not yet known.
    if state.conditioning_raster_size.is_none() {
        let dir = match state.workflow {
            DiffusionWorkflow::AzgaarJson => {
                let base = PathBuf::from(&state.working_dir).join("conditioning");
                base.to_string_lossy().into_owned()
            }
            DiffusionWorkflow::ConditioningFolder => state.conditioning_dir.clone(),
        };
        state.conditioning_raster_size = probe_conditioning_tiff_size(&dir);
    }

    // Lazily load (or reload) the world preview texture.
    if state.preview_loaded_for != state.conditioning_raster_size {
        state.preview_texture = None;
        if let Some(_) = state.conditioning_raster_size {
            let tiff_path = match state.workflow {
                DiffusionWorkflow::AzgaarJson => {
                    PathBuf::from(&state.working_dir).join("conditioning/heightmap.tif")
                }
                DiffusionWorkflow::ConditioningFolder => {
                    PathBuf::from(&state.conditioning_dir).join("heightmap.tif")
                }
            };
            if let Some((pixels, tw, th)) =
                bevy_landscape::bake::load_tiff_thumbnail(&tiff_path, 512)
            {
                // Apply a terrain colour ramp: deep blue → beach → green → white.
                let color_pixels: Vec<egui::Color32> =
                    pixels.iter().map(|&g| terrain_color_ramp(g)).collect();
                let img = egui::ColorImage {
                    size: [tw, th],
                    source_size: egui::vec2(tw as f32, th as f32),
                    pixels: color_pixels,
                };
                state.preview_texture = Some(ui.ctx().load_texture(
                    "world_preview",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
        state.preview_loaded_for = state.conditioning_raster_size;
    }

    if state.generated_overview_texture.is_none() {
        if let Some(path) = &state.generated_overview_image {
            state.generated_overview_texture = load_preview_texture(
                ui.ctx(),
                path,
                format!(
                    "diffusion_generated_overview_{}",
                    state.generated_preview_revision
                ),
            );
        }
    }
    if state.generated_selected_texture.is_none() {
        if let Some(path) = &state.generated_selected_image {
            state.generated_selected_texture = load_preview_texture(
                ui.ctx(),
                path,
                format!(
                    "diffusion_generated_selected_{}",
                    state.generated_preview_revision
                ),
            );
        }
    }

    let azgaar_crop = state.conditioning_raster_size.and_then(|(src_w, src_h)| {
        compute_azgaar_crop_window(
            src_w,
            src_h,
            state.azgaar_region_cells,
            state.azgaar_center_x,
            state.azgaar_center_y,
        )
    });
    let effective_region_cells = azgaar_crop
        .map(|crop| crop.side)
        .unwrap_or(state.azgaar_region_cells);

    // Lazily load a high-resolution crop thumbnail for the selected region.
    // This is separate from the low-res world thumbnail; it reads only the crop
    // strips so the zoomed view doesn't look blurry.
    if let Some(crop) = azgaar_crop {
        let new_key = (
            crop.x0,
            crop.y0,
            crop.side,
            crop.source_width,
            crop.source_height,
        );
        if state.crop_preview_key != Some(new_key) {
            state.crop_preview_texture = None;
            let tiff_path = match state.workflow {
                DiffusionWorkflow::AzgaarJson => {
                    PathBuf::from(&state.working_dir).join("conditioning/heightmap.tif")
                }
                DiffusionWorkflow::ConditioningFolder => {
                    PathBuf::from(&state.conditioning_dir).join("heightmap.tif")
                }
            };
            if let Some((pixels, tw, th)) = bevy_landscape::bake::load_tiff_crop_thumbnail(
                &tiff_path,
                crop.x0,
                crop.y0,
                crop.side,
                512,
            ) {
                let color_pixels: Vec<egui::Color32> =
                    pixels.iter().map(|&g| terrain_color_ramp(g)).collect();
                let img = egui::ColorImage {
                    size: [tw, th],
                    source_size: egui::vec2(tw as f32, th as f32),
                    pixels: color_pixels,
                };
                state.crop_preview_texture = Some(ui.ctx().load_texture(
                    "world_crop_preview",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
            }
            state.crop_preview_key = Some(new_key);
        }
    } else {
        state.crop_preview_texture = None;
        state.crop_preview_key = None;
    }

    let mut pick_target = None;
    let mut preview_requested = false;
    let mut probe_requested = false;
    let mut start_requested = false;
    let mut clear_requested = false;
    let running = state.is_running();
    let log_visible = matches!(state.run_state, DiffusionRunState::Running { .. });
    let log_reserved_height = if log_visible { 320.0 } else { 0.0 };
    let controls_max_height = (ui.available_height() - log_reserved_height).max(180.0);

    egui::ScrollArea::vertical()
        .id_salt("diffusion_controls_scroll")
        .max_height(controls_max_height)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Diffusion Pipeline");
                if running {
                    ui.add_space(8.0);
                    ui.spinner();
                    ui.label("Running");
                } else if matches!(state.run_state, DiffusionRunState::PickingFile { .. }) {
                    ui.add_space(8.0);
                    ui.spinner();
                    ui.label("Waiting for file selection");
                }
            });
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Workflow");
                ui.selectable_value(
                    &mut state.workflow,
                    DiffusionWorkflow::AzgaarJson,
                    "Azgaar JSON",
                );
                ui.selectable_value(
                    &mut state.workflow,
                    DiffusionWorkflow::ConditioningFolder,
                    "Conditioning Folder",
                );
            });
            ui.add_space(6.0);

            egui::Grid::new("diffusion_paths")
                .num_columns(3)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    path_row(
                        ui,
                        "Terrain Diffusion repo",
                        &mut state.repo_path,
                        "Path to the terrain-diffusion checkout",
                        &mut pick_target,
                        Some(DiffusionPickTarget::RepoPath),
                        !running,
                    );
                    text_row(
                        ui,
                        "Python executable",
                        &mut state.python_bin,
                        "python3 or path to a venv Python with torch + deps",
                        !running,
                    );
                    path_row(
                        ui,
                        "Work directory",
                        &mut state.working_dir,
                        "Temporary TIFFs and generated GeoTIFFs",
                        &mut pick_target,
                        Some(DiffusionPickTarget::WorkingDir),
                        !running,
                    );
                    path_row(
                        ui,
                        "Tile output",
                        &mut state.tile_output_dir,
                        "Baked landscape tile hierarchy",
                        &mut pick_target,
                        Some(DiffusionPickTarget::TileOutputDir),
                        !running,
                    );
                    match state.workflow {
                        DiffusionWorkflow::AzgaarJson => {
                            path_row(
                                ui,
                                "Azgaar full JSON",
                                &mut state.azgaar_json_path,
                                "Tools -> Export -> Export To JSON -> Full",
                                &mut pick_target,
                                Some(DiffusionPickTarget::AzgaarJson),
                                !running,
                            );
                        }
                        DiffusionWorkflow::ConditioningFolder => {
                            path_row(
                                ui,
                                "Conditioning folder",
                                &mut state.conditioning_dir,
                                "Folder with heightmap.tif / temperature.tif / ...",
                                &mut pick_target,
                                Some(DiffusionPickTarget::ConditioningDir),
                                !running,
                            );
                        }
                    }
                });

            ui.add_space(8.0);
            ui.heading("Model & Scale");
            ui.separator();

            egui::Grid::new("diffusion_model")
                .num_columns(2)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Model");
                    ui.horizontal(|ui| {
                        ui.add_enabled_ui(!running, |ui| {
                            ui.text_edit_singleline(&mut state.model_path);
                            if ui.small_button("30m").clicked() {
                                state.model_path = "xandergos/terrain-diffusion-30m".into();
                                state.world_scale = 30.0;
                                state.conditioning_scale_km = 7.7;
                            }
                            if ui.small_button("90m").clicked() {
                                state.model_path = "xandergos/terrain-diffusion-90m".into();
                                state.world_scale = 90.0;
                                state.conditioning_scale_km = 23.0;
                            }
                        });
                    });
                    ui.end_row();

            ui.label("Azgaar scale (km/px)")
                .on_hover_text(
                    "Used only for Azgaar JSON conversion. Suggested values are ~7.7 for 30m and 23 for 90m models.",
                );
            ui.add_enabled(
                !running && matches!(state.workflow, DiffusionWorkflow::AzgaarJson),
                egui::DragValue::new(&mut state.conditioning_scale_km)
                    .speed(0.1)
                    .range(1.0..=500.0),
            );
            ui.end_row();

            if matches!(state.workflow, DiffusionWorkflow::AzgaarJson) {
                ui.label("Azgaar region")
                    .on_hover_text(
                        "Square crop size in conditioning cells. The diffusion export expands each conditioning cell into 256 output pixels, so power-of-two sizes keep the current tile baker happy.",
                    );
                ui.add_enabled_ui(!running, |ui| {
                    egui::ComboBox::from_id_salt("diffusion_azgaar_region_cells")
                        .selected_text(format!("{} cells/side", state.azgaar_region_cells))
                        .show_ui(ui, |ui| {
                            for &cells in AZGAAR_REGION_CELL_OPTIONS {
                                ui.selectable_value(
                                    &mut state.azgaar_region_cells,
                                    cells,
                                    format!("{cells}"),
                                );
                            }
                        });
                });
                ui.end_row();

                ui.label("Region center X");
                ui.add_enabled(
                    !running,
                    egui::Slider::new(&mut state.azgaar_center_x, 0.0..=1.0).show_value(true),
                );
                ui.end_row();

                ui.label("Region center Y");
                ui.add_enabled(
                    !running,
                    egui::Slider::new(&mut state.azgaar_center_y, 0.0..=1.0).show_value(true),
                );
                ui.end_row();


                ui.label("Selected region span");
                ui.label(format!(
                    "{:.1} km/side",
                    effective_region_cells as f32 * state.conditioning_scale_km
                ));
                ui.end_row();

                ui.label("Generated raster");
                ui.label(format!(
                    "{} x {} px",
                    effective_region_cells * 256,
                    effective_region_cells * 256
                ));
                ui.end_row();

                ui.label("Loaded terrain span");
                ui.label(format!(
                    "{:.1} km/side",
                    effective_region_cells as f32 * 256.0 * state.world_scale / 1000.0
                ));
                ui.end_row();

                let ideal_world_scale = state.conditioning_scale_km * 1000.0 / 256.0;
                ui.label("Ideal world scale");
                ui.label(format!("{ideal_world_scale:.2} m/px"));
                ui.end_row();
            }

            ui.label("World Scale (m/px)")
                .on_hover_text("LOD0 cell size in your Bevy landscape after baking.");
            ui.add_enabled(
                !running,
                egui::DragValue::new(&mut state.world_scale)
                    .speed(1.0)
                    .range(0.1..=10_000.0),
            );
            ui.end_row();

            ui.label("Final Height Span (m)")
                .on_hover_text("Total world-space elevation range after baking and reload.");
            ui.add_enabled(
                !running,
                egui::DragValue::new(&mut state.final_height_span_m)
                    .speed(16.0)
                    .range(16.0..=200_000.0),
            );
            ui.end_row();

            ui.label("Base Height Scale");
            let base_height_scale = if state.world_scale > 0.0 {
                state.final_height_span_m / state.world_scale
            } else {
                0.0
            };
            ui.label(format!("{base_height_scale:.2}"));
            ui.end_row();

            ui.label("View distance (rings)");
            ui.add_enabled(
                !running,
                egui::DragValue::new(&mut state.clipmap_levels)
                    .speed(1)
                    .range(4..=MAX_SUPPORTED_CLIPMAP_LEVELS as u32),
            );
            ui.end_row();

            ui.label("Bake smooth σ");
            ui.add_enabled(
                !running,
                egui::DragValue::new(&mut state.smooth_sigma)
                    .speed(0.05)
                    .range(0.0..=10.0),
            );
            ui.end_row();
                });

            // World preview at full dialog width — outside the grid so it can expand freely.
            if matches!(state.workflow, DiffusionWorkflow::AzgaarJson) {
                let raster_size = state.conditioning_raster_size;
                let (src_w, src_h) = raster_size.unwrap_or((1, 1));
                let preview_w = ui.available_width();
                let aspect = src_w as f32 / src_h as f32;
                let preview_h = (preview_w / aspect).max(40.0);

                ui.add_space(8.0);
                ui.label("World preview");
                if raster_size.is_some() {
                    let (response, painter) = ui.allocate_painter(
                        egui::vec2(preview_w, preview_h),
                        if running {
                            egui::Sense::hover()
                        } else {
                            egui::Sense::click_and_drag()
                        },
                    );
                    let rect = response.rect;

                    if !running {
                        if let Some(pos) = response.interact_pointer_pos() {
                            state.azgaar_center_x =
                                ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                            state.azgaar_center_y =
                                ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                        }
                    }
                    // Recompute after any click so the overlay reflects the new position
                    // on the same frame rather than lagging one frame behind.
                    let azgaar_crop_view = compute_azgaar_crop_window(
                        src_w,
                        src_h,
                        state.azgaar_region_cells,
                        state.azgaar_center_x,
                        state.azgaar_center_y,
                    );

                    if let Some(tex) = &state.preview_texture {
                        painter.image(
                            tex.id(),
                            rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    } else {
                        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(50, 70, 45));
                    }
                    painter.rect_stroke(
                        rect,
                        2.0,
                        egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
                        egui::StrokeKind::Outside,
                    );

                    let crop_area_fraction = azgaar_crop_view.map(|c| c.area_fraction());
                    if let Some(crop_rect) = azgaar_crop_view.map(|c| c.preview_rect(rect)) {
                        painter.rect_filled(
                            crop_rect,
                            0.0,
                            egui::Color32::from_rgba_premultiplied(255, 200, 50, 70),
                        );
                        painter.rect_stroke(
                            crop_rect,
                            0.0,
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 220, 50)),
                            egui::StrokeKind::Outside,
                        );
                    }

                    ui.add_space(6.0);
                    ui.label("Selected region");
                    let (zoom_response, zoom_painter) = ui.allocate_painter(
                        egui::vec2(preview_w, preview_w),
                        egui::Sense::hover(),
                    );
                    let zoom_rect = zoom_response.rect;
                    if let Some(tex) = &state.crop_preview_texture {
                        zoom_painter.image(
                            tex.id(),
                            zoom_rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    } else if let (Some(tex), Some(crop)) =
                        (&state.preview_texture, azgaar_crop_view)
                    {
                        zoom_painter.image(tex.id(), zoom_rect, crop.uv_rect(), egui::Color32::WHITE);
                    } else {
                        zoom_painter.rect_filled(zoom_rect, 2.0, egui::Color32::from_rgb(50, 70, 45));
                    }
                    zoom_painter.rect_stroke(
                        zoom_rect,
                        2.0,
                        egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
                        egui::StrokeKind::Outside,
                    );

                    let coverage_area = crop_area_fraction.unwrap_or(0.0) * 100.0;
                    let local_cells =
                        azgaar_crop_view.map(|c| c.side).unwrap_or(state.azgaar_region_cells);
                    ui.label(format!(
                        "{src_w}×{src_h} source  |  {:.0}×{:.0} km crop  |  {coverage_area:.1}% area",
                        local_cells as f32 * state.conditioning_scale_km,
                        local_cells as f32 * state.conditioning_scale_km,
                    ));
                    if local_cells != state.azgaar_region_cells {
                        ui.small(format!(
                            "Requested {} cells, clamped to {} to fit the current {}×{} source.",
                            state.azgaar_region_cells, local_cells, src_w, src_h,
                        ));
                    }
                } else {
                    ui.label("(run once to see world preview)");
                }

                if !running {
                    if let Some((w, h)) = raster_size {
                        let max_cells = max_azgaar_crop_side(w, h).unwrap_or(0);
                        if let Some(whole_cells) = AZGAAR_REGION_CELL_OPTIONS
                            .iter()
                            .copied()
                            .filter(|&c| c <= max_cells)
                            .last()
                        {
                            if ui
                                .button(format!("⛶ Whole World ({whole_cells} cells)"))
                                .on_hover_text(
                                    "Center crop and set cells to the largest available square.",
                                )
                                .clicked()
                            {
                                state.azgaar_region_cells = whole_cells;
                                state.azgaar_center_x = 0.5;
                                state.azgaar_center_y = 0.5;
                            }
                        }
                    }
                }

                let preview_mode = azgaar_crop.map(generated_preview_mode_for_crop);
                let preview_key = build_preview_key(
                    &build_job_config(state),
                    state.conditioning_raster_size,
                    azgaar_crop,
                    preview_mode,
                );
                let preview_is_current =
                    state.generated_preview_key.as_deref() == Some(preview_key.as_str());

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !state.is_busy(),
                            egui::Button::new("Generate Selected Diffusion Preview"),
                        )
                        .clicked()
                    {
                        preview_requested = true;
                    }
                    if preview_is_current {
                        ui.small("Generated previews match the current region.");
                    } else if state.generated_preview_key.is_some() {
                        ui.small("Generated previews are stale for the current region/settings.");
                    }
                });

                if let Some(tex) = &state.generated_overview_texture {
                    let size = tex.size_vec2();
                    let tex_aspect = (size.x / size.y).max(0.01);
                    ui.add_space(4.0);
                    ui.label("Generated world overview");
                    ui.add(egui::Image::new((
                        tex.id(),
                        egui::vec2(preview_w, (preview_w / tex_aspect).max(40.0)),
                    )));
                }

                if let Some(tex) = &state.generated_selected_texture {
                    ui.add_space(6.0);
                    ui.label(
                        state
                            .generated_selected_mode
                            .unwrap_or(GeneratedPreviewMode::Coarse)
                            .label(),
                    );
                    ui.add(egui::Image::new((tex.id(), egui::vec2(preview_w, preview_w))));
                }
            }

            ui.add_space(8.0);
            ui.heading("Runtime");
            ui.separator();

            egui::Grid::new("diffusion_runtime")
                .num_columns(2)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Device")
                        .on_hover_text("For ROCm-backed PyTorch, use Auto or GPU. PyTorch exposes ROCm through the `cuda` device string.");
                    egui::ComboBox::from_id_salt("diffusion_device")
                        .selected_text(state.device.label())
                        .show_ui(ui, |ui| {
                            ui.add_enabled_ui(!running, |ui| {
                                ui.selectable_value(&mut state.device, DiffusionDevice::Auto, DiffusionDevice::Auto.label());
                                ui.selectable_value(&mut state.device, DiffusionDevice::Gpu, DiffusionDevice::Gpu.label());
                                ui.selectable_value(&mut state.device, DiffusionDevice::Cpu, DiffusionDevice::Cpu.label());
                            });
                        });
                    ui.end_row();

                    ui.label("Dtype");
                    egui::ComboBox::from_id_salt("diffusion_dtype")
                        .selected_text(state.dtype.label())
                        .show_ui(ui, |ui| {
                            ui.add_enabled_ui(!running, |ui| {
                                ui.selectable_value(&mut state.dtype, DiffusionDtype::Fp32, "fp32");
                                ui.selectable_value(&mut state.dtype, DiffusionDtype::Bf16, "bf16");
                                ui.selectable_value(&mut state.dtype, DiffusionDtype::Fp16, "fp16");
                            });
                        });
                    ui.end_row();

                    text_grid_row(ui, "Seed", &mut state.seed, "Optional", !running);
                    text_grid_row(
                        ui,
                        "Batch sizes",
                        &mut state.batch_size,
                        "Example: 1,2,4,8",
                        !running,
                    );
                    text_grid_row(
                        ui,
                        "Cache size",
                        &mut state.cache_size,
                        "Example: 1G",
                        !running,
                    );
                    text_grid_row(
                        ui,
                        "SNR",
                        &mut state.snr,
                        "0.2,0.2,1.0,0.2,1.0",
                        !running,
                    );
                    text_grid_row(
                        ui,
                        "Extra env",
                        &mut state.extra_env,
                        "HSA_OVERRIDE_GFX_VERSION=11.0.0",
                        !running,
                    );

                    ui.label("torch.compile");
                    ui.add_enabled_ui(!running, |ui| {
                        ui.checkbox(&mut state.use_compile, "")
                            .on_hover_text("Leave disabled for the safest first pass on ROCm.");
                    });
                    ui.end_row();
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let can_start = !state.is_busy();
                if ui
                    .add_enabled(can_start, egui::Button::new("Probe Runtime"))
                    .clicked()
                {
                    probe_requested = true;
                }

                if ui
                    .add_enabled(can_start, egui::Button::new("Generate & Load"))
                    .clicked()
                {
                    start_requested = true;
                }

                if matches!(
                    state.run_state,
                    DiffusionRunState::Running { finished: true, .. }
                ) && ui.button("Clear Log").clicked()
                {
                    clear_requested = true;
                }
            });

            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(
                    "This first integration pass runs the upstream CLI in the background, then bakes the generated GeoTIFF into the existing landscape tile format.",
                )
                .small()
                .color(egui::Color32::GRAY),
            );
        });

    draw_run_log(ui, state);

    if let Some(target) = pick_target {
        spawn_file_picker(state, target);
    }
    if preview_requested {
        start_preview(state);
    }
    if probe_requested {
        start_probe(state);
    }
    if start_requested {
        start_run(state);
    }
    if clear_requested {
        state.run_state = DiffusionRunState::Idle;
    }
}

fn draw_run_log(ui: &mut egui::Ui, state: &DiffusionPanelState) {
    let Some((kind, finished, error, reloaded, log_lines, progress, stage)) =
        (match &state.run_state {
            DiffusionRunState::Running {
                kind,
                finished,
                error,
                reloaded,
                log_lines,
                progress,
                stage,
                ..
            } => Some((
                *kind,
                *finished,
                error.as_ref(),
                *reloaded,
                log_lines,
                *progress,
                stage.as_str(),
            )),
            _ => None,
        })
    else {
        return;
    };

    ui.add_space(8.0);
    ui.heading(kind.title());
    ui.separator();

    if !finished {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(if stage.is_empty() {
                kind.running_label()
            } else {
                stage
            });
        });
        // Progress bar — only show once there's meaningful progress.
        if progress > 0.01 {
            ui.add(
                egui::ProgressBar::new(progress)
                    .show_percentage()
                    .animate(true),
            );
        } else {
            // Indeterminate pulse before first stage marker arrives.
            ui.add(egui::ProgressBar::new(0.0).animate(true));
        }
    } else if let Some(err) = error {
        let summary = err.lines().next().unwrap_or(err.as_str());
        ui.colored_label(
            egui::Color32::RED,
            format!("{}: {summary}", kind.failure_label()),
        )
        .on_hover_text(err.as_str());
    } else if reloaded {
        ui.colored_label(
            egui::Color32::GREEN,
            "Diffusion pipeline complete — terrain updated!",
        );
        ui.label(
            egui::RichText::new("Use File → Save Landscape… to persist this generated terrain.")
                .small()
                .color(egui::Color32::GRAY),
        );
    } else if kind == DiffusionRunKind::Preview {
        ui.colored_label(egui::Color32::GREEN, "Diffusion previews are ready.");
    } else if kind == DiffusionRunKind::Probe {
        ui.colored_label(egui::Color32::GREEN, "Diffusion runtime probe succeeded.");
    } else {
        ui.label("Diffusion pipeline finished.");
    }

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Log")
                .small()
                .color(egui::Color32::GRAY),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("📋 Copy log").clicked() {
                ui.ctx().output_mut(|o| {
                    o.commands
                        .push(egui::OutputCommand::CopyText(log_lines.join("\n")));
                });
            }
        });
    });

    egui::ScrollArea::vertical()
        .id_salt("diffusion_run_log")
        .max_height(240.0)
        .stick_to_bottom(!finished)
        .show(ui, |ui| {
            for line in log_lines {
                let color = if line.starts_with("✗") || line.contains("ERROR") {
                    egui::Color32::from_rgb(255, 180, 60)
                } else if line.starts_with('✓') {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::LIGHT_GRAY
                };
                ui.colored_label(color, line);
            }
        });
}

fn start_run(state: &mut DiffusionPanelState) {
    if state.is_busy() {
        return;
    }

    if let Err(err) = validate_pipeline_settings(state) {
        set_finished_error(state, DiffusionRunKind::Generate, err);
        return;
    }

    let config = build_job_config(state);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || run_pipeline(config, tx));

    state.run_state = DiffusionRunState::Running {
        kind: DiffusionRunKind::Generate,
        log_lines: Vec::new(),
        rx: Mutex::new(rx),
        finished: false,
        error: None,
        reloaded: false,
        progress: 0.0,
        stage: "Starting…".into(),
    };
    // Force texture reload after run completes.
    state.preview_loaded_for = None;
}

fn start_preview(state: &mut DiffusionPanelState) {
    if state.is_busy() {
        return;
    }

    if let Err(err) = validate_preview_settings(state) {
        set_finished_error(state, DiffusionRunKind::Preview, err);
        return;
    }

    let config = build_job_config(state);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || run_preview(config, tx));

    state.run_state = DiffusionRunState::Running {
        kind: DiffusionRunKind::Preview,
        log_lines: Vec::new(),
        rx: Mutex::new(rx),
        finished: false,
        error: None,
        reloaded: false,
        progress: 0.0,
        stage: "Preparing previews…".into(),
    };
    state.preview_loaded_for = None;
    state.preview_texture = None;
}

fn start_probe(state: &mut DiffusionPanelState) {
    if state.is_busy() {
        return;
    }

    if let Err(err) = validate_probe_settings(state) {
        set_finished_error(state, DiffusionRunKind::Probe, err);
        return;
    }

    let config = build_job_config(state);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || run_probe(config, tx));

    state.run_state = DiffusionRunState::Running {
        kind: DiffusionRunKind::Probe,
        log_lines: Vec::new(),
        rx: Mutex::new(rx),
        finished: false,
        error: None,
        reloaded: false,
        progress: 0.0,
        stage: "Probing…".into(),
    };
}

fn set_finished_error(state: &mut DiffusionPanelState, kind: DiffusionRunKind, err: String) {
    state.run_state = DiffusionRunState::Running {
        kind,
        log_lines: vec![format!("✗ {err}")],
        rx: Mutex::new(mpsc::channel().1),
        finished: true,
        error: Some(err),
        reloaded: false,
        progress: 0.0,
        stage: String::new(),
    };
}

fn build_job_config(state: &DiffusionPanelState) -> DiffusionJobConfig {
    DiffusionJobConfig {
        workflow: state.workflow,
        repo_path: state.repo_path.clone(),
        python_bin: state.python_bin.clone(),
        model_path: state.model_path.clone(),
        azgaar_json_path: state.azgaar_json_path.clone(),
        azgaar_region_cells: state.azgaar_region_cells,
        azgaar_center_x: state.azgaar_center_x,
        azgaar_center_y: state.azgaar_center_y,
        conditioning_dir: state.conditioning_dir.clone(),
        conditioning_scale_km: state.conditioning_scale_km,
        working_dir: state.working_dir.clone(),
        tile_output_dir: state.tile_output_dir.clone(),
        world_scale: state.world_scale,
        final_height_span_m: state.final_height_span_m,
        smooth_sigma: state.smooth_sigma,
        device: state.device,
        dtype: state.dtype,
        snr: state.snr.clone(),
        batch_size: state.batch_size.clone(),
        cache_size: state.cache_size.clone(),
        seed: state.seed.clone(),
        use_compile: state.use_compile,
        extra_env: state.extra_env.clone(),
    }
}

fn validate_probe_settings(state: &DiffusionPanelState) -> Result<(), String> {
    if state.repo_path.trim().is_empty() {
        return Err("Terrain Diffusion repo path is required.".into());
    }
    if state.python_bin.trim().is_empty() {
        return Err("Python executable is required.".into());
    }
    Ok(())
}

fn validate_preview_settings(state: &DiffusionPanelState) -> Result<(), String> {
    validate_probe_settings(state)?;
    if state.model_path.trim().is_empty() {
        return Err("Model path is required.".into());
    }
    if state.working_dir.trim().is_empty() {
        return Err("Work directory is required.".into());
    }
    match state.workflow {
        DiffusionWorkflow::AzgaarJson => {
            if state.azgaar_json_path.trim().is_empty() {
                return Err("Azgaar full JSON export path is required.".into());
            }
            if !AZGAAR_REGION_CELL_OPTIONS.contains(&state.azgaar_region_cells) {
                return Err(format!(
                    "Azgaar region must be one of {} cells per side.",
                    AZGAAR_REGION_CELL_OPTIONS
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        DiffusionWorkflow::ConditioningFolder => {
            return Err("Generated diffusion previews are currently implemented for Azgaar JSON workflow only.".into());
        }
    }
    Ok(())
}

fn validate_pipeline_settings(state: &DiffusionPanelState) -> Result<(), String> {
    validate_probe_settings(state)?;
    if state.model_path.trim().is_empty() {
        return Err("Model path is required.".into());
    }
    if state.tile_output_dir.trim().is_empty() {
        return Err("Tile output directory is required.".into());
    }
    if state.working_dir.trim().is_empty() {
        return Err("Work directory is required.".into());
    }
    if state.world_scale <= 0.0 {
        return Err("World scale must be greater than zero.".into());
    }
    if state.final_height_span_m <= 0.0 {
        return Err("Final height span must be greater than zero.".into());
    }
    match state.workflow {
        DiffusionWorkflow::AzgaarJson => {
            if state.azgaar_json_path.trim().is_empty() {
                return Err("Azgaar full JSON export path is required.".into());
            }
            if !AZGAAR_REGION_CELL_OPTIONS.contains(&state.azgaar_region_cells) {
                return Err(format!(
                    "Azgaar region must be one of {} cells per side.",
                    AZGAAR_REGION_CELL_OPTIONS
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        DiffusionWorkflow::ConditioningFolder => {
            if state.conditioning_dir.trim().is_empty() {
                return Err("Conditioning folder path is required.".into());
            }
        }
    }
    Ok(())
}

fn poll_file_picker(state: &mut DiffusionPanelState) {
    let result = if let DiffusionRunState::PickingFile { rx, .. } = &state.run_state {
        rx.lock().ok().and_then(|guard| guard.try_recv().ok())
    } else {
        return;
    };

    let Some(maybe_path) = result else {
        return;
    };

    let target = match std::mem::replace(&mut state.run_state, DiffusionRunState::Idle) {
        DiffusionRunState::PickingFile { target, .. } => target,
        other => {
            state.run_state = other;
            return;
        }
    };

    if let Some(path) = maybe_path {
        let value = path.display().to_string();
        match target {
            DiffusionPickTarget::RepoPath => {
                state.repo_path = value;
                state.python_bin = detect_python_bin(Path::new(&state.repo_path))
                    .display()
                    .to_string();
            }
            DiffusionPickTarget::AzgaarJson => state.azgaar_json_path = value,
            DiffusionPickTarget::ConditioningDir => state.conditioning_dir = value,
            DiffusionPickTarget::WorkingDir => state.working_dir = value,
            DiffusionPickTarget::TileOutputDir => state.tile_output_dir = value,
        }
    }
}

/// Update progress and stage label from a single log line.
///
/// Recognises:
/// - Bake level lines:  `"Level 0: 45%  (18/40)"`
/// - tqdm lines:        `"  45%|"` or `" 45%|█"` (diffusion inference)
/// - Stage markers:     key phrases that appear in the pipeline log
fn update_progress_from_log(line: &str, progress: &mut f32, stage: &mut String) {
    // Stage detection — update whenever a key phrase is seen.
    if line.contains("Converting Azgaar export") {
        *stage = "Converting world data…".into();
        *progress = 0.02;
    } else if line.contains("Cropping") && line.contains("conditioning") {
        *stage = "Cropping region…".into();
        *progress = 0.18;
    } else if line.contains("Baking") && line.contains("tiles") {
        *stage = "Baking tiles…".into();
        *progress = 0.80;
    } else if line.contains("terrain_diffusion") && line.contains("generate") {
        *stage = "AI inference…".into();
        *progress = 0.20;
    } else if line.contains("Sea level") {
        *stage = "Writing metadata…".into();
    }

    // Percentage extraction — works for both bake "Level N: X%" and tqdm "X%|".
    // Find the last `%` in the line and look backwards for the number before it.
    if let Some(pct_pos) = line.rfind('%') {
        let before = &line[..pct_pos];
        // Walk backwards past any whitespace to find the number.
        let num_str: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if let Ok(pct) = num_str.parse::<f32>() {
            let pct = pct.clamp(0.0, 100.0);
            // Map within-stage percentage to the overall pipeline range.
            let new_p = if stage.starts_with("Baking") {
                // Bake is roughly 80..100% of the overall pipeline.
                0.80 + pct * 0.002 // 0.80 + (pct/100)*0.20
            } else if stage.starts_with("AI") {
                // Inference is roughly 20..80%.
                0.20 + pct * 0.006 // 0.20 + (pct/100)*0.60
            } else {
                *progress // don't regress for other stages
            };
            if new_p > *progress {
                *progress = new_p.min(0.99);
            }
        }
    }
}

/// Parse "Output shape: WxH" from a line of azgaar-to-tiff output.
fn parse_output_shape(line: &str) -> Option<(u32, u32)> {
    let idx = line.find("Output shape: ")?;
    let rest = &line[idx + "Output shape: ".len()..];
    let (w_str, rest) = rest.split_once('x')?;
    let h_str = rest.split_ascii_whitespace().next()?;
    let w = w_str.trim().parse().ok()?;
    let h = h_str.trim().parse().ok()?;
    Some((w, h))
}

/// Try to read the pixel dimensions of the conditioning heightmap TIFF.
/// Returns `None` if the file doesn't exist or can't be read.
/// Map a grayscale height value [0..255] to a terrain colour ramp.
/// Ramp: deep ocean → shallow → beach → lowland → highland → snow.
fn terrain_color_ramp(g: u8) -> egui::Color32 {
    let t = g as f32 / 255.0;
    // Colour stops: (threshold, R, G, B)
    const STOPS: &[(f32, u8, u8, u8)] = &[
        (0.00, 10, 20, 100),   // deep ocean
        (0.30, 30, 80, 180),   // shallow
        (0.38, 180, 175, 120), // beach
        (0.42, 60, 130, 40),   // lowland green
        (0.60, 80, 100, 50),   // highland
        (0.75, 100, 90, 70),   // rocky
        (0.88, 220, 220, 215), // high snow
        (1.00, 255, 255, 255), // peak
    ];
    let mut lo = STOPS[0];
    let mut hi = STOPS[STOPS.len() - 1];
    for w in STOPS.windows(2) {
        if t >= w[0].0 && t <= w[1].0 {
            lo = w[0];
            hi = w[1];
            break;
        }
    }
    let span = hi.0 - lo.0;
    let f = if span > 0.0 { (t - lo.0) / span } else { 0.0 };
    let lerp = |a: u8, b: u8| (a as f32 + f * (b as f32 - a as f32)) as u8;
    egui::Color32::from_rgb(lerp(lo.1, hi.1), lerp(lo.2, hi.2), lerp(lo.3, hi.3))
}

fn probe_conditioning_tiff_size(conditioning_dir: &str) -> Option<(u32, u32)> {
    let path = PathBuf::from(conditioning_dir).join("heightmap.tif");
    if !path.exists() {
        return None;
    }
    let f = std::fs::File::open(&path).ok()?;
    let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).ok()?;
    dec.dimensions().ok()
}

fn poll_run(
    state: &mut DiffusionPanelState,
    active_config: &TerrainConfig,
    active_library: &MaterialLibrary,
    reload_tx: &mut MessageWriter<ReloadTerrainRequest>,
) {
    loop {
        let msg = if let DiffusionRunState::Running { rx, .. } = &state.run_state {
            rx.lock().ok().and_then(|guard| guard.try_recv().ok())
        } else {
            return;
        };

        match msg {
            None => break,
            Some(DiffusionMsg::Log(line)) => {
                if let DiffusionRunState::Running {
                    log_lines,
                    progress,
                    stage,
                    ..
                } = &mut state.run_state
                {
                    update_progress_from_log(&line, progress, stage);
                    log_lines.push(line);
                }
            }
            Some(DiffusionMsg::Done(result)) => {
                if let DiffusionRunState::Running {
                    kind,
                    log_lines,
                    finished,
                    error,
                    reloaded,
                    progress,
                    stage,
                    ..
                } = &mut state.run_state
                {
                    match result {
                        Ok(DiffusionOutcome::Generated(tile_root)) => {
                            log_lines.push("✓ Diffusion run complete — reloading terrain…".into());
                            *progress = 1.0;
                            *stage = "Done".into();
                            trigger_reload(
                                &tile_root,
                                state.world_scale,
                                state.final_height_span_m,
                                state.clipmap_levels,
                                active_config,
                                active_library,
                                reload_tx,
                            );
                            *reloaded = true;
                        }
                        Ok(DiffusionOutcome::Probed) => {
                            if *kind == DiffusionRunKind::Probe {
                                log_lines.push("✓ Diffusion runtime probe complete.".into());
                                *progress = 1.0;
                                *stage = "Done".into();
                            }
                        }
                        Ok(DiffusionOutcome::Previewed(artifacts)) => {
                            log_lines.push("✓ Diffusion selected preview generated.".into());
                            *progress = 1.0;
                            *stage = "Done".into();
                            state.generated_overview_image = artifacts.overview_image;
                            state.generated_selected_image = Some(artifacts.selected_image);
                            state.generated_selected_mode = Some(artifacts.selected_mode);
                            state.generated_preview_key = Some(artifacts.preview_key);
                            state.generated_preview_revision =
                                state.generated_preview_revision.wrapping_add(1);
                            state.generated_overview_texture = None;
                            state.generated_selected_texture = None;
                        }
                        Err(err) => {
                            log_lines.push(format!("✗ {err}"));
                            *error = Some(err);
                        }
                    }
                    *finished = true;
                }
                // After marking done, try to extract conditioning raster size from log.
                if let DiffusionRunState::Running {
                    log_lines, error, ..
                } = &state.run_state
                {
                    if error.is_none() {
                        if let Some(sz) = log_lines.iter().rev().find_map(|l| parse_output_shape(l))
                        {
                            state.conditioning_raster_size = Some(sz);
                        }
                    }
                }
                break;
            }
        }
    }
}

fn spawn_file_picker(state: &mut DiffusionPanelState, target: DiffusionPickTarget) {
    if state.is_busy() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new();
        dialog = match target {
            DiffusionPickTarget::RepoPath
            | DiffusionPickTarget::ConditioningDir
            | DiffusionPickTarget::WorkingDir
            | DiffusionPickTarget::TileOutputDir => dialog.set_title("Select folder"),
            DiffusionPickTarget::AzgaarJson => dialog
                .set_title("Select Azgaar full JSON export")
                .add_filter("JSON", &["json"]),
        };
        let result = match target {
            DiffusionPickTarget::RepoPath
            | DiffusionPickTarget::ConditioningDir
            | DiffusionPickTarget::WorkingDir
            | DiffusionPickTarget::TileOutputDir => dialog.pick_folder(),
            DiffusionPickTarget::AzgaarJson => dialog.pick_file(),
        };
        let _ = tx.send(result);
    });
    state.run_state = DiffusionRunState::PickingFile {
        rx: Mutex::new(rx),
        target,
    };
}

fn run_pipeline(config: DiffusionJobConfig, tx: mpsc::Sender<DiffusionMsg>) {
    let result = run_pipeline_inner(&config, &tx).map(DiffusionOutcome::Generated);
    let _ = tx.send(DiffusionMsg::Done(result));
}

fn run_probe(config: DiffusionJobConfig, tx: mpsc::Sender<DiffusionMsg>) {
    let result = run_probe_inner(&config, &tx).map(|()| DiffusionOutcome::Probed);
    let _ = tx.send(DiffusionMsg::Done(result));
}

fn run_preview(config: DiffusionJobConfig, tx: mpsc::Sender<DiffusionMsg>) {
    let result = run_preview_inner(&config, &tx).map(DiffusionOutcome::Previewed);
    let _ = tx.send(DiffusionMsg::Done(result));
}

fn run_probe_inner(
    config: &DiffusionJobConfig,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<(), String> {
    let repo_path = PathBuf::from(&config.repo_path);
    if !repo_path.exists() {
        return Err(format!(
            "Terrain Diffusion repo not found: {}",
            repo_path.display()
        ));
    }
    let package_dir = repo_path.join("terrain_diffusion");
    if !package_dir.exists() {
        return Err(format!(
            "terrain_diffusion package directory not found under {}",
            repo_path.display()
        ));
    }

    send_log(
        tx,
        format!(
            "Probing terrain-diffusion runtime from {}",
            repo_path.display()
        ),
    );

    run_logged_command(
        &config.python_bin,
        &["--version".to_string()],
        &repo_path,
        &config.extra_env,
        tx,
    )?;
    run_logged_command(
        &config.python_bin,
        &[
            "-c".to_string(),
            r#"import os, sys; print(f"python={sys.executable}"); print(f"cwd={os.getcwd()}"); print(f"venv={os.environ.get('VIRTUAL_ENV', '<unset>')}")"#.to_string(),
        ],
        &repo_path,
        &config.extra_env,
        tx,
    )?;
    run_logged_command(
        &config.python_bin,
        &[
            "-c".to_string(),
            r#"import importlib.util, terrain_diffusion; spec = importlib.util.find_spec("terrain_diffusion"); locations = list(spec.submodule_search_locations or []); print(f"terrain_diffusion_origin={spec.origin}"); print(f"terrain_diffusion_path={locations[0] if locations else '<none>'}")"#.to_string(),
        ],
        &repo_path,
        &config.extra_env,
        tx,
    )?;
    run_logged_command(
        &config.python_bin,
        &[
            "-c".to_string(),
            r#"import torch; print(f"torch={torch.__version__}"); print(f"hip={getattr(torch.version, 'hip', None)}"); print(f"cuda_available={torch.cuda.is_available()}"); print(f"device_count={torch.cuda.device_count()}"); print(f"device0={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "device0=<none>")"#.to_string(),
        ],
        &repo_path,
        &config.extra_env,
        tx,
    )?;
    run_logged_command(
        &config.python_bin,
        &[
            "-m".to_string(),
            "terrain_diffusion".to_string(),
            "--help".to_string(),
        ],
        &repo_path,
        &config.extra_env,
        tx,
    )?;

    Ok(())
}

fn run_preview_inner(
    config: &DiffusionJobConfig,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<DiffusionPreviewArtifacts, String> {
    let repo_path = PathBuf::from(&config.repo_path);
    if !repo_path.exists() {
        return Err(format!(
            "Terrain Diffusion repo not found: {}",
            repo_path.display()
        ));
    }

    let working_dir = PathBuf::from(&config.working_dir);
    std::fs::create_dir_all(&working_dir).map_err(|e| {
        format!(
            "Failed to create work directory {}: {e}",
            working_dir.display()
        )
    })?;
    let working_dir = working_dir.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve work directory {}: {e}",
            working_dir.display()
        )
    })?;

    let conditioning_dir = working_dir.join("conditioning");
    match config.workflow {
        DiffusionWorkflow::AzgaarJson => {
            ensure_preview_azgaar_conditioning(config, &repo_path, &conditioning_dir, tx)?;
        }
        DiffusionWorkflow::ConditioningFolder => {
            return Err(
                "Generated diffusion previews are currently implemented for Azgaar JSON workflow only."
                    .into(),
            );
        }
    }

    let conditioning_dir_str = conditioning_dir.to_string_lossy().into_owned();
    let Some((source_width, source_height)) = probe_conditioning_tiff_size(&conditioning_dir_str)
    else {
        return Err(format!(
            "Failed to read conditioning TIFF dimensions from {}",
            conditioning_dir.display()
        ));
    };
    let crop = compute_azgaar_crop_window(
        source_width,
        source_height,
        config.azgaar_region_cells,
        config.azgaar_center_x,
        config.azgaar_center_y,
    )
    .ok_or_else(|| {
        format!(
            "Conditioning TIFF dimensions are invalid for preview: {}x{}",
            source_width, source_height
        )
    })?;
    let selected_mode = generated_preview_mode_for_crop(crop);

    let preview_dir = working_dir.join("preview");
    std::fs::create_dir_all(&preview_dir).map_err(|e| {
        format!(
            "Failed to create preview directory {}: {e}",
            preview_dir.display()
        )
    })?;
    let selected_image = preview_dir.join("generated_selected.png");

    send_log(
        tx,
        "Generating fast diffusion preview for the selected region…",
    );
    run_preview_command(
        config,
        &repo_path,
        &conditioning_dir,
        crop,
        &selected_image,
        tx,
    )?;

    Ok(DiffusionPreviewArtifacts {
        overview_image: None,
        selected_image,
        selected_mode,
        preview_key: build_preview_key(
            config,
            Some((source_width, source_height)),
            Some(crop),
            Some(selected_mode),
        ),
    })
}

fn run_pipeline_inner(
    config: &DiffusionJobConfig,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<PathBuf, String> {
    let repo_path = PathBuf::from(&config.repo_path);
    if !repo_path.exists() {
        return Err(format!(
            "Terrain Diffusion repo not found: {}",
            repo_path.display()
        ));
    }

    let working_dir = PathBuf::from(&config.working_dir);
    std::fs::create_dir_all(&working_dir).map_err(|e| {
        format!(
            "Failed to create work directory {}: {e}",
            working_dir.display()
        )
    })?;
    let working_dir = working_dir.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve work directory {}: {e}",
            working_dir.display()
        )
    })?;

    let tile_output_dir = PathBuf::from(&config.tile_output_dir);
    if let Some(parent) = tile_output_dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create parent directory for {}: {e}",
                tile_output_dir.display()
            )
        })?;
    }
    let tile_output_dir = tile_output_dir
        .canonicalize()
        .unwrap_or_else(|_| tile_output_dir.clone());

    let conditioning_dir = match config.workflow {
        DiffusionWorkflow::AzgaarJson => working_dir.join("conditioning"),
        DiffusionWorkflow::ConditioningFolder => PathBuf::from(&config.conditioning_dir)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(&config.conditioning_dir)),
    };
    let cropped_conditioning_dir = working_dir.join("conditioning_region");
    let generated_tiff = working_dir.join("generated_heightmap.tif");

    let export_conditioning_dir = match config.workflow {
        DiffusionWorkflow::AzgaarJson => {
            std::fs::create_dir_all(&conditioning_dir).map_err(|e| {
                format!(
                    "Failed to create conditioning directory {}: {e}",
                    conditioning_dir.display()
                )
            })?;
            send_log(
                tx,
                format!(
                    "Converting Azgaar export to conditioning TIFFs in {}",
                    conditioning_dir.display()
                ),
            );
            let scale = format!("{:.3}", config.conditioning_scale_km);
            let args = vec![
                "-m".to_string(),
                "terrain_diffusion".to_string(),
                "azgaar-to-tiff".to_string(),
                config.azgaar_json_path.clone(),
                conditioning_dir.display().to_string(),
                "--scale".to_string(),
                scale,
            ];
            run_logged_command(&config.python_bin, &args, &repo_path, &config.extra_env, tx)?;
            crop_azgaar_conditioning_region(
                config,
                &repo_path,
                &conditioning_dir,
                &cropped_conditioning_dir,
                tx,
            )?;
            cropped_conditioning_dir.clone()
        }
        DiffusionWorkflow::ConditioningFolder => {
            if !conditioning_dir.exists() {
                return Err(format!(
                    "Conditioning folder not found: {}",
                    conditioning_dir.display()
                ));
            }
            conditioning_dir.clone()
        }
    };

    send_log(
        tx,
        format!(
            "Generating diffusion heightmap with model {}",
            config.model_path
        ),
    );
    validate_synthetic_climate_assets(&repo_path)?;
    let mut args = vec![
        "-m".to_string(),
        "terrain_diffusion".to_string(),
        "tiff-export".to_string(),
        config.model_path.clone(),
        export_conditioning_dir.display().to_string(),
        generated_tiff.display().to_string(),
        "--snr".to_string(),
        config.snr.clone(),
        "--cache-size".to_string(),
        config.cache_size.clone(),
        "--batch-size".to_string(),
        config.batch_size.clone(),
        "--dtype".to_string(),
        config.dtype.label().to_string(),
    ];
    if config.use_compile {
        args.push("--compile".to_string());
    } else {
        args.push("--no-compile".to_string());
    }
    if let Some(device) = config.device.cli_value() {
        args.push("--device".to_string());
        args.push(device.to_string());
    }
    if !config.seed.trim().is_empty() {
        args.push("--seed".to_string());
        args.push(config.seed.trim().to_string());
    }
    run_logged_command(&config.python_bin, &args, &repo_path, &config.extra_env, tx)?;

    send_log(
        tx,
        format!(
            "Baking {} into landscape tiles at {}",
            generated_tiff.display(),
            tile_output_dir.display()
        ),
    );
    let base_height_scale = config.final_height_span_m / config.world_scale;
    let bake_config = BakeConfig {
        height_path: generated_tiff,
        bump_path: None,
        output_dir: tile_output_dir.clone(),
        height_scale: base_height_scale,
        bump_scale: None,
        world_scale: config.world_scale,
        tile_size: 256,
        flip_green: false,
        smooth_sigma: config.smooth_sigma,
        // Azgaar-to-tiff encodes heights as raw INT16 metres; i16(0) = 0 m = sea level.
        // After tiff_result_to_f32 this maps to (0 − i16::MIN) / u16::MAX ≈ 0.5.
        sea_level_decoded: Some((0.0_f32 - i16::MIN as f32) / u16::MAX as f32),
    };
    bevy_landscape::bake::bake_heightmap(bake_config, |line| send_log(tx, line))?;

    Ok(tile_output_dir)
}

fn run_logged_command(
    program: &str,
    args: &[String],
    cwd: &Path,
    extra_env: &str,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<(), String> {
    send_log(
        tx,
        format!("$ (cd {} && {} {})", cwd.display(), program, args.join(" ")),
    );

    let mut child = Command::new(program);
    child
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    apply_python_environment(&mut child, program);
    apply_extra_environment(&mut child, extra_env, tx)?;

    let mut child = child
        .spawn()
        .map_err(|e| format!("Failed to start `{program}`: {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| format!("Failed to capture stdout for `{program}`"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| format!("Failed to capture stderr for `{program}`"))?;

    let tx_stdout = tx.clone();
    let out_handle = std::thread::spawn(move || stream_reader(stdout, tx_stdout));
    let tx_stderr = tx.clone();
    let err_handle = std::thread::spawn(move || stream_reader(stderr, tx_stderr));

    let status = child
        .wait()
        .map_err(|e| format!("Failed while waiting for `{program}`: {e}"))?;

    let _ = out_handle.join();
    let _ = err_handle.join();

    if !status.success() {
        return Err(format!(
            "Command failed with status {status}: {program} {}",
            args.join(" ")
        ));
    }
    Ok(())
}

fn run_preview_command(
    config: &DiffusionJobConfig,
    repo_path: &Path,
    conditioning_dir: &Path,
    crop: AzgaarCropWindow,
    selected_image: &Path,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<(), String> {
    let script = r#"
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from terrain_diffusion.common.cli_helpers import parse_cache_size
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.inference.tiff_export import CHANNEL_FILES, PADDING, _load_and_pad
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.models.edm_unet import EDMUnet2D


def downsample_mean(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.asarray(arr, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None]
    return torch.nn.functional.avg_pool2d(
        tensor,
        kernel_size=stride,
        stride=stride,
        ceil_mode=True,
    )[0, 0].cpu().numpy()


def save_relief(arr: np.ndarray, resolution_m: float, out_path: Path, relief: float) -> None:
    rgb = get_relief_map(arr, None, None, None, resolution=resolution_m, relief=relief)
    plt.imsave(out_path, np.clip(rgb, 0.0, 1.0))


def upsample_preview(arr: np.ndarray, target_side: int) -> tuple[np.ndarray, float]:
    longest = max(arr.shape)
    if longest >= target_side:
        return arr, 1.0
    scale = target_side / longest
    out_h = max(1, int(round(arr.shape[0] * scale)))
    out_w = max(1, int(round(arr.shape[1] * scale)))
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None]
    upsampled = torch.nn.functional.interpolate(
        tensor,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return upsampled, scale


conditioning_dir = Path(sys.argv[1])
selected_image = Path(sys.argv[2])
model_path = sys.argv[3]
snr = sys.argv[4]
batch_size = sys.argv[5]
cache_size = sys.argv[6]
seed = None if sys.argv[7] == "" else int(sys.argv[7])
device = None if sys.argv[8] == "" else sys.argv[8]
dtype = None if sys.argv[9] == "fp32" else sys.argv[9]
selected_max_side = max(1, int(sys.argv[10]))
crop_x0 = int(sys.argv[11])
crop_y0 = int(sys.argv[12])
crop_side = int(sys.argv[13])
conditioning_scale_km = float(sys.argv[14])

if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

batch_sizes = [int(x) for x in batch_size.split(",")] if "," in batch_size else int(batch_size)
world_config = {
    **WorldPipeline.load_config(model_path),
    "seed": seed,
    "latents_batch_size": batch_sizes,
    "log_mode": "info",
    "torch_compile": False,
    "dtype": dtype,
    "caching_strategy": "direct",
    "cache_limit": parse_cache_size(cache_size),
}
world = WorldPipeline(**world_config)
world.coarse_model = EDMUnet2D.from_pretrained(model_path, subfolder=WorldPipeline.COARSE_MODEL_FOLDER)
world._apply_dtype_and_compile()
world.to(device)

if snr:
    snr_vals = [float(x.strip()) for x in snr.split(",")]
    world.set_cond_snr(snr_vals)

src_h = src_w = None
for filename, channel, internal_scale, default_value in CHANNEL_FILES:
    path = conditioning_dir / filename
    if not path.exists():
        continue
    with rasterio.open(path) as ds:
        if src_h is None:
            src_h, src_w = ds.height, ds.width
    padded = _load_and_pad(path, channel, internal_scale, default_value)
    world.set_custom_conditioning_import(channel, padded, 0, 0, default_value=default_value)

if src_h is None or src_w is None:
    raise RuntimeError(f"No conditioning TIFFs found in {conditioning_dir}")

world._init_tile_store(None, None, None, None)
world._init_conditioning()
world.coarse = world._build_coarse_stage()

with world:
    ci0, ci1 = PADDING + crop_y0, PADDING + crop_y0 + crop_side
    cj0, cj1 = PADDING + crop_x0, PADDING + crop_x0 + crop_side
    coarse_raw = world.coarse[:, ci0:ci1, cj0:cj1]
    coarse_elev = (coarse_raw[:-1] / (coarse_raw[-1:] + 1e-8))[0].detach().cpu().numpy()
    coarse_elev = np.sign(coarse_elev) * np.square(coarse_elev)

    crop_stride = max(1, math.ceil(max(coarse_elev.shape) / selected_max_side))
    selected_preview = downsample_mean(coarse_elev, crop_stride)
    selected_resolution = conditioning_scale_km * 1000.0 * crop_stride
    target_side = min(selected_max_side, max(256, max(selected_preview.shape)))
    selected_preview, display_scale = upsample_preview(selected_preview, target_side)
    selected_resolution /= display_scale
    print(
        f"Generated selected coarse preview from coarse field {coarse_elev.shape[1]}x{coarse_elev.shape[0]} "
        f"-> {selected_preview.shape[1]}x{selected_preview.shape[0]} stride={crop_stride}"
    )

    save_relief(selected_preview, selected_resolution, selected_image, relief=1.0)
"#;

    let args = vec![
        "-c".to_string(),
        script.to_string(),
        conditioning_dir.display().to_string(),
        selected_image.display().to_string(),
        config.model_path.clone(),
        config.snr.clone(),
        config.batch_size.clone(),
        config.cache_size.clone(),
        config.seed.trim().to_string(),
        config
            .device
            .cli_value()
            .map(str::to_string)
            .unwrap_or_default(),
        config.dtype.label().to_string(),
        GENERATED_PREVIEW_SELECTED_MAX_SIDE.to_string(),
        crop.x0.to_string(),
        crop.y0.to_string(),
        crop.side.to_string(),
        format!("{:.6}", config.conditioning_scale_km),
    ];
    run_logged_command(&config.python_bin, &args, repo_path, &config.extra_env, tx)
}

fn validate_synthetic_climate_assets(repo_path: &Path) -> Result<(), String> {
    let required = [
        "data/global/etopo_10m.tif",
        "data/global/wc2.1_10m_bio_1.tif",
        "data/global/wc2.1_10m_bio_4.tif",
        "data/global/wc2.1_10m_bio_12.tif",
        "data/global/wc2.1_10m_bio_15.tif",
    ];
    let missing: Vec<String> = required
        .iter()
        .map(|rel| repo_path.join(rel))
        .filter(|path| !path.exists())
        .map(|path| path.display().to_string())
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    Err(format!(
        "terrain-diffusion is missing required climate reference files:\n{}\nThe upstream CLI prompts interactively for WorldClim downloads, which the editor does not support. Install those files under `{}` before running `tiff-export`.",
        missing.join("\n"),
        repo_path.join("data/global").display()
    ))
}

fn crop_azgaar_conditioning_region(
    config: &DiffusionJobConfig,
    repo_path: &Path,
    source_dir: &Path,
    output_dir: &Path,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<(), String> {
    if output_dir.exists() {
        std::fs::remove_dir_all(output_dir).map_err(|e| {
            format!(
                "Failed to clear previous Azgaar crop directory {}: {e}",
                output_dir.display()
            )
        })?;
    }
    std::fs::create_dir_all(output_dir).map_err(|e| {
        format!(
            "Failed to create Azgaar crop directory {}: {e}",
            output_dir.display()
        )
    })?;

    send_log(
        tx,
        format!(
            "Cropping Azgaar conditioning to {}x{} cells around ({:.2}, {:.2})",
            config.azgaar_region_cells,
            config.azgaar_region_cells,
            config.azgaar_center_x,
            config.azgaar_center_y,
        ),
    );

    let script = r#"
from pathlib import Path
import shutil
import sys

import rasterio
from rasterio.windows import Window

source_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
requested_side = int(sys.argv[3])
center_x = float(sys.argv[4])
center_y = float(sys.argv[5])

files = [
    "heightmap.tif",
    "temperature.tif",
    "temperature_std.tif",
    "precipitation.tif",
    "precipitation_cv.tif",
]
existing = [name for name in files if (source_dir / name).exists()]
if not existing:
    raise SystemExit(f"No conditioning TIFFs found in {source_dir}")

with rasterio.open(source_dir / existing[0]) as ds:
    width = ds.width
    height = ds.height

min_dim = min(width, height)
max_side = 1 << (min_dim.bit_length() - 1)
if max_side < 2:
    raise SystemExit(
        f"Conditioning rasters are too small to crop for baking: {width}x{height}"
    )

side = min(requested_side, max_side)
if side != requested_side:
    print(
        f"Requested {requested_side} conditioning cells, clamped to {side} to fit source {width}x{height}"
    )

px = (width - 1) * center_x
py = (height - 1) * center_y
x0 = int(round(px - side / 2))
y0 = int(round(py - side / 2))
x0 = max(0, min(x0, width - side))
y0 = max(0, min(y0, height - side))
window = Window(x0, y0, side, side)
print(f"Cropping source {width}x{height} -> {side}x{side} at x={x0}, y={y0}")

for name in existing:
    src = source_dir / name
    dst = output_dir / name
    with rasterio.open(src) as ds:
        if ds.width != width or ds.height != height:
            raise SystemExit(
                f"Mismatched conditioning raster size for {src}: {ds.width}x{ds.height}, expected {width}x{height}"
            )
        profile = ds.profile.copy()
        profile.update(
            width=side,
            height=side,
            transform=ds.window_transform(window),
        )
        data = ds.read(1, window=window)
        with rasterio.open(dst, "w", **profile) as out:
            out.write(data, 1)
        print(f"Wrote {dst} ({side}x{side})")
"#;
    let args = vec![
        "-c".to_string(),
        script.to_string(),
        source_dir.display().to_string(),
        output_dir.display().to_string(),
        config.azgaar_region_cells.to_string(),
        format!("{:.6}", config.azgaar_center_x),
        format!("{:.6}", config.azgaar_center_y),
    ];
    run_logged_command(&config.python_bin, &args, repo_path, &config.extra_env, tx)
}

fn stream_reader<R: std::io::Read>(reader: R, tx: mpsc::Sender<DiffusionMsg>) {
    let reader = BufReader::new(reader);
    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if !line.is_empty() {
            let _ = tx.send(DiffusionMsg::Log(line.to_string()));
        }
    }
}

fn trigger_reload(
    output_dir: &Path,
    world_scale: f32,
    final_height_span_m: f32,
    requested_clipmap_levels: u32,
    active_config: &TerrainConfig,
    active_library: &MaterialLibrary,
    reload_tx: &mut MessageWriter<ReloadTerrainRequest>,
) {
    let (world_min, world_max) = scan_world_bounds(output_dir, world_scale);
    let max_mip = scan_max_mip(&output_dir.join("height"));

    let mut new_config = active_config.clone();
    new_config.world_scale = world_scale;
    new_config.height_scale = final_height_span_m;
    new_config.clipmap_levels = derive_clipmap_levels(requested_clipmap_levels, max_mip);

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
    let fallback_half = 8192.0 * world_scale;
    let fallback = (Vec2::splat(-fallback_half), Vec2::splat(fallback_half));
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

    let tile_world_size = TILE_SIZE as f32 * world_scale;
    (
        Vec2::new(
            min_tx as f32 * tile_world_size,
            min_ty as f32 * tile_world_size,
        ),
        Vec2::new(
            (max_tx + 1) as f32 * tile_world_size,
            (max_ty + 1) as f32 * tile_world_size,
        ),
    )
}

fn derive_clipmap_levels(requested: u32, max_mip: u8) -> u32 {
    requested.max(max_mip as u32 + 1).max(1)
}

fn detect_repo_path() -> PathBuf {
    let candidates = [
        PathBuf::from("/home/avataren/src/terrain-diffusion"),
        PathBuf::from("/home/avataren/terrain-diffusion"),
        PathBuf::from("../terrain-diffusion"),
    ];
    candidates
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("/home/avataren/src/terrain-diffusion"))
}

fn detect_python_bin(repo_path: &Path) -> PathBuf {
    let candidates = [
        repo_path.join(".venv").join("bin").join("python"),
        repo_path.join(".venv").join("bin").join("python3"),
        repo_path.join("venv").join("bin").join("python"),
        repo_path.join("venv").join("bin").join("python3"),
    ];
    candidates
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("python3"))
}

fn apply_python_environment(command: &mut Command, program: &str) {
    let program_path = Path::new(program);
    let Some(bin_dir) = program_path.parent() else {
        return;
    };
    let Some(venv_root) = bin_dir.parent() else {
        return;
    };
    if bin_dir.file_name().and_then(|n| n.to_str()) != Some("bin") {
        return;
    }

    command.env("VIRTUAL_ENV", venv_root);
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![bin_dir.to_path_buf()];
    paths.extend(std::env::split_paths(&existing));
    if let Ok(path) = std::env::join_paths(paths) {
        command.env("PATH", path);
    }
}

fn apply_extra_environment(
    command: &mut Command,
    raw_env: &str,
    tx: &mpsc::Sender<DiffusionMsg>,
) -> Result<(), String> {
    let pairs = parse_env_pairs(raw_env)?;
    let specified_keys: HashSet<String> = pairs.iter().map(|(key, _)| key.clone()).collect();
    for (key, value) in pairs {
        command.env(&key, &value);
        send_log(tx, format!("env {key}={value}"));
    }
    apply_runtime_cache_defaults(command, &specified_keys, tx);
    Ok(())
}

fn apply_runtime_cache_defaults(
    command: &mut Command,
    specified_keys: &HashSet<String>,
    tx: &mpsc::Sender<DiffusionMsg>,
) {
    let cache_root = std::env::temp_dir().join("landscape-terrain-diffusion");
    apply_default_env_dir(
        command,
        specified_keys,
        tx,
        "NUMBA_CACHE_DIR",
        &cache_root.join("numba"),
    );
    apply_default_env_dir(
        command,
        specified_keys,
        tx,
        "MPLCONFIGDIR",
        &cache_root.join("matplotlib"),
    );
}

fn apply_default_env_dir(
    command: &mut Command,
    specified_keys: &HashSet<String>,
    tx: &mpsc::Sender<DiffusionMsg>,
    key: &str,
    path: &Path,
) {
    if specified_keys.contains(key) || std::env::var_os(key).is_some() {
        return;
    }
    if let Err(err) = std::fs::create_dir_all(path) {
        send_log(
            tx,
            format!(
                "Warning: failed to create default {key} directory {}: {err}",
                path.display()
            ),
        );
        return;
    }
    command.env(key, path);
    send_log(tx, format!("env {key}={} (default)", path.display()));
}

fn parse_env_pairs(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for entry in raw
        .split(['\n', ';'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let Some((key, value)) = entry.split_once('=') else {
            return Err(format!("Invalid env entry `{entry}`. Expected KEY=VALUE."));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err("Environment variable name cannot be empty.".into());
        }
        out.push((key.to_string(), value.trim().to_string()));
    }
    Ok(out)
}

fn send_log(tx: &mpsc::Sender<DiffusionMsg>, line: impl Into<String>) {
    let _ = tx.send(DiffusionMsg::Log(line.into()));
}

fn path_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut String,
    hint: &str,
    pick_target: &mut Option<DiffusionPickTarget>,
    target: Option<DiffusionPickTarget>,
    enabled: bool,
) {
    ui.label(label);
    ui.add_enabled(
        enabled,
        egui::TextEdit::singleline(value)
            .hint_text(hint)
            .desired_width(280.0),
    );
    match target {
        Some(target) => {
            if ui
                .add_enabled(enabled, egui::Button::new("Browse…"))
                .clicked()
            {
                *pick_target = Some(target);
            }
        }
        None => {
            ui.add_enabled(false, egui::Button::new("Browse…"));
        }
    }
    ui.end_row();
}

fn text_row(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str, enabled: bool) {
    ui.label(label);
    ui.add_enabled(
        enabled,
        egui::TextEdit::singleline(value)
            .hint_text(hint)
            .desired_width(280.0),
    );
    ui.label("");
    ui.end_row();
}

fn text_grid_row(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str, enabled: bool) {
    ui.label(label);
    ui.add_enabled(
        enabled,
        egui::TextEdit::singleline(value)
            .hint_text(hint)
            .desired_width(280.0),
    );
    ui.end_row();
}

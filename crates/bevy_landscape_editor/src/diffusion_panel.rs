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

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum DiffusionWorkflow {
    #[default]
    AzgaarJson,
    ConditioningFolder,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
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

#[derive(Clone, Copy, Default, PartialEq, Eq)]
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
}

impl DiffusionRunKind {
    fn title(self) -> &'static str {
        match self {
            Self::Generate => "Run Log",
            Self::Probe => "Probe Log",
        }
    }

    fn running_label(self) -> &'static str {
        match self {
            Self::Generate => "Terrain diffusion job is running...",
            Self::Probe => "Diffusion runtime probe is running...",
        }
    }

    fn failure_label(self) -> &'static str {
        match self {
            Self::Generate => "Diffusion pipeline failed",
            Self::Probe => "Diffusion runtime probe failed",
        }
    }
}

enum DiffusionOutcome {
    Generated(PathBuf),
    Probed,
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
}

impl DiffusionPanelState {
    pub(crate) fn new() -> Self {
        let repo_path = detect_repo_path();
        let working_dir = "assets/terrain_diffusion".to_string();
        Self {
            workflow: DiffusionWorkflow::AzgaarJson,
            repo_path: repo_path.display().to_string(),
            python_bin: detect_python_bin(&repo_path).display().to_string(),
            model_path: "xandergos/terrain-diffusion-90m".into(),
            azgaar_json_path: String::new(),
            conditioning_dir: format!("{working_dir}/conditioning"),
            conditioning_scale_km: 23.0,
            working_dir,
            tile_output_dir: "assets/tiles_diffusion".into(),
            world_scale: 90.0,
            final_height_span_m: 8192.0,
            smooth_sigma: 1.0,
            clipmap_levels: TerrainConfig::default().clipmap_levels,
            device: DiffusionDevice::Auto,
            dtype: DiffusionDtype::Fp32,
            snr: "0.2,0.2,1.0,0.2,1.0".into(),
            batch_size: "1,2,4,8".into(),
            cache_size: "1G".into(),
            seed: String::new(),
            use_compile: false,
            extra_env: "HSA_OVERRIDE_GFX_VERSION=11.0.0".into(),
            run_state: DiffusionRunState::Idle,
            prefs_applied: false,
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
    let mut pick_target = None;
    let mut probe_requested = false;
    let mut start_requested = false;
    let mut clear_requested = false;
    let running = state.is_running();

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

    draw_run_log(ui, state);

    if let Some(target) = pick_target {
        spawn_file_picker(state, target);
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
    let Some((kind, finished, error, reloaded, log_lines)) = (match &state.run_state {
        DiffusionRunState::Running {
            kind,
            finished,
            error,
            reloaded,
            log_lines,
            ..
        } => Some((*kind, *finished, error.as_ref(), *reloaded, log_lines)),
        _ => None,
    }) else {
        return;
    };

    ui.add_space(8.0);
    ui.heading(kind.title());
    ui.separator();

    if !finished {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(kind.running_label());
        });
    } else if let Some(err) = error {
        ui.colored_label(
            egui::Color32::RED,
            format!("{}: {err}", kind.failure_label()),
        );
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
    } else if kind == DiffusionRunKind::Probe {
        ui.colored_label(egui::Color32::GREEN, "Diffusion runtime probe succeeded.");
    } else {
        ui.label("Diffusion pipeline finished.");
    }

    egui::ScrollArea::vertical()
        .id_salt("diffusion_run_log")
        .max_height(240.0)
        .stick_to_bottom(true)
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
    };
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
    };
}

fn build_job_config(state: &DiffusionPanelState) -> DiffusionJobConfig {
    DiffusionJobConfig {
        workflow: state.workflow,
        repo_path: state.repo_path.clone(),
        python_bin: state.python_bin.clone(),
        model_path: state.model_path.clone(),
        azgaar_json_path: state.azgaar_json_path.clone(),
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
                if let DiffusionRunState::Running { log_lines, .. } = &mut state.run_state {
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
                    ..
                } = &mut state.run_state
                {
                    match result {
                        Ok(DiffusionOutcome::Generated(tile_root)) => {
                            log_lines.push("✓ Diffusion run complete — reloading terrain…".into());
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
                            }
                        }
                        Err(err) => {
                            log_lines.push(format!("✗ {err}"));
                            *error = Some(err);
                        }
                    }
                    *finished = true;
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

    let tile_output_dir = PathBuf::from(&config.tile_output_dir);
    if let Some(parent) = tile_output_dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create parent directory for {}: {e}",
                tile_output_dir.display()
            )
        })?;
    }

    let conditioning_dir = match config.workflow {
        DiffusionWorkflow::AzgaarJson => working_dir.join("conditioning"),
        DiffusionWorkflow::ConditioningFolder => PathBuf::from(&config.conditioning_dir),
    };
    let generated_tiff = working_dir.join("generated_heightmap.tif");

    match config.workflow {
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
        }
        DiffusionWorkflow::ConditioningFolder => {
            if !conditioning_dir.exists() {
                return Err(format!(
                    "Conditioning folder not found: {}",
                    conditioning_dir.display()
                ));
            }
        }
    }

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
        conditioning_dir.display().to_string(),
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

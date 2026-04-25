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
    load_level, DetailSynthesisConfig, MaterialLibrary, ReloadTerrainRequest, TerrainConfig,
    TerrainMetadata, TerrainSourceDesc,
};
use bevy_landscape_clouds::CloudsConfig;
use bevy_landscape_generator::GeneratorParams;
use bevy_landscape_water::{OceanFftSettings, WaterSettings};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{mpsc, Mutex};

use crate::sky_panel::TimeOfDay;

// ---------------------------------------------------------------------------
// Serialisable DTOs for settings types that contain non-serde Bevy types.
// These only include fields that are meaningfully edited in the UI panels.
// ---------------------------------------------------------------------------

/// Serialisable snapshot of `WaterSettings` (excludes internal/engine fields).
#[derive(Serialize, Deserialize)]
struct WaterSettingsDto {
    amplitude: f32,
    wave_speed: f32,
    wave_direction: [f32; 2],
    clarity: f32,
    deep_color: [f32; 3],
    shallow_color: [f32; 3],
    edge_scale: f32,
    edge_color: [f32; 3],
    refraction_strength: f32,
    foam_threshold: f32,
    foam_color: [f32; 3],
    shoreline_foam_depth: f32,
    shore_wave_damp_width: f32,
    jacobian_foam_strength: f32,
    capillary_strength: f32,
    macro_noise_amplitude: f32,
    macro_noise_scale: f32,
}

impl From<&WaterSettings> for WaterSettingsDto {
    fn from(s: &WaterSettings) -> Self {
        fn to_rgb(c: Color) -> [f32; 3] {
            let l = c.to_linear();
            [l.red, l.green, l.blue]
        }
        Self {
            amplitude: s.amplitude,
            wave_speed: s.wave_speed,
            wave_direction: [s.wave_direction.x, s.wave_direction.y],
            clarity: s.clarity,
            deep_color: to_rgb(s.deep_color),
            shallow_color: to_rgb(s.shallow_color),
            edge_scale: s.edge_scale,
            edge_color: to_rgb(s.edge_color),
            refraction_strength: s.refraction_strength,
            foam_threshold: s.foam_threshold,
            foam_color: to_rgb(s.foam_color),
            shoreline_foam_depth: s.shoreline_foam_depth,
            shore_wave_damp_width: s.shore_wave_damp_width,
            jacobian_foam_strength: s.jacobian_foam_strength,
            capillary_strength: s.capillary_strength,
            macro_noise_amplitude: s.macro_noise_amplitude,
            macro_noise_scale: s.macro_noise_scale,
        }
    }
}

impl WaterSettingsDto {
    fn apply_to(&self, s: &mut WaterSettings) {
        fn from_rgb(rgb: [f32; 3]) -> Color {
            Color::linear_rgb(rgb[0], rgb[1], rgb[2])
        }
        s.amplitude = self.amplitude;
        s.wave_speed = self.wave_speed;
        s.wave_direction = Vec2::new(self.wave_direction[0], self.wave_direction[1]);
        s.clarity = self.clarity;
        s.deep_color = from_rgb(self.deep_color);
        s.shallow_color = from_rgb(self.shallow_color);
        s.edge_scale = self.edge_scale;
        s.edge_color = from_rgb(self.edge_color);
        s.refraction_strength = self.refraction_strength;
        s.foam_threshold = self.foam_threshold;
        s.foam_color = from_rgb(self.foam_color);
        s.shoreline_foam_depth = self.shoreline_foam_depth;
        s.shore_wave_damp_width = self.shore_wave_damp_width;
        s.jacobian_foam_strength = self.jacobian_foam_strength;
        s.capillary_strength = self.capillary_strength;
        s.macro_noise_amplitude = self.macro_noise_amplitude;
        s.macro_noise_scale = self.macro_noise_scale;
    }
}

/// Serialisable snapshot of `OceanFftSettings`.
#[derive(Serialize, Deserialize)]
struct OceanFftDto {
    enabled: bool,
    size: u32,
    world_size: f32,
    wind_speed: f32,
    wind_direction: [f32; 2],
    amplitude: f32,
    choppy: f32,
    seed: u32,
    strength: f32,
}

impl From<&OceanFftSettings> for OceanFftDto {
    fn from(s: &OceanFftSettings) -> Self {
        Self {
            enabled: s.enabled,
            size: s.size,
            world_size: s.world_size,
            wind_speed: s.wind_speed,
            wind_direction: [s.wind_direction.x, s.wind_direction.y],
            amplitude: s.amplitude,
            choppy: s.choppy,
            seed: s.seed,
            strength: s.strength,
        }
    }
}

impl OceanFftDto {
    fn apply_to(&self, s: &mut OceanFftSettings) {
        s.enabled = self.enabled;
        s.size = self.size;
        s.world_size = self.world_size;
        s.wind_speed = self.wind_speed;
        s.wind_direction = Vec2::new(self.wind_direction[0], self.wind_direction[1]);
        s.amplitude = self.amplitude;
        s.choppy = self.choppy;
        s.seed = self.seed;
        s.strength = self.strength;
    }
}

/// Serialisable snapshot of both water resources combined.
#[derive(Serialize, Deserialize)]
struct WaterDesc {
    settings: WaterSettingsDto,
    fft: OceanFftDto,
}

/// Serialisable snapshot of `DetailSynthesisConfig`.
#[derive(Serialize, Deserialize)]
struct SynthesisDto {
    enabled: bool,
    max_amplitude: f32,
    lacunarity: f32,
    gain: f32,
    erosion_strength: f32,
    seed: [f32; 2],
    slope_mask_threshold_deg: f32,
    slope_mask_falloff_deg: f32,
    normal_detail_strength: f32,
}

impl From<&DetailSynthesisConfig> for SynthesisDto {
    fn from(c: &DetailSynthesisConfig) -> Self {
        Self {
            enabled: c.enabled,
            max_amplitude: c.max_amplitude,
            lacunarity: c.lacunarity,
            gain: c.gain,
            erosion_strength: c.erosion_strength,
            seed: [c.seed.x, c.seed.y],
            slope_mask_threshold_deg: c.slope_mask_threshold_deg,
            slope_mask_falloff_deg: c.slope_mask_falloff_deg,
            normal_detail_strength: c.normal_detail_strength,
        }
    }
}

impl SynthesisDto {
    fn apply_to(&self, c: &mut DetailSynthesisConfig) {
        c.enabled = self.enabled;
        c.max_amplitude = self.max_amplitude;
        c.lacunarity = self.lacunarity;
        c.gain = self.gain;
        c.erosion_strength = self.erosion_strength;
        c.seed = Vec2::new(self.seed[0], self.seed[1]);
        c.slope_mask_threshold_deg = self.slope_mask_threshold_deg;
        c.slope_mask_falloff_deg = self.slope_mask_falloff_deg;
        c.normal_detail_strength = self.normal_detail_strength;
    }
}

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
    mut tod: ResMut<TimeOfDay>,
    water_settings: Option<ResMut<WaterSettings>>,
    fft_settings: Option<ResMut<OceanFftSettings>>,
    synthesis: Option<ResMut<DetailSynthesisConfig>>,
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
                    level_desc.sky = serde_json::to_value(&*tod).ok();
                    if let (Some(ws), Some(fft)) = (&water_settings, &fft_settings) {
                        level_desc.water = serde_json::to_value(WaterDesc {
                            settings: WaterSettingsDto::from(ws.as_ref()),
                            fft: OceanFftDto::from(fft.as_ref()),
                        })
                        .ok();
                    }
                    if let Some(syn) = &synthesis {
                        level_desc.synthesis =
                            serde_json::to_value(SynthesisDto::from(syn.as_ref())).ok();
                    }
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
                Some(LevelIoOp::Load) => {
                    match load_level(&path) {
                        Ok(level_desc) => {
                            if let Some(cc) = level_desc
                                .clouds
                                .as_ref()
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                            {
                                *clouds_config = cc;
                            }
                            if let Some(hours) = level_desc
                                .sky
                                .as_ref()
                                .and_then(|v| serde_json::from_value::<TimeOfDay>(v.clone()).ok())
                            {
                                *tod = hours;
                            }
                            if let Some(wd) = level_desc
                                .water
                                .as_ref()
                                .and_then(|v| serde_json::from_value::<WaterDesc>(v.clone()).ok())
                            {
                                if let Some(mut ws) = water_settings {
                                    wd.settings.apply_to(ws.as_mut());
                                }
                                if let Some(mut fft) = fft_settings {
                                    wd.fft.apply_to(fft.as_mut());
                                }
                            }
                            if let Some(sd) = level_desc.synthesis.as_ref().and_then(|v| {
                                serde_json::from_value::<SynthesisDto>(v.clone()).ok()
                            }) {
                                if let Some(mut syn) = synthesis {
                                    sd.apply_to(syn.as_mut());
                                }
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
                    }
                }
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

use std::f32::consts::PI;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};

use bevy::prelude::*;
use bevy::render::{extract_resource::ExtractResource, render_resource::ShaderType};
use serde::{Deserialize, Serialize};

/// Ticks executed per render frame while erosion is accumulating.
pub const TICKS_PER_FRAME: u32 = 5;

#[derive(Resource, Clone, Debug, Serialize, Deserialize, ExtractResource)]
#[serde(default)]
pub struct ErosionParams {
    pub enabled: bool,
    pub iterations: u32,
    /// Simulation time-step.
    pub dt: f32,
    pub gravity: f32,
    /// Cell size in simulation space.
    pub pipe_length: f32,
    /// Virtual pipe cross-section area A (paper eq. 2). Default 5.0.
    pub pipe_area: f32,
    pub rain_rate: f32,
    pub evaporation_rate: f32,
    pub sediment_capacity: f32,
    pub erosion_rate: f32,
    pub deposition_rate: f32,
    pub min_slope: f32,
    /// Maximum water depth at which erosion is fully active (lmax Kdmax).
    pub erosion_depth_max: f32,
    pub hardness_influence: f32,
    pub thermal_enabled: bool,
    pub repose_angle: f32,
    pub talus_rate: f32,
    pub thermal_iterations: u32,
    pub particle_enabled: bool,
    pub num_particles: u32,
    pub particle_max_steps: u32,
    pub particle_inertia: f32,
    /// 0 = eroded height, 1 = water depth, 2 = sediment, 3 = velocity magnitude, 4 = hardness
    pub debug_view: u32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            enabled: false,
            // 2000 iterations gives the river network time to develop at the
            // reduced erosion rates above (was 500, which completed too quickly
            // with the old over-aggressive defaults).
            iterations: 2000,
            dt: 0.02,
            gravity: 9.81,
            pipe_length: 1.0,
            pipe_area: 1.0,
            rain_rate: 0.003,
            evaporation_rate: 0.015,
            // Reduced from 1.0: high capacity + fast flows on a [0,1] heightmap
            // eroded the entire terrain to flat mush within ~50 ticks.
            sediment_capacity: 0.5,
            // Reduced from 0.5: allows gradual channel incision without
            // stripping all material in the first pass.
            erosion_rate: 0.15,
            // Reduced from 1.0: slower deposition lets sediment travel further
            // downstream, producing alluvial fans and longer river paths.
            deposition_rate: 0.3,
            min_slope: 0.01,
            // Reduced from 0.05: erosion at full strength with a thinner water
            // film promotes channel incision over broad sheet wash.
            erosion_depth_max: 0.02,
            hardness_influence: 0.5,
            thermal_enabled: true,
            repose_angle: 35.0,
            talus_rate: 0.3,
            thermal_iterations: 3,
            particle_enabled: false,
            num_particles: 50_000,
            particle_max_steps: 48,
            particle_inertia: 0.05,
            debug_view: 0,
        }
    }
}

/// Shared state between main world and render world (via Arc atomics).
/// Tracks whether a new erosion run is needed and how many ticks have run.
#[derive(Resource, Clone)]
pub struct ErosionControlState {
    /// True when erosion needs to (re)start from tick 0.
    pub dirty: Arc<AtomicBool>,
    /// How many ticks have been dispatched so far in the current run.
    pub ticks_done: Arc<AtomicU32>,
}

impl ErosionControlState {
    pub fn new_dirty() -> Self {
        Self {
            dirty: Arc::new(AtomicBool::new(true)),
            ticks_done: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn mark_dirty(&self) {
        self.ticks_done.store(0, Ordering::Release);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn ticks_done(&self) -> u32 {
        self.ticks_done.load(Ordering::Acquire)
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }
}

impl ExtractResource for ErosionControlState {
    type Source = ErosionControlState;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone() // Arc clone — same atomics, both worlds share state
    }
}

/// GPU-side uniform matching the WGSL `ErosionParams` struct.
/// Layout: 80 bytes (multiple of 16, no padding required).
#[derive(Clone, Resource, ExtractResource, ShaderType)]
pub struct ErosionUniform {
    pub resolution:           UVec2,
    pub dt:                   f32,
    pub gravity:              f32,
    pub pipe_length:          f32,
    pub rain_rate:            f32,
    pub evaporation_rate:     f32,
    pub sediment_capacity:    f32,
    pub erosion_rate:         f32,
    pub deposition_rate:      f32,
    pub min_slope:            f32,
    pub hardness_influence:   f32,
    pub repose_angle_radians: f32,
    pub talus_rate:           f32,
    pub num_particles:        u32,
    pub max_steps:            u32,
    pub inertia:              f32,
    pub frame_seed:           u32,
    pub pipe_area:            f32,
    pub erosion_depth_max:    f32,
}

impl Default for ErosionUniform {
    fn default() -> Self {
        Self::from_params(&ErosionParams::default(), 1024, 42)
    }
}

impl ErosionUniform {
    /// `hardness_seed` should be stable per terrain (e.g. generator seed).
    ///
    /// The simulation runs in **pixel-grid units** (`pipe_length` = 1 cell).
    ///
    /// Resolution compensation:
    /// - `pipe_area` is scaled by `N / 512` so that flux (which depends on
    ///   per-pixel Δh ∝ 1/N) is resolution-independent.
    /// - The shader computes *physical slope* (via `gradient2` multiplied by
    ///   `N / 512`) for sediment capacity, making erosion depth consistent
    ///   across resolutions.
    /// - Particle erosion: `num_particles` scales by `(N/512)²` to maintain
    ///   constant particle density per physical area, and `max_steps` scales
    ///   by `N/512` so each particle covers the same physical distance.
    pub fn from_params(p: &ErosionParams, resolution: u32, hardness_seed: u32) -> Self {
        let res_scale = resolution as f32 / 512.0;
        Self {
            resolution:           UVec2::splat(resolution),
            dt:                   p.dt,
            gravity:              p.gravity,
            pipe_length:          p.pipe_length,
            rain_rate:            p.rain_rate,
            evaporation_rate:     p.evaporation_rate,
            sediment_capacity:    p.sediment_capacity,
            erosion_rate:         p.erosion_rate,
            deposition_rate:      p.deposition_rate,
            min_slope:            p.min_slope,
            hardness_influence:   p.hardness_influence,
            repose_angle_radians: p.repose_angle * (PI / 180.0),
            talus_rate:           p.talus_rate,
            num_particles:        ((p.num_particles as f32) * res_scale * res_scale) as u32,
            max_steps:            ((p.particle_max_steps as f32) * res_scale).ceil() as u32,
            inertia:              p.particle_inertia,
            frame_seed:           hardness_seed,
            pipe_area:            p.pipe_area * res_scale,
            erosion_depth_max:    p.erosion_depth_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_params_scale_with_resolution() {
        let base = ErosionParams::default();
        for &res in &[256u32, 512, 1024, 2048] {
            let u = ErosionUniform::from_params(&base, res, 42);
            let rs = res as f32 / 512.0;

            // num_particles ∝ (N/512)²
            let expected_np = ((base.num_particles as f32) * rs * rs) as u32;
            assert_eq!(u.num_particles, expected_np, "num_particles at res {res}");

            // max_steps ∝ N/512
            let expected_ms = ((base.particle_max_steps as f32) * rs).ceil() as u32;
            assert_eq!(u.max_steps, expected_ms, "max_steps at res {res}");

            // pipe_area ∝ N/512
            let expected_pa = base.pipe_area * rs;
            assert!((u.pipe_area - expected_pa).abs() < 1e-6, "pipe_area at res {res}");

            // evaporation_rate / rs must stay < 1.0 for default params
            assert!(
                base.evaporation_rate / rs < 1.0,
                "evaporation would kill water instantly at res {res}"
            );
        }
    }
}

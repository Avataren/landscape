//! CPU port of `assets/shaders/detail_synthesis.wgsl`.
//!
//! Used by the collision-mesh builder so that physics geometry matches the
//! GPU-synthesized visual terrain. The math (hash, gradient noise, rotation,
//! erosion-shaped fBM) is a literal translation of the WGSL functions; results
//! agree to within float-rounding error.

use crate::terrain::detail_synthesis::DetailSynthesisConfig;
use bevy::prelude::Vec2;

const GRADIENT_EPSILON: f32 = 0.37;
const EROSION_RESPONSE: f32 = 3.5;
const U32_TO_UNIT: f32 = 2.328_306_4e-10; // 1.0 / 4294967295.0

/// PCG 2D integer hash — bit-identical to the `pcg2d` function in
/// `assets/shaders/detail_synthesis.wgsl`.  The output is the source of the
/// gradient vectors at integer grid corners; using an integer hash (instead
/// of WGSL's `fract(sin(x) * 43758)`) guarantees CPU and GPU produce the
/// exact same noise field.
#[inline]
fn pcg2d(v_in: [u32; 2]) -> [u32; 2] {
    let mut v = [
        v_in[0].wrapping_mul(1664525).wrapping_add(1013904223),
        v_in[1].wrapping_mul(1664525).wrapping_add(1013904223),
    ];
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(1664525));
    v[1] = v[1].wrapping_add(v[0].wrapping_mul(1664525));
    v[0] ^= v[0] >> 16;
    v[1] ^= v[1] >> 16;
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(1664525));
    v[1] = v[1].wrapping_add(v[0].wrapping_mul(1664525));
    v[0] ^= v[0] >> 16;
    v[1] ^= v[1] >> 16;
    v
}

#[inline]
fn hash_grad(pi: [i32; 2]) -> Vec2 {
    // Unit-length gradient — must match the WGSL `hash_grad` exactly.
    // Mapping a single hash channel into an angle in [0, 2π) and emitting
    // (cos θ, sin θ) gives an isotropic gradient with no lattice bias.
    // `i32 as u32` is a bit-cast (matches WGSL `bitcast<vec2<u32>>`).
    let h = pcg2d([pi[0] as u32, pi[1] as u32]);
    let theta = (h[0] as f32) * U32_TO_UNIT * std::f32::consts::TAU;
    Vec2::new(theta.cos(), theta.sin())
}

#[inline]
fn gradient_noise(p: Vec2) -> f32 {
    let pf = Vec2::new(p.x.floor(), p.y.floor());
    let i = [pf.x as i32, pf.y as i32];
    let f = p - pf;
    let u = f * f * f * (f * (f * 6.0 - Vec2::splat(15.0)) + Vec2::splat(10.0));
    let a = hash_grad([i[0], i[1]]).dot(f);
    let b = hash_grad([i[0] + 1, i[1]]).dot(f - Vec2::new(1.0, 0.0));
    let c = hash_grad([i[0], i[1] + 1]).dot(f - Vec2::new(0.0, 1.0));
    let d = hash_grad([i[0] + 1, i[1] + 1]).dot(f - Vec2::new(1.0, 1.0));
    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    ab + (cd - ab) * u.y
}

#[inline]
fn rot2(p: Vec2) -> Vec2 {
    Vec2::new(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y)
}

fn erosion_shaped_fbm(base: Vec2, octaves: u32, lac: f32, g: f32, erosion: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 0.5;
    let mut pos = base;
    let mut acc_grad = Vec2::ZERO;
    for _ in 0..octaves {
        let n = gradient_noise(pos);
        let grad = Vec2::new(
            gradient_noise(pos + Vec2::new(GRADIENT_EPSILON, 0.0)) - n,
            gradient_noise(pos + Vec2::new(0.0, GRADIENT_EPSILON)) - n,
        ) / GRADIENT_EPSILON;
        acc_grad += grad * amplitude;
        let atten = 1.0 + (1.0 / (1.0 + acc_grad.dot(acc_grad) * EROSION_RESPONSE) - 1.0) * erosion;
        value += amplitude * n * atten;
        pos = rot2(pos) * lac;
        amplitude *= g;
    }
    value
}

/// Picks the octave count for a collision sample given its cell spacing,
/// matching the GPU-side rule but capped so the finest octave wavelength
/// stays at or above `2 × cell_size` (Nyquist) to avoid sampling aliasing.
pub fn octave_count_for_cell(cell_size_ws: f32, source_spacing: f32, lacunarity: f32) -> u32 {
    if cell_size_ws >= source_spacing {
        return 0;
    }
    let base_wl = source_spacing * 0.5;
    let fine_wl = (cell_size_ws * 2.0).max(0.01);
    let raw = (base_wl / fine_wl).log2().floor() as i32;
    let visual = raw.clamp(1, 6) as u32;

    // Nyquist for collision sampling: keep finest octave wavelength ≥ 2·cell_size.
    let nyquist_wl = cell_size_ws * 2.0;
    let mut wl = base_wl;
    let mut allowed = 0u32;
    while wl >= nyquist_wl && allowed < visual {
        allowed += 1;
        wl /= lacunarity.max(1.001);
    }
    allowed
}

/// Adds a synthesised height residual at `world_pos` matching what the GPU
/// detail-synthesis pass produces. `slope_deg` is the local source-heightmap
/// slope (in degrees); pass 0 to disable the slope mask.
pub fn synthesise_residual(
    world_pos: Vec2,
    octaves: u32,
    slope_deg: f32,
    cfg: &DetailSynthesisConfig,
    source_spacing: f32,
) -> f32 {
    if !cfg.enabled || octaves == 0 || cfg.max_amplitude <= 0.0 {
        return 0.0;
    }
    let base_freq = 2.0 / source_spacing.max(0.001);
    let p = (world_pos + cfg.seed) * base_freq;
    let n = erosion_shaped_fbm(p, octaves, cfg.lacunarity, cfg.gain, cfg.erosion_strength);
    let slope_mask = 1.0
        - smoothstep(
            cfg.slope_mask_threshold_deg,
            cfg.slope_mask_threshold_deg + cfg.slope_mask_falloff_deg.max(0.1),
            slope_deg,
        );
    n * cfg.max_amplitude * slope_mask
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

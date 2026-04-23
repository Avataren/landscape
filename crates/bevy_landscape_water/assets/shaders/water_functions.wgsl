#define_import_path bevy_landscape_water::water_functions

#ifdef PREPASS_PIPELINE
#import bevy_render::globals::Globals
@group(0) @binding(1) var<uniform> globals: Globals;
#else
#import bevy_pbr::mesh_view_bindings::globals
#endif

#import bevy_landscape_water::water_bindings::material

const PI: f32  = 3.14159265358979323846;
const G:  f32  = 9.81;

struct WaveResult {
    height: f32,
    normal: vec3<f32>,
}

// ---------------------------------------------------------------------------
// 8 Gerstner wave layers: vec4(dir.x, dir.y, wavelength_m, Q_artist).
//
// GPU Gems §1 design rules:
//   • Wavelengths are irrational multiples of each other (prime-ish numbers)
//     to avoid constructive-interference beating.
//   • Amplitudes proportional to wavelength so A/L ≈ AMP_RATIO (constant
//     steepness-to-length ratio) — computed in gerstner_wave() from L.
//   • All directions within ±45° of +X wind direction.
//   • Q_artist ∈ [0,1] → Q_eff = Q_artist / (ω × A × numWaves), guaranteeing
//     Σ(Q_eff × ω × A) ≤ 1 (no surface self-intersection).
// ---------------------------------------------------------------------------
// dir.xy = unit direction (XZ), z = wavelength (m), w = Q_artist ∈ [0,1]
const WAVE_0: vec4<f32> = vec4( 1.000,  0.000, 83.7, 0.70);  // primary swell +X
const WAVE_1: vec4<f32> = vec4( 0.921,  0.391, 61.3, 0.65);  // +25°
const WAVE_2: vec4<f32> = vec4( 0.707,  0.707, 47.1, 0.60);  // +45°
const WAVE_3: vec4<f32> = vec4( 0.966, -0.259, 53.9, 0.60);  // -15°
const WAVE_4: vec4<f32> = vec4( 0.809,  0.588, 31.7, 0.50);  // +36°
const WAVE_5: vec4<f32> = vec4( 0.891, -0.454, 23.3, 0.45);  // -27°
const WAVE_6: vec4<f32> = vec4( 0.978,  0.208, 17.1, 0.40);  // +12° (chop)
const WAVE_7: vec4<f32> = vec4( 0.743, -0.669, 11.9, 0.35);  // -42° (chop)

const NUM_WAVES: f32 = 8.0;
// Amplitude-to-wavelength ratio: A = AMP_RATIO × L.
// ~0.006 gives realistic ocean proportions without excessive displacement.
const AMP_RATIO: f32 = 0.006;

// ---------------------------------------------------------------------------
// GPU Gems §1.3.3 — per-wave attenuation by pixel footprint.
//
// A wave whose crest-to-crest length L is smaller than the screen-space pixel
// footprint P_ws causes aliasing (it changes faster than once per pixel).
// The Nyquist limit is L = 2 × P_ws; we start fading at L = 4 × P_ws so the
// transition is smooth well before aliasing appears.
//
//   weight = 0  when L ≤ 2 × pixel_size   (sub-Nyquist — suppress entirely)
//   weight = 1  when L ≥ 4 × pixel_size   (fully resolved — full contribution)
// ---------------------------------------------------------------------------
fn wave_lod_weight(wavelength: f32, pixel_size: f32) -> f32 {
    return smoothstep(pixel_size * 2.0, pixel_size * 4.0, wavelength);
}

fn gerstner_wave(
    p:         vec2<f32>,
    t:         f32,
    params:    vec4<f32>,
    base_amp:  f32,
    lod_w:     f32,
) -> WaveResult {
    let dir      = params.xy;
    let L        = params.z;
    let Q_artist = params.w;

    let omega = 2.0 * PI / L;
    let amp   = base_amp * AMP_RATIO * L;
    // Deep-water dispersion relation: phase speed = sqrt(g / omega).
    let phase = material.wave_speed * sqrt(G * omega);

    let f    = omega * dot(dir, p) + phase * t;
    let sinf = sin(f);
    let cosf = cos(f);

    // GPU Gems §1 Q normalisation: ensures Σ(Q_eff×ω×A) ≤ 1.
    let Q_eff = Q_artist / (omega * amp * NUM_WAVES);

    let wa = omega * amp;
    // Normal accumulation terms (GPU Gems eq. 9):
    let nx = Q_eff * wa * dir.x * sinf;
    let ny = wa * cosf;
    let nz = Q_eff * wa * dir.y * sinf;

    // Scale both height and normal contribution by the LOD weight so that
    // waves below the pixel Nyquist limit fade out smoothly.
    return WaveResult(amp * sinf * lod_w, vec3(nx, ny, nz) * lod_w);
}

// Sum 8 Gerstner waves; return combined height and analytic surface normal.
// pixel_size: world-space size of one screen pixel at the current fragment
//             (computed from dpdx/dpdy in the fragment shader).
//             Pass 0.0 to disable LOD filtering (e.g. vertex shader path).
fn get_wave_result(p: vec2<f32>, pixel_size: f32) -> WaveResult {
    let t   = globals.time;
    let amp = material.amplitude;

    var h    = 0.0;
    var sumn = vec3(0.0);

    let w0 = gerstner_wave(p, t, WAVE_0, amp, wave_lod_weight(WAVE_0.z, pixel_size)); h += w0.height; sumn += w0.normal;
    let w1 = gerstner_wave(p, t, WAVE_1, amp, wave_lod_weight(WAVE_1.z, pixel_size)); h += w1.height; sumn += w1.normal;
    let w2 = gerstner_wave(p, t, WAVE_2, amp, wave_lod_weight(WAVE_2.z, pixel_size)); h += w2.height; sumn += w2.normal;
    let w3 = gerstner_wave(p, t, WAVE_3, amp, wave_lod_weight(WAVE_3.z, pixel_size)); h += w3.height; sumn += w3.normal;
    let w4 = gerstner_wave(p, t, WAVE_4, amp, wave_lod_weight(WAVE_4.z, pixel_size)); h += w4.height; sumn += w4.normal;
    let w5 = gerstner_wave(p, t, WAVE_5, amp, wave_lod_weight(WAVE_5.z, pixel_size)); h += w5.height; sumn += w5.normal;
    let w6 = gerstner_wave(p, t, WAVE_6, amp, wave_lod_weight(WAVE_6.z, pixel_size)); h += w6.height; sumn += w6.normal;
    let w7 = gerstner_wave(p, t, WAVE_7, amp, wave_lod_weight(WAVE_7.z, pixel_size)); h += w7.height; sumn += w7.normal;

    // GPU Gems §1 eq. 9 final normal assembly:
    //   N = (-Σ nx, 1 - Σ ny, -Σ nz), normalised.
    return WaveResult(h, normalize(vec3(-sumn.x, 1.0 - sumn.y, -sumn.z)));
}

// Convenience for vertex shader (no pixel-footprint filtering needed there).
fn get_wave_height(p: vec2<f32>) -> f32 {
    return get_wave_result(p, 0.0).height;
}

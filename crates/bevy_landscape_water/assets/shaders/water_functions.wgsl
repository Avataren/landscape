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
    /// World-space Y displacement (height above rest plane).
    height:  f32,
    /// World-space XZ lateral displacement (Gerstner horizontal offset).
    xz_disp: vec2<f32>,
    /// Analytic surface normal (evaluated at pre-displacement position).
    normal:  vec3<f32>,
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
//     Σ(Q_eff × ω × A) ≤ 1 (no surface self-intersection / folding).
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
// GPU Gems §1.3.3 — per-wave attenuation by pixel/vertex footprint.
//
// A wave whose crest-to-crest length L is smaller than the footprint P
// causes aliasing.  Nyquist limit is L = 2P; fade starts at 4P.
//   weight = 0  when L ≤ 2P   (sub-Nyquist — suppress entirely)
//   weight = 1  when L ≥ 4P   (fully resolved — full contribution)
// ---------------------------------------------------------------------------
fn wave_lod_weight(wavelength: f32, footprint: f32) -> f32 {
    return smoothstep(footprint * 2.0, footprint * 4.0, wavelength);
}

// ---------------------------------------------------------------------------
// Single Gerstner wave (GPU Gems §1 equations 4 and 9).
//
// Returns:
//   height  = A × sin(ωD·P + φt)                   (Y displacement)
//   xz_disp = Q_eff × A × D × cos(ωD·P + φt)       (lateral displacement)
//   normal  = terms for full Gerstner normal (eq. 9)
//
// The full normal requires the Q term on ny:
//   N = normalise(-Σ(wa×D_x×cos), 1 - Σ(Q_eff×wa×sin), -Σ(wa×D_z×cos))
// This is ONLY correct when XZ displacement is also applied to the geometry.
// ---------------------------------------------------------------------------
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
    // Deep-water dispersion: phase speed c = sqrt(g/ω).
    let phase = material.wave_speed * sqrt(G * omega);

    let f    = omega * dot(dir, p) + phase * t;
    let sinf = sin(f);
    let cosf = cos(f);

    // Q normalisation (GPU Gems §1): Σ(Q_eff×ω×A) ≤ 1  →  no surface folding.
    let Q_eff = Q_artist / (omega * amp * NUM_WAVES);
    let wa    = omega * amp;

    // Lateral displacement: Q_eff × A × D × cos(f)   (eq. 4)
    let xz = Q_eff * amp * dir * cosf;

    // Full Gerstner normal terms (eq. 9):
    //   nx = wa × D_x × cos   (no Q)
    //   ny = Q_eff × wa × sin (Q term lives here)
    //   nz = wa × D_z × cos   (no Q)
    let nx = wa * dir.x * cosf;
    let ny = Q_eff * wa * sinf;
    let nz = wa * dir.y * cosf;

    return WaveResult(
        amp * sinf * lod_w,
        xz * lod_w,
        vec3(nx, ny, nz) * lod_w,
    );
}

// ---------------------------------------------------------------------------
// Sum 8 Gerstner waves.
//
// footprint: world-space size of one pixel (fragment shader) or one vertex
//            grid cell (vertex shader) — drives per-wave LOD attenuation.
//            Pass 0.0 to disable LOD filtering (all waves active).
//
// The returned normal is the full Gerstner surface normal (GPU Gems eq. 9)
// which is only physically correct when the caller also applies xz_disp to
// the geometry.  Evaluate at the PRE-DISPLACEMENT world XZ position.
// ---------------------------------------------------------------------------
fn get_wave_result(p: vec2<f32>, footprint: f32) -> WaveResult {
    let t   = globals.time;
    let amp = material.amplitude;

    var h    = 0.0;
    var xz   = vec2(0.0);
    var sumn = vec3(0.0);

    let w0 = gerstner_wave(p, t, WAVE_0, amp, wave_lod_weight(WAVE_0.z, footprint)); h += w0.height; xz += w0.xz_disp; sumn += w0.normal;
    let w1 = gerstner_wave(p, t, WAVE_1, amp, wave_lod_weight(WAVE_1.z, footprint)); h += w1.height; xz += w1.xz_disp; sumn += w1.normal;
    let w2 = gerstner_wave(p, t, WAVE_2, amp, wave_lod_weight(WAVE_2.z, footprint)); h += w2.height; xz += w2.xz_disp; sumn += w2.normal;
    let w3 = gerstner_wave(p, t, WAVE_3, amp, wave_lod_weight(WAVE_3.z, footprint)); h += w3.height; xz += w3.xz_disp; sumn += w3.normal;
    let w4 = gerstner_wave(p, t, WAVE_4, amp, wave_lod_weight(WAVE_4.z, footprint)); h += w4.height; xz += w4.xz_disp; sumn += w4.normal;
    let w5 = gerstner_wave(p, t, WAVE_5, amp, wave_lod_weight(WAVE_5.z, footprint)); h += w5.height; xz += w5.xz_disp; sumn += w5.normal;
    let w6 = gerstner_wave(p, t, WAVE_6, amp, wave_lod_weight(WAVE_6.z, footprint)); h += w6.height; xz += w6.xz_disp; sumn += w6.normal;
    let w7 = gerstner_wave(p, t, WAVE_7, amp, wave_lod_weight(WAVE_7.z, footprint)); h += w7.height; xz += w7.xz_disp; sumn += w7.normal;

    // Full Gerstner normal (GPU Gems §1 eq. 9):
    //   N = normalise(-Σ(wa×D_x×cos), 1 - Σ(Q_eff×wa×sin), -Σ(wa×D_z×cos))
    return WaveResult(h, xz, normalize(vec3(-sumn.x, 1.0 - sumn.y, -sumn.z)));
}

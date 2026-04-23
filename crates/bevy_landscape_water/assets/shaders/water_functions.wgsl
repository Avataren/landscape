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
    height:  f32,
    xz_disp: vec2<f32>,
    normal:  vec3<f32>,
}

struct DetailWaveResult {
    slope:        vec2<f32>,
    crest:        f32,
    slope_energy: f32,
}

// Local directions are expressed relative to +X, then rotated into the
// configured wind direction in world space.
const GEOM_WAVE_0: vec4<f32> = vec4( 1.000,  0.000, 92.0, 0.74);
const GEOM_WAVE_1: vec4<f32> = vec4( 0.966,  0.259, 68.0, 0.70);
const GEOM_WAVE_2: vec4<f32> = vec4( 0.829, -0.559, 51.0, 0.63);
const GEOM_WAVE_3: vec4<f32> = vec4( 0.707,  0.707, 39.0, 0.58);
const GEOM_WAVE_4: vec4<f32> = vec4( 0.924, -0.383, 29.0, 0.52);
const GEOM_WAVE_5: vec4<f32> = vec4( 0.616,  0.788, 21.0, 0.45);

const DETAIL_WAVE_0: vec4<f32> = vec4( 0.986,  0.169, 11.0, 1.00);
const DETAIL_WAVE_1: vec4<f32> = vec4( 0.954, -0.301,  8.0, 0.92);
const DETAIL_WAVE_2: vec4<f32> = vec4( 0.857,  0.516,  6.1, 0.82);
const DETAIL_WAVE_3: vec4<f32> = vec4( 0.766, -0.643,  4.5, 0.74);
const DETAIL_WAVE_4: vec4<f32> = vec4( 0.999,  0.035,  3.3, 0.68);
const DETAIL_WAVE_5: vec4<f32> = vec4( 0.940,  0.342,  2.4, 0.58);
const DETAIL_WAVE_6: vec4<f32> = vec4( 0.899, -0.438,  1.7, 0.48);
const DETAIL_WAVE_7: vec4<f32> = vec4( 0.707,  0.707,  1.2, 0.38);

// Normal-only swell waves — evaluated in the fragment shader, never in the
// vertex shader.  Their normal contributions break up the periodic repetition
// that becomes visible at distance when only the 3–4 longest geometry waves
// remain after LOD filtering.
//
// Direction xy: local frame (x=along wind, y=across) — rotated by wind.
// z: wavelength (world units).  Ratios with 92 m dominant wave are irrational
// (√2 ≈ 131/92, golden ≈ 173/92) so the patterns never spatially re-align.
// w: unused (kept for struct alignment / future use).
const SWELL_WAVE_0: vec4<f32> = vec4( 0.906,  0.423,  47.0, 0.0);  // 25° — gap filler 39-51m
const SWELL_WAVE_1: vec4<f32> = vec4( 0.574,  0.819, 131.0, 0.0);  // 55° cross-swell  (131/92 ≈ √2)
const SWELL_WAVE_2: vec4<f32> = vec4( 0.766, -0.643, 173.0, 0.0);  // -40° broad swell (173/92 ≈ φ×1.16)

const NUM_GEOM_WAVES:      f32 = 6.0;
const GEOM_AMP_RATIO:      f32 = 0.0048;
// Swell waves use the same ratio but are scaled down so they add variation
// without fighting the primary geometry normal.
const SWELL_AMP_SCALE:     f32 = 0.40;
const DETAIL_AMP_RATIO:    f32 = 0.0016;
const DETAIL_SHARPNESS:    f32 = 3.5;
const DETAIL_SPEED_BOOST:  f32 = 1.35;

fn wave_lod_weight(wavelength: f32, footprint: f32) -> f32 {
    if footprint <= 0.0 {
        return 1.0;
    }
    return smoothstep(footprint * 2.0, footprint * 4.0, wavelength);
}

fn dominant_wind_direction() -> vec2<f32> {
    let raw = material.wave_direction.xy;
    if dot(raw, raw) > 1e-4 {
        return normalize(raw);
    }
    return vec2(1.0, 0.0);
}

fn rotate_local_dir(local_dir: vec2<f32>) -> vec2<f32> {
    let wind    = dominant_wind_direction();
    let lateral = vec2(-wind.y, wind.x);
    return normalize(wind * local_dir.x + lateral * local_dir.y);
}

fn wave_phase_offset(params: vec4<f32>) -> f32 {
    return dot(params.xy, vec2(1.37, 2.91)) + params.z * 0.173 + params.w * 4.123;
}

fn normal_to_slope(normal: vec3<f32>) -> vec2<f32> {
    let inv_y = 1.0 / max(normal.y, 0.001);
    return vec2(-normal.x * inv_y, -normal.z * inv_y);
}

fn slope_to_normal(slope: vec2<f32>) -> vec3<f32> {
    return normalize(vec3(-slope.x, 1.0, -slope.y));
}

fn combine_surface_normal(base_normal: vec3<f32>, detail_slope: vec2<f32>) -> vec3<f32> {
    return slope_to_normal(normal_to_slope(base_normal) + detail_slope);
}

fn gerstner_wave(
    p:         vec2<f32>,
    t:         f32,
    params:    vec4<f32>,
    base_amp:  f32,
    lod_w:     f32,
) -> WaveResult {
    let dir      = rotate_local_dir(params.xy);
    let L        = params.z;
    let Q_artist = params.w;

    let omega = 2.0 * PI / L;
    let amp   = base_amp * GEOM_AMP_RATIO * L;
    let phase = material.wave_speed * sqrt(G * omega);

    let f    = omega * dot(dir, p) + phase * t + wave_phase_offset(params);
    let sinf = sin(f);
    let cosf = cos(f);

    let Q_eff = Q_artist / (omega * amp * NUM_GEOM_WAVES);
    let wa    = omega * amp;
    let xz = Q_eff * amp * dir * cosf;
    let nx = wa * dir.x * cosf;
    let ny = Q_eff * wa * sinf;
    let nz = wa * dir.y * cosf;

    return WaveResult(
        amp * sinf * lod_w,
        xz * lod_w,
        vec3(nx, ny, nz) * lod_w,
    );
}

fn get_wave_result(p: vec2<f32>, footprint: f32) -> WaveResult {
    let t   = globals.time;
    let amp = material.amplitude;

    var h    = 0.0;
    var xz   = vec2(0.0);
    var sumn = vec3(0.0);

    let w0 = gerstner_wave(p, t, GEOM_WAVE_0, amp, wave_lod_weight(GEOM_WAVE_0.z, footprint)); h += w0.height; xz += w0.xz_disp; sumn += w0.normal;
    let w1 = gerstner_wave(p, t, GEOM_WAVE_1, amp, wave_lod_weight(GEOM_WAVE_1.z, footprint)); h += w1.height; xz += w1.xz_disp; sumn += w1.normal;
    let w2 = gerstner_wave(p, t, GEOM_WAVE_2, amp, wave_lod_weight(GEOM_WAVE_2.z, footprint)); h += w2.height; xz += w2.xz_disp; sumn += w2.normal;
    let w3 = gerstner_wave(p, t, GEOM_WAVE_3, amp, wave_lod_weight(GEOM_WAVE_3.z, footprint)); h += w3.height; xz += w3.xz_disp; sumn += w3.normal;
    let w4 = gerstner_wave(p, t, GEOM_WAVE_4, amp, wave_lod_weight(GEOM_WAVE_4.z, footprint)); h += w4.height; xz += w4.xz_disp; sumn += w4.normal;
    let w5 = gerstner_wave(p, t, GEOM_WAVE_5, amp, wave_lod_weight(GEOM_WAVE_5.z, footprint)); h += w5.height; xz += w5.xz_disp; sumn += w5.normal;

    return WaveResult(h, xz, normalize(vec3(-sumn.x, 1.0 - sumn.y, -sumn.z)));
}

fn detail_wave(
    p:        vec2<f32>,
    t:        f32,
    params:   vec4<f32>,
    base_amp: f32,
    lod_w:    f32,
) -> DetailWaveResult {
    let dir        = rotate_local_dir(params.xy);
    let wavelength = params.z;
    let strength   = params.w;
    let omega      = 2.0 * PI / wavelength;
    let amp        = base_amp * DETAIL_AMP_RATIO * wavelength * strength;
    let phase      = material.wave_speed * DETAIL_SPEED_BOOST * sqrt(G * omega);
    let f          = omega * dot(dir, p) + phase * t + wave_phase_offset(params) * 1.71;

    let crest_base = clamp(0.5 + 0.5 * sin(f), 0.0, 1.0);
    let crest      = pow(max(crest_base, 1e-4), DETAIL_SHARPNESS);
    let dcrest_df  =
        0.5 * DETAIL_SHARPNESS * pow(max(crest_base, 1e-4), DETAIL_SHARPNESS - 1.0) * cos(f);
    let slope      = dir * (amp * omega * dcrest_df) * lod_w;

    return DetailWaveResult(slope, crest * lod_w, length(slope));
}

fn get_detail_wave_result(p: vec2<f32>, footprint: f32) -> DetailWaveResult {
    let t   = globals.time;
    let amp = material.amplitude;

    var slope        = vec2(0.0);
    var crest        = 0.0;
    var slope_energy = 0.0;

    let d0 = detail_wave(p, t, DETAIL_WAVE_0, amp, wave_lod_weight(DETAIL_WAVE_0.z, footprint * 0.85)); slope += d0.slope; crest = max(crest, d0.crest); slope_energy += d0.slope_energy;
    let d1 = detail_wave(p, t, DETAIL_WAVE_1, amp, wave_lod_weight(DETAIL_WAVE_1.z, footprint * 0.85)); slope += d1.slope; crest = max(crest, d1.crest); slope_energy += d1.slope_energy;
    let d2 = detail_wave(p, t, DETAIL_WAVE_2, amp, wave_lod_weight(DETAIL_WAVE_2.z, footprint * 0.85)); slope += d2.slope; crest = max(crest, d2.crest); slope_energy += d2.slope_energy;
    let d3 = detail_wave(p, t, DETAIL_WAVE_3, amp, wave_lod_weight(DETAIL_WAVE_3.z, footprint * 0.85)); slope += d3.slope; crest = max(crest, d3.crest); slope_energy += d3.slope_energy;
    let d4 = detail_wave(p, t, DETAIL_WAVE_4, amp, wave_lod_weight(DETAIL_WAVE_4.z, footprint * 0.85)); slope += d4.slope; crest = max(crest, d4.crest); slope_energy += d4.slope_energy;
    let d5 = detail_wave(p, t, DETAIL_WAVE_5, amp, wave_lod_weight(DETAIL_WAVE_5.z, footprint * 0.85)); slope += d5.slope; crest = max(crest, d5.crest); slope_energy += d5.slope_energy;
    let d6 = detail_wave(p, t, DETAIL_WAVE_6, amp, wave_lod_weight(DETAIL_WAVE_6.z, footprint * 0.85)); slope += d6.slope; crest = max(crest, d6.crest); slope_energy += d6.slope_energy;
    let d7 = detail_wave(p, t, DETAIL_WAVE_7, amp, wave_lod_weight(DETAIL_WAVE_7.z, footprint * 0.85)); slope += d7.slope; crest = max(crest, d7.crest); slope_energy += d7.slope_energy;

    return DetailWaveResult(slope, crest, slope_energy);
}

// Returns a normalised surface normal contributed by the swell waves only.
// Only called in the fragment shader — never in the vertex shader — so it
// adds macro variation to the shading without altering the mesh geometry or
// the foam height calculation.
fn get_swell_normal(p: vec2<f32>, footprint: f32) -> vec3<f32> {
    let t   = globals.time;
    let amp = material.amplitude * SWELL_AMP_SCALE;

    var sumn = vec3(0.0);
    let s0 = gerstner_wave(p, t, SWELL_WAVE_0, amp, wave_lod_weight(SWELL_WAVE_0.z, footprint));
    let s1 = gerstner_wave(p, t, SWELL_WAVE_1, amp, wave_lod_weight(SWELL_WAVE_1.z, footprint));
    let s2 = gerstner_wave(p, t, SWELL_WAVE_2, amp, wave_lod_weight(SWELL_WAVE_2.z, footprint));
    sumn += s0.normal;
    sumn += s1.normal;
    sumn += s2.normal;
    return normalize(vec3(-sumn.x, 1.0 - sumn.y, -sumn.z));
}

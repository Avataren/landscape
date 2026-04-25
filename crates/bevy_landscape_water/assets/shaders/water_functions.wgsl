#define_import_path bevy_landscape_water::water_functions

#ifdef PREPASS_PIPELINE
#import bevy_render::globals::Globals
@group(0) @binding(1) var<uniform> globals: Globals;
#else
#import bevy_pbr::mesh_view_bindings::globals
#endif

#import bevy_landscape_water::water_bindings::{material, terrain_height_samp, terrain_height_tex}

const PI: f32  = 3.14159265358979323846;
const G:  f32  = 9.81;
const MAX_TERRAIN_CLIPMAP_LEVELS: i32 = 16;

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
const GEOM_AMP_RATIO:      f32 = 0.0042;
// Swell waves use the same ratio but are scaled down so they add variation
// without fighting the primary geometry normal.
const SWELL_AMP_SCALE:     f32 = 0.26;
const DETAIL_AMP_RATIO:    f32 = 0.0021;
const DETAIL_SHARPNESS:    f32 = 4.2;
const DETAIL_SPEED_BOOST:  f32 = 2.15;

fn wave_lod_weight(wavelength: f32, footprint: f32) -> f32 {
    if footprint <= 0.0 {
        return 1.0;
    }
    return smoothstep(footprint * 2.0, footprint * 4.0, wavelength);
}

fn detail_wave_lod_weight(wavelength: f32, footprint: f32) -> f32 {
    if footprint <= 0.0 {
        return 1.0;
    }
    return smoothstep(footprint * 3.0, footprint * 6.5, wavelength);
}

fn filtered_normal_footprint(footprint: f32) -> f32 {
    return max(footprint * 1.85, footprint + 0.75);
}

fn micro_detail_fade(footprint: f32) -> f32 {
    return 1.0 - smoothstep(0.7, 3.0, footprint);
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

fn water_amplitude() -> f32 {
    return material.wave_params.x;
}

fn water_clarity() -> f32 {
    return material.wave_params.y;
}

fn water_edge_scale() -> f32 {
    return material.wave_params.z;
}

fn water_wave_speed() -> f32 {
    return material.wave_params.w;
}

fn water_refraction_strength() -> f32 {
    return material.optical_params.x;
}

fn water_foam_threshold() -> f32 {
    return material.optical_params.y;
}

fn shoreline_foam_depth() -> f32 {
    return material.optical_params.z;
}

fn shore_wave_damp_width() -> f32 {
    return material.optical_params.w;
}

fn water_surface_height() -> f32 {
    return material.terrain_params.x;
}

fn water_jacobian_foam_strength() -> f32 {
    return material.extra_params.x;
}

fn water_capillary_strength() -> f32 {
    return material.extra_params.y;
}

fn water_macro_noise_amplitude() -> f32 {
    return material.extra_params.z;
}

fn water_macro_noise_scale() -> f32 {
    return max(material.extra_params.w, 1.0);
}

fn terrain_height_scale() -> f32 {
    return material.terrain_params.y;
}

fn terrain_num_levels() -> i32 {
    return i32(material.terrain_params.z + 0.5);
}

fn terrain_data_available() -> bool {
    return terrain_num_levels() > 0
        && all(material.terrain_world_bounds.zw > material.terrain_world_bounds.xy);
}

fn terrain_in_world_bounds(world_xz: vec2<f32>) -> bool {
    if !terrain_data_available() {
        return false;
    }
    return all(world_xz >= material.terrain_world_bounds.xy)
        && all(world_xz <= material.terrain_world_bounds.zw);
}

fn terrain_level_contains(world_xz: vec2<f32>, lvl: vec4<f32>) -> bool {
    if lvl.z <= 0.0 {
        return false;
    }
    let half_span = 0.5 / lvl.z;
    let delta = abs(world_xz - lvl.xy);
    return max(delta.x, delta.y) <= half_span;
}

fn terrain_lod_for_world(world_xz: vec2<f32>) -> i32 {
    let num_levels = terrain_num_levels();
    var fallback_lod = max(num_levels - 1, 0);
    for (var lod = 0; lod < MAX_TERRAIN_CLIPMAP_LEVELS; lod = lod + 1) {
        if lod >= num_levels {
            break;
        }
        let lvl = material.terrain_clip_levels[lod];
        if terrain_level_contains(world_xz, lvl) {
            return lod;
        }
    }
    return fallback_lod;
}

fn terrain_height_at(world_xz: vec2<f32>) -> f32 {
    if !terrain_in_world_bounds(world_xz) {
        return 0.0;
    }

    let lod = terrain_lod_for_world(world_xz);
    let lvl = material.terrain_clip_levels[lod];
    let world_min = material.terrain_world_bounds.xy;
    let world_max = material.terrain_world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(world_xz, world_min, world_max);
    let uv = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    // terrain_height_tex is R32Float storing world-space metres directly.
    return textureSampleLevel(terrain_height_tex, terrain_height_samp, uv, lod, 0.0).r;
}

fn terrain_water_depth_at(world_xz: vec2<f32>) -> f32 {
    if !terrain_in_world_bounds(world_xz) {
        return 1e6;
    }
    return max(water_surface_height() - terrain_height_at(world_xz), 0.0);
}

fn shoreline_wave_attenuation(world_xz: vec2<f32>) -> f32 {
    if !terrain_in_world_bounds(world_xz) {
        return 1.0;
    }
    let depth = terrain_water_depth_at(world_xz);
    return smoothstep(0.0, max(shore_wave_damp_width(), 0.001), depth);
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
    let phase = water_wave_speed() * sqrt(G * omega);

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

// Per-wave horizontal-displacement gradient terms for Jacobian foldover.
// Returns (∂xz_disp.x/∂x, ∂xz_disp.x/∂z, ∂xz_disp.z/∂z) — note ∂xz_disp.z/∂x
// equals ∂xz_disp.x/∂z for Gerstner waves so we collapse to three components.
fn gerstner_disp_grad(
    p:        vec2<f32>,
    t:        f32,
    params:   vec4<f32>,
    base_amp: f32,
    lod_w:    f32,
) -> vec3<f32> {
    let dir      = rotate_local_dir(params.xy);
    let L        = params.z;
    let Q_artist = params.w;

    let omega = 2.0 * PI / L;
    let amp   = base_amp * GEOM_AMP_RATIO * L;
    let phase = water_wave_speed() * sqrt(G * omega);

    let f    = omega * dot(dir, p) + phase * t + wave_phase_offset(params);
    let sinf = sin(f);

    let Q_eff = Q_artist / (omega * amp * NUM_GEOM_WAVES);
    // ∂(Q_eff·amp·D_x·cos f)/∂x = -Q_eff·amp·omega·D_x² · sin f, etc.
    let k = -Q_eff * amp * omega * sinf * lod_w;
    return vec3(k * dir.x * dir.x, k * dir.x * dir.y, k * dir.y * dir.y);
}

// Aggregate displacement gradient over all geometry waves. Foldover is detected
// via the Jacobian J = (1+gxx)(1+gzz) - gxz²; J < 0 means the surface has
// folded onto itself (wave breaking).
fn get_wave_disp_grad(p: vec2<f32>, footprint: f32) -> vec3<f32> {
    let t   = globals.time;
    let amp = water_amplitude();
    var g = vec3(0.0);
    g += gerstner_disp_grad(p, t, GEOM_WAVE_0, amp, wave_lod_weight(GEOM_WAVE_0.z, footprint));
    g += gerstner_disp_grad(p, t, GEOM_WAVE_1, amp, wave_lod_weight(GEOM_WAVE_1.z, footprint));
    g += gerstner_disp_grad(p, t, GEOM_WAVE_2, amp, wave_lod_weight(GEOM_WAVE_2.z, footprint));
    g += gerstner_disp_grad(p, t, GEOM_WAVE_3, amp, wave_lod_weight(GEOM_WAVE_3.z, footprint));
    g += gerstner_disp_grad(p, t, GEOM_WAVE_4, amp, wave_lod_weight(GEOM_WAVE_4.z, footprint));
    g += gerstner_disp_grad(p, t, GEOM_WAVE_5, amp, wave_lod_weight(GEOM_WAVE_5.z, footprint));
    return g;
}

fn get_wave_result(p: vec2<f32>, footprint: f32) -> WaveResult {
    let t   = globals.time;
    let amp = water_amplitude();

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
    let phase      = water_wave_speed() * DETAIL_SPEED_BOOST * sqrt(G * omega);
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
    let amp = water_amplitude();

    var slope        = vec2(0.0);
    var crest        = 0.0;
    var slope_energy = 0.0;

    let d0 = detail_wave(p, t, DETAIL_WAVE_0, amp, detail_wave_lod_weight(DETAIL_WAVE_0.z, footprint)); slope += d0.slope; crest = max(crest, d0.crest); slope_energy += d0.slope_energy;
    let d1 = detail_wave(p, t, DETAIL_WAVE_1, amp, detail_wave_lod_weight(DETAIL_WAVE_1.z, footprint)); slope += d1.slope; crest = max(crest, d1.crest); slope_energy += d1.slope_energy;
    let d2 = detail_wave(p, t, DETAIL_WAVE_2, amp, detail_wave_lod_weight(DETAIL_WAVE_2.z, footprint)); slope += d2.slope; crest = max(crest, d2.crest); slope_energy += d2.slope_energy;
    let d3 = detail_wave(p, t, DETAIL_WAVE_3, amp, detail_wave_lod_weight(DETAIL_WAVE_3.z, footprint)); slope += d3.slope; crest = max(crest, d3.crest); slope_energy += d3.slope_energy;
    let d4 = detail_wave(p, t, DETAIL_WAVE_4, amp, detail_wave_lod_weight(DETAIL_WAVE_4.z, footprint)); slope += d4.slope; crest = max(crest, d4.crest); slope_energy += d4.slope_energy;
    let d5 = detail_wave(p, t, DETAIL_WAVE_5, amp, detail_wave_lod_weight(DETAIL_WAVE_5.z, footprint)); slope += d5.slope; crest = max(crest, d5.crest); slope_energy += d5.slope_energy;
    let d6 = detail_wave(p, t, DETAIL_WAVE_6, amp, detail_wave_lod_weight(DETAIL_WAVE_6.z, footprint)); slope += d6.slope; crest = max(crest, d6.crest); slope_energy += d6.slope_energy;
    let d7 = detail_wave(p, t, DETAIL_WAVE_7, amp, detail_wave_lod_weight(DETAIL_WAVE_7.z, footprint)); slope += d7.slope; crest = max(crest, d7.crest); slope_energy += d7.slope_energy;

    return DetailWaveResult(slope, crest, slope_energy);
}

// -----------------------------------------------------------------------
// 2-D value noise primitive shared by macro-noise and capillary-noise
// passes.  Returns (value ∈ [0,1], dvalue/dx, dvalue/dy).
// -----------------------------------------------------------------------
fn cap_hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

// Two-channel hash for voronoi cell offsets.
fn voro_hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(p, vec2(127.1, 311.7)),
        dot(p, vec2(269.5, 183.3)),
    );
    return fract(sin(q) * 43758.5453);
}

// 2D voronoi (worley) noise.  Returns:
//   .x = distance to nearest cell point (0 at cell centre, ~1 at cell edge)
//   .y = distance to second-nearest cell point
// (.y - .x) is small along Voronoi edges, large inside cells — useful for
// extracting bubble-edge silhouettes.
fn voronoi_f1f2(p: vec2<f32>) -> vec2<f32> {
    let n = floor(p);
    let f = p - n;
    var f1 = 8.0;
    var f2 = 8.0;
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let g = vec2<f32>(f32(i), f32(j));
            let o = voro_hash22(n + g);
            let r = g + o - f;
            let d = dot(r, r);
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    return vec2(sqrt(f1), sqrt(f2));
}

// Foam mask from voronoi.  Two octaves at related but irrational scales so
// the bubble pattern doesn't tile.  `coverage` ∈ [0,1] controls how much of
// the mask is filled (0 = sparse bubbles, 1 = full white).
// `cell_size` is the world-space cell size in metres.
// `drift_speed` (m/s) scrolls cells along the wind direction over time so
// foam patches translate naturally with the surface.
fn foam_voronoi_mask(
    p:           vec2<f32>,
    cell_size:   f32,
    drift_speed: f32,
    coverage:    f32,
) -> f32 {
    let scale  = 1.0 / max(cell_size, 0.05);
    let wind   = dominant_wind_direction();
    let scroll = -wind * drift_speed * globals.time;
    let q1 = (p + scroll) * scale;
    let v1 = voronoi_f1f2(q1);
    // Bubble bodies: bright at cell centres (low f1), dimmer near edges.
    let bubbles = 1.0 - smoothstep(0.18, 0.62, v1.x);
    // Edge ridges: bright lines along Voronoi cell boundaries.
    let edges   = 1.0 - smoothstep(0.0, 0.10, v1.y - v1.x);
    let layer1  = clamp(bubbles + edges * 0.35, 0.0, 1.0);

    // Second octave at ~1.7× scale, offset, decorrelates the pattern.
    let q2 = (p + scroll * 0.65 + vec2(13.7, 47.1)) * (scale * 1.73);
    let v2 = voronoi_f1f2(q2);
    let layer2 = 1.0 - smoothstep(0.20, 0.65, v2.x);

    let combined = mix(layer1, layer1 * (0.55 + 0.45 * layer2), 0.6);
    // Coverage maps the mask through smoothstep so a single slider goes from
    // sparse foam pockets to a continuous foam sheet.
    let lo = mix(0.85, 0.05, coverage);
    let hi = mix(0.95, 0.40, coverage);
    return smoothstep(lo, hi, combined);
}

fn cap_value_noise_d(p: vec2<f32>) -> vec3<f32> {
    let i  = floor(p);
    let f  = p - i;
    let u  = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);

    let a = cap_hash21(i);
    let b = cap_hash21(i + vec2(1.0, 0.0));
    let c = cap_hash21(i + vec2(0.0, 1.0));
    let d = cap_hash21(i + vec2(1.0, 1.0));

    let v    = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    let dvdx = du.x * ((b - a) + (a - b - c + d) * u.y);
    let dvdy = du.y * ((c - a) + (a - b - c + d) * u.x);
    return vec3(v, dvdx, dvdy);
}

// -----------------------------------------------------------------------
// Macro height-noise — long-wavelength stochastic FBM that breaks the
// strictly periodic Gerstner sum.  Evaluated in the vertex shader (height
// displacement) AND the fragment shader (slope contribution).  Crucially
// the longest octave is footprint-immune so it survives all the way to the
// horizon, where the high-frequency Gerstner waves alias away.
//
// Returns (height, dh/dx, dh/dz) in world units.
// -----------------------------------------------------------------------
fn macro_noise_octave(
    p:        vec2<f32>,
    t:        f32,
    wind:     vec2<f32>,
    cross:    vec2<f32>,
    wavelen:  f32,
    drift_a:  f32,
    drift_b:  f32,
    height:   f32,
    seed:     vec2<f32>,
    footprint: f32,
) -> vec3<f32> {
    // Fade if the octave is finer than the pixel footprint (cheap aliasing
    // guard — the longest octave at default settings has wavelen = 80 m so
    // it never fades at typical view distances).
    let lod_w = 1.0 - smoothstep(wavelen * 0.4, wavelen * 1.0, footprint);
    if lod_w <= 0.0 {
        return vec3(0.0);
    }
    let freq  = 1.0 / wavelen;
    let drift = wind * drift_a + cross * drift_b;
    let pp    = (p - drift * t) * freq + seed;
    let n     = cap_value_noise_d(pp);
    // Centre value around 0 so vertices both lift and sink relative to rest.
    let h     = (n.x - 0.5) * 2.0 * height * lod_w;
    let dhdx  = n.y * 2.0 * height * freq * lod_w;
    let dhdz  = n.z * 2.0 * height * freq * lod_w;
    return vec3(h, dhdx, dhdz);
}

fn macro_noise_height_grad(p: vec2<f32>, footprint: f32) -> vec3<f32> {
    let amp = water_macro_noise_amplitude();
    if amp <= 0.0 {
        return vec3(0.0);
    }
    let t     = globals.time;
    let wind  = dominant_wind_direction();
    let cross = vec2(-wind.y, wind.x);
    let base  = water_macro_noise_scale();

    var sum = vec3(0.0);
    // Three octaves at base × {1.0, 0.46, 0.22}.  Amplitude tapers ~× 0.55
    // per octave (Phillips-like roll-off) so the macro signal is dominated by
    // the longest wavelength — the bit that survives at the horizon.
    sum += macro_noise_octave(p, t, wind, cross, base * 1.00, 0.18, 0.06, amp * 1.00,  vec2( 0.0,  0.0), footprint);
    sum += macro_noise_octave(p, t, wind, cross, base * 0.46, 0.27, 0.09, amp * 0.55,  vec2(31.7, -7.1), footprint);
    sum += macro_noise_octave(p, t, wind, cross, base * 0.22, 0.42, 0.13, amp * 0.30,  vec2(-12.3, 19.5), footprint);
    return sum;
}

// -----------------------------------------------------------------------
// Capillary / micro-noise normal layer
//
// Procedural value noise FBM with analytic gradient.  Adds stochastic
// high-frequency surface roughness that the regular Gerstner sums cannot
// produce — kills the visible periodicity of pure analytic waves and gives
// sun glitter something to scintillate on.
// -----------------------------------------------------------------------
// Returns the surface-slope contribution from a 3-octave scrolling FBM.
// Per-octave amplitude is calibrated for peak per-octave slope ≈ 0.04, so the
// summed perturbation stays within the same order of magnitude as the existing
// detail-wave slope (≈0.2 sustained).
fn capillary_octave(
    p:        vec2<f32>,
    t:        f32,
    wind:     vec2<f32>,
    cross:    vec2<f32>,
    wavelen:  f32,
    drift_a:  f32,
    drift_b:  f32,
    height:   f32,
    seed:     vec2<f32>,
    footprint: f32,
) -> vec2<f32> {
    // Fade out octaves that fall below the pixel footprint to avoid alias.
    let lod_w = 1.0 - smoothstep(wavelen * 0.6, wavelen * 1.6, footprint);
    if lod_w <= 0.0 {
        return vec2(0.0);
    }
    let freq  = 1.0 / wavelen;
    let drift = wind * drift_a + cross * drift_b;
    let pp    = (p - drift * t) * freq + seed;
    let n     = cap_value_noise_d(pp);
    return vec2(n.y, n.z) * freq * height * lod_w;
}

fn capillary_slope(p: vec2<f32>, footprint: f32) -> vec2<f32> {
    let t     = globals.time;
    let wind  = dominant_wind_direction();
    let cross = vec2(-wind.y, wind.x);

    var slope = vec2(0.0);
    slope += capillary_octave(p, t, wind, cross, 3.0, 0.55, 0.18, 0.085, vec2( 0.0,  0.0), footprint);
    slope += capillary_octave(p, t, wind, cross, 1.1, 0.95, 0.22, 0.024, vec2(17.3, 41.7), footprint);
    slope += capillary_octave(p, t, wind, cross, 0.4, 1.30, 0.31, 0.0058, vec2(-9.1, 23.5), footprint);
    return slope;
}

// Returns a normalised surface normal contributed by the swell waves only.
// Only called in the fragment shader — never in the vertex shader — so it
// adds macro variation to the shading without altering the mesh geometry or
// the foam height calculation.
fn get_swell_normal(p: vec2<f32>, footprint: f32) -> vec3<f32> {
    let t   = globals.time;
    let amp = water_amplitude() * SWELL_AMP_SCALE;

    var sumn = vec3(0.0);
    let s0 = gerstner_wave(p, t, SWELL_WAVE_0, amp, wave_lod_weight(SWELL_WAVE_0.z, footprint));
    let s1 = gerstner_wave(p, t, SWELL_WAVE_1, amp, wave_lod_weight(SWELL_WAVE_1.z, footprint));
    let s2 = gerstner_wave(p, t, SWELL_WAVE_2, amp, wave_lod_weight(SWELL_WAVE_2.z, footprint));
    sumn += s0.normal;
    sumn += s1.normal;
    sumn += s2.normal;
    return normalize(vec3(-sumn.x, 1.0 - sumn.y, -sumn.z));
}

// detail_synthesis.wgsl
// GPU compute pass: writes procedural height residual (world-space metres) into
// one layer of the R32Float detail texture array.
//
// Dispatch once per synthesis LOD level: ceil(res/8) × ceil(res/8) workgroups.
// The output is read by terrain_vertex.wgsl to add sub-metre detail on top of
// the coarse source heightmap.

// ---- Noise primitives (matching terrain_noise.wgsl) ----

const GRADIENT_EPSILON: f32 = 0.37;
const EROSION_RESPONSE:  f32 = 3.5;

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3)),
    );
    return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(hash2(i + vec2<f32>(0.0, 0.0)), f - vec2<f32>(0.0, 0.0));
    let b = dot(hash2(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
    let c = dot(hash2(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
    let d = dot(hash2(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Domain rotation ~36.87° between octaves to reduce grid alignment artefacts.
fn rot2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y);
}

// Gradient-attenuated fBM mimicking erosion.
// When erosion == 0, the attenuation factor is always 1 → plain fBM.
fn erosion_shaped_fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32, erosion: f32) -> f32 {
    var value    = 0.0;
    var amplitude = 0.5;
    var pos      = base;
    var acc_grad = vec2<f32>(0.0, 0.0);
    for (var i = 0u; i < octaves; i++) {
        let n = gradient_noise(pos);
        let grad = vec2<f32>(
            gradient_noise(pos + vec2<f32>(GRADIENT_EPSILON, 0.0)) - n,
            gradient_noise(pos + vec2<f32>(0.0, GRADIENT_EPSILON)) - n,
        ) / GRADIENT_EPSILON;
        acc_grad += grad * amplitude;
        let atten = mix(1.0, 1.0 / (1.0 + dot(acc_grad, acc_grad) * EROSION_RESPONSE), erosion);
        value    += amplitude * n * atten;
        pos       = rot2(pos) * lac;
        amplitude *= g;
    }
    return value;
}

// ---- Synthesis uniform (std140 — 12 × f32/u32 = 48 bytes) ----

struct SynthesisParams {
    clip_center_x:      f32,   // world-space X of clipmap ring centre
    clip_center_z:      f32,   // world-space Z of clipmap ring centre
    texel_world_size:   f32,   // world-space size of one clipmap texel at this LOD
    clipmap_res:        f32,   // texture resolution (e.g. 512.0)
    octave_count:       u32,   // fBM octave count for this LOD
    source_lod_spacing: f32,   // world-space texel size of the source heightmap
    max_amplitude:      f32,   // maximum residual height in world-space metres
    lacunarity:         f32,   // fBM frequency multiplier per octave
    gain:               f32,   // fBM amplitude multiplier per octave
    erosion_strength:   f32,   // 0 = plain fBM, 1 = fully erosion-shaped
    seed_x:             f32,   // XZ domain offset for per-terrain uniqueness
    seed_z:             f32,
}

@group(0) @binding(0) var<uniform> params: SynthesisParams;
// Per-layer D2 view of the detail R32Float array (one dispatch per LOD layer).
@group(0) @binding(1) var detail_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn synthesize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res   = u32(params.clipmap_res);
    let res_i = i32(res);
    if gid.x >= res || gid.y >= res { return; }

    let half_i = res_i / 2;
    let col    = i32(gid.x);
    let row    = i32(gid.y);

    // Integer texel-space centre of this LOD ring.
    let center_tx = i32(floor(params.clip_center_x / params.texel_world_size));
    let center_tz = i32(floor(params.clip_center_z / params.texel_world_size));

    // Global texel index for this thread (column/row scan across the ring).
    let gx = center_tx - half_i + col;
    let gz = center_tz - half_i + row;

    // World-space XZ for noise evaluation.
    let world_x = f32(gx) * params.texel_world_size;
    let world_z  = f32(gz) * params.texel_world_size;

    // Toroidal write coordinate — mirrors CPU rem_euclid convention:
    //   tx = gx.rem_euclid(N)
    let tx = ((gx % res_i) + res_i) % res_i;
    let tz = ((gz % res_i) + res_i) % res_i;

    // Base frequency: start at half-Nyquist of the source heightmap, i.e. the
    // finest feature the source can represent.  fBM octaves double the frequency
    // each step, filling down toward the LOD Nyquist (2 × texel_world_size).
    let base_freq = 2.0 / params.source_lod_spacing;
    let p = vec2<f32>(world_x + params.seed_x, world_z + params.seed_z) * base_freq;

    // erosion_shaped_fbm output is approximately in [-0.5, 0.5].
    let n        = erosion_shaped_fbm(p, params.octave_count, params.lacunarity, params.gain, params.erosion_strength);
    let detail_m = n * params.max_amplitude;

    textureStore(detail_out, vec2<i32>(tx, tz), vec4<f32>(detail_m, 0.0, 0.0, 0.0));
}

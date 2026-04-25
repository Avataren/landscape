// detail_synthesis.wgsl
// GPU compute pass: writes full height (bilinear source + fBM detail) in
// world-space metres into one layer of the R32Float clipmap height array.
//
// Dispatch once per synthesis LOD level: ceil(res/8) × ceil(res/8) workgroups.
// The output replaces the CPU-uploaded tile data for fine LODs, so the vertex
// shader reads full detail directly from height_tex without a separate residual.

// ---- Noise primitives ----
//
// Uses an integer PCG hash so the noise field is bit-identical between the
// GPU compute pass and the CPU collision-mesh builder (synthesis_cpu.rs).
// The float `fract(sin(x) * 43758)` hack used previously was not portable —
// CPU and GPU `sin()` differ in the low bits, which `fract()` amplifies.

const GRADIENT_EPSILON: f32 = 0.37;
const EROSION_RESPONSE:  f32 = 3.5;
const U32_TO_UNIT:       f32 = 2.3283064e-10; // 1.0 / 4294967295.0

fn pcg2d(v_in: vec2<u32>) -> vec2<u32> {
    var v = v_in * vec2<u32>(1664525u) + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return v;
}

fn hash_grad(pi: vec2<i32>) -> vec2<f32> {
    let h = pcg2d(bitcast<vec2<u32>>(pi));
    return vec2<f32>(h) * (2.0 * U32_TO_UNIT) - vec2<f32>(1.0);
}

fn gradient_noise(p: vec2<f32>) -> f32 {
    let pf = floor(p);
    let i  = vec2<i32>(pf);
    let f  = p - pf;
    let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a  = dot(hash_grad(i + vec2<i32>(0, 0)), f);
    let b  = dot(hash_grad(i + vec2<i32>(1, 0)), f - vec2<f32>(1.0, 0.0));
    let c  = dot(hash_grad(i + vec2<i32>(0, 1)), f - vec2<f32>(0.0, 1.0));
    let d  = dot(hash_grad(i + vec2<i32>(1, 1)), f - vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn rot2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y);
}

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

// ---- Synthesis uniform (std140 — 20 × f32/u32 = 80 bytes, 16-aligned) ----

struct SynthesisParams {
    clip_center_x:        f32,   //  0 world-space X of clipmap ring centre
    clip_center_z:        f32,   //  4 world-space Z of clipmap ring centre
    texel_world_size:     f32,   //  8 world-space size of one clipmap texel
    clipmap_res:          f32,   // 12 texture resolution (e.g. 512.0)
    octave_count:         u32,   // 16 fBM octave count for this LOD
    source_lod_spacing:   f32,   // 20 world-space texel size of the source heightmap
    max_amplitude:        f32,   // 24 maximum detail height in world-space metres
    lacunarity:           f32,   // 28 fBM frequency multiplier per octave
    gain:                 f32,   // 32 fBM amplitude multiplier per octave
    erosion_strength:     f32,   // 36 0 = plain fBM, 1 = fully erosion-shaped
    seed_x:               f32,   // 40 XZ domain offset for per-terrain uniqueness
    seed_z:               f32,   // 44
    source_origin_x:      f32,   // 48 world-space X of source heightmap origin
    source_origin_z:      f32,   // 52 world-space Z of source heightmap origin
    source_extent_x:      f32,   // 56 world-space X size of source heightmap
    source_extent_z:      f32,   // 60 world-space Z size of source heightmap
    height_scale:         f32,   // 64 [0,1] → world-space metres multiplier
    slope_mask_threshold: f32,   // 68 slope angle (°) where detail starts fading
    slope_mask_falloff:   f32,   // 72 fade band width (°) above threshold
    _pad:                 f32,   // 76
}

@group(0) @binding(0) var<uniform> params: SynthesisParams;
// Per-layer D2 storage view of the R32Float clipmap height array.
@group(0) @binding(1) var clipmap_out: texture_storage_2d<r32float, write>;
// Source heightmap (R16Unorm, full world coverage at max_mip_level resolution).
@group(0) @binding(2) var source_heightmap: texture_2d<f32>;
@group(0) @binding(3) var source_samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn synthesize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res   = u32(params.clipmap_res);
    let res_i = i32(res);
    if gid.x >= res || gid.y >= res { return; }

    let half_i = res_i / 2;
    let col    = i32(gid.x);
    let row    = i32(gid.y);

    let center_tx = i32(floor(params.clip_center_x / params.texel_world_size));
    let center_tz = i32(floor(params.clip_center_z / params.texel_world_size));

    let gx = center_tx - half_i + col;
    let gz = center_tz - half_i + row;

    let world_x = f32(gx) * params.texel_world_size;
    let world_z = f32(gz) * params.texel_world_size;

    // Toroidal write coordinate.
    let tx = ((gx % res_i) + res_i) % res_i;
    let tz = ((gz % res_i) + res_i) % res_i;

    // ---- Source heightmap sampling ----
    // One source texel in UV space (for finite-difference slope estimation).
    let src_du = params.source_lod_spacing / params.source_extent_x;
    let src_dv = params.source_lod_spacing / params.source_extent_z;

    let src_uv = clamp(
        (vec2<f32>(world_x, world_z) - vec2<f32>(params.source_origin_x, params.source_origin_z))
            / vec2<f32>(params.source_extent_x, params.source_extent_z),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );

    let h_norm     = textureSampleLevel(source_heightmap, source_samp, src_uv, 0.0).r;
    let h_norm_x   = textureSampleLevel(source_heightmap, source_samp,
                         clamp(src_uv + vec2<f32>(src_du, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;
    let h_norm_z   = textureSampleLevel(source_heightmap, source_samp,
                         clamp(src_uv + vec2<f32>(0.0, src_dv), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;

    let base_h = h_norm * params.height_scale;

    // ---- Slope masking ----
    // Finite-difference slope from the source heightmap in world-space m/m.
    // Steep macro-terrain (cliffs) gets no fBM — real cliff faces are smooth rock.
    let dh_dx = (h_norm_x - h_norm) / params.source_lod_spacing * params.height_scale;
    let dh_dz = (h_norm_z - h_norm) / params.source_lod_spacing * params.height_scale;
    let slope_deg = degrees(atan(sqrt(dh_dx * dh_dx + dh_dz * dh_dz)));
    let slope_mask = 1.0 - smoothstep(
        params.slope_mask_threshold,
        params.slope_mask_threshold + max(params.slope_mask_falloff, 0.1),
        slope_deg,
    );

    // ---- fBM detail ----
    let base_freq = 2.0 / params.source_lod_spacing;
    let p = vec2<f32>(world_x + params.seed_x, world_z + params.seed_z) * base_freq;
    let n = erosion_shaped_fbm(
        p, params.octave_count, params.lacunarity, params.gain, params.erosion_strength,
    );
    let detail_m = n * params.max_amplitude * slope_mask;

    textureStore(clipmap_out, vec2<i32>(tx, tz), vec4<f32>(base_h + detail_m, 0.0, 0.0, 0.0));
}

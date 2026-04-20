const CONTINENT_OCTAVES: u32 = 4u;
const WARP_OCTAVES: u32 = 3u;
const CHANNEL_OCTAVES: u32 = 4u;
const EROSION_RESPONSE: f32 = 3.5;
const GRADIENT_EPSILON: f32 = 0.37;

struct GeneratorParams {
    resolution:          vec2<u32>,
    octaves:             u32,
    seed:                u32,
    offset:              vec2<f32>,
    frequency:           f32,
    lacunarity:          f32,
    gain:                f32,
    height_scale:        f32,
    continent_frequency: f32,
    continent_strength:  f32,
    ridge_strength:      f32,
    warp_frequency:      f32,
    warp_strength:       f32,
    erosion_strength:    f32,
    grayscale:           u32,
}

struct DownsampleParams {
    src_resolution: vec2<u32>,
    dst_resolution: vec2<u32>,
}

struct NormalParams {
    resolution: vec2<u32>,
    effective_height_scale: f32,
    lod_scale: f32,
}

@group(0) @binding(0) var<uniform> params: GeneratorParams;
@group(1) @binding(0) var preview_output: texture_storage_2d<rgba32float, read_write>;

// === Preview normalization resources (used by the 3-pass normalized preview) ===
// binding 6: [min_bits, max_bits] as u32 (IEEE 754 — valid for positive floats)
// binding 7: raw scalar heights before normalization
@group(1) @binding(6) var<storage, read_write> minmax_buf: array<atomic<u32>, 2>;
@group(1) @binding(7) var raw_heights: texture_storage_2d<r32float, read_write>;

var<workgroup> wg_min: atomic<u32>;
var<workgroup> wg_max: atomic<u32>;

@group(0) @binding(1) var<uniform> export_params: GeneratorParams;
@group(1) @binding(1) var export_output: texture_storage_2d<r32float, write>;

@group(0) @binding(2) var<uniform> downsample_params: DownsampleParams;
@group(1) @binding(2) var downsample_src: texture_2d<f32>;
@group(1) @binding(3) var downsample_dst: texture_storage_2d<r32float, write>;

@group(0) @binding(3) var<uniform> normal_params: NormalParams;
@group(1) @binding(4) var normal_src: texture_2d<f32>;
@group(1) @binding(5) var normal_dst: texture_storage_2d<rgba8snorm, write>;

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn remap01(v: f32) -> f32 {
    return saturate(v * 0.5 + 0.5);
}

// IQ-style gradient hash: symmetric treatment of both x and y.
// Uses sin() for reliable cross-platform consistency.
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

// Domain rotation between fBM octaves breaks axis-aligned grid patterns.
// ~36.87° rotation (3-4-5 triangle), as recommended by IQ.
fn rot2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y);
}

fn fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = base;

    for (var i = 0u; i < octaves; i++) {
        value += amplitude * gradient_noise(pos);
        pos = rot2(pos) * lac;
        amplitude *= g;
    }

    return value;
}

fn ridged_fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var total = 0.0;
    var pos = base;

    for (var i = 0u; i < octaves; i++) {
        value += amplitude * (1.0 - abs(gradient_noise(pos)));
        total += amplitude;
        pos = rot2(pos) * lac;
        amplitude *= g;
    }

    if total > 0.0 {
        return value / total;
    }
    return 0.0;
}

fn erosion_shaped_fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32, erosion: f32) -> f32 {
    if erosion <= 0.0 {
        return fbm(base, octaves, lac, g);
    }

    var value = 0.0;
    var amplitude = 0.5;
    var pos = base;
    var acc_grad = vec2<f32>(0.0, 0.0);

    for (var i = 0u; i < octaves; i++) {
        let n = gradient_noise(pos);
        let grad = vec2<f32>(
            gradient_noise(pos + vec2<f32>(GRADIENT_EPSILON, 0.0)) - n,
            gradient_noise(pos + vec2<f32>(0.0, GRADIENT_EPSILON)) - n,
        ) / GRADIENT_EPSILON;
        acc_grad += grad * amplitude;

        let attenuation = mix(
            1.0,
            1.0 / (1.0 + dot(acc_grad, acc_grad) * EROSION_RESPONSE),
            erosion,
        );
        value += amplitude * n * attenuation;

        pos = rot2(pos) * lac;
        amplitude *= g;
    }

    return value;
}

fn continent_mask(pos: vec2<f32>) -> f32 {
    let continents = remap01(fbm(pos, CONTINENT_OCTAVES, 2.02, 0.55));
    return smoothstep(0.28, 0.72, continents);
}

fn domain_warp(base_uv: vec2<f32>, warp_frequency: f32, warp_strength: f32) -> vec2<f32> {
    let clamped_strength = clamp(warp_strength, 0.0, 2.0);
    if clamped_strength <= 0.0 {
        return base_uv;
    }

    let warp_pos = base_uv * max(warp_frequency, 0.05);
    let warp = vec2<f32>(
        fbm(warp_pos + vec2<f32>(5.2, 1.3), WARP_OCTAVES, 2.0, 0.5),
        fbm(warp_pos + vec2<f32>(8.3, -2.8), WARP_OCTAVES, 2.0, 0.5),
    );
    return base_uv + warp * clamped_strength;
}

fn terrain_height_for(params_ref: GeneratorParams, uv: vec2<f32>) -> f32 {
    let seed_off = vec2<f32>(
        f32(params_ref.seed) * 0.47316,
        f32(params_ref.seed) * 0.31419,
    );
    let base_uv = uv + params_ref.offset + seed_off;
    let detail_uv = domain_warp(base_uv, params_ref.warp_frequency, params_ref.warp_strength);
    let detail_pos = detail_uv * max(params_ref.frequency, 0.001);
    let octaves = max(params_ref.octaves, 1u);
    let lacunarity = max(params_ref.lacunarity, 1.01);
    let gain = clamp(params_ref.gain, 0.05, 0.95);
    let erosion = saturate(params_ref.erosion_strength);

    let detail = erosion_shaped_fbm(detail_pos, octaves, lacunarity, gain, erosion);

    var ridge_octaves = 1u;
    if octaves > 1u { ridge_octaves = octaves - 1u; }
    let ridges = ridged_fbm(
        detail_pos * 0.85 + vec2<f32>(17.13, 17.13),
        ridge_octaves,
        lacunarity,
        gain,
    );

    let base_height = remap01(detail);
    let mountainous = mix(base_height, ridges, saturate(params_ref.ridge_strength));

    let continent = continent_mask(
        base_uv * max(params_ref.continent_frequency, 0.05) + vec2<f32>(31.7, -22.9),
    );
    let continental_height = mountainous * (0.18 + 0.82 * continent) + continent * 0.18 - 0.09;

    var height = mix(
        mountainous,
        continental_height,
        saturate(params_ref.continent_strength),
    );

    let channels = ridged_fbm(
        (detail_uv + vec2<f32>(-13.5, 21.4)) * (params_ref.frequency * 0.55 + 0.35),
        CHANNEL_OCTAVES,
        2.05,
        0.55,
    );
    let channel_mask = channels * channels;
    let channel_mask_pow4 = channel_mask * channel_mask;
    let highlands = smoothstep(0.28, 0.82, mountainous);
    height -= erosion * channel_mask_pow4 * highlands * (0.03 + 0.11 * continent);

    return saturate(height);
}

fn clamp_texel(coord: vec2<i32>, resolution: vec2<u32>) -> vec2<i32> {
    return clamp(
        coord,
        vec2<i32>(0, 0),
        vec2<i32>(i32(resolution.x) - 1, i32(resolution.y) - 1),
    );
}

fn sample_height_texel(src: texture_2d<f32>, coord: vec2<i32>, resolution: vec2<u32>) -> f32 {
    return textureLoad(src, clamp_texel(coord, resolution), 0).x;
}

fn preview_palette(h: f32) -> vec3<f32> {
    let contrast_h = smoothstep(0.14, 0.9, pow(h, 0.72));
    let foothills = vec3<f32>(0.10, 0.12, 0.09);
    let uplands = vec3<f32>(0.30, 0.38, 0.21);
    let rock = vec3<f32>(0.60, 0.53, 0.39);
    let peaks = vec3<f32>(0.94, 0.92, 0.88);

    let low_to_mid = mix(foothills, uplands, smoothstep(0.06, 0.38, contrast_h));
    let mid_to_high = mix(low_to_mid, rock, smoothstep(0.34, 0.68, contrast_h));
    return mix(mid_to_high, peaks, smoothstep(0.64, 0.94, contrast_h));
}

fn preview_color(uv: vec2<f32>) -> vec3<f32> {
    let res = vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
    let texel = 1.0 / res;

    let h = terrain_height_for(params, uv);
    let hx0 = terrain_height_for(params, uv - vec2<f32>(texel.x, 0.0));
    let hx1 = terrain_height_for(params, uv + vec2<f32>(texel.x, 0.0));
    let hy0 = terrain_height_for(params, uv - vec2<f32>(0.0, texel.y));
    let hy1 = terrain_height_for(params, uv + vec2<f32>(0.0, texel.y));

    let dx = (hx1 - hx0) * 4.6;
    let dy = (hy1 - hy0) * 4.6;
    let normal = normalize(vec3<f32>(-dx, 0.48, -dy));
    let light_dir = normalize(vec3<f32>(-0.62, 0.74, 0.28));

    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = 0.24 + 0.34 * (1.0 - normal.y);
    let ridge_boost = smoothstep(0.12, 0.78, 1.0 - normal.y) * 0.30;
    let shade = pow(saturate(ambient + diffuse * 0.78 + ridge_boost), 0.88);

    let base = preview_palette(h);
    let height_boost = smoothstep(0.22, 0.92, h) * 0.12;
    return clamp(base * shade + vec3<f32>(height_boost), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y {
        return;
    }

    let res = vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5, 0.5)) / res;

    var color: vec3<f32>;
    if params.grayscale != 0u {
        let h = terrain_height_for(params, uv);
        color = vec3<f32>(h, h, h);
    } else {
        color = preview_color(uv);
    }

    textureStore(preview_output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}

// === Normalized preview — 3-pass pipeline ===

// Pass 1: write one raw height value per texel into raw_heights.
@compute @workgroup_size(8, 8, 1)
fn preview_generate_raw(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y {
        return;
    }
    let res = vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + 0.5) / res;
    let h = terrain_height_for(params, uv);
    textureStore(raw_heights, vec2<i32>(id.xy), vec4<f32>(h, 0.0, 0.0, 1.0));
}

// Pass 2: two-level parallel reduction — workgroup shared memory, then one global atomic
// per workgroup.  minmax_buf must be pre-initialised to [0x7F7FFFFF, 0] on the CPU
// before dispatch (done in prepare_normalization_bind_group each frame).
@compute @workgroup_size(8, 8, 1)
fn preview_reduce_minmax(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    if local_idx == 0u {
        atomicStore(&wg_min, 0x7F7FFFFFu);
        atomicStore(&wg_max, 0u);
    }
    workgroupBarrier();

    if id.x < params.resolution.x && id.y < params.resolution.y {
        let h = textureLoad(raw_heights, vec2<i32>(id.xy)).x;
        // IEEE 754: positive floats in [0,1] compare correctly as unsigned integers.
        let bits = bitcast<u32>(h);
        atomicMin(&wg_min, bits);
        atomicMax(&wg_max, bits);
    }
    workgroupBarrier();

    if local_idx == 0u {
        atomicMin(&minmax_buf[0], atomicLoad(&wg_min));
        atomicMax(&minmax_buf[1], atomicLoad(&wg_max));
    }
}

// Pass 3: read raw heights, normalise to [0,1] using the global min/max, then write
// the final display colour (hillshade palette or greyscale) to preview_output.
@compute @workgroup_size(8, 8, 1)
fn preview_normalize_display(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y {
        return;
    }

    let min_h = bitcast<f32>(atomicLoad(&minmax_buf[0]));
    let max_h = bitcast<f32>(atomicLoad(&minmax_buf[1]));
    let range = max_h - min_h;

    let h_raw = textureLoad(raw_heights, vec2<i32>(id.xy)).x;
    let h = select(h_raw, (h_raw - min_h) / range, range > 1e-6);

    var color: vec3<f32>;
    if params.grayscale != 0u {
        color = vec3<f32>(h, h, h);
    } else {
        // Re-compute finite-difference neighbours for hillshading using raw (unnormalised)
        // heights so the gradient magnitude is unchanged; use normalised h for palette only.
        let res = vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
        let texel = 1.0 / res;
        let uv = (vec2<f32>(f32(id.x), f32(id.y)) + 0.5) / res;

        let hx0 = terrain_height_for(params, uv - vec2<f32>(texel.x, 0.0));
        let hx1 = terrain_height_for(params, uv + vec2<f32>(texel.x, 0.0));
        let hy0 = terrain_height_for(params, uv - vec2<f32>(0.0, texel.y));
        let hy1 = terrain_height_for(params, uv + vec2<f32>(0.0, texel.y));

        let dx = (hx1 - hx0) * 4.6;
        let dy = (hy1 - hy0) * 4.6;
        let normal = normalize(vec3<f32>(-dx, 0.48, -dy));
        let light_dir = normalize(vec3<f32>(-0.62, 0.74, 0.28));

        let diffuse = max(dot(normal, light_dir), 0.0);
        let ambient = 0.24 + 0.34 * (1.0 - normal.y);
        let ridge_boost = smoothstep(0.12, 0.78, 1.0 - normal.y) * 0.30;
        let shade = pow(saturate(ambient + diffuse * 0.78 + ridge_boost), 0.88);

        let base = preview_palette(h);
        let height_boost = smoothstep(0.22, 0.92, h) * 0.12;
        color = clamp(base * shade + vec3<f32>(height_boost), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    textureStore(preview_output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn generate_height(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= export_params.resolution.x || id.y >= export_params.resolution.y {
        return;
    }

    let res = vec2<f32>(f32(export_params.resolution.x), f32(export_params.resolution.y));
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5, 0.5)) / res;
    let h = terrain_height_for(export_params, uv);

    textureStore(export_output, vec2<i32>(id.xy), vec4<f32>(h, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn downsample_height(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= downsample_params.dst_resolution.x || id.y >= downsample_params.dst_resolution.y {
        return;
    }

    let base = vec2<i32>(i32(id.x) * 2, i32(id.y) * 2);
    let h00 = sample_height_texel(downsample_src, base + vec2<i32>(0, 0), downsample_params.src_resolution);
    let h10 = sample_height_texel(downsample_src, base + vec2<i32>(1, 0), downsample_params.src_resolution);
    let h01 = sample_height_texel(downsample_src, base + vec2<i32>(0, 1), downsample_params.src_resolution);
    let h11 = sample_height_texel(downsample_src, base + vec2<i32>(1, 1), downsample_params.src_resolution);
    let h = (h00 + h10 + h01 + h11) * 0.25;

    textureStore(downsample_dst, vec2<i32>(id.xy), vec4<f32>(h, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn derive_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= normal_params.resolution.x || id.y >= normal_params.resolution.y {
        return;
    }

    let center = vec2<i32>(id.xy);
    let hf_m1_m1 = sample_height_texel(normal_src, center + vec2<i32>(-1, -1), normal_params.resolution);
    let hf_0_m1 = sample_height_texel(normal_src, center + vec2<i32>(0, -1), normal_params.resolution);
    let hf_1_m1 = sample_height_texel(normal_src, center + vec2<i32>(1, -1), normal_params.resolution);
    let hf_m1_0 = sample_height_texel(normal_src, center + vec2<i32>(-1, 0), normal_params.resolution);
    let hf_1_0 = sample_height_texel(normal_src, center + vec2<i32>(1, 0), normal_params.resolution);
    let hf_m1_1 = sample_height_texel(normal_src, center + vec2<i32>(-1, 1), normal_params.resolution);
    let hf_0_1 = sample_height_texel(normal_src, center + vec2<i32>(0, 1), normal_params.resolution);
    let hf_1_1 = sample_height_texel(normal_src, center + vec2<i32>(1, 1), normal_params.resolution);

    let gx = (
        hf_1_m1 + 2.0 * hf_1_0 + hf_1_1
        - hf_m1_m1 - 2.0 * hf_m1_0 - hf_m1_1
    ) * (1.0 / 8.0) * normal_params.effective_height_scale;
    let gz = (
        hf_m1_1 + 2.0 * hf_0_1 + hf_1_1
        - hf_m1_m1 - 2.0 * hf_0_m1 - hf_1_m1
    ) * (1.0 / 8.0) * normal_params.effective_height_scale;

    let nx = -gx;
    let nz = -gz;
    let len = max(sqrt(nx * nx + normal_params.lod_scale * normal_params.lod_scale + nz * nz), 1e-6);
    let packed = vec2<f32>(clamp(nx / len, -1.0, 1.0), clamp(nz / len, -1.0, 1.0));
    textureStore(normal_dst, center, vec4<f32>(packed, 0.0, 1.0));
}

// Heightfield generator compute shader.
// Generates a terrain preview that matches the CPU export path:
// continent shaping, domain warp, ridged mountains, and erosion-inspired carving.

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
}

@group(0) @binding(0) var<uniform> params: GeneratorParams;
@group(1) @binding(0) var output: texture_storage_2d<rgba32float, read_write>;

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn remap01(v: f32) -> f32 {
    return saturate(v * 0.5 + 0.5);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract(vec2<f32>(p3.x + p3.y, p3.x + p3.z) * p3.zy);
}

fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let ga = hash22(i + vec2<f32>(0.0, 0.0)) * 2.0 - 1.0;
    let gb = hash22(i + vec2<f32>(1.0, 0.0)) * 2.0 - 1.0;
    let gc = hash22(i + vec2<f32>(0.0, 1.0)) * 2.0 - 1.0;
    let gd = hash22(i + vec2<f32>(1.0, 1.0)) * 2.0 - 1.0;

    let va = dot(ga, f - vec2<f32>(0.0, 0.0));
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));

    return mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}

fn fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = base;

    for (var i = 0u; i < octaves; i++) {
        value += amplitude * gradient_noise(pos);
        pos *= lac;
        amplitude *= g;
    }

    return value;
}

fn ridged_noise(p: vec2<f32>) -> f32 {
    return 1.0 - abs(gradient_noise(p));
}

fn ridged_fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var total = 0.0;
    var pos = base;

    for (var i = 0u; i < octaves; i++) {
        value += amplitude * ridged_noise(pos);
        total += amplitude;
        pos *= lac;
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

        pos *= lac;
        amplitude *= g;
    }

    return value;
}

fn continent_mask(pos: vec2<f32>) -> f32 {
    let continents = remap01(fbm(pos, CONTINENT_OCTAVES, 2.02, 0.55));
    return smoothstep(0.28, 0.72, continents);
}

fn domain_warp(base_uv: vec2<f32>) -> vec2<f32> {
    let warp_strength = clamp(params.warp_strength, 0.0, 2.0);
    if warp_strength <= 0.0 {
        return base_uv;
    }

    let warp_pos = base_uv * max(params.warp_frequency, 0.05);
    let warp = vec2<f32>(
        fbm(warp_pos + vec2<f32>(5.2, 1.3), WARP_OCTAVES, 2.0, 0.5),
        fbm(warp_pos + vec2<f32>(8.3, -2.8), WARP_OCTAVES, 2.0, 0.5),
    );
    return base_uv + warp * warp_strength;
}

fn terrain_height(uv: vec2<f32>) -> f32 {
    let seed_off = vec2<f32>(
        f32(params.seed) * 0.47316,
        f32(params.seed) * 0.31419,
    );
    let base_uv = uv + params.offset + seed_off;
    let detail_uv = domain_warp(base_uv);
    let detail_pos = detail_uv * max(params.frequency, 0.001);
    let octaves = max(params.octaves, 1u);
    let lacunarity = max(params.lacunarity, 1.01);
    let gain = clamp(params.gain, 0.05, 0.95);
    let erosion = saturate(params.erosion_strength);

    let detail = erosion_shaped_fbm(detail_pos, octaves, lacunarity, gain, erosion);

    var ridge_octaves = 1u;
    if octaves > 1u {
        ridge_octaves = octaves - 1u;
    }
    let ridges = ridged_fbm(
        detail_pos * 0.85 + vec2<f32>(17.13, 17.13),
        ridge_octaves,
        lacunarity,
        gain,
    );

    let base_height = remap01(detail);
    let mountainous = mix(base_height, ridges, saturate(params.ridge_strength));

    let continent = continent_mask(
        base_uv * max(params.continent_frequency, 0.05) + vec2<f32>(31.7, -22.9),
    );
    let continental_height = mountainous * (0.18 + 0.82 * continent) + continent * 0.18 - 0.09;

    var height = mix(
        mountainous,
        continental_height,
        saturate(params.continent_strength),
    );

    let channels = ridged_fbm(
        (detail_uv + vec2<f32>(-13.5, 21.4)) * (params.frequency * 0.55 + 0.35),
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

@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y {
        return;
    }

    let res = vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
    let uv = vec2<f32>(f32(id.x), f32(id.y)) / res;
    let h = terrain_height(uv);

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(h, h, h, 1.0));
}

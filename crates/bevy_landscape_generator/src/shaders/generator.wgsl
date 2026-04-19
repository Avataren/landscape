// Heightfield generator compute shader.
// Generates FBM gradient noise into an rgba32float storage texture.
// R channel holds normalised height [0, 1].  G/B are mirrored for greyscale preview.

struct GeneratorParams {
    resolution:  vec2<u32>,
    octaves:     u32,
    seed:        u32,
    offset:      vec2<f32>,
    frequency:   f32,
    lacunarity:  f32,
    gain:        f32,
    height_scale: f32,
    pad:         vec2<f32>,   // explicit 8-byte pad → struct size = 48 bytes
}

@group(0) @binding(0) var<uniform> params: GeneratorParams;
@group(1) @binding(0) var output: texture_storage_2d<rgba32float, read_write>;

// Hash → deterministic vec2 from vec2 input (IQ's series-based hash).
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract(vec2<f32>(p3.x + p3.y, p3.x + p3.z) * p3.zy);
}

// Gradient noise in [-1, 1].
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic smoothstep for C2 continuity.
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

// Fractal Brownian Motion over gradient noise.
fn fbm(base: vec2<f32>, octaves: u32, lac: f32, g: f32) -> f32 {
    var value     = 0.0;
    var amplitude = 0.5;
    var pos       = base;

    for (var i = 0u; i < octaves; i++) {
        value     += amplitude * gradient_noise(pos);
        pos       *= lac;
        amplitude *= g;
    }

    return value;
}

@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y {
        return;
    }

    let res = vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
    let uv  = vec2<f32>(f32(id.x), f32(id.y)) / res;

    // Deterministic seed offset (two independent irrational multiples).
    let seed_off = vec2<f32>(
        f32(params.seed) * 0.47316,
        f32(params.seed) * 0.31419,
    );
    let pos = (uv + params.offset + seed_off) * params.frequency;

    // Map gradient noise range ~[-1, 1] → [0, 1].
    var h = fbm(pos, params.octaves, params.lacunarity, params.gain);
    h = clamp(h * 0.5 + 0.5, 0.0, 1.0);

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(h, h, h, 1.0));
}

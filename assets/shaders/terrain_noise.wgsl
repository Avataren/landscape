// terrain_noise.wgsl — shared noise primitives for terrain shaders.
// Imported by detail_synthesis.wgsl and any future shaders that need fBM.

const GRADIENT_EPSILON: f32 = 0.37;
const EROSION_RESPONSE: f32 = 3.5;

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

// Domain rotation ~36.87° between fBM octaves (3-4-5 triangle, per IQ).
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

// Gradient-attenuated fBM that mimics erosion — steep regions are suppressed.
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
        let atten = mix(1.0, 1.0 / (1.0 + dot(acc_grad, acc_grad) * EROSION_RESPONSE), erosion);
        value += amplitude * n * atten;
        pos = rot2(pos) * lac;
        amplitude *= g;
    }
    return value;
}

// Band weight: smoothly zero-out bands the LOD mesh can't represent.
// wavelength and spacing are in the same unit (world-space meters).
fn band_weight(wavelength: f32, lod_spacing: f32) -> f32 {
    return smoothstep(lod_spacing * 2.0, lod_spacing * 4.0, wavelength);
}

// Source cutoff: don't synthesize what the 30m source heightmap already has.
// source_spacing is the source texel size in world units.
fn source_fade(wavelength: f32, source_spacing: f32) -> f32 {
    return 1.0 - smoothstep(source_spacing, source_spacing * 2.0, wavelength);
}

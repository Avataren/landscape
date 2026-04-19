use bevy::math::{Vec2, Vec3};

use crate::params::GeneratorParams;

const CONTINENT_OCTAVES: u32 = 4;
const WARP_OCTAVES: u32 = 3;
const CHANNEL_OCTAVES: u32 = 4;
const EROSION_RESPONSE: f32 = 3.5;
const GRADIENT_EPSILON: f32 = 0.37;

pub fn sample_height(params: &GeneratorParams, uv_x: f32, uv_y: f32) -> f32 {
    let uv = Vec2::new(uv_x, uv_y);
    let seed_off = Vec2::new(params.seed as f32 * 0.47316, params.seed as f32 * 0.31419);
    let base_uv = uv + params.offset + seed_off;
    let detail_uv = domain_warp(base_uv, params);
    let detail_pos = detail_uv * params.frequency.max(0.001);

    let detail = erosion_shaped_fbm(
        detail_pos,
        params.octaves.max(1),
        params.lacunarity.max(1.01),
        params.gain.clamp(0.05, 0.95),
        params.erosion_strength.clamp(0.0, 1.0),
    );
    let ridge_octaves = params.octaves.saturating_sub(1).max(1);
    let ridges = ridged_fbm(
        detail_pos * 0.85 + Vec2::splat(17.13),
        ridge_octaves,
        params.lacunarity.max(1.01),
        params.gain.clamp(0.05, 0.95),
    );

    let base_height = remap01(detail);
    let mountainous = lerp(base_height, ridges, params.ridge_strength.clamp(0.0, 1.0));

    let continent =
        continent_mask(base_uv * params.continent_frequency.max(0.05) + Vec2::new(31.7, -22.9));
    let continental_height = mountainous * (0.18 + 0.82 * continent) + continent * 0.18 - 0.09;
    let mut height = lerp(
        mountainous,
        continental_height,
        params.continent_strength.clamp(0.0, 1.0),
    );

    let channels = ridged_fbm(
        (detail_uv + Vec2::new(-13.5, 21.4)) * (params.frequency * 0.55 + 0.35),
        CHANNEL_OCTAVES,
        2.05,
        0.55,
    );
    let channel_mask = channels * channels * channels * channels;
    let highlands = smoothstep(0.28, 0.82, mountainous);
    height -= params.erosion_strength.clamp(0.0, 1.0)
        * channel_mask
        * highlands
        * (0.03 + 0.11 * continent);

    saturate(height)
}

fn domain_warp(base_uv: Vec2, params: &GeneratorParams) -> Vec2 {
    let warp_strength = params.warp_strength.clamp(0.0, 2.0);
    if warp_strength <= f32::EPSILON {
        return base_uv;
    }

    let warp_pos = base_uv * params.warp_frequency.max(0.05);
    let warp = Vec2::new(
        fbm(warp_pos + Vec2::new(5.2, 1.3), WARP_OCTAVES, 2.0, 0.5),
        fbm(warp_pos + Vec2::new(8.3, -2.8), WARP_OCTAVES, 2.0, 0.5),
    );

    base_uv + warp * warp_strength
}

fn continent_mask(pos: Vec2) -> f32 {
    let continents = remap01(fbm(pos, CONTINENT_OCTAVES, 2.02, 0.55));
    smoothstep(0.28, 0.72, continents)
}

fn erosion_shaped_fbm(
    mut p: Vec2,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    erosion_strength: f32,
) -> f32 {
    if erosion_strength <= f32::EPSILON {
        return fbm(p, octaves, lacunarity, gain);
    }

    let mut value = 0.0;
    let mut amplitude = 0.5;
    let mut acc_grad = Vec2::ZERO;

    for _ in 0..octaves {
        let n = gradient_noise(p);
        let grad = Vec2::new(
            gradient_noise(p + Vec2::X * GRADIENT_EPSILON) - n,
            gradient_noise(p + Vec2::Y * GRADIENT_EPSILON) - n,
        ) / GRADIENT_EPSILON;
        acc_grad += grad * amplitude;

        let attenuation = lerp(
            1.0,
            1.0 / (1.0 + acc_grad.length_squared() * EROSION_RESPONSE),
            erosion_strength,
        );
        value += amplitude * n * attenuation;

        p *= lacunarity;
        amplitude *= gain;
    }

    value
}

fn ridged_fbm(mut p: Vec2, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 0.5f32;
    let mut total = 0.0f32;

    for _ in 0..octaves {
        value += amplitude * ridged_noise(p);
        total += amplitude;
        p *= lacunarity;
        amplitude *= gain;
    }

    if total > 0.0 {
        value / total
    } else {
        0.0
    }
}

fn ridged_noise(p: Vec2) -> f32 {
    1.0 - gradient_noise(p).abs()
}

fn fbm(mut p: Vec2, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 0.5f32;
    for _ in 0..octaves {
        value += amplitude * gradient_noise(p);
        p *= lacunarity;
        amplitude *= gain;
    }
    value
}

fn hash22(p: Vec2) -> Vec2 {
    let mut p3 = Vec3::new(p.x * 0.1031, p.y * 0.1030, p.x * 0.0973);
    p3 = fract3(p3);
    p3 += p3.dot(Vec3::new(p3.y, p3.z, p3.x) + 33.33);
    Vec2::new(fract1((p3.x + p3.y) * p3.z), fract1((p3.x + p3.z) * p3.y))
}

fn gradient_noise(p: Vec2) -> f32 {
    let i = p.floor();
    let f = fract2(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let ga = hash22(i + Vec2::new(0.0, 0.0)) * 2.0 - Vec2::ONE;
    let gb = hash22(i + Vec2::new(1.0, 0.0)) * 2.0 - Vec2::ONE;
    let gc = hash22(i + Vec2::new(0.0, 1.0)) * 2.0 - Vec2::ONE;
    let gd = hash22(i + Vec2::new(1.0, 1.0)) * 2.0 - Vec2::ONE;

    let va = ga.dot(f - Vec2::new(0.0, 0.0));
    let vb = gb.dot(f - Vec2::new(1.0, 0.0));
    let vc = gc.dot(f - Vec2::new(0.0, 1.0));
    let vd = gd.dot(f - Vec2::new(1.0, 1.0));

    lerp(lerp(va, vb, u.x), lerp(vc, vd, u.x), u.y)
}

fn remap01(v: f32) -> f32 {
    saturate(v * 0.5 + 0.5)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

fn fract1(v: f32) -> f32 {
    v - v.floor()
}

fn fract2(v: Vec2) -> Vec2 {
    v - v.floor()
}

fn fract3(v: Vec3) -> Vec3 {
    v - v.floor()
}

fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[cfg(test)]
mod tests {
    use super::{fract1, sample_height};
    use crate::params::GeneratorParams;

    #[test]
    fn sampled_height_stays_normalized() {
        let params = GeneratorParams::default();
        for y in 0..32 {
            for x in 0..32 {
                let h = sample_height(&params, x as f32 / 31.0, y as f32 / 31.0);
                assert!((0.0..=1.0).contains(&h), "height out of range: {h}");
            }
        }
    }

    #[test]
    fn seed_changes_shape() {
        let params_a = GeneratorParams::default();
        let mut params_b = GeneratorParams::default();
        params_b.seed += 1;

        let a = sample_height(&params_a, 0.37, 0.61);
        let b = sample_height(&params_b, 0.37, 0.61);
        assert_ne!(a, b);
    }

    #[test]
    fn fract_matches_shader_semantics_for_negative_values() {
        assert!((fract1(-0.2) - 0.8).abs() < 1e-6);
        assert!((fract1(-1.75) - 0.25).abs() < 1e-6);
        assert!((fract1(1.75) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn negative_domain_sampling_stays_continuous() {
        let params = GeneratorParams::default();
        let a = sample_height(&params, -0.7501, 0.12);
        let b = sample_height(&params, -0.7499, 0.12);
        assert!(
            (a - b).abs() < 0.01,
            "unexpected discontinuity across negative coordinate seam: {a} vs {b}"
        );
    }
}

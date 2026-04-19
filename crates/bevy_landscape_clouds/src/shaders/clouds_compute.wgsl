struct Config {
    render_resolution: vec2<f32>,
    time: f32,
    planet_radius: f32,
    camera_translation: vec3<f32>,
    _pad1: f32,
    previous_camera_translation: vec3<f32>,
    _pad_prev: f32,
    inverse_camera_view: mat4x4<f32>,
    inverse_camera_projection: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    wind_displacement: vec3<f32>,
    _pad2: f32,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    cloud_heights: vec4<f32>,
    noise_scales: vec4<f32>,
    softness: vec4<f32>,
    ambient_top: vec4<f32>,
    ambient_bottom: vec4<f32>,
    phase_steps: vec4<f32>,
    shadow_params: vec4<f32>,
    march: vec4<u32>,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(1) @binding(0) var cloud_render_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var cloud_history_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(2) var cloud_atlas_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(3) var cloud_worley_texture: texture_storage_3d<rgba32float, read_write>;

const PI: f32 = 3.141592653589793;
const EPSILON: f32 = 1e-5;
const MAX_CLOUD_DISTANCE: f32 = 80000.0;
const HORIZON_TRACE_DISTANCE: f32 = 45000.0;
const DISTANT_DETAIL_START: f32 = 12000.0;
const DISTANT_DETAIL_END: f32 = 36000.0;
const DISTANT_SHADOW_START: f32 = 8000.0;
const DISTANT_SHADOW_END: f32 = 24000.0;
const CLOUD_ATLAS_SIZE: u32 = 512u;
const CLOUD_WORLEY_SIZE: u32 = 32u;
const CLOUD_WORLEY_SIZE_F32: f32 = 32.0;

struct Ray {
    step_distance: f32,
    dir_length: f32,
    start: f32,
}

fn linearstep(start: f32, end: f32, value: f32) -> f32 {
    return clamp((value - start) / max(end - start, EPSILON), 0.0, 1.0);
}

fn linearstep0(edge: f32, value: f32) -> f32 {
    return min(value / max(edge, EPSILON), 1.0);
}

fn remap(value: f32, start: f32, end: f32) -> f32 {
    return (value - start) / max(end - start, EPSILON);
}

fn hash13(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 1031.1031);
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

fn value_hash(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.1031);
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

fn hash_based_noise(x: vec3<f32>, tile: f32) -> f32 {
    let p = floor(x);
    var f = fract(x);
    f = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(value_hash(p % tile), value_hash((p + vec3<f32>(1.0, 0.0, 0.0)) % tile), f.x),
            mix(
                value_hash((p + vec3<f32>(0.0, 1.0, 0.0)) % tile),
                value_hash((p + vec3<f32>(1.0, 1.0, 0.0)) % tile),
                f.x,
            ),
            f.y,
        ),
        mix(
            mix(
                value_hash((p + vec3<f32>(0.0, 0.0, 1.0)) % tile),
                value_hash((p + vec3<f32>(1.0, 0.0, 1.0)) % tile),
                f.x,
            ),
            mix(
                value_hash((p + vec3<f32>(0.0, 1.0, 1.0)) % tile),
                value_hash((p + vec3<f32>(1.0, 1.0, 1.0)) % tile),
                f.x,
            ),
            f.y,
        ),
        f.z,
    );
}

fn voronoi(x: vec3<f32>, tile: f32) -> f32 {
    let p = floor(x);
    let f = fract(x);
    var res = 100.0;

    for (var k = -1.0; k < 1.1; k += 1.0) {
        for (var j = -1.0; j < 1.1; j += 1.0) {
            for (var i = -1.0; i < 1.1; i += 1.0) {
                let b = vec3<f32>(i, j, k);
                let c = (p + b) % vec3<f32>(tile);
                let r = b - f + hash13(c);
                res = min(res, dot(r, r));
            }
        }
    }

    return 1.0 - res;
}

fn tilable_voronoi(p: vec3<f32>, octaves: i32, start_freq: f32) -> f32 {
    var freq = start_freq;
    var amplitude = 1.0;
    var noise = 0.0;
    var weight = 0.0;

    for (var i = 0; i < octaves; i += 1) {
        noise += amplitude * voronoi(p * freq, freq);
        freq *= 2.0;
        weight += amplitude;
        amplitude *= 0.5;
    }

    return noise / max(weight, EPSILON);
}

fn tilable_perlin_fbm(p: vec3<f32>, octaves: i32, start_freq: f32) -> f32 {
    var freq = start_freq;
    var amplitude = 1.0;
    var noise = 0.0;
    var weight = 0.0;

    for (var i = 0; i < octaves; i += 1) {
        noise += amplitude * hash_based_noise(p * freq, freq);
        freq *= 2.0;
        weight += amplitude;
        amplitude *= 0.5;
    }

    return noise / max(weight, EPSILON);
}

fn render_clouds_atlas(coord: vec3<f32>) -> vec4<f32> {
    let mfbm = 0.9;
    let mvor = 0.7;
    return vec4<f32>(
        mix(1.0, tilable_perlin_fbm(coord, 7, 4.0), mfbm) *
            mix(1.0, tilable_voronoi(coord, 8, 9.0), mvor),
        0.625 * tilable_voronoi(coord, 3, 15.0) +
            0.250 * tilable_voronoi(coord, 3, 19.0) +
            0.125 * tilable_voronoi(coord, 3, 23.0) -
            1.0,
        1.0 - tilable_voronoi(coord + 0.5, 6, 9.0),
        1.0,
    );
}

fn render_clouds_worley(coord: vec3<f32>) -> vec4<f32> {
    let r = tilable_voronoi(coord, 16, 3.0);
    let g = tilable_voronoi(coord, 4, 8.0);
    let b = tilable_voronoi(coord, 4, 16.0);
    let c = max(0.0, 1.0 - (r + g * 0.5 + b * 0.25) / 1.75);
    return vec4<f32>(c, c, c, 1.0);
}

fn wrap_atlas(v: f32) -> u32 {
    let s = f32(CLOUD_ATLAS_SIZE);
    return u32(((v % s) + s) % s);
}

fn wrap_worley(v: vec3<f32>) -> vec3<f32> {
    let s = CLOUD_WORLEY_SIZE_F32;
    return ((v % s) + s) % s;
}

fn cloud_map_base(p: vec3<f32>, normalized_height: f32) -> f32 {
    let scale = 0.00005 * config.noise_scales.x;
    let sx = p.x * scale * config.render_resolution.x;
    let sz = p.z * scale * config.render_resolution.y;
    let cloud = textureLoad(
        cloud_atlas_texture,
        vec2<u32>(wrap_atlas(sx), wrap_atlas(sz)),
    ).rgb;

    let n = normalized_height * normalized_height * cloud.b + pow(1.0 - normalized_height, 16.0);
    return remap(cloud.r - n, cloud.g, 1.0);
}

fn cloud_map_detail(position: vec3<f32>) -> f32 {
    let p = position * (0.0016 * config.noise_scales.x * config.noise_scales.y);
    let p1 = wrap_worley(p);
    let a = textureLoad(
        cloud_worley_texture,
        vec3<u32>(u32(p1.x), u32(p1.y), u32(p1.z)),
    ).r;
    let p2 = (p1 + 1.0) % CLOUD_WORLEY_SIZE_F32;
    let b = textureLoad(
        cloud_worley_texture,
        vec3<u32>(u32(p2.x), u32(p2.y), u32(p2.z)),
    ).r;
    return mix(a, b, fract(p.y));
}

fn cloud_gradient(normalized_height: f32) -> f32 {
    return linearstep(0.0, 0.1, normalized_height) - linearstep(0.8, 1.2, normalized_height);
}

fn distant_detail_weight(distance_to_sample: f32) -> f32 {
    return 1.0 - smoothstep(DISTANT_DETAIL_START, DISTANT_DETAIL_END, distance_to_sample);
}

fn distant_shadow_quality(distance_to_sample: f32) -> f32 {
    return 1.0 - smoothstep(DISTANT_SHADOW_START, DISTANT_SHADOW_END, distance_to_sample);
}

fn get_cloud_map_density(pos: vec3<f32>, normalized_height: f32, detail_weight: f32) -> f32 {
    var m = cloud_map_base(pos, normalized_height) * cloud_gradient(normalized_height);
    let detail_strength = smoothstep(1.0, 0.5, m);
    if detail_strength > 0.0 && detail_weight > 0.01 {
        m -= cloud_map_detail(pos) * detail_strength * config.noise_scales.z * detail_weight;
    }

    m = smoothstep(0.0, config.noise_scales.w, m + config.cloud_heights.z - 1.0);
    m *= linearstep0(config.softness.x, normalized_height);
    return clamp(m * config.cloud_heights.w, 0.0, 1.0);
}

fn get_normalized_height(pos: vec3<f32>) -> f32 {
    let clouds_height = config.cloud_heights.y - config.cloud_heights.x;
    return (length(pos) - (config.planet_radius + config.cloud_heights.x)) / max(clouds_height, EPSILON);
}

fn sky_ray_origin() -> vec3<f32> {
    return config.camera_translation - config.wind_displacement + vec3<f32>(0.0, config.planet_radius, 0.0);
}

// Distance from a point at planet-center radius `r` along `ray_dir` to a shell
// at altitude `alt` above the surface (shell radius = planet_radius + alt).
// Returns the nearest positive intersection distance, or -1.0 if none exists.
fn intersect_shell(r: f32, ray_dir_y: f32, alt: f32) -> f32 {
    let R   = config.planet_radius + alt;
    let mu  = r * ray_dir_y;
    let disc = mu * mu + R * R - r * r;
    if disc < 0.0 { return -1.0; }
    let sq = sqrt(disc);
    let t1 = -mu - sq;
    let t2 = -mu + sq;
    if t1 >= 0.0 { return t1; }
    if t2 >= 0.0 { return t2; }
    return -1.0;
}

fn get_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> Ray {
    // ray_origin.y approximates the camera's distance from the planet centre
    // (valid because x,z << planet_radius for any terrain-scale position).
    let r       = ray_origin.y;
    let cam_alt = r - config.planet_radius;
    let miss    = Ray(0.0, 0.0, MAX_CLOUD_DISTANCE + 1.0);

    var start: f32;
    var end: f32;

    if cam_alt < config.cloud_heights.x {
        // Camera is below the cloud layer — only upward rays can hit it.
        if ray_dir.y <= 0.0 { return miss; }
        let d_bot = intersect_shell(r, ray_dir.y, config.cloud_heights.x);
        let d_top = intersect_shell(r, ray_dir.y, config.cloud_heights.y);
        if d_bot < 0.0 || d_top < 0.0 { return miss; }
        start = d_bot;
        end   = d_top;
    } else if cam_alt > config.cloud_heights.y {
        // Camera is above the cloud layer — only downward rays can hit it.
        if ray_dir.y >= 0.0 { return miss; }
        let d_top = intersect_shell(r, ray_dir.y, config.cloud_heights.y);
        let d_bot = intersect_shell(r, ray_dir.y, config.cloud_heights.x);
        if d_top < 0.0 || d_bot < 0.0 { return miss; }
        start = d_top; // top shell is closer when descending
        end   = d_bot;
    } else {
        // Camera is inside the cloud layer — start right here.
        start = 0.0;
        if ray_dir.y >= 0.0 {
            let d_top = intersect_shell(r, ray_dir.y, config.cloud_heights.y);
            if d_top < 0.0 { return miss; }
            end = d_top;
        } else {
            let d_bot = intersect_shell(r, ray_dir.y, config.cloud_heights.x);
            if d_bot < 0.0 { return miss; }
            end = d_bot;
        }
    }

    let horizon_limit = mix(
        HORIZON_TRACE_DISTANCE,
        MAX_CLOUD_DISTANCE,
        smoothstep(-0.02, 0.18, ray_dir.y),
    );
    end = min(end, horizon_limit);
    if end <= start { return miss; }

    let step_distance = (end - start) / f32(max(config.march.x, 1u));
    let hashed_offset = hash13(ray_dir + fract(config.time));
    let dir_length    = start - step_distance * hashed_offset;
    return Ray(step_distance, dir_length, start);
}

fn henyey_greenstein(ray_dot_sun: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / max(pow(1.0 + g2 - 2.0 * g * ray_dot_sun, 1.5), EPSILON);
}

fn volumetric_shadow(origin: vec3<f32>, ray_dot_sun: f32, shadow_quality: f32) -> f32 {
    if shadow_quality <= 0.01 {
        return 1.0;
    }

    var step_size = config.shadow_params.x * mix(2.5, 1.0, shadow_quality);
    var distance_along_ray = step_size * 0.5;
    var transmittance = 1.0;
    let step_count = max(
        1u,
        u32(round(mix(2.0, f32(max(config.march.y, 1u)), shadow_quality))),
    );

    for (var step: u32 = 0u; step < max(config.march.y, 1u); step += 1u) {
        if step >= step_count {
            break;
        }
        let pos = origin + config.sun_direction.xyz * distance_along_ray;
        let normalized_height = get_normalized_height(pos);
        if normalized_height > 1.0 {
            return transmittance;
        }

        let density = get_cloud_map_density(pos, normalized_height, shadow_quality);
        transmittance *= exp(-density * step_size);
        step_size *= config.shadow_params.y;
        distance_along_ray += step_size;
    }

    return transmittance;
}

fn get_ray_direction(pixel: vec2<f32>) -> vec3<f32> {
    let rect_relative = pixel / config.render_resolution;
    let ndc_xy = (rect_relative * 2.0 - vec2<f32>(1.0, 1.0)) * vec2<f32>(1.0, -1.0);
    let ray_clip = vec4<f32>(ndc_xy, -1.0, 1.0);
    let ray_eye = config.inverse_camera_projection * ray_clip;
    let ray_world = config.inverse_camera_view * vec4<f32>(ray_eye.xy, -1.0, 0.0);
    return normalize(ray_world.xyz);
}

fn render_clouds(pixel: vec2<f32>) -> vec4<f32> {
    let ray_origin = sky_ray_origin();
    let ray_dir = get_ray_direction(pixel);
    let ray = get_ray(ray_origin, ray_dir);
    if ray.start > MAX_CLOUD_DISTANCE {
        return vec4<f32>(0.0);
    }

    let ray_dot_sun = dot(ray_dir, config.sun_direction.xyz);
    let scattering = mix(
        henyey_greenstein(ray_dot_sun, config.phase_steps.x),
        henyey_greenstein(ray_dot_sun, config.phase_steps.y),
        config.phase_steps.z,
    );

    var dir_length = ray.dir_length;
    var scattered_light = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var step: u32 = 0u; step < max(config.march.x, 1u); step += 1u) {
        let world_position = ray_origin + dir_length * ray_dir;
        let normalized_height = clamp(get_normalized_height(world_position), 0.0, 1.0);
        let sample_distance = max(dir_length, ray.start);
        let detail_weight = distant_detail_weight(sample_distance);
        let shadow_quality = distant_shadow_quality(sample_distance);
        let density = get_cloud_map_density(world_position, normalized_height, detail_weight);

        if density > 0.0 {
            let daylight = mix(0.04, 1.0, config.shadow_params.z);
            let ambient_light = mix(config.ambient_bottom, config.ambient_top, normalized_height) * daylight;
            let shadow = mix(
                0.82,
                volumetric_shadow(world_position, ray_dot_sun, shadow_quality),
                shadow_quality,
            );
            let scattering_light = density
                * (ambient_light.rgb
                    + config.sun_color.rgb * scattering * shadow);
            let delta_transmittance = exp(-density * ray.step_distance);
            let integrated_scattering =
                scattering_light * (1.0 - delta_transmittance) / max(density, EPSILON);
            scattered_light += transmittance * integrated_scattering;
            transmittance *= delta_transmittance;
        }

        if transmittance <= config.phase_steps.w {
            break;
        }

        dir_length += ray.step_distance;
    }

    let opacity = clamp(1.0 - transmittance, 0.0, 1.0);
    if opacity <= 0.002 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(scattered_light * opacity, opacity);
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.z == 0u && invocation_id.x < CLOUD_ATLAS_SIZE && invocation_id.y < CLOUD_ATLAS_SIZE {
        let atlas_coord = vec2<f32>(f32(invocation_id.x), f32(invocation_id.y)) + vec2<f32>(0.5);
        let coord = vec3<f32>(atlas_coord / f32(CLOUD_ATLAS_SIZE), 0.5);
        textureStore(cloud_atlas_texture, invocation_id.xy, render_clouds_atlas(coord));
    }

    if invocation_id.x < CLOUD_WORLEY_SIZE && invocation_id.y < CLOUD_WORLEY_SIZE && invocation_id.z < CLOUD_WORLEY_SIZE {
        let coord = (vec3<f32>(invocation_id.xyz) + vec3<f32>(0.5)) / CLOUD_WORLEY_SIZE_F32;
        textureStore(cloud_worley_texture, invocation_id.xyz, render_clouds_worley(coord));
    }
}

fn sample_history_bilinear(uv: vec2<f32>) -> vec4<f32> {
    let dims = config.render_resolution;
    let px = uv * dims - vec2<f32>(0.5);
    let f = fract(px);
    let p0 = vec2<u32>(u32(clamp(floor(px.x), 0.0, dims.x - 1.0)),
                       u32(clamp(floor(px.y), 0.0, dims.y - 1.0)));
    let p1x = min(p0.x + 1u, u32(dims.x) - 1u);
    let p1y = min(p0.y + 1u, u32(dims.y) - 1u);
    let s00 = textureLoad(cloud_history_texture, p0);
    let s10 = textureLoad(cloud_history_texture, vec2<u32>(p1x, p0.y));
    let s01 = textureLoad(cloud_history_texture, vec2<u32>(p0.x, p1y));
    let s11 = textureLoad(cloud_history_texture, vec2<u32>(p1x, p1y));
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if invocation_id.x >= u32(config.render_resolution.x) || invocation_id.y >= u32(config.render_resolution.y) {
        return;
    }

    let pixel = vec2<f32>(f32(invocation_id.x), f32(invocation_id.y)) + vec2<f32>(0.5);
    let current = render_clouds(pixel);
    let history_factor = config.shadow_params.w;

    var blended = current;
    if history_factor > 0.01 {
        let ray_dir = get_ray_direction(pixel);

        // Intersect ray with cloud layer midpoint in sky-shifted space.
        let cloud_mid_y_sky = config.planet_radius + (config.cloud_heights.x + config.cloud_heights.y) * 0.5;
        let ray_origin_sky = sky_ray_origin();

        if abs(ray_dir.y) > 0.001 {
            let t = (cloud_mid_y_sky - ray_origin_sky.y) / ray_dir.y;
            if t > 0.0 && t < MAX_CLOUD_DISTANCE {
                // Convert sky-shifted world pos back to actual world space.
                // sky_ray_origin = camera_translation - wind_displacement + (0, planet_radius, 0)
                // actual_world = sky_world - (0, planet_radius, 0) + wind_displacement
                //              = camera_translation + ray_dir * t  (wind terms cancel)
                let world_pos = config.camera_translation + ray_dir * t;

                // Reproject through the previous frame's view-projection matrix (world -> clip).
                let prev_clip = config.previous_view_proj * vec4<f32>(world_pos, 1.0);
                if prev_clip.w > EPSILON {
                    let prev_ndc = prev_clip.xy / prev_clip.w;
                    // NDC -> UV: X right, Y up in NDC; Y flipped for texture (top=0).
                    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);

                    if all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
                        let previous = sample_history_bilinear(prev_uv);
                        blended = mix(current, previous, history_factor);
                    }
                }
            }
        }
    }

    textureStore(cloud_render_texture, invocation_id.xy, blended);
    textureStore(cloud_history_texture, invocation_id.xy, blended);
}

#import bevy_pbr::{
    forward_io::Vertex,
    mesh_functions,
    mesh_view_bindings as view_bindings,
    mesh_view_bindings::{environment_map_uniform, light_probes, lights, view},
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::atmosphere::types::Atmosphere
#import bevy_pbr::atmosphere::bruneton_functions::transmittance_lut_r_mu_to_uv
#import bevy_render::maths::fast_acos_4

struct CloudMaterial {
    wind_velocity_time: vec4<f32>,
    wind_offset_density: vec4<f32>,
    shape_coverage: vec4<f32>,
    detail_warp: vec4<f32>,
    softness: vec4<f32>,
    lighting: vec4<f32>,
    phase_and_steps: vec4<f32>,
    march: vec4<u32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: CloudMaterial;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) local_position: vec3<f32>,
    @location(2) @interpolate(flat) instance_index: u32,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

struct BoxHit {
    entry: f32,
    exit: f32,
}

struct WeatherSample {
    coverage: f32,
    mask: f32,
    cloud_type: f32,
    warp: vec2<f32>,
}

const PI: f32 = 3.141592653589793;
const HALF_BOX: vec3<f32> = vec3<f32>(0.5, 0.5, 0.5);
const EPSILON: f32 = 1e-5;
const MAX_CLOUD_VIEW_DISTANCE: f32 = 80_000.0;
const MAX_CLOUD_SHADOW_DISTANCE: f32 = 12_000.0;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.local_position = vertex.position;
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);
    out.instance_index = vertex.instance_index;
    return out;
}

fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return out_min + (value - in_min) * (out_max - out_min) / max(in_max - in_min, EPSILON);
}

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 127.1) * 43758.5453123);
}

fn hash13(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}

fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise2d(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let frac_p = fract(p);
    let u = frac_p * frac_p * (vec2<f32>(3.0) - 2.0 * frac_p);

    let n00 = hash12(cell + vec2<f32>(0.0, 0.0));
    let n10 = hash12(cell + vec2<f32>(1.0, 0.0));
    let n01 = hash12(cell + vec2<f32>(0.0, 1.0));
    let n11 = hash12(cell + vec2<f32>(1.0, 1.0));
    let nx0 = mix(n00, n10, u.x);
    let nx1 = mix(n01, n11, u.x);
    return mix(nx0, nx1, u.y);
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn compress_hdr(color: vec3<f32>, strength: f32) -> vec3<f32> {
    let luma = luminance(max(color, vec3<f32>(0.0)));
    return color / (1.0 + luma * strength);
}

fn chroma_only(color: vec3<f32>) -> vec3<f32> {
    let peak = max(max(color.r, color.g), color.b);
    return color / max(peak, EPSILON);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let frac_p = fract(p);
    let u = frac_p * frac_p * (vec3<f32>(3.0) - 2.0 * frac_p);

    let n000 = hash13(cell + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(cell + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(cell + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(cell + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(cell + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(cell + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(cell + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(cell + vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

fn fbm_base(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    for (var octave: u32 = 0u; octave < 3u; octave += 1u) {
        value += amplitude * noise3d(p * frequency);
        frequency *= 2.03;
        amplitude *= 0.5;
    }
    return value;
}

fn fbm_detail(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.6;
    var frequency = 1.0;
    for (var octave: u32 = 0u; octave < 2u; octave += 1u) {
        value += amplitude * noise3d(p * frequency);
        frequency *= 2.17;
        amplitude *= 0.45;
    }
    return value;
}

fn fbm_weather(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.55;
    var frequency = 1.0;
    for (var octave: u32 = 0u; octave < 3u; octave += 1u) {
        value += amplitude * noise2d(p * frequency);
        frequency *= 2.11;
        amplitude *= 0.52;
    }
    return value;
}

fn sample_weather(world_pos: vec3<f32>, wind: vec3<f32>) -> WeatherSample {
    let weather_uv = (world_pos.xz + wind.xz) * material.detail_warp.x;
    let warp = vec2<f32>(
        fbm_weather(weather_uv + vec2<f32>(13.2, -4.7)),
        fbm_weather(weather_uv + vec2<f32>(-7.1, 9.4)),
    ) - vec2<f32>(0.5);
    let warped_uv = weather_uv + warp * 0.85;
    let coverage = clamp(fbm_weather(warped_uv), 0.0, 1.0);
    let cluster = clamp(fbm_weather(warped_uv * 0.47 + vec2<f32>(17.3, 5.8)), 0.0, 1.0);
    let cloud_type = clamp(fbm_weather(warped_uv * 0.31 + vec2<f32>(-11.4, 8.2)), 0.0, 1.0);

    let target_coverage = mix(0.58, 0.34, material.shape_coverage.z);
    let mask = smoothstep(target_coverage, target_coverage + 0.18, coverage + cluster * 0.26);
    return WeatherSample(coverage, mask, cloud_type, warp);
}

fn intersect_height_slab(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    bottom_height: f32,
    top_height: f32,
    max_distance: f32,
) -> BoxHit {
    if abs(ray_dir.y) < EPSILON {
        if ray_origin.y >= bottom_height && ray_origin.y <= top_height {
            return BoxHit(0.0, max_distance);
        }
        return BoxHit(1.0, 0.0);
    }

    let t0 = (bottom_height - ray_origin.y) / ray_dir.y;
    let t1 = (top_height - ray_origin.y) / ray_dir.y;
    let entry = min(t0, t1);
    let exit = max(t0, t1);
    return BoxHit(max(entry, 0.0), min(exit, max_distance));
}

fn cloud_gradient(height_t: f32) -> f32 {
    let bottom = smoothstep(0.0, material.softness.y, height_t);
    let top = 1.0 - smoothstep(1.0 - material.softness.z, 1.0, height_t);
    let billow = smoothstep(0.08, 0.65, height_t) * (1.0 - smoothstep(0.70, 1.0, height_t));
    return clamp(bottom * top * (0.75 + 0.25 * billow), 0.0, 1.0);
}

fn normalized_height_in_layer(world_y: f32, cloud_bottom_height: f32, cloud_top_height: f32) -> f32 {
    return clamp(
        (world_y - cloud_bottom_height) / max(cloud_top_height - cloud_bottom_height, EPSILON),
        0.0,
        1.0,
    );
}

fn sample_density(world_pos: vec3<f32>, normalized_height: f32) -> f32 {
    if normalized_height <= 0.0 || normalized_height >= 1.0 {
        return 0.0;
    }
    let wind = material.wind_offset_density.xyz;
    let weather = sample_weather(world_pos, wind);
    if weather.mask <= 0.01 {
        return 0.0;
    }

    let height_t = clamp(normalized_height, 0.0, 1.0);
    let gradient = cloud_gradient(mix(height_t, sqrt(height_t), weather.cloud_type * 0.45));
    if gradient <= 0.0 {
        return 0.0;
    }

    let shear = vec3<f32>(0.35, 0.0, -0.25) * (height_t - 0.5) * material.detail_warp.w;
    let world_sample = world_pos + vec3<f32>(wind.x, wind.y * 0.2, wind.z) + shear;
    let base_pos = world_sample * material.shape_coverage.x
        + vec3<f32>(weather.warp.x, 0.0, weather.warp.y) * material.detail_warp.y;
    let detail_pos = (world_sample + vec3<f32>(weather.warp.x, height_t * 0.08, weather.warp.y) * 300.0)
        * material.shape_coverage.y;
    let base = fbm_base(base_pos);
    let detail = fbm_detail(detail_pos * 1.9 + vec3<f32>(5.2, 11.3, 3.7));
    let core_threshold = mix(0.56, 0.28, weather.mask) - weather.cloud_type * 0.06;
    var density = smoothstep(core_threshold, core_threshold + material.softness.x, base + weather.mask * 0.34);
    density *= gradient * weather.mask;

    let erosion = smoothstep(0.54, 0.86, detail);
    density -= erosion * material.shape_coverage.w * mix(0.45, 0.9, 1.0 - gradient);
    density += smoothstep(0.72, 0.92, base) * weather.mask * 0.22;
    density *= mix(0.92, 1.20, weather.coverage) * mix(0.92, 1.10, weather.cloud_type);

    return clamp(density, 0.0, 1.0) * material.wind_offset_density.w;
}

fn sample_shadow_density(world_pos: vec3<f32>, normalized_height: f32) -> f32 {
    if normalized_height <= 0.0 || normalized_height >= 1.0 {
        return 0.0;
    }
    let wind = material.wind_offset_density.xyz;
    let weather = sample_weather(world_pos, wind);
    if weather.mask <= 0.01 {
        return 0.0;
    }

    let height_t = clamp(normalized_height, 0.0, 1.0);
    let gradient = cloud_gradient(mix(height_t, sqrt(height_t), weather.cloud_type * 0.45));
    if gradient <= 0.0 {
        return 0.0;
    }

    let shear = vec3<f32>(0.35, 0.0, -0.25) * (height_t - 0.5) * material.detail_warp.w;
    let base_pos = (world_pos + vec3<f32>(wind.x, wind.y * 0.2, wind.z) + shear) * material.shape_coverage.x
        + vec3<f32>(weather.warp.x, 0.0, weather.warp.y) * material.detail_warp.y;
    let base = fbm_base(base_pos);
    let core_threshold = mix(0.60, 0.34, weather.mask) - weather.cloud_type * 0.05;
    var density = smoothstep(core_threshold, core_threshold + material.softness.x, base + weather.mask * 0.28);
    density *= gradient * weather.mask * mix(0.82, 1.05, weather.coverage);
    return clamp(density, 0.0, 1.0) * material.wind_offset_density.w;
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, EPSILON), 1.5);
    return (1.0 - g2) / (4.0 * PI * denom);
}

fn dual_lobe_phase(cos_theta: f32) -> f32 {
    return mix(
        henyey_greenstein(cos_theta, material.phase_and_steps.x),
        henyey_greenstein(cos_theta, material.phase_and_steps.y),
        clamp(material.phase_and_steps.z, 0.0, 1.0),
    );
}

fn sample_view_diffuse_environment(sample_dir_ws: vec3<f32>) -> vec3<f32> {
    if light_probes.view_cubemap_index < 0 {
        return vec3<f32>(0.0);
    }

    var sample_dir = (environment_map_uniform.transform * vec4<f32>(sample_dir_ws, 1.0)).xyz;
    sample_dir.z = -sample_dir.z;

#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
    return textureSampleLevel(
        view_bindings::diffuse_environment_maps[light_probes.view_cubemap_index],
        view_bindings::environment_map_sampler,
        sample_dir,
        0.0,
    ).rgb * light_probes.intensity_for_view;
#else
    return textureSampleLevel(
        view_bindings::diffuse_environment_map,
        view_bindings::environment_map_sampler,
        sample_dir,
        0.0,
    ).rgb * light_probes.intensity_for_view;
#endif
}

fn clamp_to_surface(atmosphere: Atmosphere, position: vec3<f32>) -> vec3<f32> {
    let min_radius = atmosphere.bottom_radius + 1.0;
    let radius = length(position);
    if radius < min_radius {
        return normalize(position) * min_radius;
    }
    return position;
}

fn calculate_visible_sun_ratio(
    atmosphere: Atmosphere,
    radius: f32,
    mu: f32,
    sun_angular_size: f32,
) -> f32 {
    let horizon_cos = -sqrt(max(1.0 - (atmosphere.bottom_radius * atmosphere.bottom_radius) /
        max(radius * radius, EPSILON), 0.0));
    let horizon_angle = fast_acos_4(horizon_cos);
    let sun_zenith_angle = fast_acos_4(clamp(mu, -1.0, 1.0));

    if sun_zenith_angle + sun_angular_size * 0.5 <= horizon_angle {
        return 1.0;
    }

    if sun_zenith_angle - sun_angular_size * 0.5 >= horizon_angle {
        return 0.0;
    }

    let d = (horizon_angle - sun_zenith_angle) / max(sun_angular_size * 0.5, EPSILON);
    return clamp(0.5 + d * 0.5, 0.0, 1.0);
}

fn sample_atmosphere_transmittance(
    world_pos: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    sun_disk_angular_size: f32,
) -> vec3<f32> {
#ifdef ATMOSPHERE
    let atmosphere = view_bindings::atmosphere_data.atmosphere;
    let scene_units_to_m = view_bindings::atmosphere_data.settings.scene_units_to_m;
    let planet_origin = vec3<f32>(0.0, atmosphere.bottom_radius, 0.0);
    let atmosphere_pos = world_pos * scene_units_to_m + planet_origin;
    let clamped_pos = clamp_to_surface(atmosphere, atmosphere_pos);
    let radius = length(clamped_pos);
    let local_up = normalize(clamped_pos);
    let mu_light = dot(sun_dir_ws, local_up);
    let uv = transmittance_lut_r_mu_to_uv(atmosphere, radius, mu_light);
    let transmittance = textureSampleLevel(
        view_bindings::atmosphere_transmittance_texture,
        view_bindings::atmosphere_transmittance_sampler,
        uv,
        0.0,
    ).rgb;
    let sun_visibility = calculate_visible_sun_ratio(
        atmosphere,
        radius,
        mu_light,
        sun_disk_angular_size,
    );
    return transmittance * sun_visibility;
#else
    return vec3<f32>(1.0);
#endif
}

fn cloud_shadow(
    sample_world: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    cloud_bottom_height: f32,
    cloud_top_height: f32,
) -> f32 {
    let shadow_steps = max(material.march.y, 1u);
    let hit = intersect_height_slab(
        sample_world,
        sun_dir_ws,
        cloud_bottom_height,
        cloud_top_height,
        MAX_CLOUD_SHADOW_DISTANCE,
    );
    if hit.exit <= hit.entry {
        return 1.0;
    }

    let start = hit.entry;
    let end = hit.exit;
    let step_length = max((end - start) / f32(shadow_steps), 0.002);

    var t = start + step_length * 0.5;
    var transmittance = 1.0;

    for (var step: u32 = 0u; step < shadow_steps; step += 1u) {
        if t >= end || transmittance <= 0.03 {
            break;
        }

        let world_pos = sample_world + sun_dir_ws * t;
        let normalized_height = normalized_height_in_layer(world_pos.y, cloud_bottom_height, cloud_top_height);
        let density = sample_shadow_density(world_pos, normalized_height);
        transmittance *= exp(-density * material.lighting.w * step_length * 90.0);
        t += step_length;
    }

    return transmittance;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var out: FragmentOutput;

    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let camera_world = view.world_position;
    if !is_front {
        discard;
    }

    let ray_dir_world = normalize(in.world_position.xyz - camera_world);
    if ray_dir_world.y <= EPSILON {
        discard;
    }

    let cloud_bottom_height = (world_from_local * vec4<f32>(0.0, -0.5, 0.0, 1.0)).y;
    let cloud_top_height = (world_from_local * vec4<f32>(0.0, 0.5, 0.0, 1.0)).y;
    let hit = intersect_height_slab(
        camera_world,
        ray_dir_world,
        cloud_bottom_height,
        cloud_top_height,
        MAX_CLOUD_VIEW_DISTANCE,
    );

    let segment_start = hit.entry;
    let segment_end = hit.exit;
    if segment_end <= segment_start {
        discard;
    }

    let step_mix = smoothstep(0.2, 0.95, 1.0 - abs(ray_dir_world.y));
    let view_steps = max(
        u32(round(mix(f32(max(material.march.x, 1u)), f32(max(material.march.z, material.march.x)), step_mix))),
        1u,
    );
    let segment_length = segment_end - segment_start;
    let step_length = segment_length / f32(view_steps);
    let jitter = hash11(dot(in.position.xy, vec2<f32>(0.067, 0.113)) + material.wind_velocity_time.w)
        * material.softness.w;

    let has_sun = lights.n_directional_lights > 0u;
    let sun_dir_ws = select(vec3<f32>(0.0, 1.0, 0.0), lights.directional_lights[0].direction_to_light, has_sun);
    let sun_color = select(vec3<f32>(0.0), lights.directional_lights[0].color.rgb, has_sun);
    let sun_disk_size = select(0.0, lights.directional_lights[0].sun_disk_angular_size, has_sun);
    let phase = dual_lobe_phase(dot(ray_dir_world, sun_dir_ws));
    let sun_tint = chroma_only(sun_color);

    var transmittance = 1.0;
    var scattered_light = vec3<f32>(0.0);
    var t = segment_start + step_length * (0.35 + jitter);

    for (var step: u32 = 0u; step < view_steps; step += 1u) {
        if t >= segment_end || transmittance <= 0.01 {
            break;
        }

        let per_step_jitter = (hash11(
            dot(in.position.xy, vec2<f32>(0.013, 0.079))
                + f32(step) * 17.0
                + material.wind_velocity_time.w * 0.23,
        ) - 0.5) * step_length * 0.65;
        let sample_t = clamp(t + per_step_jitter, segment_start, segment_end);
        let sample_world = camera_world + ray_dir_world * sample_t;
        let normalized_height = normalized_height_in_layer(sample_world.y, cloud_bottom_height, cloud_top_height);
        let density = sample_density(sample_world, normalized_height);

        if density > 0.012 {
            let scene_units_to_m =
#ifdef ATMOSPHERE
                view_bindings::atmosphere_data.settings.scene_units_to_m;
#else
                1.0;
#endif

            let atmosphere = sample_atmosphere_transmittance(sample_world, sun_dir_ws, sun_disk_size);
            let shadow = select(
                1.0,
                select(
                    1.0,
                    cloud_shadow(sample_world, sun_dir_ws, cloud_bottom_height, cloud_top_height),
                    density > 0.08,
                ),
                has_sun,
            );

            let local_up = normalize(
#ifdef ATMOSPHERE
                sample_world * scene_units_to_m
                    + vec3<f32>(0.0, view_bindings::atmosphere_data.atmosphere.bottom_radius, 0.0)
#else
                vec3<f32>(0.0, 1.0, 0.0)
#endif
            );
            let sky_up = sample_view_diffuse_environment(local_up);
            let sky_horizon = sample_view_diffuse_environment(normalize(mix(local_up, -sun_dir_ws, 0.55)));
            let ambient_env = compress_hdr(mix(sky_up, sky_horizon, 0.35), 0.18);
            let powder = 1.0 - exp(-density * step_length * 42.0);
            let ambient = ambient_env * material.lighting.x * mix(0.85, 1.35, powder);
            let direct = sun_tint * atmosphere * shadow * phase * material.lighting.y
                * (1.0 + powder * material.phase_and_steps.w);
            let extinction = max(density * material.lighting.z * step_length * 64.0, EPSILON);
            let delta_transmittance = exp(-extinction);
            let integrated_scattering = (ambient + direct) * (1.0 - delta_transmittance);

            scattered_light += transmittance * integrated_scattering;
            transmittance *= delta_transmittance;
        }

        t += step_length;
    }

    scattered_light = compress_hdr(scattered_light, 0.35);
    let alpha = clamp(1.0 - transmittance, 0.0, 1.0);
    if alpha <= 0.002 {
        discard;
    }

    out.color = vec4<f32>(scattered_light, alpha);
    return out;
}

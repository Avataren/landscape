// terrain_fragment.wgsl
// Terrain surface shading integrated with Bevy's clustered forward rendering.
//
// Supports all Bevy light types:
//   - Directional lights  (iterated globally, with cascade shadows)
//   - Point lights        (clustered, with optional shadows)
//   - Spot lights         (clustered, with optional shadows)
//
// Albedo is a world-aligned macro colour map (baked data) or a procedural
// slope/altitude blend (fallback).  Lambertian diffuse BRDF; no specular
// (terrain reads best with diffuse-only shading).

#import bevy_pbr::{
    mesh_view_bindings::{lights, view, clusterable_objects},
    mesh_view_types::{
        DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    },
    shadows::{fetch_directional_shadow, fetch_point_shadow, fetch_spot_shadow},
    clustered_forward::{
        fragment_cluster_index,
        unpack_clusterable_object_index_ranges,
        get_clusterable_object_id,
    },
}

// Must match TerrainMaterialUniforms in material.rs exactly.
struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,
    morph_start_ratio:  f32,
    ring_patches:       f32,
    num_lod_levels:     f32,
    patch_resolution:   f32,
    world_bounds:       vec4<f32>,
    bounds_fade:        vec4<f32>, // x = fade dist, y = use_macro_color, z = flip_v, w = show_wireframe
    clip_levels: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:  texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var macro_color_tex:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var macro_color_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

// Must match TerrainVOut in terrain_vertex.wgsl.
struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       macro_xz_ws:  vec2<f32>,
    @location(3)       patch_uv:     vec2<f32>,
    @location(4) @interpolate(flat) lod_level: u32,
    @location(5)       morph_alpha:  f32,
}

// ---------------------------------------------------------------------------
// Albedo helpers
// ---------------------------------------------------------------------------

fn procedural_albedo(slope: f32, h_norm: f32) -> vec3<f32> {
    let grass = vec3<f32>(0.28, 0.52, 0.18);
    let dirt  = vec3<f32>(0.50, 0.40, 0.28);
    let rock  = vec3<f32>(0.44, 0.38, 0.32);
    let snow  = vec3<f32>(0.90, 0.93, 0.98);

    var c = mix(grass, dirt, smoothstep(0.12, 0.30, slope));
    c     = mix(c,    rock,  smoothstep(0.30, 0.52, slope));
    c     = mix(c,    snow,  smoothstep(0.62, 0.82, h_norm));
    return c;
}

fn in_world_bounds(world_xz: vec2<f32>) -> bool {
    return all(world_xz >= terrain.world_bounds.xy)
        && all(world_xz <= terrain.world_bounds.zw);
}

fn macro_color_uv(world_xz: vec2<f32>) -> vec2<f32> {
    let world_min  = terrain.world_bounds.xy;
    let world_span = max(terrain.world_bounds.zw - world_min, vec2<f32>(1.0, 1.0));
    var uv = (world_xz - world_min) / world_span;
    if terrain.bounds_fade.z > 0.5 {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}

// ---------------------------------------------------------------------------
// Sky hemisphere ambient
//
// Approximates diffuse IBL from the atmosphere without cubemap sampling.
// The sky and ground bounce values are in *display-space* (0–1 fractions).
// Dividing by view.exposure converts them to the physical-unit scale used by
// the rest of the lighting, so that when the final (direct + ambient) is
// multiplied by view.exposure the hemisphere values survive as their original
// display-space fractions.
//
// Adjust these to match the scene's sky colour and ground albedo.
// ---------------------------------------------------------------------------

const SKY_AMBIENT:    vec3<f32> = vec3<f32>(0.20, 0.26, 0.40); // clear-sky blue, ~25 % brightness
const GROUND_BOUNCE:  vec3<f32> = vec3<f32>(0.04, 0.03, 0.02); // dark earth, ~4 % brightness

fn hemisphere_ambient(world_normal: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    // sky_t = 1 for normals pointing straight up, 0 for straight down.
    let sky_t  = saturate(world_normal.y * 0.5 + 0.5);
    let irr    = mix(GROUND_BOUNCE, SKY_AMBIENT, sky_t);
    return albedo * irr;
}

// ---------------------------------------------------------------------------
// Per-pixel normal from height finite differences
//
// Mirrors the vertex shader's normal_at() exactly, but runs in the fragment
// stage so every pixel gets an independent normal sample — no vertex
// interpolation artefacts.  Uses the height clipmap (always populated) rather
// than the baked RG8Snorm normal array (which may be zero when
// procedural_fallback is false and no baked normal tiles are loaded).
// ---------------------------------------------------------------------------

fn height_at_frag(lod: u32, xz: vec2<f32>) -> f32 {
    let lvl        = terrain.clip_levels[lod];
    let world_min  = terrain.world_bounds.xy;
    let world_max  = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz  = clamp(xz, world_min, world_max);
    let uv         = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    return textureSample(height_tex, height_samp, uv, i32(lod)).r
        * terrain.height_scale;
}

fn pixel_normal(lod: u32, xz: vec2<f32>) -> vec3<f32> {
    let eps = terrain.clip_levels[lod].w; // texel world size at this LOD
    let h   = height_at_frag(lod, xz);
    let h_r = height_at_frag(lod, xz + vec2<f32>(eps, 0.0));
    let h_u = height_at_frag(lod, xz + vec2<f32>(0.0, eps));
    return normalize(vec3<f32>(h - h_r, eps, h - h_u));
}

// ---------------------------------------------------------------------------
// Attenuation — same formula as pbr_lighting::getDistanceAttenuation
// ---------------------------------------------------------------------------

fn distance_attenuation(dist_sq: f32, inv_range_sq: f32) -> f32 {
    let factor       = dist_sq * inv_range_sq;
    let smooth_factor = saturate(1.0 - factor * factor);
    return (smooth_factor * smooth_factor) / max(dist_sq, 0.0001);
}

fn patch_grid_wireframe(patch_uv: vec2<f32>) -> f32 {
    let grid_uv = patch_uv * terrain.patch_resolution;
    let grid_fw = max(fwidth(grid_uv), vec2<f32>(1e-4, 1e-4));
    let quad_dist = abs(fract(grid_uv - 0.5) - 0.5) / grid_fw;
    let quad_line = 1.0 - saturate(min(quad_dist.x, quad_dist.y));

    let patch_fw = max(fwidth(patch_uv), vec2<f32>(1e-4, 1e-4));
    let patch_dist = min(patch_uv, 1.0 - patch_uv) / patch_fw;
    let patch_line = 1.0 - saturate(min(patch_dist.x, patch_dist.y) - 0.5);

    return max(quad_line * 0.75, patch_line);
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

@fragment
fn fragment(in: TerrainVOut) -> @location(0) vec4<f32> {
    // Discard patches that extend beyond the heightmap footprint.
    if !in_world_bounds(in.macro_xz_ws) {
        discard;
    }

    // --- Per-pixel normal from height finite differences ---
    // `vertex_n` is the morph-blended vertex normal used for shadow bias
    // (stable, no high-frequency noise).  `n` is recomputed per-pixel from
    // the height clipmap — same formula as the vertex shader's normal_at(),
    // but evaluated at the exact fragment world position rather than at the
    // nearest vertex, eliminating vertex-interpolation artefacts on curved slopes.
    //
    // Normal blending mirrors the vertex shader: at morph_alpha=0 the fine LOD
    // normal is used exclusively; at alpha=1 (the seam boundary) only the coarse
    // LOD normal is used.  This eliminates the lighting band that appears when
    // adjacent LOD patches compute normals from different clipmap resolutions.
    // Using world_pos.xz (the morphed position) ensures both sides of the seam
    // sample the same clipmap point when alpha=1.
    let vertex_n   = normalize(in.world_normal);
    let coarse_lod = min(in.lod_level + 1u, u32(terrain.num_lod_levels) - 1u);
    let frag_xz    = in.world_pos.xz;
    let n_fine     = pixel_normal(in.lod_level, frag_xz);
    let n_coarse   = pixel_normal(coarse_lod,   frag_xz);
    let n          = normalize(mix(n_fine, n_coarse, in.morph_alpha));

    // --- Albedo ---
    let slope  = 1.0 - abs(dot(n, vec3<f32>(0.0, 1.0, 0.0)));
    let h_norm = clamp(in.world_pos.y / terrain.height_scale, 0.0, 1.0);

    var albedo = procedural_albedo(slope, h_norm);
    if terrain.bounds_fade.y > 0.5 {
        // textureSample (not textureSampleLevel) lets the GPU pick the correct
        // mip based on screen-space derivatives — better antialiasing at distance.
        albedo = textureSample(
            macro_color_tex, macro_color_samp,
            macro_color_uv(in.macro_xz_ws),
        ).rgb;
    }

    // --- Shared values ---
    // view_z: view-space depth, used for shadow cascade selection and
    // cluster z-slice lookup.  Negative for fragments in front of camera.
    let view_z = dot(
        vec4<f32>(view.view_from_world[0].z,
                  view.view_from_world[1].z,
                  view.view_from_world[2].z,
                  view.view_from_world[3].z),
        in.world_pos,
    );
    let is_orthographic = view.clip_from_view[3].w == 1.0;

    // Lambertian BRDF numerator: albedo / π (denominator is cancelled by
    // the π in the irradiance-to-radiance conversion already baked into
    // light.color.rgb for directional lights, and absent for point/spot
    // which pass colour × intensity directly).
    let diffuse = albedo / 3.14159;

    var direct = vec3<f32>(0.0);

    // -----------------------------------------------------------------------
    // Directional lights  (global, not clustered)
    // light.color.rgb is pre-multiplied by illuminance (lux) in the GPU buffer.
    // -----------------------------------------------------------------------
    for (var i: u32 = 0u; i < lights.n_directional_lights; i++) {
        let light = lights.directional_lights[i];
        let ndotl = max(dot(n, light.direction_to_light), 0.0);

        var shadow = 1.0;
        if (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            // Shadow bias uses vertex_n (per-vertex, more stable for self-shadow).
            shadow = fetch_directional_shadow(i, in.world_pos, vertex_n, view_z);
        }

        direct += diffuse * ndotl * shadow * light.color.rgb;
    }

    // -----------------------------------------------------------------------
    // Clustered point and spot lights
    // color_inverse_square_range.rgb = colour × intensity (cd/m²)
    // color_inverse_square_range.w   = 1 / range²
    // -----------------------------------------------------------------------
    let cluster_index = fragment_cluster_index(in.clip_pos.xy, view_z, is_orthographic);
    let ranges        = unpack_clusterable_object_index_ranges(cluster_index);

    // --- Point lights ---
    for (var i: u32 = ranges.first_point_light_index_offset;
             i < ranges.first_spot_light_index_offset; i++) {
        let light_id  = get_clusterable_object_id(i);
        let light     = &clusterable_objects.data[light_id];
        let to_frag   = (*light).position_radius.xyz - in.world_pos.xyz;
        let dist_sq   = dot(to_frag, to_frag);
        let L         = normalize(to_frag);
        let ndotl     = max(dot(n, L), 0.0);
        let atten     = distance_attenuation(dist_sq, (*light).color_inverse_square_range.w);

        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_point_shadow(light_id, in.world_pos, vertex_n);
        }

        direct += diffuse * ndotl * atten * shadow * (*light).color_inverse_square_range.rgb;
    }

    // --- Spot lights ---
    for (var i: u32 = ranges.first_spot_light_index_offset;
             i < ranges.first_reflection_probe_index_offset; i++) {
        let light_id  = get_clusterable_object_id(i);
        let light     = &clusterable_objects.data[light_id];
        let to_frag   = (*light).position_radius.xyz - in.world_pos.xyz;
        let dist_sq   = dot(to_frag, to_frag);
        let L         = normalize(to_frag);
        let ndotl     = max(dot(n, L), 0.0);
        let atten     = distance_attenuation(dist_sq, (*light).color_inverse_square_range.w);

        // Spot cone attenuation.
        // light_custom_data.xy = normalised spot direction (xz components);
        // light_custom_data.zw = precomputed scale and offset for cone falloff.
        var spot_dir = vec3<f32>((*light).light_custom_data.x, 0.0, (*light).light_custom_data.y);
        spot_dir.y = sqrt(max(0.0, 1.0 - spot_dir.x * spot_dir.x - spot_dir.z * spot_dir.z));
        if ((*light).flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
            spot_dir.y = -spot_dir.y;
        }
        let cd           = dot(-spot_dir, L);
        let cone_raw     = saturate(cd * (*light).light_custom_data.z + (*light).light_custom_data.w);
        let cone_atten   = cone_raw * cone_raw;

        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_spot_shadow(
                light_id, in.world_pos, vertex_n, (*light).shadow_map_near_z,
            );
        }

        direct += diffuse * ndotl * atten * cone_atten * shadow
            * (*light).color_inverse_square_range.rgb;
    }

    // --- Ambient ---
    // Hemisphere ambient (sky/ground bounce) gives physically motivated diffuse
    // sky light without requiring cubemap sampling.  The result is in display
    // space; dividing by view.exposure puts it in the same physical-unit scale
    // as `direct` so the final `* view.exposure` restores the display fraction.
    // lights.ambient_color is the GlobalAmbientLight flat term (may be zero).
    let sky_ambient_physical = hemisphere_ambient(n, albedo) / max(view.exposure, 1e-10);
    let flat_ambient         = albedo * lights.ambient_color.rgb;
    let ambient              = sky_ambient_physical + flat_ambient;

    // Apply camera exposure (physical luminance → display values).
    var out_rgb = (direct + ambient) * view.exposure;

    if terrain.bounds_fade.w > 0.5 {
        let wire = patch_grid_wireframe(in.patch_uv);
        let wire_color = vec3<f32>(0.98, 0.94, 0.35);
        out_rgb = mix(out_rgb, wire_color, wire * 0.92);
    }

    return vec4<f32>(out_rgb, 1.0);
}

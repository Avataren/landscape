// terrain_fragment.wgsl
// Terrain surface shading: iterates all scene directional lights directly via
// mesh_view_bindings so the terrain reacts to every light in the scene without
// storing any light data in the terrain uniform.

#import bevy_pbr::{
    mesh_view_bindings::{lights, view},
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    shadows::fetch_directional_shadow,
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
    bounds_fade:        vec4<f32>, // x = fade dist, y = use_macro_color, z = flip_v
    clip_levels: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var macro_color_tex:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var macro_color_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

// Must match TerrainVOut in terrain_vertex.wgsl.
struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       macro_xz_ws:  vec2<f32>,
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
// Fragment
// ---------------------------------------------------------------------------

@fragment
fn fragment(in: TerrainVOut) -> @location(0) vec4<f32> {
    // Discard patches that extend beyond the heightmap footprint.
    if !in_world_bounds(in.macro_xz_ws) {
        discard;
    }

    // --- Albedo ---
    let n      = normalize(in.world_normal);
    let slope  = 1.0 - abs(dot(n, vec3<f32>(0.0, 1.0, 0.0)));
    let h_norm = clamp(in.world_pos.y / terrain.height_scale, 0.0, 1.0);

    var albedo = procedural_albedo(slope, h_norm);
    if terrain.bounds_fade.y > 0.5 {
        albedo = textureSampleLevel(
            macro_color_tex, macro_color_samp,
            macro_color_uv(in.macro_xz_ws), 0.0,
        ).rgb;
    }

    // --- View-space depth used for shadow cascade selection ---
    let view_z = dot(
        vec4<f32>(view.view_from_world[0].z,
                  view.view_from_world[1].z,
                  view.view_from_world[2].z,
                  view.view_from_world[3].z),
        in.world_pos,
    );

    // --- Directional lights ---
    // light.color.rgb is pre-multiplied by illuminance in the GPU buffer.
    // Lambertian diffuse: albedo * NdotL * color / π.
    var direct = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < lights.n_directional_lights; i++) {
        let light  = lights.directional_lights[i];
        let ndotl  = max(dot(n, light.direction_to_light), 0.0);

        var shadow = 1.0;
        if (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_directional_shadow(i, in.world_pos, in.world_normal, view_z);
        }

        direct += (albedo / 3.14159) * ndotl * shadow * light.color.rgb;
    }

    // --- Ambient (GlobalAmbientLight + environment map contribution baked in) ---
    let ambient = albedo * lights.ambient_color.rgb;

    // Apply camera exposure (converts physical luminance to display values).
    let out_rgb = (direct + ambient) * view.exposure;

    return vec4<f32>(out_rgb, 1.0);
}

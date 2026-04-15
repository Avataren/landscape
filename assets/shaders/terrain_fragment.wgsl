// terrain_fragment.wgsl
// Slope + altitude layer blending with directional lighting + cascade shadows.

#import bevy_pbr::{
    mesh_view_bindings as view_bindings,
    shadows,
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
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
    bounds_fade:        vec4<f32>,
    sun_direction:      vec4<f32>,  // xyz = world-space toward-sun unit vector
    clip_levels: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var macro_color_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var macro_color_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

// Must match TerrainVOut in terrain_vertex.wgsl.
struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       macro_xz_ws:  vec2<f32>,
}

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

fn macro_color_uv(world_xz: vec2<f32>) -> vec2<f32> {
    let world_min = terrain.world_bounds.xy;
    let world_span = max(terrain.world_bounds.zw - world_min, vec2<f32>(1.0, 1.0));
    var uv = (world_xz - world_min) / world_span;
    // bounds_fade.z > 0.5 → flip V to correct EXR files exported with V=0 at bottom.
    if terrain.bounds_fade.z > 0.5 {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}

fn macro_color_in_bounds(world_xz: vec2<f32>) -> bool {
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw;
    return all(world_xz >= world_min) && all(world_xz <= world_max);
}

@fragment
fn fragment(in: TerrainVOut) -> @location(0) vec4<f32> {
    // Discard fragments outside the heightmap footprint so the mesh rings
    // that extend beyond world_bounds don't render as flat green terrain.
    if !macro_color_in_bounds(in.macro_xz_ws) {
        discard;
    }

    let n      = normalize(in.world_normal);
    let up     = vec3<f32>(0.0, 1.0, 0.0);
    let slope  = 1.0 - abs(dot(n, up));                         // 0=flat, 1=vertical
    let h_norm = clamp(in.world_pos.y / terrain.height_scale, 0.0, 1.0);
    let use_macro_color = terrain.bounds_fade.y > 0.5;
    let macro_in_bounds = macro_color_in_bounds(in.macro_xz_ws);

    var c = procedural_albedo(slope, h_norm);
    if use_macro_color && macro_in_bounds {
        c = textureSampleLevel(macro_color_tex, macro_color_samp, macro_color_uv(in.macro_xz_ws), 0.0).rgb;
    }

    // --- Directional sun from scene light ---
    let sun   = normalize(terrain.sun_direction.xyz);
    let ndotl = max(dot(n, sun), 0.0);

    // --- Cascade shadow from Bevy's shadow maps ---
    // view_z is the view-space depth used to select the correct cascade.
    // Computed by dotting the Z-row of view_from_world against world position.
    let view_z = dot(vec4<f32>(
        view_bindings::view.view_from_world[0].z,
        view_bindings::view.view_from_world[1].z,
        view_bindings::view.view_from_world[2].z,
        view_bindings::view.view_from_world[3].z,
    ), in.world_pos);

    var shadow = 1.0;
    if view_bindings::lights.n_directional_lights > 0u {
        let flags = view_bindings::lights.directional_lights[0].flags;
        if (flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_directional_shadow(0u, in.world_pos, n, view_z);
        }
    }

    // Ambient is unshadowed; direct component is attenuated by shadow.
    let lit = 0.18 + ndotl * 0.82 * shadow;

    // --- Subtle macro variation to break visual repetition on the procedural fallback ---
    let macro_v = select(
        sin(in.world_pos.x * 0.008) * sin(in.world_pos.z * 0.011) * 0.03,
        0.0,
        use_macro_color,
    );

    return vec4<f32>(c * lit + macro_v, 1.0);
}

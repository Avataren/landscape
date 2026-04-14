// terrain_fragment.wgsl
// Slope + altitude layer blending with simple directional lighting.
// Self-contained: no external common imports.

// Must match TerrainMaterialUniforms in material.rs exactly.
struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,
    morph_start_ratio:  f32,
    ring_patches:       f32,
    pad0:               f32,
    pad1:               f32,
    pad2:               f32,
    pad3:               f32,
    clip_levels: array<vec4<f32>, 8>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

// Must match TerrainVOut in terrain_vertex.wgsl.
struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
}

@fragment
fn fragment(in: TerrainVOut) -> @location(0) vec4<f32> {
    let n      = normalize(in.world_normal);
    let up     = vec3<f32>(0.0, 1.0, 0.0);
    let slope  = 1.0 - abs(dot(n, up));                         // 0=flat, 1=vertical
    let h_norm = clamp(in.world_pos.y / terrain.height_scale, 0.0, 1.0);

    // --- Material layers ---
    let grass = vec3<f32>(0.28, 0.52, 0.18);
    let dirt  = vec3<f32>(0.50, 0.40, 0.28);
    let rock  = vec3<f32>(0.44, 0.38, 0.32);
    let snow  = vec3<f32>(0.90, 0.93, 0.98);

    var c = mix(grass, dirt, smoothstep(0.12, 0.30, slope));
    c     = mix(c,    rock,  smoothstep(0.30, 0.52, slope));
    c     = mix(c,    snow,  smoothstep(0.62, 0.82, h_norm));

    // --- Simple directional sun ---
    let sun = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let lit = 0.18 + max(dot(n, sun), 0.0) * 0.82;

    // --- Subtle macro variation to break visual repetition ---
    let macro_v = sin(in.world_pos.x * 0.008) * sin(in.world_pos.z * 0.011) * 0.03;

    return vec4<f32>(c * lit + macro_v, 1.0);
}

// terrain_common.wgsl
// Shared definitions and helpers imported by vertex and fragment shaders.

// ---------------------------------------------------------------------------
// Terrain material uniforms (group 2, binding 2)
// Must match TerrainMaterialUniforms in material.rs exactly.
// ---------------------------------------------------------------------------
struct TerrainParams {
    height_scale:    f32,   // world-space Y multiplier for [0,1] height values
    world_size:      f32,   // world-space XZ extent covered by the height texture
    world_offset_x:  f32,   // world-space X of the texture's minimum corner
    world_offset_z:  f32,   // world-space Z of the texture's minimum corner
}

// ---------------------------------------------------------------------------
// Sample height texture at a world-space XZ position.
// Returns a [0, height_scale] world-space Y value.
// ---------------------------------------------------------------------------
fn sample_height_world(
    tex:    texture_2d<f32>,
    samp:   sampler,
    params: TerrainParams,
    world_xz: vec2<f32>,
) -> f32 {
    let uv = world_to_height_uv(params, world_xz);
    return textureSampleLevel(tex, samp, uv, 0.0).r * params.height_scale;
}

// Convert world XZ to height texture UV [0,1].
fn world_to_height_uv(params: TerrainParams, world_xz: vec2<f32>) -> vec2<f32> {
    return (world_xz - vec2<f32>(params.world_offset_x, params.world_offset_z)) / params.world_size;
}

// ---------------------------------------------------------------------------
// Reconstruct a world-space normal from height texture finite differences.
// eps_world = world-space step size for the central difference.
// ---------------------------------------------------------------------------
fn reconstruct_normal(
    tex:       texture_2d<f32>,
    samp:      sampler,
    params:    TerrainParams,
    world_xz:  vec2<f32>,
    eps_world: f32,
) -> vec3<f32> {
    let h   = textureSampleLevel(tex, samp, world_to_height_uv(params, world_xz), 0.0).r;
    let h_r = textureSampleLevel(tex, samp, world_to_height_uv(params, world_xz + vec2<f32>(eps_world, 0.0)), 0.0).r;
    let h_u = textureSampleLevel(tex, samp, world_to_height_uv(params, world_xz + vec2<f32>(0.0, eps_world)), 0.0).r;
    // Cross-product of tangent vectors scaled by height_scale
    return normalize(vec3<f32>(
        (h - h_r) * params.height_scale,
        eps_world,
        (h - h_u) * params.height_scale,
    ));
}

// ---------------------------------------------------------------------------
// Smoothstep-based terrain layer blend helpers
// ---------------------------------------------------------------------------
fn blend_rock(slope: f32) -> f32   { return smoothstep(0.30, 0.52, slope); }
fn blend_dirt(slope: f32) -> f32   { return smoothstep(0.12, 0.30, slope) * (1.0 - blend_rock(slope)); }
fn blend_snow(h_norm: f32) -> f32  { return smoothstep(0.62, 0.82, h_norm); }

fn terrain_albedo(slope: f32, h_norm: f32) -> vec3<f32> {
    let grass = vec3<f32>(0.30, 0.52, 0.20);
    let dirt  = vec3<f32>(0.50, 0.40, 0.28);
    let rock  = vec3<f32>(0.44, 0.38, 0.32);
    let snow  = vec3<f32>(0.90, 0.93, 0.98);

    var c = mix(grass, dirt, blend_dirt(slope));
    c = mix(c, rock, blend_rock(slope));
    c = mix(c, snow, blend_snow(h_norm));
    return c;
}

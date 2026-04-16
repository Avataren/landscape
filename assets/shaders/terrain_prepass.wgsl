// terrain_prepass.wgsl
// Depth-only prepass for terrain shadow casting.
//
// Replicates the height sampling and CDLOD morphing from terrain_vertex.wgsl so
// that the shadow-map geometry matches the visible displaced surface exactly.
// Only outputs clip position (and world_position required by VertexOutput).

#import bevy_pbr::{
    mesh_functions::get_world_from_local,
    view_transformations::position_world_to_clip,
    prepass_io::{Vertex, VertexOutput},
}

struct MaterialSlotGpu {
    tint_vis: vec4<f32>,
    ranges:   vec4<f32>,
}

struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,
    morph_start_ratio:  f32,
    ring_patches:       f32,
    num_lod_levels:     f32,
    patch_resolution:   f32,
    world_bounds:       vec4<f32>,
    bounds_fade:        vec4<f32>,
    debug_flags:        vec4<f32>,
    clip_levels: array<vec4<f32>, 16>,
    slot_header: vec4<f32>,
    slots:       array<MaterialSlotGpu, 8>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:  texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

fn bounds_fade_at(xz: vec2<f32>) -> f32 {
    let fade_dist = max(terrain.bounds_fade.x, 1.0);
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw;
    let edge_dist = min(
        min(xz.x - world_min.x, world_max.x - xz.x),
        min(xz.y - world_min.y, world_max.y - xz.y),
    );
    return smoothstep(0.0, fade_dist, edge_dist);
}

fn height_at(lod: u32, xz: vec2<f32>) -> f32 {
    let lvl = terrain.clip_levels[lod];
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(xz, world_min, world_max);
    let uv = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    let h = textureSampleLevel(height_tex, height_samp, uv, i32(lod), 0.0).r
        * terrain.height_scale;
    return h * bounds_fade_at(xz);
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    let model = get_world_from_local(v.instance_index);

    let patch_size_ws = length(model[0].xyz);

    let lod_f     = round(log2(patch_size_ws / terrain.base_patch_size));
    let lod_level = u32(clamp(lod_f, 0.0, 15.0));

    let max_lod_idx = u32(terrain.num_lod_levels) - 1u;
    let coarse_lod  = min(lod_level + 1u, max_lod_idx);

    let world_xz_orig = (model * vec4<f32>(v.position, 1.0)).xz;

    // --- Geomorphing (identical to terrain_vertex.wgsl) ---
    let lvl_fine     = terrain.clip_levels[lod_level];
    let half_ring_ws = 0.5 / lvl_fine.z;
    let ring_center  = lvl_fine.xy;

    let vertex_delta     = abs(world_xz_orig - ring_center);
    let dist_from_center = max(vertex_delta.x, vertex_delta.y);

    let morph_start_ws = half_ring_ws * terrain.morph_start_ratio;
    let boundary_t = clamp(
        (dist_from_center - morph_start_ws) / max(half_ring_ws - morph_start_ws, 0.001),
        0.0, 1.0,
    );
    let boundary_alpha = boundary_t * boundary_t * (3.0 - 2.0 * boundary_t);

    let boundary_dist_ws = half_ring_ws - dist_from_center;
    let boundary_lock = select(
        0.0, 1.0,
        boundary_dist_ws <= (0.5 * terrain.clip_levels[lod_level].w + 1e-4),
    );
    let morph_alpha = max(boundary_alpha, boundary_lock);

    let fine_step_ws   = terrain.clip_levels[lod_level].w;
    let coarse_step_ws = fine_step_ws * 2.0;
    let coarse_world_xz = floor(world_xz_orig / coarse_step_ws) * coarse_step_ws;
    let world_xz = mix(world_xz_orig, coarse_world_xz, morph_alpha);

    // --- Height blend (same as main pass) ---
    let h   = mix(height_at(lod_level, world_xz), height_at(coarse_lod, world_xz), morph_alpha);
    let pos = vec3<f32>(world_xz.x, h, world_xz.y);

    var out: VertexOutput;
    out.world_position = vec4<f32>(pos, 1.0);
    out.position       = position_world_to_clip(pos);
    return out;
}

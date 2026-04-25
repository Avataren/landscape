// terrain_vertex.wgsl
// Phase 5 — per-level clipmap texture array + CDLOD geomorphing.
//
// The clipmap (height_tex) stores world-space metres directly (R32Float).
// Fine LOD layers are written by the detail_synthesis compute pass which
// combines bilinear source height + fBM residual.  Coarse LOD layers are
// written by the CPU tile upload system.  No separate detail texture needed.

#import bevy_pbr::{
    mesh_functions::get_world_from_local,
    view_transformations::position_world_to_clip,
    forward_io::Vertex,
}

struct MaterialSlotGpu {
    tint_vis: vec4<f32>,
    ranges:   vec4<f32>,
}

struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,   // lod0_mesh_spacing; LOD = log2(level_scale_ws / base_patch_size)
    morph_start_ratio:  f32,
    ring_patches:       f32,
    num_lod_levels:     f32,
    patch_resolution:   f32,
    world_bounds:       vec4<f32>, // (min_x, min_z, max_x, max_z)
    bounds_fade:        vec4<f32>, // x = fade distance, y = use_macro_color, z = flip_v, w = show_wireframe
    debug_flags:        vec4<f32>, // x = show_normals_only, yzw reserved
    clip_levels: array<vec4<f32>, 32>,
    slot_header: vec4<f32>,
    slots:       array<MaterialSlotGpu, 8>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)  var height_tex:   texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1)  var height_samp:  sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2)  var<uniform> terrain: TerrainParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(5)  var normal_tex:   texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6)  var normal_samp:  sampler;

// ---------------------------------------------------------------------------
// Vertex → fragment interface
// ---------------------------------------------------------------------------

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
// Helpers
// ---------------------------------------------------------------------------

fn in_world_bounds(world_xz: vec2<f32>) -> bool {
    return all(world_xz >= terrain.world_bounds.xy)
        && all(world_xz <= terrain.world_bounds.zw);
}

/// Sample height in world-space metres from the R32Float clipmap array.
/// Returns 0.0 for out-of-bounds positions (flat ground plane at terrain edge).
fn height_at(lod: u32, xz: vec2<f32>) -> f32 {
    if !in_world_bounds(xz) {
        return 0.0;
    }
    let lvl = terrain.clip_levels[lod];
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(xz, world_min, world_max);
    let uv = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    // height_tex stores metres directly (R32Float written by compute or CPU upload).
    return textureSampleLevel(height_tex, height_samp, uv, i32(lod), 0.0).r;
}

/// Sample the baked RGBA8Snorm normal array and return a morph-blended world-space normal.
fn baked_normal_v(lod: u32, xz: vec2<f32>, alpha: f32) -> vec3<f32> {
    let lvl       = terrain.clip_levels[lod];
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(xz, world_min, world_max);
    let uv        = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    let rgba      = textureSampleLevel(normal_tex, normal_samp, uv, i32(lod), 0.0);
    let xz_n      = mix(rgba.rg, rgba.ba, alpha);
    let y2        = max(1.0 - dot(xz_n, xz_n), 0.0);
    return normalize(vec3<f32>(xz_n.x, sqrt(y2), xz_n.y));
}

/// Blend height between fine and coarse clipmap levels by morph_alpha.
fn blended_height(lod: u32, coarse_lod: u32, alpha: f32, xz: vec2<f32>) -> f32 {
    return mix(height_at(lod, xz), height_at(coarse_lod, xz), alpha);
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vertex(v: Vertex) -> TerrainVOut {
    let model = get_world_from_local(v.instance_index);
    let patch_size_ws = length(model[0].xyz); // = level_scale_ws
    let lod_f = round(log2(patch_size_ws / terrain.base_patch_size));
    let lod_level = u32(clamp(lod_f, 0.0, 15.0));

    let max_lod_idx = u32(terrain.num_lod_levels) - 1u;
    let coarse_lod  = min(lod_level + 1u, max_lod_idx);

    let world_xz_orig = (model * vec4<f32>(v.position, 1.0)).xz;

    // --- Geomorphing ---
    let lvl_fine    = terrain.clip_levels[lod_level];
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
        0.0,
        1.0,
        boundary_dist_ws <= (0.5 * terrain.clip_levels[lod_level].w + 1e-4),
    );

    let morph_alpha = max(boundary_alpha, boundary_lock);

    let fine_step_ws   = terrain.clip_levels[lod_level].w;
    let coarse_step_ws = fine_step_ws * 2.0;
    let coarse_world_xz = floor(world_xz_orig / coarse_step_ws) * coarse_step_ws;

    let world_xz = mix(world_xz_orig, coarse_world_xz, morph_alpha);

    let h   = blended_height(lod_level, coarse_lod, morph_alpha, world_xz);
    let pos = vec3<f32>(world_xz.x, h, world_xz.y);

    let nrm = baked_normal_v(lod_level, world_xz, morph_alpha);

    var out: TerrainVOut;
    out.clip_pos     = position_world_to_clip(pos);
    out.world_pos    = vec4<f32>(pos, 1.0);
    out.world_normal = nrm;
    out.macro_xz_ws  = world_xz_orig;
    out.patch_uv     = v.uv;
    out.lod_level    = lod_level;
    out.morph_alpha  = morph_alpha;
    return out;
}

// terrain_vertex.wgsl
// Phase 5 — per-level clipmap texture array + CDLOD geomorphing.
//
// KEY: to eliminate height discontinuities at LOD ring boundaries, we blend
// height samples between the fine level and the coarser level using morph_alpha.
// When alpha=0 we read the fine level; when alpha=1 we read the coarse level.
// This guarantees that the outer boundary of LOD L and the inner boundary of
// LOD L+1 always agree on height for the same world XZ position.

#import bevy_pbr::{
    mesh_functions::get_world_from_local,
    view_transformations::position_world_to_clip,
    forward_io::Vertex,
    mesh_view_bindings::view,
}

// ---------------------------------------------------------------------------
// Bindings — match TerrainMaterialUniforms in material.rs exactly.
//
// clip_levels[L] = (origin_x, origin_z, inv_ring_span, texel_world_size)
// ---------------------------------------------------------------------------

struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,   // patch_resolution * world_scale (LOD-0 patch side)
    morph_start_ratio:  f32,
    ring_patches:       f32,
    num_lod_levels:     f32,   // active LOD count; used to clamp coarse index
    pad1:               f32,
    pad2:               f32,
    pad3:               f32,
    clip_levels: array<vec4<f32>, 8>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:  texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

// ---------------------------------------------------------------------------
// Vertex → fragment interface
// ---------------------------------------------------------------------------

struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample height from the given LOD level's clipmap layer.
fn height_at(lod: u32, xz: vec2<f32>) -> f32 {
    let lvl = terrain.clip_levels[lod];
    // lvl.xy = world-space bottom-left corner of the clipmap region
    // lvl.z  = 1 / ring_span  →  maps world XZ into [0,1] UV
    let uv = clamp((xz - lvl.xy) * lvl.z, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(height_tex, height_samp, uv, i32(lod), 0.0).r
           * terrain.height_scale;
}

/// Blend height between fine (lod) and coarse (lod+1) levels by morph_alpha.
/// This ensures both sides of a LOD ring boundary read the same height value,
/// eliminating the crack caused by different texture resolutions.
fn blended_height(lod: u32, coarse_lod: u32, alpha: f32, xz: vec2<f32>) -> f32 {
    let h_fine   = height_at(lod,        xz);
    let h_coarse = height_at(coarse_lod, xz);
    return mix(h_fine, h_coarse, alpha);
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vertex(v: Vertex) -> TerrainVOut {
    let model = get_world_from_local(v.instance_index);

    // --- Patch world size from model X-scale. ---
    // Transform encodes: scale = (patch_size_ws, 1, patch_size_ws).
    let patch_size_ws = length(model[0].xyz);

    // --- Derive LOD level. ---
    // patch_size_ws = base_patch_size × 2^L  →  L = log2(size / base)
    let lod_f     = round(log2(patch_size_ws / terrain.base_patch_size));
    let lod_level = u32(clamp(lod_f, 0.0, 7.0));

    // Coarse LOD index for height blending: clamped so we never read layer N.
    let max_lod_idx = u32(terrain.num_lod_levels) - 1u;
    let coarse_lod  = min(lod_level + 1u, max_lod_idx);

    // --- World XZ before morphing (for the camera-distance test). ---
    let world_xz_orig = (model * vec4<f32>(v.position, 1.0)).xz;

    // --- Geomorphing ---
    // Blend vertices toward the 2× coarser grid near the outer ring edge so
    // T-junction seams are eliminated.
    //
    // The mesh vertices lie in [0,1] local space with spacing 1/patch_resolution.
    // base_patch_size = patch_resolution * world_scale; with world_scale=1,
    // fine_step = 1 / terrain.base_patch_size.
    let fine_step   = 1.0 / terrain.base_patch_size;
    let coarse_step = fine_step * 2.0;

    let local_xz  = v.position.xz;
    let coarse_xz = round(local_xz / coarse_step) * coarse_step;

    // Chebyshev camera distance → square morph band matching ring geometry.
    let half_ring   = patch_size_ws * terrain.ring_patches * 0.5;
    let morph_start = half_ring * terrain.morph_start_ratio;
    let morph_end   = half_ring;
    let cam_delta   = abs(view.world_position.xz - world_xz_orig);
    let cam_dist    = max(cam_delta.x, cam_delta.y);
    let morph_alpha = clamp(
        (cam_dist - morph_start) / max(morph_end - morph_start, 0.001),
        0.0, 1.0,
    );

    let morphed_xz    = mix(local_xz, coarse_xz, morph_alpha);
    let morphed_local = vec3<f32>(morphed_xz.x, 0.0, morphed_xz.y);

    // --- World XZ after morphing. ---
    let world_xz = (model * vec4<f32>(morphed_local, 1.0)).xz;

    // --- Height blended between fine and coarse clipmap levels. ---
    // At alpha=0: fine level only.  At alpha=1: coarse level only.
    // This guarantees the outer edge of LOD L exactly matches the inner edge
    // of LOD L+1, which samples only from its own (= the coarse) level.
    let h   = blended_height(lod_level, coarse_lod, morph_alpha, world_xz);
    let pos = vec3<f32>(world_xz.x, h, world_xz.y);

    // --- Normal via central differences (using the same blended sampling). ---
    let eps = terrain.clip_levels[lod_level].w;   // one texel in world units at this LOD
    let h_r = blended_height(lod_level, coarse_lod, morph_alpha, world_xz + vec2<f32>(eps, 0.0));
    let h_u = blended_height(lod_level, coarse_lod, morph_alpha, world_xz + vec2<f32>(0.0, eps));
    let nrm = normalize(vec3<f32>(h - h_r, eps, h - h_u));

    var out: TerrainVOut;
    out.clip_pos     = position_world_to_clip(pos);
    out.world_pos    = vec4<f32>(pos, 1.0);
    out.world_normal = nrm;
    return out;
}

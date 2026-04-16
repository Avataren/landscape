// terrain_vertex.wgsl
// Phase 5 — per-level clipmap texture array + CDLOD geomorphing.
//
// KEY: to eliminate height discontinuities at LOD ring boundaries, we blend
// height samples between the fine level and the coarser level using morph_alpha.
// When alpha=0 we read the fine level; when alpha=1 we read the coarse level.
// This guarantees that the outer boundary of LOD L and the inner boundary of
// LOD L+1 always agree on height for the same world XZ position.

#import bevy_pbr::{
    view_transformations::position_world_to_clip,
    forward_io::Vertex,
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
    patch_resolution:   f32,   // quads per patch edge
    world_bounds:       vec4<f32>, // (min_x, min_z, max_x, max_z)
    bounds_fade:        vec4<f32>, // x = fade distance, y = use_macro_color, z = flip_v, w = show_wireframe
    clip_levels: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:  texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var normal_tex: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var normal_samp: sampler;

// ---------------------------------------------------------------------------
// Patch storage buffer — one entry per draw instance, indexed by instance_index.
// Matches PatchDescriptorGpu in src/terrain/render/gpu_types.rs exactly.
// ---------------------------------------------------------------------------

struct PatchDescriptor {
    origin_ws:     vec2<f32>,
    patch_size_ws: f32,
    lod_level:     u32,
    morph_start:   f32,
    morph_end:     f32,
    patch_kind:    u32,
    _pad0:         u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(7)
var<storage, read> patch_descriptors: array<PatchDescriptor>;

// ---------------------------------------------------------------------------
// Vertex → fragment interface
// ---------------------------------------------------------------------------

struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       macro_xz_ws:  vec2<f32>,
    @location(3)       patch_uv:     vec2<f32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Sample height from the given LOD level's clipmap layer and fade it out once
/// the world position leaves the baked dataset footprint.
fn height_at(lod: u32, xz: vec2<f32>) -> f32 {
    let lvl = terrain.clip_levels[lod];
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(xz, world_min, world_max);
    // lvl.z = 1 / ring_span,  lvl.w = texel_world_size (scale_L)
    //
    // Shift by +0.5 texels before computing UV so that integer world-space
    // vertex positions land exactly at texel CENTRES rather than at the
    // boundary between two texels.  Without this shift a vertex at integer
    // world coordinate n maps to UV = n / N (exactly between texel n-1 and
    // texel n), and the linear filter straddles the toroidal seam at UV = 0.5
    // — mixing heights from opposite ends of the ring window.  The resulting
    // wrong height leaks into normal computation and produces dark bands that
    // move with the seam as the clip center shifts (shimmering).
    //
    // With the half-texel offset every sample point is at (n + 0.5) / N which
    // sits squarely inside texel n; the seam at UV = 0.5 can only be reached by
    // a non-integer n + 0.5 = N/2, impossible for integer n.
    let uv = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    let sampled_height = textureSampleLevel(height_tex, height_samp, uv, i32(lod), 0.0).r
        * terrain.height_scale;
    return sampled_height * bounds_fade_at(xz);
}

fn normal_at(lod: u32, xz: vec2<f32>) -> vec3<f32> {
    // Derive the world-space normal from height finite-differences.
    // Sampling one texel step in +X and +Z gives the surface tangent vectors;
    // their cross product is the outward normal.
    //   tangent_x = (eps, h_r - h, 0)
    //   tangent_z = (0,   h_u - h, eps)
    //   normal    = tangent_x × tangent_z = (h - h_r, eps, h - h_u)
    let eps = terrain.clip_levels[lod].w; // texel world size at this LOD
    let h   = height_at(lod, xz);
    let h_r = height_at(lod, xz + vec2<f32>(eps, 0.0));
    let h_u = height_at(lod, xz + vec2<f32>(0.0, eps));
    return normalize(vec3<f32>(h - h_r, eps, h - h_u));
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
    // --- Read patch descriptor from storage buffer. ---
    let pd = patch_descriptors[v.instance_index];
    let patch_size_ws = pd.patch_size_ws;
    let lod_level     = pd.lod_level;

    // Coarse LOD index for height blending: clamped so we never read layer N.
    let max_lod_idx = u32(terrain.num_lod_levels) - 1u;
    let coarse_lod  = min(lod_level + 1u, max_lod_idx);

    // --- World XZ before morphing. ---
    // Mesh vertices are in [0,1] local XZ; scale by patch_size_ws and offset
    // by origin to get world space.
    let world_xz_orig = pd.origin_ws + v.position.xz * patch_size_ws;

    // --- Geomorphing ---
    // Blend vertices toward the 2× coarser grid near the outer ring edge so
    // T-junction seams are eliminated.
    //
    // IMPORTANT: we use the vertex's distance from the RING CENTER (not from
    // the camera) to compute morph_alpha.  The ring center comes from
    // clip_levels, which is always aligned to the camera grid.  This guarantees
    // that every vertex on the outer ring boundary gets alpha = 1.0 regardless
    // of the camera's offset within its grid cell.
    //
    // Using camera distance instead would leave outer-boundary vertices with
    // alpha < 1 when the camera is offset from the clip center (up to
    // 2^(clipmap_levels-1) world units), creating residual height cracks.
    let lvl_fine    = terrain.clip_levels[lod_level];
    // half_ring_ws = ring_span / 2 = 0.5 / inv_ring_span
    let half_ring_ws = 0.5 / lvl_fine.z;
    // clip_levels[L].xy = ring center in world space (clip_center * texel_ws).
    let ring_center  = lvl_fine.xy;

    let vertex_delta     = abs(world_xz_orig - ring_center);
    let dist_from_center = max(vertex_delta.x, vertex_delta.y);  // Chebyshev

    let morph_start_ws = half_ring_ws * terrain.morph_start_ratio;
    let boundary_t = clamp(
        (dist_from_center - morph_start_ws) / max(half_ring_ws - morph_start_ws, 0.001),
        0.0, 1.0,
    );
    // Keep morphing strictly boundary-driven so each LOD uses its own data
    // through the interior of the ring. Camera-distance morphing can force
    // far rings to sample much coarser levels too early, which shows distant
    // "ghost" silhouettes when intermediate levels are still flat/unloaded.
    let boundary_alpha = boundary_t * boundary_t * (3.0 - 2.0 * boundary_t);

    // Boundary lock: force alpha = 1.0 for vertices that are on the outermost
    // edge (within half a fine texel). This guarantees exact coarse-grid snap
    // on the stitch edge and prevents occasional sub-texel cracks.
    let boundary_dist_ws = half_ring_ws - dist_from_center;
    let boundary_lock = select(
        0.0,
        1.0,
        boundary_dist_ws <= (0.5 * terrain.clip_levels[lod_level].w + 1e-4),
    );

    let morph_alpha = max(boundary_alpha, boundary_lock);

    // Snap in world space to the globally anchored 2x coarser grid.
    let fine_step_ws   = terrain.clip_levels[lod_level].w;
    let coarse_step_ws = fine_step_ws * 2.0;
    // Snap to the same globally anchored coarse lattice that the next LOD ring
    // uses. `round()` can pick the higher coarse cell for odd fine-grid edges,
    // which opens a 1-fine-texel crack until the coarser ring catches up.
    let coarse_world_xz = floor(world_xz_orig / coarse_step_ws) * coarse_step_ws;

    // --- World XZ after morphing. ---
    let world_xz = mix(world_xz_orig, coarse_world_xz, morph_alpha);

    // --- Height blended between fine and coarse clipmap levels. ---
    // At alpha=0: fine level only.  At alpha=1: coarse level only.
    // This guarantees the outer edge of LOD L exactly matches the inner edge
    // of LOD L+1, which samples only from its own (= the coarse) level.
    let h   = blended_height(lod_level, coarse_lod, morph_alpha, world_xz);
    let pos = vec3<f32>(world_xz.x, h, world_xz.y);

    let n_fine = normal_at(lod_level, world_xz);
    let n_coarse = normal_at(coarse_lod, world_xz);
    let nrm = normalize(mix(n_fine, n_coarse, morph_alpha));

    var out: TerrainVOut;
    out.clip_pos     = position_world_to_clip(pos);
    out.world_pos    = vec4<f32>(pos, 1.0);
    out.world_normal = nrm;
    out.macro_xz_ws  = world_xz_orig;
    out.patch_uv     = v.uv;
    return out;
}

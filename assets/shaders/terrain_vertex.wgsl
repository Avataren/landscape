// terrain_vertex.wgsl
// Geomorphing CDLOD vertex shader.
// Blends fine-LOD vertices toward the next-coarser grid at ring boundaries,
// eliminating T-junction seams between LOD levels.

#import bevy_pbr::{
    mesh_functions::get_world_from_local,
    view_transformations::position_world_to_clip,
    forward_io::Vertex,
    mesh_view_bindings::view,
}

// ---------------------------------------------------------------------------
// Material bindings — must match TerrainMaterialUniforms in material.rs
// ---------------------------------------------------------------------------
struct TerrainParams {
    height_scale:      f32,   // world-space Y for a texel value of 1.0
    world_size:        f32,   // XZ extent (in world units) the texture covers
    world_offset_x:    f32,   // world X of texture left edge
    world_offset_z:    f32,   // world Z of texture bottom edge
    ring_patches:      f32,   // patches per ring edge (from TerrainConfig)
    morph_start_ratio: f32,   // fraction of ring at which morphing begins
    patch_resolution:  f32,   // vertices per patch edge
    pad0:              f32,   // alignment padding
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> terrain: TerrainParams;

// ---------------------------------------------------------------------------
// Vertex → fragment interface (custom struct avoids VertexOutput conditionals)
// ---------------------------------------------------------------------------
struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn world_xz_to_uv(xz: vec2<f32>) -> vec2<f32> {
    return clamp(
        (xz - vec2<f32>(terrain.world_offset_x, terrain.world_offset_z)) / terrain.world_size,
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
}

fn height_at(xz: vec2<f32>) -> f32 {
    return textureSampleLevel(height_tex, height_samp, world_xz_to_uv(xz), 0.0).r
           * terrain.height_scale;
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
@vertex
fn vertex(v: Vertex) -> TerrainVOut {
    let model = get_world_from_local(v.instance_index);

    // 1. Compute world XZ before any morphing (used for camera-distance test).
    //    The entity Transform encodes:
    //      translation = (patch_origin_ws.x, 0, patch_origin_ws.z)
    //      scale       = (patch_size_ws, 1, patch_size_ws)
    let world_xz_orig = (model * vec4<f32>(v.position, 1.0)).xz;

    // 2. Geomorphing: blend fine vertices toward the 2× coarser grid when
    //    the vertex is near the outer edge of its LOD ring, so the boundary
    //    vertices align exactly with the next coarser level's vertices.
    //
    //    patch_size_ws is the X scale of the model matrix.
    let patch_size_ws = length(model[0].xyz);

    // Local-space step sizes (positions run [0,1] with patch_resolution steps).
    let step       = 1.0 / terrain.patch_resolution;
    let coarse_step = step * 2.0;

    // Snap local XZ to the next-coarser grid.
    let local_xz = v.position.xz;
    let coarse_xz = round(local_xz / coarse_step) * coarse_step;

    // Morph factor: 0 at morph_start distance, 1 at ring edge.
    // Use Chebyshev distance (max of |Δx|, |Δz|) instead of Euclidean so the
    // morph band forms a perfect square frame matching the ring geometry.
    // Euclidean distance is larger at corners (×√2), which would cause the
    // coarser ring's inner corner vertices to start morphing prematurely,
    // creating a visible seam at the ring corners.
    let half_ring    = patch_size_ws * terrain.ring_patches * 0.5;
    let morph_start  = half_ring * terrain.morph_start_ratio;
    let morph_end    = half_ring;
    let cam_delta    = abs(view.world_position.xz - world_xz_orig);
    let cam_dist     = max(cam_delta.x, cam_delta.y); // Chebyshev distance
    let morph_alpha  = clamp(
        (cam_dist - morph_start) / max(morph_end - morph_start, 0.001),
        0.0, 1.0,
    );

    // Blend local position toward the coarser grid.
    let morphed_local_xz = mix(local_xz, coarse_xz, morph_alpha);
    let morphed_local    = vec3<f32>(morphed_local_xz.x, 0.0, morphed_local_xz.y);

    // 3. Transform morphed local position to world XZ.
    let world_xz = (model * vec4<f32>(morphed_local, 1.0)).xz;

    // 4. Sample height and displace Y.
    let h   = height_at(world_xz);
    let pos = vec3<f32>(world_xz.x, h, world_xz.y);

    // 5. Reconstruct normal by central differences (world-space, no transform needed).
    let eps = terrain.world_size / 256.0;
    let h_r = height_at(world_xz + vec2<f32>(eps, 0.0));
    let h_u = height_at(world_xz + vec2<f32>(0.0, eps));
    let nrm = normalize(vec3<f32>(h - h_r, eps, h - h_u));

    var out: TerrainVOut;
    out.clip_pos     = position_world_to_clip(pos);
    out.world_pos    = vec4<f32>(pos, 1.0);
    out.world_normal = nrm;
    return out;
}

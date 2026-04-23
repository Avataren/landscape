#import bevy_pbr::{
    mesh_functions,
    skinning,
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::mesh_view_bindings::view

#import bevy_landscape_water::water_functions as water_fn
#import bevy_landscape_water::water_bindings

#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#else
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#endif

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

#ifdef SKINNED
    var model = skinning::skin_model(vertex.joint_indices, vertex.joint_weights);
#else
    var model = mesh_functions::get_world_from_local(vertex.instance_index);
#endif

#ifdef VERTEX_UVS
#ifdef SKINNED
    out.world_normal = skinning::skin_normals(model, vertex.normal);
#else
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index
    );
#endif
#endif

    let world_position = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(vertex.position, 1.0)
    );

    // Vertex Y displacement from Gerstner waves.
    // Only worthwhile on close geometry; beyond 500 m the wave-height
    // variation is sub-pixel and skipping it saves vertex-shader work.
    var height = 0.0;
#if QUALITY > 2
    let w_pos    = world_position.xz;
    let cam_dist = distance(w_pos, vec2<f32>(view.world_position.x, view.world_position.z));
    if cam_dist < 500.0 {
        // Pass pixel_size=0 — no LOD filtering in the vertex shader.
        height = water_fn::get_wave_height(w_pos);
    }
#endif

    out.world_position = world_position + vec4<f32>(out.world_normal * height, 0.0);
    out.position       = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        model,
        vertex.tangent,
        vertex.instance_index
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    return out;
}

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

    // Full Gerstner displacement (GPU Gems §1 eq. 4):
    //   P' = (x + Σ Q_eff·A·D_x·cos, y + Σ A·sin, z + Σ Q_eff·A·D_z·cos)
    //
    // Approximate vertex spacing from camera distance so short-wavelength
    // waves are LOD-filtered on coarse rings just as they are in the fragment
    // shader.  vertex_size ≈ 4m (fine ring) growing to 64m at the horizon.
    let orig_world_xz = world_position.xz;
    let cam_dist      = distance(orig_world_xz, vec2<f32>(view.world_position.x, view.world_position.z));
    let vertex_size   = clamp(cam_dist / 128.0, 4.0, 64.0);
    let wave = water_fn::get_wave_result(orig_world_xz, vertex_size);

    out.world_position = world_position + vec4<f32>(wave.xz_disp.x, wave.height, wave.xz_disp.y, 0.0);
    out.position       = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_UVS
    // Store the pre-displacement world XZ so the fragment shader can evaluate
    // Gerstner normals at the correct (undisplaced) position.
    out.uv = orig_world_xz;
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

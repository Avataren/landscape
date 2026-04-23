#import bevy_pbr::{
    pbr_functions::alpha_discard,
    pbr_fragment::pbr_input_from_standard_material,
    mesh_view_bindings::view,
}

#ifdef DEPTH_PREPASS
#import bevy_pbr::{
    prepass_utils,
    view_transformations::{frag_coord_to_ndc, position_ndc_to_world},
}
#endif

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}
#endif

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output
#endif

#import bevy_landscape_water::water_bindings
#import bevy_landscape_water::water_functions as water_fn

@fragment
fn fragment(
#ifdef MESHLET_MESH_MATERIAL_PASS
    @builtin(position) frag_coord: vec4<f32>,
#else
    p_in: VertexOutput,
    @builtin(front_facing) is_front: bool,
#endif
) -> FragmentOutput {
#ifdef MESHLET_MESH_MATERIAL_PASS
    let p_in = resolve_vertex_output(frag_coord);
    let is_front = true;
#endif

    var in = p_in;

    // -----------------------------------------------------------------------
    // Depth prepass — discard water hidden behind terrain; derive water depth.
    // -----------------------------------------------------------------------
    var terrain_depth_m = 1.0; // fallback: 1 m (avoids shoreline foam without prepass)
#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
    let raw_depth = prepass_utils::prepass_depth(in.position, 0u);
    // Reversed-Z: 1 = near, 0 = far. raw_depth == 0 means no opaque geometry.
    if raw_depth > 0.0 {
        let terrain_ndc     = frag_coord_to_ndc(vec4(in.position.xy, raw_depth, 1.0));
        let terrain_world_y = position_ndc_to_world(terrain_ndc).y;
        if terrain_world_y > in.world_position.y + 0.05 {
            discard;
        }
        terrain_depth_m = max(in.world_position.y - terrain_world_y, 0.0);
    }
#endif
#endif

    // -----------------------------------------------------------------------
    // Gerstner normal — LOD-filtered by screen-space pixel footprint.
    //
    // dpdx / dpdy give the world-space displacement per screen pixel in X and Y.
    // The larger component sets the effective pixel footprint, which drives the
    // per-wave attenuation in water_functions::wave_lod_weight().  This removes
    // high-frequency wave normals that would alias as noise at distance.
    // -----------------------------------------------------------------------
    let w_pos = in.world_position.xz;
    let dpx   = dpdx(in.world_position.xyz);
    let dpz   = dpdy(in.world_position.xyz);
    let pixel_size = max(length(dpx.xz), length(dpz.xz));

    let wave = water_fn::get_wave_result(w_pos, pixel_size);
    in.world_normal = wave.normal;

#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let deep_color            = water_bindings::material.deep_color;
    let shallow_color         = water_bindings::material.shallow_color;
    let water_clarity         = water_bindings::material.clarity;
    let foam_threshold        = water_bindings::material.foam_threshold;
    let foam_color            = water_bindings::material.foam_color;
    let shoreline_foam_depth  = water_bindings::material.shoreline_foam_depth;

    // -----------------------------------------------------------------------
    // Beer's law colour absorption through the water column.
    // -----------------------------------------------------------------------
#ifdef DEPTH_PREPASS
    let beer_depth = terrain_depth_m + 0.5;
#else
    let view_to_frag = normalize(in.world_position.xyz - view.world_position.xyz);
    let vert_view    = saturate(-view_to_frag.y);
    let beer_depth   = vert_view * 12.0 + 1.0;
#endif
    let absorb       = vec3<f32>(water_clarity * 3.0, water_clarity * 1.5, water_clarity * 0.5);
    let beer         = exp(-beer_depth * absorb);

    let absorbed_rgb = mix(deep_color.rgb, shallow_color.rgb, beer);
    var water_color  = vec4<f32>(absorbed_rgb, deep_color.a);

    // -----------------------------------------------------------------------
    // Foam — wave crest + shoreline.
    //
    // foam_threshold is normalised to [0..1] where 1.0 = top of the wave
    // range and 0.0 = always foam.  The effective max wave height is
    // approximated as amplitude × 2.0  (≈ AMP_RATIO × amplitude × Σλ).
    //
    // Shoreline foam: shallow areas where waves break at the shore.
    //   Only available when the depth prepass provides terrain_depth_m.
    // -----------------------------------------------------------------------
    let max_wave_h = water_bindings::material.amplitude * 2.0;
    let norm_h     = saturate(wave.height / max(max_wave_h, 0.001));
    // Transition width = (1 - foam_threshold) / 2 so it scales with the threshold.
    let transition  = max((1.0 - foam_threshold) * 0.5, 0.02);
    let crest_foam  = smoothstep(foam_threshold - transition, foam_threshold + transition, norm_h);
    let crest_w     = crest_foam * crest_foam;

#ifdef DEPTH_PREPASS
    // Fade from full foam at depth=0 to no foam at depth=shoreline_foam_depth.
    let shore_t = 1.0 - saturate(terrain_depth_m / max(shoreline_foam_depth, 0.01));
    let shore_w = shore_t * shore_t;   // squared: sharper shoreline edge
#else
    let shore_w = 0.0;
#endif

    let foam_weight = max(crest_w, shore_w);
    water_color     = mix(water_color, foam_color, foam_weight);

    pbr_input.material.base_color *= water_color;
    pbr_input.material.base_color  = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}

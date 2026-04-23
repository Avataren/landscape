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
    // Use the derivative of the pre-displacement world XZ (stored in uv by the
    // vertex shader) for pixel footprint.  Using dpdx(world_position) instead
    // would pick up the lateral Gerstner XZ displacement which is discontinuous
    // at mesh triangle edges, creating a visible grid of foam lines.
    let w_pos = in.uv;
    let pixel_size = clamp(max(length(dpdx(in.uv)), length(dpdy(in.uv))), 0.35, 12.0);

    let wave         = water_fn::get_wave_result(w_pos, pixel_size);
    let detail_wave  = water_fn::get_detail_wave_result(w_pos, pixel_size);
    let water_normal = water_fn::combine_surface_normal(wave.normal, detail_wave.slope);
    in.world_normal  = water_normal;

#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let deep_color            = water_bindings::material.deep_color;
    let shallow_color         = water_bindings::material.shallow_color;
    let edge_color            = water_bindings::material.edge_color;
    let edge_scale            = max(water_bindings::material.edge_scale, 0.001);
    let water_clarity         = water_bindings::material.clarity;
    let refraction_strength   = water_bindings::material.refraction_strength;
    let foam_threshold        = water_bindings::material.foam_threshold;
    let foam_color            = water_bindings::material.foam_color;
    let shoreline_foam_depth  = water_bindings::material.shoreline_foam_depth;

    // -----------------------------------------------------------------------
    // Water absorption through the actual view path, not just vertical depth.
    // -----------------------------------------------------------------------
    let view_to_camera = normalize(view.world_position.xyz - in.world_position.xyz);
#ifdef DEPTH_PREPASS
    let path_cos   = max(dot(view_to_camera, vec3(0.0, 1.0, 0.0)), 0.08);
    let beer_depth = terrain_depth_m / path_cos + 0.35;
#else
    let vert_view  = clamp(1.0 - view_to_camera.y, 0.0, 1.0);
    let beer_depth = vert_view * 14.0 + 1.5;
#endif
    let clarity      = clamp(water_clarity, 0.0, 1.0);
    let clarity_sq   = clarity * clarity;
    let absorb       = mix(
        vec3<f32>(0.18, 0.082, 0.038),
        vec3<f32>(0.028, 0.013, 0.006),
        clarity_sq
    );
    let beer         = exp(-beer_depth * absorb);
    let absorbed_rgb = mix(deep_color.rgb, shallow_color.rgb, beer);

    // -----------------------------------------------------------------------
    // Foam — wave crest + shoreline.
    //
    // foam_threshold is normalised to [0..1] where 1.0 = top of the wave
    // range.  max_wave_h is set to amplitude × 1.5 so the typical tallest
    // geometry wave (Σ AMP_RATIO × λ ≈ 1.44 × amplitude) maps to norm_h ≈ 1.
    //
    // We use only geometry wave height for crest foam.  The detail-wave
    // "breaking" term based on detail_wave.crest / slope_energy is omitted:
    // detail_wave.crest is a max over 8 waves and is near-1 at ~72 % of all
    // near-camera pixels, making the breaking foam fire everywhere → solid
    // white foam.  Crest foam from actual wave height is predictable and
    // physically correct.
    //
    // Shoreline foam: shallow areas where waves break at the shore.
    //   Only available when the depth prepass provides terrain_depth_m.
    // -----------------------------------------------------------------------
    let max_wave_h    = water_bindings::material.amplitude * 1.5;
    let norm_h        = clamp(wave.height / max(max_wave_h, 0.001), 0.0, 1.0);
    let transition    = max((1.0 - foam_threshold) * 0.5, 0.02);
    let crest_foam    = smoothstep(foam_threshold - transition, foam_threshold + transition, norm_h);
    let crest_w       = crest_foam * crest_foam;
    let foam_far_fade = 1.0 - smoothstep(3.5, 10.0, pixel_size);

#ifdef DEPTH_PREPASS
    // Fade from full foam at depth=0 to no foam at depth=shoreline_foam_depth.
    let shore_t = 1.0 - clamp(terrain_depth_m / max(shoreline_foam_depth, 0.01), 0.0, 1.0);
    let shore_w = shore_t * shore_t;   // squared: sharper shoreline edge
#else
    let shore_w = 0.0;
#endif

    let foam_weight = clamp(max(crest_w * foam_far_fade, shore_w), 0.0, 1.0);

    // Thin-water edge scattering brightens the shoreline without making deep
    // water milky. This also gives the eye a cleaner transition at the coast.
#ifdef DEPTH_PREPASS
    let shoreline_edge = exp(-terrain_depth_m * edge_scale * 6.0);
#else
    let shoreline_edge = 0.0;
#endif
    let grazing        = pow(1.0 - clamp(dot(water_normal, view_to_camera), 0.0, 1.0), 5.0);
    let edge_scatter   = shoreline_edge * (0.2 + 0.8 * grazing);
    let water_rgb      = mix(absorbed_rgb, edge_color.rgb, edge_scatter * 0.32);
    var water_color    = vec4<f32>(mix(water_rgb, foam_color.rgb, foam_weight), 1.0);

    pbr_input.material.base_color *= water_color;
    pbr_input.material.base_color  = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    pbr_input.material.base_color.a = 1.0;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.reflectance = vec3<f32>(0.25);
    pbr_input.material.ior = 1.333;
    pbr_input.material.perceptual_roughness = clamp(
        0.028 +
        (1.0 - water_normal.y) * 0.24 +
        detail_wave.slope_energy * 0.18 +
        smoothstep(2.0, 12.0, pixel_size) * 0.06 +
        foam_weight * 0.22,
        0.025,
        0.42
    );
    pbr_input.material.specular_transmission = clamp(0.96 * clarity_sq * (1.0 - foam_weight * 0.85), 0.0, 0.98);
    pbr_input.material.diffuse_transmission = 0.07 * clarity * (1.0 - foam_weight) * (1.0 - grazing * 0.5);
    pbr_input.material.thickness = clamp(
        beer_depth * mix(0.22, 1.35, clamp(refraction_strength / 24.0, 0.0, 1.0)),
        0.25,
        48.0
    );
    pbr_input.material.attenuation_distance = mix(8.0, 120.0, clarity_sq);
    pbr_input.material.attenuation_color = vec4<f32>(
        mix(deep_color.rgb, shallow_color.rgb, 0.55 + shoreline_edge * 0.2),
        1.0
    );

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

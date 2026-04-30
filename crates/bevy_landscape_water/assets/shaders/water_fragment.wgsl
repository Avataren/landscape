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
    var prepass_surface_depth_m = 1.0; // fallback optical thickness without prepass
    var prepass_shore_depth_m = 1e6;
    var has_prepass_terrain = false;
    let rest_surface_y = water_fn::water_surface_height();
#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
    let raw_depth = prepass_utils::prepass_depth(in.position, 0u);
    // Reversed-Z: 1 = near, 0 = far. raw_depth == 0 means no opaque geometry.
    if raw_depth > 0.0 {
        let terrain_ndc     = frag_coord_to_ndc(vec4(in.position.xy, raw_depth, 1.0));
        let terrain_world_y = position_ndc_to_world(terrain_ndc).y;
        let terrain_above_water = terrain_world_y > rest_surface_y + 0.05;
        let terrain_closer_than_water = raw_depth > in.position.z + 1e-5;

        // Only discard when dry land is actually in front of this water
        // fragment. Distant coastline visible behind a crest must not erase it.
        if terrain_above_water && terrain_closer_than_water {
            discard;
        }

        // Above-water terrain behind the wave should not drive shoreline foam
        // or optical depth for the current pixel; fall back to local bathymetry.
        if !terrain_above_water {
            has_prepass_terrain = true;
            prepass_shore_depth_m = max(rest_surface_y - terrain_world_y, 0.0);
            prepass_surface_depth_m = max(in.world_position.y - terrain_world_y, 0.0);
        }
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
    let terrain_depth_valid = water_fn::terrain_in_world_bounds(w_pos);
    let terrain_height_y = water_fn::terrain_height_at(w_pos);
    let clipmap_shore_depth = max(rest_surface_y - terrain_height_y, 0.0);
    let clipmap_surface_depth = max(in.world_position.y - terrain_height_y, 0.0);
    let shore_wave_attn = water_fn::shoreline_wave_attenuation(w_pos);
    let shore_depth_m = select(
        select(1e6, clipmap_shore_depth, terrain_depth_valid),
        prepass_shore_depth_m,
        has_prepass_terrain,
    );
    let water_depth_m = select(
        select(1.0, clipmap_surface_depth, terrain_depth_valid),
        prepass_surface_depth_m,
        has_prepass_terrain,
    );
    let pixel_size = clamp(max(length(dpdx(in.uv)), length(dpdy(in.uv))), 0.35, 12.0);
    let normal_footprint = water_fn::filtered_normal_footprint(pixel_size);
    let micro_detail_fade = water_fn::micro_detail_fade(pixel_size);
    let far_field_filter = 1.0 - micro_detail_fade;

    let wave         = water_fn::get_wave_result(w_pos, normal_footprint);
    let detail_wave  = water_fn::get_detail_wave_result(w_pos, normal_footprint);
    // Swell normals: 3 cross-swell waves (47 m, 131 m, 173 m) computed only in
    // the fragment shader.  These break the periodic repetition visible at
    // distance when only the 3 longest geometry waves remain after LOD filter.
    let swell_n      = water_fn::get_swell_normal(w_pos, normal_footprint);

    // Tessendorf FFT contributions — sampled at the same world XZ used by
    // the vertex shader for displacement, so geometry and shading agree.
    let fft         = water_fn::sample_fft_displacement(w_pos);
    let fft_slope   = water_fn::fft_height_slope(w_pos);
    let fft_s       = water_fn::fft_strength();
    let gerstner_w  = 1.0 - fft_s;
    // Blend the Gerstner surface normal with the FFT one (the FFT slope is
    // converted to a normal via slope_to_normal).  When fft_strength = 0
    // this collapses back to pure Gerstner; when 1 the FFT carries the
    // surface shape entirely.
    let gerst_normal = wave.normal;
    let fft_normal   = water_fn::slope_to_normal(fft_slope);
    let blended_norm = normalize(gerst_normal * gerstner_w + fft_normal * fft_s);
    let macro_normal = normalize(mix(
        vec3<f32>(0.0, 1.0, 0.0),
        blended_norm,
        shore_wave_attn * mix(1.0, 0.72, far_field_filter),
    ));
    // Detail and swell slopes are analytic Gerstner sums (8 + 3 directional
    // cosines).  When FFT carries the surface, summing them on top prints a
    // visible diamond cross-hatch interference pattern, so gate them by the
    // Gerstner weight — at fft_strength = 1.0 they vanish entirely.
    let detail_slope = detail_wave.slope * shore_wave_attn * micro_detail_fade * gerstner_w;
    let detail_slope_energy = detail_wave.slope_energy * shore_wave_attn * micro_detail_fade * gerstner_w;
    let swell_slope  = water_fn::normal_to_slope(swell_n) * shore_wave_attn * mix(1.0, 0.78, far_field_filter) * gerstner_w;
    // Capillary noise: stochastic high-frequency normal perturbation that
    // breaks the periodic look of pure analytic waves.  Faded by the same
    // micro_detail_fade so it disappears on coarse rings.
    let capillary = water_fn::capillary_slope(w_pos, normal_footprint)
        * shore_wave_attn
        * micro_detail_fade
        * water_fn::water_capillary_strength();
    // Macro slope — distance-immune.  Adds slow stochastic tilt to the
    // surface normal even at the horizon, so distant water doesn't read as
    // flat or as a tiled Gerstner pattern.
    let macro_grad   = water_fn::macro_noise_height_grad(w_pos, normal_footprint);
    let macro_slope  = vec2<f32>(macro_grad.y, macro_grad.z) * shore_wave_attn;
    let water_normal = water_fn::combine_surface_normal(
        macro_normal,
        detail_slope + swell_slope + capillary + macro_slope,
    );
    in.world_normal  = water_normal;

#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // The SSAO texture at these pixels contains terrain occlusion, not water
    // surface occlusion. Water is a flat transmissive surface with no self-AO.
    pbr_input.diffuse_occlusion = vec3<f32>(1.0);
    pbr_input.specular_occlusion = 1.0;

    let deep_color            = water_bindings::material.deep_color;
    let shallow_color         = water_bindings::material.shallow_color;
    let edge_color            = water_bindings::material.edge_color;
    let edge_scale            = max(water_fn::water_edge_scale(), 0.001);
    let water_clarity         = water_fn::water_clarity();
    let refraction_strength   = water_fn::water_refraction_strength();
    let foam_threshold        = water_fn::water_foam_threshold();
    let foam_color            = water_bindings::material.foam_color;
    let shoreline_foam_depth  = water_fn::shoreline_foam_depth();

    // -----------------------------------------------------------------------
    // Fresnel — Schlick approximation for water (IOR = 1.333, F0 ≈ 0.02).
    //
    // At normal incidence water is ~2% reflective (mostly shows absorbed depth
    // colour). At grazing angles it approaches 100% reflective (mirror).  This
    // is the dominant visual cue that distinguishes water from flat paint.
    //
    //   fresnel = F0 + (1 - F0) × (1 - cosθ)^5,   F0 = 0.02
    //
    // Uses: (a) roughness — lower roughness at grazing → sharper SSR
    //       (b) base_color — attenuate at grazing so SSR reflections read
    //           clearly without fighting the absorption colour
    // -----------------------------------------------------------------------
    let view_to_camera = normalize(view.world_position.xyz - in.world_position.xyz);
    let cos_view       = clamp(dot(view_to_camera, water_normal), 0.0, 1.0);
    let fresnel        = 0.02 + 0.98 * pow(1.0 - cos_view, 5.0);

    // -----------------------------------------------------------------------
    // Water absorption through the actual view path, not just vertical depth.
    //
    // In the deferred GBuffer pass (PREPASS_PIPELINE is defined), terrain
    // depth from the prepass is unreliable — water writes its own depth there.
    // Fall back to view-angle-based depth approximation in that case.
    // -----------------------------------------------------------------------
#ifdef DEPTH_PREPASS
    let path_cos   = max(dot(view_to_camera, vec3(0.0, 1.0, 0.0)), 0.08);
    let beer_depth = water_depth_m / path_cos + 0.35;
#else
    let vert_view  = clamp(1.0 - view_to_camera.y, 0.0, 1.0);
    let beer_depth = vert_view * 14.0 + 1.5;
#endif
    let clarity      = clamp(water_clarity, 0.0, 1.0);
    let clarity_sq   = clarity * clarity;
    let absorb       = mix(
        vec3<f32>(0.11, 0.050, 0.022),
        vec3<f32>(0.016, 0.007, 0.003),
        clarity_sq
    );
    let beer         = exp(-beer_depth * absorb * 0.72);
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
    let max_wave_h    = water_fn::water_amplitude() * 1.5;
    let blended_h     = wave.height * gerstner_w + fft.x * fft_s;
    let norm_h        = clamp((blended_h * shore_wave_attn) / max(max_wave_h, 0.001), 0.0, 1.0);
    let transition    = max((1.0 - foam_threshold) * 0.5, 0.02);
    let crest_foam    = smoothstep(foam_threshold - transition, foam_threshold + transition, norm_h);
    let crest_raw     = crest_foam * crest_foam;
    let foam_far_fade = 1.0 - smoothstep(3.5, 10.0, pixel_size);

    // -----------------------------------------------------------------------
    // Voronoi foam pattern.
    //
    // Modulates the crest / shoreline foam masks with an organic bubble &
    // edge texture so foam reads as discrete patches rather than smooth
    // discs / continuous strips along the waterline.  Two world-space scales
    // (tighter for crests, broader for shore) and independent scroll drift.
    // -----------------------------------------------------------------------
    let crest_pattern  = water_fn::foam_voronoi_mask(
        w_pos,
        1.5,    // 1.5 m cell — bubble cluster scale
        0.35,   // crest foam drifts with the wave train
        clamp(crest_raw, 0.0, 1.0),
    );
    let crest_w        = crest_raw * mix(0.25, 1.05, crest_pattern);

    // -----------------------------------------------------------------------
    // Jacobian (foldover) foam — wave-breaking detection.
    //
    // For Gerstner waves with horizontal displacement, the surface "folds"
    // onto itself where the Jacobian determinant of the displacement field
    // J = (1+gxx)·(1+gzz) - gxz²  drops below ~0; large Q values amplify
    // foldover.  The result is streaky organic foam at the lee side of crests
    // — what real ocean foam actually does — rather than smooth disc-on-crest.
    // -----------------------------------------------------------------------
    let disp_grad = water_fn::get_wave_disp_grad(w_pos, normal_footprint) * shore_wave_attn;
    let gerst_jac = (1.0 + disp_grad.x) * (1.0 + disp_grad.z) - disp_grad.y * disp_grad.y;
    // The FFT pipeline writes the Jacobian determinant of its choppy
    // displacement field directly into the alpha channel — much more accurate
    // than the per-wave analytic version since it includes phase-coherent
    // foldovers from the entire spectrum.
    let fft_jac     = mix(1.0, fft.w, fft_s) * shore_wave_attn + (1.0 - shore_wave_attn);
    let jacobian    = gerst_jac * gerstner_w + fft_jac * fft_s;
    // Foam appears as J drops below 1; classic ocean shaders trigger near 0.
    // We band-pass between 0.55 (faint streaks) and -0.05 (full foldover).
    let fold_foam = smoothstep(0.55, -0.05, jacobian);
    let fold_w    = fold_foam * fold_foam * foam_far_fade
        * water_fn::water_jacobian_foam_strength();

    // Fade from full foam at depth=0 to no foam at depth=shoreline_foam_depth.
    let shore_t      = 1.0 - clamp(shore_depth_m / max(shoreline_foam_depth, 0.01), 0.0, 1.0);
    let shore_raw    = shore_t * shore_t;
    // Voronoi pattern at a coarser cell size so shoreline foam reads as
    // overlapping bubble masses rather than a uniform white strip.  The
    // pattern drifts slowly inland (scroll along wind), giving life to
    // otherwise-static contact lines.
    let shore_pattern = water_fn::foam_voronoi_mask(
        w_pos,
        2.4,    // larger cells → broader foam patches at shore
        0.15,   // slow drift
        clamp(shore_raw, 0.0, 1.0),
    );
    // Mix the patterned mask back with the raw factor so the very-shallowest
    // pixels stay solid white (avoids visible noise right at the contact).
    let shore_w = mix(shore_raw * shore_pattern, shore_raw, pow(shore_t, 4.0));

    let foam_weight = clamp(max(max(crest_w * foam_far_fade, fold_w), shore_w), 0.0, 1.0);

    // Thin-water edge scattering brightens the shoreline without making deep
    // water milky. This also gives the eye a cleaner transition at the coast.
    let shoreline_edge = exp(-shore_depth_m * edge_scale * 6.0);
    let edge_scatter   = shoreline_edge * (0.2 + 0.8 * (1.0 - cos_view));
    let water_rgb      = mix(absorbed_rgb, edge_color.rgb, edge_scatter * 0.32);

    // Fresnel: at grazing angles the absorbed depth colour yields to specular
    // sky reflections (via PBR IBL).  Attenuate modestly so the water colour
    // reads at all view angles, not just steep/overhead.
    let water_rgb_fresnel = water_rgb * (1.0 - fresnel * 0.45);
    var water_color = vec4<f32>(mix(water_rgb_fresnel, foam_color.rgb, foam_weight), 1.0);

    pbr_input.material.base_color *= water_color;
    pbr_input.material.base_color  = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    pbr_input.material.base_color.a = 1.0;
    pbr_input.material.metallic = 0.0;
    // Correct F0 for water (IOR 1.333): reflectance = sqrt(0.02/0.16) ≈ 0.354.
    pbr_input.material.reflectance = vec3<f32>(0.35);
    pbr_input.material.ior = 1.333;
    // Roughness: rises with wave slope; Fresnel drives it toward zero at
    // grazing (mirror-flat horizon = classic Fresnel water look).
    let specular_aa = smoothstep(0.75, 4.5, pixel_size) * (0.10 + 0.14 * fresnel)
        + far_field_filter * 0.05;
    pbr_input.material.perceptual_roughness = clamp(
        mix(
            0.025 +
            (1.0 - water_normal.y) * 0.20 +
            detail_slope_energy * 0.14 +
            smoothstep(2.0, 12.0, pixel_size) * 0.06 +
            foam_weight * 0.22,
            0.01,           // mirror-flat at grazing angles
            fresnel
        ) + specular_aa,
        0.01,
        0.50
    );
    // Fresnel drives specular transmission: at grazing angles water is a
    // mirror (fresnel→1 → transmission→0); looking straight down it's
    // mostly transparent (fresnel≈0.02 → full Beer's-law transmission).
    pbr_input.material.specular_transmission = clamp(
        (1.0 - fresnel) * 0.96 * clarity_sq * (1.0 - foam_weight * 0.85),
        0.0, 0.98
    );
    pbr_input.material.diffuse_transmission = 0.07 * clarity * (1.0 - foam_weight) * fresnel * 0.1;
    // Distance-calibrated thickness keeps the refracted UV offset at roughly
    // `refraction_strength` pixels regardless of how close the water is to the
    // camera.  The PBR SST offset scales as thickness / view_distance, so:
    //   target_thick = refraction_strength × view_dist / focal_px
    // For IOR 1.333 and a 60° FOV at 1080 p, the bend factor × focal ≈ 600.
    //
    // With a fixed large thickness (old approach), close-up water could produce
    // pixel offsets of hundreds of pixels, shooting the sample UV far off screen
    // and producing clamped-edge repeating stripes.  Setting thickness = 0 when
    // refraction_strength = 0 eliminates all artifacts at that setting.
    //
    // A 3 % screen-edge fade collapses the offset to zero at screen boundaries;
    // 3 % (≈ 30 px at 1080 p) is imperceptible.
    let view_dist     = max(length(in.world_position.xyz - view.world_position.xyz), 0.1);
    let target_thick  = clamp(refraction_strength * view_dist / 600.0, 0.0, 16.0);
    let screen_uv     = in.position.xy / view.viewport.zw;
    let screen_edge_t = min(
        min(screen_uv.x, 1.0 - screen_uv.x),
        min(screen_uv.y, 1.0 - screen_uv.y),
    );
    pbr_input.material.thickness = target_thick * smoothstep(0.0, 0.03, screen_edge_t);
    pbr_input.material.attenuation_distance = mix(16.0, 180.0, clarity_sq);
    pbr_input.material.attenuation_color    = vec4<f32>(
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

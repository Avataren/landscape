#define_import_path bevy_landscape_water::water_ssr

// Screen-space reflections for the water surface.
//
// Only compiled when DEPTH_PREPASS is active — the algorithm requires the
// depth prepass for intersection testing and view_transmission_texture for
// the reflected colour.
#ifdef DEPTH_PREPASS

#import bevy_pbr::{
    mesh_view_bindings::{view_transmission_texture, view_transmission_sampler},
    prepass_utils,
    view_transformations::{
        position_world_to_clip,
        ndc_to_uv,
        ndc_to_frag_coord,
        depth_ndc_to_view_z,
    },
}
#import bevy_landscape_water::water_bindings::material

// Step past the water surface before marching so the terrain depth values
// stored in the prepass (water writes no depth of its own) don't cause an
// immediate false positive.
const SSR_START_OFFSET_M: f32 = 1.5;

// Returns reflected scene colour (rgb) and a confidence weight (a).
//   confidence = 0 → no screen-space hit; caller falls back to PBR IBL.
//   confidence > 0 → valid hit; caller mixes in proportion.
fn screen_space_reflect(
    world_pos:   vec3<f32>,  // water fragment world position
    reflect_dir: vec3<f32>,  // unit reflection direction (world space)
    frag_coord:  vec4<f32>,  // in.position (screen xy + fragment depth)
) -> vec4<f32> {
    // ssr_params: x=enabled, y=steps, z=max_distance, w=thickness
    if material.ssr_params.x < 0.5 { return vec4(0.0); }
    let ssr_steps       = i32(clamp(material.ssr_params.y, 4.0, 128.0));
    let ssr_max_dist    = material.ssr_params.z;
    let ssr_thickness   = material.ssr_params.w;

    // Project ray start (offset past the water surface) and end into NDC.
    let start_clip = position_world_to_clip(world_pos + reflect_dir * SSR_START_OFFSET_M);
    if start_clip.w <= 0.001 { return vec4(0.0); }
    let start_ndc = start_clip.xyz / start_clip.w;

    let end_clip = position_world_to_clip(world_pos + reflect_dir * ssr_max_dist);
    if end_clip.w <= 0.001 { return vec4(0.0); }
    let end_ndc = end_clip.xyz / end_clip.w;

    let ray_ndc = end_ndc - start_ndc;

    // prev_scene_vz is initialised to a large negative sentinel representing
    // "no geometry / sky", placing the first step firmly in state A
    // ("ray ahead of geometry") so only a genuine A→B transition fires.
    var prev_ndc:      vec3<f32> = start_ndc;
    var prev_scene_vz: f32       = -1.0e9;
    var prev_ray_vz:   f32       = depth_ndc_to_view_z(start_ndc.z);

    for (var i = 1; i <= ssr_steps; i++) {
        let t          = f32(i) / f32(ssr_steps);
        let sample_ndc = start_ndc + ray_ndc * t;
        let sample_uv  = ndc_to_uv(sample_ndc.xy);

        // Stop once the ray leaves the visible screen area.
        if any(sample_uv < vec2(0.0)) || any(sample_uv > vec2(1.0)) { break; }

        let sample_frag_xy = ndc_to_frag_coord(sample_ndc.xy);
        let scene_depth    = prepass_utils::prepass_depth(
            vec4(sample_frag_xy, sample_ndc.z, 1.0), 0u
        );

        // Convert to view-space Z (linear in metres, negative forward).
        // Closer to camera = less negative = greater value.
        // Using view-space instead of NDC depth gives uniform precision at all
        // distances — reversed-Z packs distant values near 0 in NDC, making
        // the comparison unreliable beyond ~100 m.
        let ray_vz = depth_ndc_to_view_z(sample_ndc.z);

        // scene_depth == 0 is the reversed-Z far plane (sky / empty).
        var scene_vz = -1.0e9; // sentinel for sky
        if scene_depth > 0.0 {
            scene_vz = depth_ndc_to_view_z(scene_depth);

            // scene_vz > ray_vz → scene geometry is closer to the camera than
            // the ray → the ray has crossed the surface (state A → B).
            //
            // Thickness check: reject candidates where the scene is much closer
            // than the ray, which indicates a false positive from a different
            // object visible at that screen UV (the source of banding artifacts
            // at grazing angles).
            //
            // Guard on prev_scene_vz <= prev_ray_vz (state A at previous step)
            // to fire only on a genuine transition, not while inside a slab.
            if scene_vz > ray_vz
                && (scene_vz - ray_vz) < ssr_thickness
                && prev_scene_vz <= prev_ray_vz
            {
                // Bisect between the two straddling steps for sub-step precision.
                var lo = prev_ndc;
                var hi = sample_ndc;
                for (var b = 0; b < 6; b++) {
                    let mid      = (lo + hi) * 0.5;
                    let mid_xy   = ndc_to_frag_coord(mid.xy);
                    let mid_d    = prepass_utils::prepass_depth(vec4(mid_xy, mid.z, 1.0), 0u);
                    let mid_svz  = select(-1.0e9, depth_ndc_to_view_z(mid_d), mid_d > 0.0);
                    let mid_rvz  = depth_ndc_to_view_z(mid.z);
                    if mid_svz > mid_rvz {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }

                let hit_uv = ndc_to_uv(hi.xy);

                // Fade within 6 % of each screen edge to prevent hard pop-in
                // when a reflected object exits the frame.
                let edge       = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
                let confidence = smoothstep(0.0, 0.06, edge);
                if confidence <= 0.0 { return vec4(0.0); }

                let color = textureSample(
                    view_transmission_texture, view_transmission_sampler, hit_uv
                );
                return vec4(color.rgb, confidence);
            }
        }

        prev_ndc      = sample_ndc;
        prev_scene_vz = scene_vz;
        prev_ray_vz   = ray_vz;
    }

    return vec4(0.0); // no hit — IBL provides the fallback
}

#endif // DEPTH_PREPASS

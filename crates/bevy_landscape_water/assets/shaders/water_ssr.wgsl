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
    view_transformations::{position_world_to_clip, ndc_to_uv, ndc_to_frag_coord},
}

// Coarse linear steps cover the full ray; bisection tightens each hit.
const SSR_LINEAR_STEPS: i32 = 32;
const SSR_BISECT_STEPS: i32 = 6;
// Step past the water surface before marching so the terrain depth values
// stored in the prepass (water writes no depth of its own) don't trigger
// an immediate false positive.
const SSR_START_OFFSET_M: f32 = 1.5;
// Maximum world-space ray length. Reflections beyond this fall back to IBL.
const SSR_MAX_DISTANCE_M: f32 = 300.0;

// Returns reflected scene colour (rgb) and a confidence weight (a).
//   confidence = 0 → no screen-space hit; caller falls back to PBR IBL.
//   confidence > 0 → valid hit; caller mixes in proportion.
fn screen_space_reflect(
    world_pos:   vec3<f32>,  // water fragment world position
    reflect_dir: vec3<f32>,  // unit reflection direction (world space)
    frag_coord:  vec4<f32>,  // in.position (screen xy + fragment depth)
) -> vec4<f32> {
    // Project ray start (offset past water surface) and end into NDC.
    let start_clip = position_world_to_clip(world_pos + reflect_dir * SSR_START_OFFSET_M);
    if start_clip.w <= 0.001 { return vec4(0.0); }
    let start_ndc = start_clip.xyz / start_clip.w;

    let end_clip = position_world_to_clip(world_pos + reflect_dir * SSR_MAX_DISTANCE_M);
    if end_clip.w <= 0.001 { return vec4(0.0); }
    let end_ndc = end_clip.xyz / end_clip.w;

    let ray_ndc = end_ndc - start_ndc;

    // Initialise prev_scene_depth to 0 (= "sky / no geometry") so the first
    // step begins in state A ("ray ahead of geometry"), preventing the start
    // offset position from immediately triggering a transition.
    var prev_ndc:          vec3<f32> = start_ndc;
    var prev_scene_depth:  f32       = 0.0;

    for (var i = 1; i <= SSR_LINEAR_STEPS; i++) {
        let t          = f32(i) / f32(SSR_LINEAR_STEPS);
        let sample_ndc = start_ndc + ray_ndc * t;
        let sample_uv  = ndc_to_uv(sample_ndc.xy);

        // Stop once the ray leaves the visible screen area.
        if any(sample_uv < vec2(0.0)) || any(sample_uv > vec2(1.0)) { break; }

        let sample_frag_xy = ndc_to_frag_coord(sample_ndc.xy);
        let scene_depth    = prepass_utils::prepass_depth(
            vec4(sample_frag_xy, sample_ndc.z, 1.0), 0u
        );

        // scene_depth == 0 is the reversed-Z far plane (sky / empty).
        if scene_depth > 0.0 {
            // Reversed-Z: higher depth value = closer to camera.
            // scene_depth > sample_ndc.z → scene geometry is closer than the
            // ray sample → the ray has crossed the surface (state A → B).
            // Guard on prev_scene_depth <= prev_ndc.z (state A at previous
            // step) to fire only on a genuine transition, not inside slabs.
            if scene_depth > sample_ndc.z && prev_scene_depth <= prev_ndc.z {
                // Bisect between the two straddling steps for sub-step precision.
                var lo = prev_ndc;
                var hi = sample_ndc;
                for (var b = 0; b < SSR_BISECT_STEPS; b++) {
                    let mid    = (lo + hi) * 0.5;
                    let mid_xy = ndc_to_frag_coord(mid.xy);
                    let mid_d  = prepass_utils::prepass_depth(vec4(mid_xy, mid.z, 1.0), 0u);
                    if mid_d > mid.z {
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

        prev_ndc         = sample_ndc;
        prev_scene_depth = scene_depth;
    }

    return vec4(0.0); // no hit — IBL provides the fallback
}

#endif // DEPTH_PREPASS

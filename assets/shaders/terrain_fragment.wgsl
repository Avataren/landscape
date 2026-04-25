// terrain_fragment.wgsl
// Terrain surface shading integrated with Bevy's clustered forward rendering.
//
// Supports all Bevy light types:
//   - Directional lights  (iterated globally, with cascade shadows)
//   - Point lights        (clustered, with optional shadows)
//   - Spot lights         (clustered, with optional shadows)
//
// Albedo is a world-aligned macro colour map (baked data) or a procedural
// slope/altitude blend (fallback).  Lambertian diffuse BRDF; no specular
// (terrain reads best with diffuse-only shading).

// Module alias lets us access conditional bindings (e.g. diffuse_environment_map
// vs diffuse_environment_maps) without name-importing symbols that may not exist
// depending on MULTIPLE_LIGHT_PROBES_IN_ARRAY / ENVIRONMENT_MAP defines.
#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_pbr::{
    mesh_view_bindings::{lights, view, clusterable_objects, light_probes},
    mesh_view_types::{
        DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    },
    shadows::{fetch_directional_shadow, fetch_point_shadow, fetch_spot_shadow},
    clustered_forward::{
        fragment_cluster_index,
        unpack_clusterable_object_index_ranges,
        get_clusterable_object_id,
    },
}

// Must match TerrainMaterialUniforms in material.rs exactly.
struct MaterialSlotGpu {
    tint_vis: vec4<f32>,   // rgb = tint, a = visibility (0/1)
    ranges:   vec4<f32>,   // x = alt_min, y = alt_max, z = slope_min°, w = slope_max°
    uv_scale: vec4<f32>,   // x = fine_scale_m, y = coarse_scale_mul, z = has_tex (0/1), w = reserved
}

struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,
    morph_start_ratio:  f32,
    ring_patches:       f32,
    num_lod_levels:     f32,
    patch_resolution:   f32,
    world_bounds:       vec4<f32>,
    bounds_fade:        vec4<f32>, // x = fade dist, y = use_macro_color, z = flip_v, w = show_wireframe
    debug_flags:        vec4<f32>, // x = show_normals_only, yzw reserved
    clip_levels: array<vec4<f32>, 32>,
    slot_header: vec4<f32>,                     // x = slot count
    slots:       array<MaterialSlotGpu, 8>,
    synthesis_norm:  vec4<f32>,  // x=seed_x, y=seed_z, z=base_freq, w=octaves
    synthesis_norm2: vec4<f32>,  // x=lacunarity, y=gain, z=erosion, w=normal_strength
    source_meta:     vec4<f32>,  // xy = source world origin, zw = source world extent
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)  var height_tex:       texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1)  var height_samp:      sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2)  var<uniform> terrain: TerrainParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(3)  var macro_color_tex:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4)  var macro_color_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(5)  var normal_tex:       texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6)  var normal_samp:      sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(7)  var pbr_albedo_arr:   texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8)  var pbr_albedo_samp:  sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(9)  var pbr_normal_arr:   texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var pbr_normal_samp:  sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var pbr_orm_arr:      texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var pbr_orm_samp:     sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var source_height_tex:  texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(14) var source_height_samp: sampler;

// Must match TerrainVOut in terrain_vertex.wgsl.
struct TerrainVOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec4<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       macro_xz_ws:  vec2<f32>,
    @location(3)       patch_uv:     vec2<f32>,
    @location(4) @interpolate(flat) lod_level: u32,
    @location(5)       morph_alpha:  f32,
}

// ---------------------------------------------------------------------------
// Albedo helpers
// ---------------------------------------------------------------------------

fn procedural_albedo(slope: f32, h_norm: f32) -> vec3<f32> {
    let grass = vec3<f32>(0.28, 0.52, 0.18);
    let dirt  = vec3<f32>(0.50, 0.40, 0.28);
    let rock  = vec3<f32>(0.44, 0.38, 0.32);
    let snow  = vec3<f32>(0.90, 0.93, 0.98);

    var c = mix(grass, dirt, smoothstep(0.12, 0.30, slope));
    c     = mix(c,    rock,  smoothstep(0.30, 0.52, slope));
    c     = mix(c,    snow,  smoothstep(0.62, 0.82, h_norm));
    return c;
}

// Smooth band weight — 1 inside [lo, hi], falling off over `fade` on either side.
// Used to drive altitude / slope masks without hard edges.
fn band(v: f32, lo: f32, hi: f32, fade: f32) -> f32 {
    let lo_w = smoothstep(lo - fade, lo + fade, v);
    let hi_w = 1.0 - smoothstep(hi - fade, hi + fade, v);
    return clamp(lo_w * hi_w, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Hash / noise / sampling utilities
// ---------------------------------------------------------------------------

fn hash1(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash1(i), hash1(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash1(i + vec2<f32>(0.0, 1.0)), hash1(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y,
    );
}

// 2×2 rotation matrix (column-major).
fn rot2(a: f32) -> mat2x2<f32> {
    let s = sin(a); let c = cos(a);
    return mat2x2<f32>(c, s, -s, c);
}

// Stochastic (no-tile) colour sample via per-cell random rotation + offset.
// Bilinear blend over 4 grid corners; each corner rotates the local UV and
// applies a random translation so adjacent cells sample different orientations.
// Gradient vectors are rotated to give correct mip LOD (textureSampleGrad).
fn sample_color_notile(
    tex:   texture_2d_array<f32>,
    samp:  sampler,
    uv:    vec2<f32>,
    layer: i32,
) -> vec4<f32> {
    let i  = floor(uv);
    let f  = fract(uv);
    let bl = f * f * (3.0 - 2.0 * f);
    let dx = dpdx(uv); let dy = dpdy(uv);

    let c00 = i;
    let c10 = i + vec2<f32>(1.0, 0.0);
    let c01 = i + vec2<f32>(0.0, 1.0);
    let c11 = i + vec2<f32>(1.0, 1.0);

    let R00 = rot2(hash1(c00) * 6.2831853);
    let R10 = rot2(hash1(c10) * 6.2831853);
    let R01 = rot2(hash1(c01) * 6.2831853);
    let R11 = rot2(hash1(c11) * 6.2831853);

    // Rotation-only: no translation offset. Large random offsets (hash2) jump to
    // completely different texture regions per cell → blotchy colour patches on
    // high-contrast textures. Rotation alone breaks the periodic grid without
    // sampling discontinuous colour regions across cell boundaries.
    let uv00 = R00 * (uv - c00) + c00;
    let uv10 = R10 * (uv - c10) + c10;
    let uv01 = R01 * (uv - c01) + c01;
    let uv11 = R11 * (uv - c11) + c11;

    let s00 = textureSampleGrad(tex, samp, uv00, layer, R00 * dx, R00 * dy);
    let s10 = textureSampleGrad(tex, samp, uv10, layer, R10 * dx, R10 * dy);
    let s01 = textureSampleGrad(tex, samp, uv01, layer, R01 * dx, R01 * dy);
    let s11 = textureSampleGrad(tex, samp, uv11, layer, R11 * dx, R11 * dy);

    let w00 = (1.0 - bl.x) * (1.0 - bl.y);
    let w10 = bl.x          * (1.0 - bl.y);
    let w01 = (1.0 - bl.x) * bl.y;
    let w11 = bl.x          * bl.y;
    return s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11;
}

// Stochastic normal-map sample with inverse rotation applied to tangent-space XY.
// Blending normals sampled at large random offsets causes ts_xy vectors from
// different surface regions to cancel.  Rotation-based blending avoids this:
// each cell samples the same local detail at a random orientation, then ts_xy
// is rotated back by the inverse (= transpose) rotation before accumulation,
// so all four contributions are in a consistent tangent-space frame.
fn sample_normal_notile(uv: vec2<f32>, layer: i32) -> vec3<f32> {
    let i  = floor(uv);
    let f  = fract(uv);
    let bl = f * f * (3.0 - 2.0 * f);
    let dx = dpdx(uv); let dy = dpdy(uv);

    let c00 = i;
    let c10 = i + vec2<f32>(1.0, 0.0);
    let c01 = i + vec2<f32>(0.0, 1.0);
    let c11 = i + vec2<f32>(1.0, 1.0);

    let R00 = rot2(hash1(c00) * 6.2831853);
    let R10 = rot2(hash1(c10) * 6.2831853);
    let R01 = rot2(hash1(c01) * 6.2831853);
    let R11 = rot2(hash1(c11) * 6.2831853);

    let uv00 = R00 * (uv - c00) + c00;
    let uv10 = R10 * (uv - c10) + c10;
    let uv01 = R01 * (uv - c01) + c01;
    let uv11 = R11 * (uv - c11) + c11;

    let w00 = (1.0 - bl.x) * (1.0 - bl.y);
    let w10 = bl.x          * (1.0 - bl.y);
    let w01 = (1.0 - bl.x) * bl.y;
    let w11 = bl.x          * bl.y;

    let rg00 = textureSampleGrad(pbr_normal_arr, pbr_normal_samp, uv00, layer, R00*dx, R00*dy).rg;
    let rg10 = textureSampleGrad(pbr_normal_arr, pbr_normal_samp, uv10, layer, R10*dx, R10*dy).rg;
    let rg01 = textureSampleGrad(pbr_normal_arr, pbr_normal_samp, uv01, layer, R01*dx, R01*dy).rg;
    let rg11 = textureSampleGrad(pbr_normal_arr, pbr_normal_samp, uv11, layer, R11*dx, R11*dy).rg;

    let xy00 = transpose(R00) * (rg00 * 2.0 - 1.0);
    let xy10 = transpose(R10) * (rg10 * 2.0 - 1.0);
    let xy01 = transpose(R01) * (rg01 * 2.0 - 1.0);
    let xy11 = transpose(R11) * (rg11 * 2.0 - 1.0);

    let ts00 = vec3<f32>(xy00, sqrt(max(0.0, 1.0 - dot(xy00, xy00))));
    let ts10 = vec3<f32>(xy10, sqrt(max(0.0, 1.0 - dot(xy10, xy10))));
    let ts01 = vec3<f32>(xy01, sqrt(max(0.0, 1.0 - dot(xy01, xy01))));
    let ts11 = vec3<f32>(xy11, sqrt(max(0.0, 1.0 - dot(xy11, xy11))));

    return normalize(ts00 * w00 + ts10 * w10 + ts01 * w01 + ts11 * w11);
}

// Triplanar normal-map sample blended in world space.
// Top (Y) projection uses plain textureSample: on steep faces it has near-
// zero weight, and XZ stochastic cells stretch into tall vertical strips on
// near-vertical surfaces. Side (X, Z) projections use stochastic sampling
// because their UVs lie in the plane of the cliff face — no stretch — so
// stochastic cells cover the surface uniformly and break tiling properly.
// Sign-correct the face-normal component so ±X and ±Z faces both work.
fn sample_triplanar_normal(
    wpos:  vec3<f32>,
    n:     vec3<f32>,
    layer: i32,
    scale: f32,
) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(4.0));
    w /= w.x + w.y + w.z + 1e-6;

    // Top (Y): plain on XZ. tangent=+X, bitangent=+Z → world=(ts.x, ts.z, ts.y)
    let rg_y = textureSample(pbr_normal_arr, pbr_normal_samp, wpos.xz / scale, layer).rg * 2.0 - 1.0;
    let b_y  = sqrt(max(0.0, 1.0 - dot(rg_y, rg_y)));
    let n_y  = normalize(vec3<f32>(rg_y.x, b_y, rg_y.y));

    // Side X: stochastic on ZY. Flip U by sign(n.x) for ±X faces.
    // tangent=+Z, bitangent=+Y → world=(ts.z·sign, ts.y, ts.x)
    let ts_x = sample_normal_notile(wpos.zy / scale * vec2<f32>(sign(n.x), 1.0), layer);
    let n_x  = normalize(vec3<f32>(ts_x.z * sign(n.x), ts_x.y, ts_x.x));

    // Side Z: stochastic on XY. Flip U by sign(n.z) for ±Z faces.
    // tangent=+X, bitangent=+Y → world=(ts.x, ts.y, ts.z·sign)
    let ts_z = sample_normal_notile(wpos.xy / scale * vec2<f32>(sign(n.z), 1.0), layer);
    let n_z  = normalize(vec3<f32>(ts_z.x, ts_z.y, ts_z.z * sign(n.z)));

    return normalize(n_y * w.y + n_x * w.x + n_z * w.z);
}

// Triplanar albedo sample — no UV stretching on vertical cliff faces.
// Top (Y) projection uses plain textureSample (low weight on steep faces;
// stochastic XZ cells stretch badly on near-vertical surfaces).
// Side (X, Z) projections use stochastic sampling to break tiling on cliffs.
fn sample_triplanar(
    tex:   texture_2d_array<f32>,
    samp:  sampler,
    wpos:  vec3<f32>,
    n:     vec3<f32>,
    layer: i32,
    scale: f32,
) -> vec4<f32> {
    var w = pow(abs(n), vec3<f32>(4.0));
    w /= w.x + w.y + w.z + 1e-6;
    let cy = textureSample(tex, samp, wpos.xz / scale, layer);
    let cx = sample_color_notile(tex, samp, wpos.zy / scale, layer);
    let cz = sample_color_notile(tex, samp, wpos.xy / scale, layer);
    return cy * w.y + cx * w.x + cz * w.z;
}

// Two-octave noise offset in [-1, 1] used to perturb altitude/slope values
// before band evaluation — makes zone edges irregular instead of sharp rings.
fn zone_noise(world_xz: vec2<f32>) -> f32 {
    return (value_noise(world_xz / 150.0) * 0.7
          + value_noise(world_xz /  40.0) * 0.3) * 2.0 - 1.0;
}

// Sample PBR albedo textures blended by altitude + slope weights.
// When a slot has a real texture (uv_scale.z > 0.5) the array layer is
// sampled using world-space tiling; otherwise the slot tint is used.
// Perturb the macro terrain normal with per-slot PBR normal maps.
// Each slot's tangent-space detail normal is decoded from Rgba8Unorm
// (stored as nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5) and blended by the same
// altitude+slope weights used for albedo.  Returns `base_n` unchanged when
// no slots are active or none have texture data.
fn apply_normal_detail(
    base_n:    vec3<f32>,
    world_xz:  vec2<f32>,
    world_y:   f32,
    slope_deg: f32,
) -> vec3<f32> {
    let count = u32(terrain.slot_header.x);
    if count == 0u {
        return base_n;
    }

    let alt_fade   = max(terrain.height_scale * 0.05, 1.0);
    let slope_fade = 3.0;

    let znoise      = zone_noise(world_xz);
    let noisy_slope = slope_deg + znoise * 6.0;
    let noisy_alt   = world_y   + znoise * terrain.height_scale * 0.015;

    let ref_up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
                        abs(base_n.y) < 0.99);
    let tangent   = normalize(cross(ref_up, base_n));
    let bitangent = cross(base_n, tangent);

    // Blend from XZ-planar to triplanar on steep slopes.
    // Triplanar samples all three world-axis projections so vertical cliff
    // faces get proper detail normals instead of degenerate stretched UVs.
    let tri_t = smoothstep(25.0, 50.0, slope_deg);
    let wpos  = vec3<f32>(world_xz.x, world_y, world_xz.y);

    var sum_n      = vec3<f32>(0.0);
    var sum_weight = 0.0;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let slot = terrain.slots[i];
        let vis  = slot.tint_vis.w;
        if vis <= 0.0 {
            continue;
        }
        let w_alt   = band(noisy_alt,   slot.ranges.x, slot.ranges.y, alt_fade);
        let w_slope = band(noisy_slope, slot.ranges.z, slot.ranges.w, slope_fade);
        let w       = vis * w_alt * w_slope;

        var perturbed: vec3<f32>;
        if slot.uv_scale.z > 0.5 {
            let fine_scale   = max(slot.uv_scale.x, 0.01);
            let coarse_scale = fine_scale * max(slot.uv_scale.y, 2.0);
            let uv   = world_xz / fine_scale;
            let ts_n = sample_normal_notile(uv, i32(i));
            let ts_n_amp = normalize(vec3<f32>(ts_n.xy * 2.0, ts_n.z));
            let n_planar = normalize(tangent * ts_n_amp.x + bitangent * ts_n_amp.y + base_n * ts_n_amp.z);

            if tri_t > 0.001 {
                let n_tri = sample_triplanar_normal(wpos, base_n, i32(i), fine_scale);
                perturbed = normalize(mix(n_planar, n_tri, tri_t));
            } else {
                perturbed = n_planar;
            }
        } else {
            perturbed = base_n;
        }

        sum_n      = sum_n + perturbed * w;
        sum_weight = sum_weight + w;
    }

    if sum_weight < 1e-4 {
        return base_n;
    }
    return normalize(sum_n / sum_weight);
}

// Weighted-average roughness from per-slot ORM maps (G channel).
// Returns 0.5 (neutral) when no slots are active or no ORM textures are loaded.
fn sample_roughness(
    world_xz:  vec2<f32>,
    world_y:   f32,
    slope_deg: f32,
) -> f32 {
    let count = u32(terrain.slot_header.x);
    if count == 0u {
        return 0.5;
    }

    let alt_fade   = max(terrain.height_scale * 0.05, 1.0);
    let slope_fade = 3.0;

    let znoise      = zone_noise(world_xz);
    let noisy_slope = slope_deg + znoise * 6.0;
    let noisy_alt   = world_y   + znoise * terrain.height_scale * 0.015;

    var sum_r      = 0.0;
    var sum_weight = 0.0;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let slot = terrain.slots[i];
        let vis  = slot.tint_vis.w;
        if vis <= 0.0 {
            continue;
        }
        let w_alt   = band(noisy_alt,   slot.ranges.x, slot.ranges.y, alt_fade);
        let w_slope = band(noisy_slope, slot.ranges.z, slot.ranges.w, slope_fade);
        let w       = vis * w_alt * w_slope;

        var r: f32;
        if slot.uv_scale.z > 0.5 {
            let fine_scale   = max(slot.uv_scale.x, 0.01);
            let coarse_scale = fine_scale * max(slot.uv_scale.y, 2.0);
            let uv = world_xz / coarse_scale;
            r = sample_color_notile(pbr_orm_arr, pbr_orm_samp, uv, i32(i)).g;
        } else {
            r = 0.5;
        }

        sum_r      = sum_r + r * w;
        sum_weight = sum_weight + w;
    }

    if sum_weight < 1e-4 {
        return 0.5;
    }
    return sum_r / sum_weight;
}

fn sample_pbr_albedo(
    world_xz:  vec2<f32>,
    world_y:   f32,
    slope_deg: f32,
    n_macro:   vec3<f32>,
    fallback:  vec3<f32>,
) -> vec3<f32> {
    let count = u32(terrain.slot_header.x);
    if count == 0u {
        return fallback;
    }

    let alt_fade   = max(terrain.height_scale * 0.05, 1.0);
    let slope_fade = 3.0;

    let znoise      = zone_noise(world_xz);
    let noisy_slope = slope_deg + znoise * 6.0;
    let noisy_alt   = world_y   + znoise * terrain.height_scale * 0.015;

    let wpos  = vec3<f32>(world_xz.x, world_y, world_xz.y);
    // Triplanar blends in on steep slopes to eliminate UV stretching on cliffs.
    let tri_t = smoothstep(50.0, 70.0, slope_deg);

    var sum_col    = vec3<f32>(0.0);
    var sum_weight = 0.0;
    var best_col   = fallback;
    var best_w     = -1.0;

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let slot = terrain.slots[i];
        let vis  = slot.tint_vis.w;
        if vis <= 0.0 {
            continue;
        }
        let w_alt   = band(noisy_alt,   slot.ranges.x, slot.ranges.y, alt_fade);
        let w_slope = band(noisy_slope, slot.ranges.z, slot.ranges.w, slope_fade);
        let w       = vis * w_alt * w_slope;

        var col: vec3<f32>;
        if slot.uv_scale.z > 0.5 {
            let fine_scale   = max(slot.uv_scale.x, 0.01);
            let coarse_scale = fine_scale * max(slot.uv_scale.y, 2.0);
            let uv_fine   = world_xz / fine_scale;
            let uv_coarse = world_xz / coarse_scale;
            let c_fine   = sample_color_notile(pbr_albedo_arr, pbr_albedo_samp, uv_fine, i32(i)).rgb;
            let c_coarse = sample_color_notile(pbr_albedo_arr, pbr_albedo_samp, uv_coarse, i32(i)).rgb;
            let col_uv   = c_fine * 0.7 + c_coarse * 0.3;
            if tri_t > 0.001 {
                let c_tri = sample_triplanar(pbr_albedo_arr, pbr_albedo_samp,
                                             wpos, n_macro, i32(i), coarse_scale).rgb;
                col = mix(col_uv, c_tri, tri_t) * slot.tint_vis.rgb;
            } else {
                col = col_uv * slot.tint_vis.rgb;
            }
        } else {
            col = slot.tint_vis.rgb;
        }

        if w > best_w {
            best_w = w;
            best_col = col;
        }

        sum_col    = sum_col + col * w;
        sum_weight = sum_weight + w;
    }

    if sum_weight < 1e-4 {
        return best_col;
    }
    return sum_col / sum_weight;
}

fn in_world_bounds(world_xz: vec2<f32>) -> bool {
    return all(world_xz >= terrain.world_bounds.xy)
        && all(world_xz <= terrain.world_bounds.zw);
}

fn macro_color_uv(world_xz: vec2<f32>) -> vec2<f32> {
    let world_min  = terrain.world_bounds.xy;
    let world_span = max(terrain.world_bounds.zw - world_min, vec2<f32>(1.0, 1.0));
    let sample_xz = clamp(world_xz, world_min, terrain.world_bounds.zw);
    var uv = (sample_xz - world_min) / world_span;
    if terrain.bounds_fade.z > 0.5 {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}

// ---------------------------------------------------------------------------
// Sky hemisphere ambient
//
// Approximates diffuse IBL from the atmosphere without cubemap sampling.
// The sky and ground bounce values are in *display-space* (0–1 fractions).
// Dividing by view.exposure converts them to the physical-unit scale used by
// the rest of the lighting, so that when the final (direct + ambient) is
// multiplied by view.exposure the hemisphere values survive as their original
// display-space fractions.
//
// Adjust these to match the scene's sky colour and ground albedo.
// ---------------------------------------------------------------------------

const SKY_AMBIENT:    vec3<f32> = vec3<f32>(0.20, 0.26, 0.40); // clear-sky blue, ~25 % brightness
const GROUND_BOUNCE:  vec3<f32> = vec3<f32>(0.04, 0.03, 0.02); // dark earth, ~4 % brightness

fn hemisphere_ambient(world_normal: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    // sky_t = 1 for normals pointing straight up, 0 for straight down.
    let sky_t  = saturate(world_normal.y * 0.5 + 0.5);
    let irr    = mix(GROUND_BOUNCE, SKY_AMBIENT, sky_t);
    return albedo * irr;
}

// ---------------------------------------------------------------------------
// Per-pixel normal from height finite differences
//
// Mirrors the vertex shader's normal_at() exactly, but runs in the fragment
// stage so every pixel gets an independent normal sample — no vertex
// interpolation artefacts.  Uses the height clipmap (always populated) rather
// than the baked RG8Snorm normal array (which may be zero when
// procedural_fallback is false and no baked normal tiles are loaded).
// ---------------------------------------------------------------------------

/// Sample height in world-space metres from the R32Float clipmap.
/// The clipmap stores metres directly — no height_scale multiply needed.
///
/// Manual bilinear: R32Float is not in the `FLOAT32_FILTERABLE` set on most
/// devices Bevy initialises by default, so `textureSample` with a linear
/// sampler silently returns nearest texels.  That makes every clipmap texel
/// a flat plateau and the FD-derived per-pixel normal then jumps at every
/// texel boundary — visible as a hard grid all over the surface.  Doing the
/// 2×2 lerp manually with `textureLoad` sidesteps the feature requirement.
fn height_at_frag(lod: u32, xz: vec2<f32>) -> f32 {
    let lvl       = terrain.clip_levels[lod];
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(xz, world_min, world_max);

    let dims_u    = textureDimensions(height_tex, 0);
    let dims_i    = vec2<i32>(dims_u);
    let dims_f    = vec2<f32>(dims_u);

    // Toroidal layer UV → continuous texel coord with the standard −0.5
    // half-texel shift that puts texel I's centre at uv = (I+0.5)/N.
    let uv     = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    let coord  = uv * dims_f - vec2<f32>(0.5);
    let i0_f   = floor(coord);
    let f      = coord - i0_f;

    // Repeat-wrap each integer corner into [0, dims).
    let i0 = vec2<i32>(i0_f);
    let x0 = ((i0.x % dims_i.x) + dims_i.x) % dims_i.x;
    let y0 = ((i0.y % dims_i.y) + dims_i.y) % dims_i.y;
    let x1 = (x0 + 1) % dims_i.x;
    let y1 = (y0 + 1) % dims_i.y;

    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), i32(lod), 0).r;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), i32(lod), 0).r;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y1), i32(lod), 0).r;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y1), i32(lod), 0).r;

    let top = mix(h00, h10, f.x);
    let bot = mix(h01, h11, f.x);
    return mix(top, bot, f.y);
}

fn pixel_normal(lod: u32, xz: vec2<f32>) -> vec3<f32> {
    let eps = terrain.clip_levels[lod].w; // texel world size at this LOD
    let h   = height_at_frag(lod, xz);
    let h_r = height_at_frag(lod, xz + vec2<f32>(eps, 0.0));
    let h_u = height_at_frag(lod, xz + vec2<f32>(0.0, eps));
    return normalize(vec3<f32>(h - h_r, eps, h - h_u));
}

// ---------------------------------------------------------------------------
// Source-heightmap macro normal
// ---------------------------------------------------------------------------
// FD on the displacement clipmap is unreliable as a "macro" normal: the
// clipmap stores `base_h + amplitude * detail`, the per-texel jumps in `base_h`
// (sampled bilinearly from a R16Unorm source) become visible whenever the fBM
// detail is small or zero, and on devices without FLOAT32_FILTERABLE the
// linear sampler degenerates to nearest, which makes every clipmap texel a
// flat plateau.
//
// Sampling the source heightmap directly with a linear sampler (R16Unorm is
// always filterable) gives a smooth macro normal that tracks the actual
// terrain shape, independent of the displacement amplitude.  fBM detail can
// then be added on top via fbm_perturb_normal without these artefacts.
fn source_macro_normal(world_xz: vec2<f32>) -> vec3<f32> {
    let origin = terrain.source_meta.xy;
    let extent = terrain.source_meta.zw;
    if extent.x <= 0.0 || extent.y <= 0.0 {
        // No source-heightmap state yet — fall back to the legacy clipmap FD.
        return pixel_normal(0u, world_xz);
    }
    let inv_extent = vec2<f32>(1.0) / extent;

    // Source texel size in world space = 2 / base_freq (synthesis sets
    // base_freq = 2 / source_spacing).
    let base_freq = max(terrain.synthesis_norm.z, 1e-6);
    let src_spacing = 2.0 / base_freq;
    let eps_uv = vec2<f32>(src_spacing) * inv_extent;
    let height_scale = terrain.height_scale;

    let uv  = clamp((world_xz - origin) * inv_extent, vec2<f32>(0.0), vec2<f32>(1.0));
    let h0n = textureSample(source_height_tex, source_height_samp, uv).r;
    let hxn = textureSample(source_height_tex, source_height_samp,
                            clamp(uv + vec2<f32>(eps_uv.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).r;
    let hzn = textureSample(source_height_tex, source_height_samp,
                            clamp(uv + vec2<f32>(0.0, eps_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).r;

    let dh_dx = (hxn - h0n) * height_scale;       // metres per source-spacing
    let dh_dz = (hzn - h0n) * height_scale;
    return normalize(vec3<f32>(-dh_dx, src_spacing, -dh_dz));
}

// ---------------------------------------------------------------------------
// Per-fragment fBM normal perturbation
// ---------------------------------------------------------------------------
// Re-evaluates the same fBM noise field used by the detail-synthesis compute
// pass and uses its analytic-ish gradient to perturb the macro normal.
// Decoupled from displacement amplitude so the surface keeps visible noise
// shading even when the geometric displacement is zero.
//
// Hash + gradient noise must stay bit-identical with detail_synthesis.wgsl
// and synthesis_cpu.rs — using a deterministic PCG so all three agree.
// ---------------------------------------------------------------------------

const FBM_GRADIENT_EPSILON: f32 = 0.37;
const FBM_EROSION_RESPONSE: f32 = 3.5;
const FBM_U32_TO_UNIT:      f32 = 2.3283064e-10;

fn fbm_pcg2d(v_in: vec2<u32>) -> vec2<u32> {
    var v = v_in * vec2<u32>(1664525u) + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return v;
}

fn fbm_hash_grad(pi: vec2<i32>) -> vec2<f32> {
    // Unit-length gradient — see detail_synthesis.wgsl for rationale.
    let h = fbm_pcg2d(bitcast<vec2<u32>>(pi));
    let theta = f32(h.x) * FBM_U32_TO_UNIT * 6.2831853;
    return vec2<f32>(cos(theta), sin(theta));
}

fn fbm_gradient_noise(p: vec2<f32>) -> f32 {
    let pf = floor(p);
    let i  = vec2<i32>(pf);
    let f  = p - pf;
    let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a  = dot(fbm_hash_grad(i + vec2<i32>(0, 0)), f);
    let b  = dot(fbm_hash_grad(i + vec2<i32>(1, 0)), f - vec2<f32>(1.0, 0.0));
    let c  = dot(fbm_hash_grad(i + vec2<i32>(0, 1)), f - vec2<f32>(0.0, 1.0));
    let d  = dot(fbm_hash_grad(i + vec2<i32>(1, 1)), f - vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm_rot2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y);
}

fn fbm_height(base: vec2<f32>, octaves: u32, lac: f32, g: f32, erosion: f32) -> f32 {
    var value     = 0.0;
    var amplitude = 0.5;
    var pos       = base;
    var acc_grad  = vec2<f32>(0.0, 0.0);
    for (var i = 0u; i < octaves; i++) {
        let n = fbm_gradient_noise(pos);
        let grad = vec2<f32>(
            fbm_gradient_noise(pos + vec2<f32>(FBM_GRADIENT_EPSILON, 0.0)) - n,
            fbm_gradient_noise(pos + vec2<f32>(0.0, FBM_GRADIENT_EPSILON)) - n,
        ) / FBM_GRADIENT_EPSILON;
        acc_grad += grad * amplitude;
        let atten = mix(1.0, 1.0 / (1.0 + dot(acc_grad, acc_grad) * FBM_EROSION_RESPONSE), erosion);
        value    += amplitude * n * atten;
        pos       = fbm_rot2(pos) * lac;
        amplitude *= g;
    }
    return value;
}

// Perturb `macro_n` by the analytic gradient of the same fBM the displacement
// compute pass uses.  Uses three fBM evaluations (centre + two world-space
// neighbours) so the perturbation magnitude is in metres-per-metre, which is
// then projected into the local tangent frame of `macro_n`.
fn fbm_perturb_normal(macro_n: vec3<f32>, world_xz: vec2<f32>, slope_deg: f32) -> vec3<f32> {
    let cfg_octaves = u32(terrain.synthesis_norm.w);
    let strength = terrain.synthesis_norm2.w;
    if cfg_octaves == 0u || strength <= 0.0 {
        return macro_n;
    }

    let base_freq = terrain.synthesis_norm.z;
    let lac       = terrain.synthesis_norm2.x;
    let g         = terrain.synthesis_norm2.y;
    let erosion   = terrain.synthesis_norm2.z;
    let seed      = terrain.synthesis_norm.xy;

    // Use at most 4 octaves for the lighting perturbation.  The displacement
    // compute pass can keep its full octave count because the clipmap stores
    // pre-evaluated heights and the FD step there matches its texel size, but
    // fragment-shader FD over arbitrarily fine wavelengths aliases — adjacent
    // pixels can land on opposite ends of an octave-5 wave and produce wildly
    // different gradients, splattering noise into the lighting.
    let octaves = min(cfg_octaves, 4u);

    // FD step = half the wavelength of the finest contributing octave.  Below
    // that, FD becomes a sample-rate aliasing operation; above it the
    // perturbation gets blurry but never spiky.
    let finest_freq = base_freq * pow(lac, f32(octaves) - 1.0);
    let eps_ws = max(0.5 / finest_freq, 0.5);

    let p0 = (world_xz + seed) * base_freq;
    let px = (world_xz + vec2<f32>(eps_ws, 0.0) + seed) * base_freq;
    let pz = (world_xz + vec2<f32>(0.0, eps_ws) + seed) * base_freq;

    let h0 = fbm_height(p0, octaves, lac, g, erosion);
    let hx = fbm_height(px, octaves, lac, g, erosion);
    let hz = fbm_height(pz, octaves, lac, g, erosion);

    // Cliff fade — keep rock faces shaded by their macro normal so the noise
    // doesn't fight strong silhouettes.
    let cliff_fade = 1.0 - smoothstep(40.0, 80.0, slope_deg);

    // (hx - h0) is dimensionless fBM output; multiplying by `strength`
    // (metres) gives metres of "virtual" displacement, then dividing by
    // eps_ws (metres) yields a unitless slope that perturbs the normal.
    let dh_dx = (hx - h0) * strength / eps_ws * cliff_fade;
    let dh_dz = (hz - h0) * strength / eps_ws * cliff_fade;

    // Build a local frame from macro_n and tilt it by the fBM gradient.
    let ref_up   = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
                          abs(macro_n.y) < 0.99);
    let tangent   = normalize(cross(ref_up, macro_n));
    let bitangent = cross(macro_n, tangent);

    return normalize(macro_n - tangent * dh_dx - bitangent * dh_dz);
}

// ─────────────────────────────────────────────────────────────────────────────

// Sample the baked RGBA8Snorm normal array for this LOD.
//
// RG = fine XZ (computed at this LOD's texel eps), BA = coarse XZ (2× eps).
// Both channels are stored in the same layer so a single texture lookup
// provides both ends of the LOD seam, halving normal-texture bandwidth vs
// sampling fine and coarse LOD layers separately (GPU Gems 2, Ch. 2).
//
// UV math mirrors `height_at_frag` exactly (same half-texel offset).
fn baked_normal_rgba(lod: u32, xz: vec2<f32>) -> vec4<f32> {
    let lvl       = terrain.clip_levels[lod];
    let world_min = terrain.world_bounds.xy;
    let world_max = terrain.world_bounds.zw - vec2<f32>(lvl.w, lvl.w);
    let sample_xz = clamp(xz, world_min, world_max);
    let uv        = fract((sample_xz + 0.5 * lvl.w) * lvl.z);
    return textureSample(normal_tex, normal_samp, uv, i32(lod));
}

// Returns the shading normal blended between fine and coarse using morph_alpha.
// Always uses per-pixel FD normals so fBM detail from the compute pass
// appears in lighting without a separate baked normal update.
fn shading_normal_blended(lod: u32, coarse_lod: u32, xz: vec2<f32>, alpha: f32) -> vec3<f32> {
    let n_fine   = pixel_normal(lod, xz);
    let n_coarse = pixel_normal(coarse_lod, xz);
    return normalize(mix(n_fine, n_coarse, alpha));
}

// ---------------------------------------------------------------------------
// Attenuation — same formula as pbr_lighting::getDistanceAttenuation
// ---------------------------------------------------------------------------

fn distance_attenuation(dist_sq: f32, inv_range_sq: f32) -> f32 {
    let factor       = dist_sq * inv_range_sq;
    let smooth_factor = saturate(1.0 - factor * factor);
    return (smooth_factor * smooth_factor) / max(dist_sq, 0.0001);
}

fn patch_grid_wireframe(patch_uv: vec2<f32>) -> f32 {
    let grid_uv = patch_uv * terrain.patch_resolution;
    let grid_fw = max(fwidth(grid_uv), vec2<f32>(1e-4, 1e-4));
    let quad_dist = abs(fract(grid_uv - 0.5) - 0.5) / grid_fw;
    let quad_line = 1.0 - saturate(min(quad_dist.x, quad_dist.y));

    let patch_fw = max(fwidth(patch_uv), vec2<f32>(1e-4, 1e-4));
    let patch_dist = min(patch_uv, 1.0 - patch_uv) / patch_fw;
    let patch_line = 1.0 - saturate(min(patch_dist.x, patch_dist.y) - 0.5);

    return max(quad_line * 0.75, patch_line);
}

fn lod_debug_color(lod: u32) -> vec3<f32> {
    switch (lod % 8u) {
        case 0u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 1u: { return vec3<f32>(0.0, 0.8, 1.0); }
        case 2u: { return vec3<f32>(1.0, 1.0, 0.0); }
        case 3u: { return vec3<f32>(1.0, 0.5, 0.0); }
        case 4u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 5u: { return vec3<f32>(0.8, 0.0, 1.0); }
        case 6u: { return vec3<f32>(0.4, 0.4, 1.0); }
        default: { return vec3<f32>(1.0, 1.0, 1.0); }
    }
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

@fragment
fn fragment(in: TerrainVOut) -> @location(0) vec4<f32> {
    // Discard fragments based on the post-morph world position. Using the
    // pre-morph XZ here lets coarse LOD boundary snaps grow visible geometry
    // beyond the terrain footprint, which shows up as black rectangles at the
    // map edge when the depth prepass and main pass disagree.
    if !in_world_bounds(in.world_pos.xz) {
        discard;
    }

    // --- Per-pixel normal from height finite differences ---
    // `vertex_n` is the morph-blended vertex normal used for shadow bias
    // (stable, no high-frequency noise).  `n` is recomputed per-pixel from
    // the height clipmap — same formula as the vertex shader's normal_at(),
    // but evaluated at the exact fragment world position rather than at the
    // nearest vertex, eliminating vertex-interpolation artefacts on curved slopes.
    //
    // Normal blending mirrors the vertex shader: at morph_alpha=0 the fine LOD
    // normal is used exclusively; at alpha=1 (the seam boundary) only the coarse
    // LOD normal is used.  This eliminates the lighting band that appears when
    // adjacent LOD patches compute normals from different clipmap resolutions.
    // Using world_pos.xz (the morphed position) ensures both sides of the seam
    // sample the same clipmap point when alpha=1.
    let vertex_n   = normalize(in.world_normal);
    let coarse_lod = min(in.lod_level + 1u, u32(terrain.num_lod_levels) - 1u);
    let frag_xz    = in.world_pos.xz;
    // Macro normal sourced directly from the source heightmap (linear sampled,
    // R16Unorm is always filterable) — bypasses both the FD-on-clipmap quantisation
    // and the FLOAT32_FILTERABLE feature requirement.
    let n_macro_raw = source_macro_normal(frag_xz);

    // Slope drives material zone weights and stays based on the unperturbed
    // macro normal — using the noisy fBM-perturbed normal here splatters rock
    // material everywhere because the gradient varies wildly per-pixel.
    let macro_ndotu = abs(dot(n_macro_raw, vec3<f32>(0.0, 1.0, 0.0)));
    let slope_deg   = degrees(acos(clamp(macro_ndotu, 0.0, 1.0)));

    // Per-fragment fBM perturbation — only affects the *lighting* normal, not
    // the slope used for material selection.  Decoupled from the geometric
    // displacement amplitude so the surface keeps shading detail when
    // amplitude is small.
    let n_macro = fbm_perturb_normal(n_macro_raw, frag_xz, slope_deg);

    // Apply per-slot PBR normal maps on top of the macro normal.
    let n = apply_normal_detail(n_macro, in.world_pos.xz, in.world_pos.y, slope_deg);

    // --- PBR texture debug (debug_flags.z) ---
    // Samples slot 0 directly at world UV — bypasses zone blending so you
    // see exactly what the GPU reads from the texture array.
    // debug_flags.z == 1: raw normal-map RGB (should look like a blue-ish normal map)
    // debug_flags.z == 2: ORM roughness G channel as greyscale
    // Red output means uv_scale.z == 0 (slot has no texture path set).
    if terrain.debug_flags.z > 0.5 {
        let s0   = terrain.slots[0];
        let uv0  = in.world_pos.xz / max(s0.uv_scale.x, 0.01);
        if s0.uv_scale.z > 0.5 {
            if terrain.debug_flags.z > 1.5 {
                let r = textureSample(pbr_orm_arr, pbr_orm_samp, uv0, 0).g;
                return vec4<f32>(r, r, r, 1.0);
            } else {
                return vec4<f32>(textureSample(pbr_normal_arr, pbr_normal_samp, uv0, 0).rgb, 1.0);
            }
        }
        return vec4<f32>(1.0, 0.0, 0.0, 1.0); // red = no texture path
    }

    // --- Debug: render normals as colour and skip lighting/material ---
    if terrain.debug_flags.x > 1.5 {
        if terrain.debug_flags.x > 2.5 {
            let base = lod_debug_color(in.lod_level);
            return vec4<f32>(mix(base, vec3<f32>(1.0), in.morph_alpha * 0.65), 1.0);
        }
        // debug_flags.x == 2: show height normalised by height_scale as greyscale.
        let h = height_at_frag(in.lod_level, frag_xz);
        let scaled = clamp(h / max(terrain.height_scale, 1.0), 0.0, 1.0);
        return vec4<f32>(scaled, scaled, scaled, 1.0);
    }
    if terrain.debug_flags.x > 0.5 {
        return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    }

    // --- Albedo ---
    let slope  = 1.0 - abs(dot(n, vec3<f32>(0.0, 1.0, 0.0)));
    let h_norm = clamp(in.world_pos.y / terrain.height_scale, 0.0, 1.0);

    // Fallback used when the material library is empty (count == 0) or all
    // slots have zero weight at this fragment.  Keeps the terrain visible.
    let fallback = procedural_albedo(slope, h_norm);
    var albedo   = sample_pbr_albedo(in.world_pos.xz, in.world_pos.y, slope_deg, n_macro, fallback);
    if terrain.bounds_fade.y > 0.5 {
        // textureSample (not textureSampleLevel) lets the GPU pick the correct
        // mip based on screen-space derivatives — better antialiasing at distance.
        albedo = textureSample(
            macro_color_tex, macro_color_samp,
            macro_color_uv(in.macro_xz_ws),
        ).rgb;
    }

    // --- Shared values ---
    // view_z: view-space depth, used for shadow cascade selection and
    // cluster z-slice lookup.  Negative for fragments in front of camera.
    let view_z = dot(
        vec4<f32>(view.view_from_world[0].z,
                  view.view_from_world[1].z,
                  view.view_from_world[2].z,
                  view.view_from_world[3].z),
        in.world_pos,
    );
    let is_orthographic = view.clip_from_view[3].w == 1.0;

    // Sample roughness and apply an Oren-Nayar A term to the diffuse.
    // Rough surfaces (rock) appear more matte; smooth surfaces (sand) brighter.
    let roughness = sample_roughness(in.world_pos.xz, in.world_pos.y, slope_deg);
    let sigma2    = roughness * roughness;
    let on_a      = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33));
    let diffuse   = albedo / 3.14159 * on_a;

    var direct = vec3<f32>(0.0);

    // -----------------------------------------------------------------------
    // Directional lights  (global, not clustered)
    // light.color.rgb is pre-multiplied by illuminance (lux) in the GPU buffer.
    // -----------------------------------------------------------------------
    for (var i: u32 = 0u; i < lights.n_directional_lights; i++) {
        let light = lights.directional_lights[i];
        let ndotl = max(dot(n, light.direction_to_light), 0.0);

        var shadow = 1.0;
        if (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            // Shadow bias uses vertex_n (per-vertex, more stable for self-shadow).
            shadow = fetch_directional_shadow(i, in.world_pos, vertex_n, view_z);
        }

        direct += diffuse * ndotl * shadow * light.color.rgb;
    }

    // -----------------------------------------------------------------------
    // Clustered point and spot lights
    // color_inverse_square_range.rgb = colour × intensity (cd/m²)
    // color_inverse_square_range.w   = 1 / range²
    // -----------------------------------------------------------------------
    let cluster_index = fragment_cluster_index(in.clip_pos.xy, view_z, is_orthographic);
    let ranges        = unpack_clusterable_object_index_ranges(cluster_index);

    // --- Point lights ---
    for (var i: u32 = ranges.first_point_light_index_offset;
             i < ranges.first_spot_light_index_offset; i++) {
        let light_id  = get_clusterable_object_id(i);
        let light     = &clusterable_objects.data[light_id];
        let to_frag   = (*light).position_radius.xyz - in.world_pos.xyz;
        let dist_sq   = dot(to_frag, to_frag);
        let L         = normalize(to_frag);
        let ndotl     = max(dot(n, L), 0.0);
        let atten     = distance_attenuation(dist_sq, (*light).color_inverse_square_range.w);

        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_point_shadow(light_id, in.world_pos, vertex_n);
        }

        direct += diffuse * ndotl * atten * shadow * (*light).color_inverse_square_range.rgb;
    }

    // --- Spot lights ---
    for (var i: u32 = ranges.first_spot_light_index_offset;
             i < ranges.first_reflection_probe_index_offset; i++) {
        let light_id  = get_clusterable_object_id(i);
        let light     = &clusterable_objects.data[light_id];
        let to_frag   = (*light).position_radius.xyz - in.world_pos.xyz;
        let dist_sq   = dot(to_frag, to_frag);
        let L         = normalize(to_frag);
        let ndotl     = max(dot(n, L), 0.0);
        let atten     = distance_attenuation(dist_sq, (*light).color_inverse_square_range.w);

        // Spot cone attenuation.
        // light_custom_data.xy = normalised spot direction (xz components);
        // light_custom_data.zw = precomputed scale and offset for cone falloff.
        var spot_dir = vec3<f32>((*light).light_custom_data.x, 0.0, (*light).light_custom_data.y);
        spot_dir.y = sqrt(max(0.0, 1.0 - spot_dir.x * spot_dir.x - spot_dir.z * spot_dir.z));
        if ((*light).flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
            spot_dir.y = -spot_dir.y;
        }
        let cd           = dot(-spot_dir, L);
        let cone_raw     = saturate(cd * (*light).light_custom_data.z + (*light).light_custom_data.w);
        let cone_atten   = cone_raw * cone_raw;

        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_spot_shadow(
                light_id, in.world_pos, vertex_n, (*light).shadow_map_near_z,
            );
        }

        direct += diffuse * ndotl * atten * cone_atten * shadow
            * (*light).color_inverse_square_range.rgb;
    }

    // --- Ambient ---
    // Sample diffuse irradiance from the atmosphere-generated cubemap when available.
    // Uses the view probe (atmosphere covers the whole world, no local probe needed).
    // Handles both bindless (MULTIPLE_LIGHT_PROBES_IN_ARRAY) and single-texture paths.
    // Falls back to the hemisphere approximation when no environment map is present.
#ifdef ENVIRONMENT_MAP
    var irradiance = vec3<f32>(0.0);
    let view_probe_idx = light_probes.view_cubemap_index;
    if view_probe_idx >= 0 {
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
        irradiance = textureSampleLevel(
            view_bindings::diffuse_environment_maps[view_probe_idx],
            view_bindings::environment_map_sampler,
            n, 0.0,
        ).rgb * light_probes.intensity_for_view;
#else
        irradiance = textureSampleLevel(
            view_bindings::diffuse_environment_map,
            view_bindings::environment_map_sampler,
            n, 0.0,
        ).rgb * light_probes.intensity_for_view;
#endif
    }
    let sky_ambient_physical = albedo * irradiance;
#else
    let sky_ambient_physical = hemisphere_ambient(n, albedo) / max(view.exposure, 1e-10);
#endif
    let flat_ambient         = albedo * lights.ambient_color.rgb;
    let ambient              = sky_ambient_physical + flat_ambient;

    // Apply camera exposure (physical luminance → display values).
    var out_rgb = (direct + ambient) * view.exposure;

    if terrain.bounds_fade.w > 0.5 {
        let wire = patch_grid_wireframe(in.patch_uv);
        let wire_color = vec3<f32>(0.98, 0.94, 0.35);
        out_rgb = mix(out_rgb, wire_color, wire * 0.92);
    }

    return vec4<f32>(out_rgb, 1.0);
}

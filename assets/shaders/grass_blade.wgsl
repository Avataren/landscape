// GPU-driven grass blade shader — two LOD passes, three PBR texture variants.
//
// Placement: vertex_index → blade_idx → world XZ via snapped camera grid,
//            heightmap LOD 0 sample → world Y.
// Appearance: one of 3 grass variants selected per blade (stable hash).
//             diffuse + opacity for colour/cutout, normal map for lighting,
//             specular for shininess.

#import bevy_pbr::view_transformations::position_world_to_clip
#import bevy_pbr::{
    mesh_view_bindings::{lights, view, clusterable_objects},
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

// ── Constants ─────────────────────────────────────────────────────────────────
const VERTS_PER_BLADE: u32 = 12u;

// ── Uniforms & textures ───────────────────────────────────────────────────────
struct GrassParams {
    camera_grid:  vec4<f32>,  // xy=cam XZ, z=grid_size, w=spacing
    clip_level:   vec4<f32>,  // xy=ring_center XZ, z=inv_span, w=texel_ws
    blade:        vec4<f32>,  // x=inner_radius_sq, y=height, z=width, w=slope_max
    alt_wind:     vec4<f32>,  // x=alt_min, y=alt_max, z=wind_time, w=wind_strength
    wind_color:   vec4<f32>,  // x=wind_scale, yz=unused, w=debug_mode
    world_bounds: vec4<f32>,  // xy=world_min XZ, zw=world_max XZ
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:       texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params:  GrassParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var diffuse_arr:      texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var diffuse_samp:     sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var normal_arr:       texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var normal_samp:      sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var opacity_arr:      texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var opacity_samp:     sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var specular_arr:     texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var specular_samp:    sampler;

// ── Vertex output ─────────────────────────────────────────────────────────────
struct VertexOutput {
    @builtin(position)                      clip_pos:  vec4<f32>,
    @location(0)                            world_pos: vec3<f32>,
    @location(1)                            world_n:   vec3<f32>,
    @location(2)                            blade_uv:  vec2<f32>,
    @location(3)                            tangent:   vec3<f32>,
    @location(4)                            bitangent: vec3<f32>,
    @location(5) @interpolate(flat)         variant:   u32,
}

// ── Hash helpers ──────────────────────────────────────────────────────────────
fn uhash(n: u32) -> u32 {
    var x = n;
    x ^= x >> 17u; x *= 0xbf324c81u;
    x ^= x >> 11u; x *= 0x68bc9c4du;
    x ^= x >> 16u;
    return x;
}
fn fhash1(n: u32) -> f32       { return f32(uhash(n)) * (1.0 / 4294967296.0); }
fn fhash2(n: u32) -> vec2<f32> { return vec2<f32>(fhash1(n), fhash1(n + 1u)); }

// ── Toroidal clipmap LOD-0 sample ─────────────────────────────────────────────
fn sample_height(xz: vec2<f32>) -> f32 {
    let wmin = params.world_bounds.xy;
    let wmax = params.world_bounds.zw;
    if any(xz < wmin) || any(xz > wmax) { return -1e9; }

    // clip_level.x stores the clipmap array layer (LOD index).
    let layer    = i32(params.clip_level.x);
    let inv_span = params.clip_level.z;
    let texel_ws = params.clip_level.w;
    let dims_u   = textureDimensions(height_tex, 0).xy;
    let dims_i   = vec2<i32>(dims_u);
    let dims_f   = vec2<f32>(dims_u);

    let sxz   = clamp(xz, wmin, wmax - vec2<f32>(texel_ws));
    let uv    = fract((sxz + 0.5 * texel_ws) * inv_span);
    let coord = uv * dims_f - vec2<f32>(0.5);
    let i0_f  = floor(coord);
    let f     = coord - i0_f;
    let i0    = vec2<i32>(i0_f);
    let x0 = ((i0.x % dims_i.x) + dims_i.x) % dims_i.x;
    let y0 = ((i0.y % dims_i.y) + dims_i.y) % dims_i.y;
    let x1 = (x0 + 1) % dims_i.x;
    let y1 = (y0 + 1) % dims_i.y;

    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), layer, 0).r;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), layer, 0).r;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y1), layer, 0).r;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y1), layer, 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// ── Vertex shader ─────────────────────────────────────────────────────────────
@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let OFFSCREEN = vec4<f32>(2.0, 2.0, 0.0, 1.0);

    let blade_idx  = vid / VERTS_PER_BLADE;
    let local_vert = vid % VERTS_PER_BLADE;

    let camera_x        = params.camera_grid.x;
    let camera_z        = params.camera_grid.y;
    let grid_size       = u32(params.camera_grid.z);
    let spacing         = params.camera_grid.w;
    let inner_radius_sq = params.blade.x;
    var blade_height    = params.blade.y;
    let blade_width     = params.blade.z;
    let slope_max       = params.blade.w;
    let alt_min         = params.alt_wind.x;
    let alt_max         = params.alt_wind.y;
    let wind_time       = params.alt_wind.z;
    let wind_strength   = params.alt_wind.w;
    let wind_scale      = params.wind_color.x;

    let gx = blade_idx % grid_size;
    let gz = blade_idx / grid_size;
    if gx >= grid_size || gz >= grid_size {
        out.clip_pos = OFFSCREEN; return out;
    }

    // World-stable integer cell coordinates.
    let base_cell_x = i32(floor(camera_x / spacing));
    let base_cell_z = i32(floor(camera_z / spacing));
    let half_i      = i32(grid_size / 2u);
    let wcx         = u32(base_cell_x + i32(gx) - half_i);
    let wcz         = u32(base_cell_z + i32(gz) - half_i);
    let stable_seed = (wcx * 2654435761u) ^ (wcz * 2246822519u + 1u);

    let snap_x = f32(base_cell_x) * spacing;
    let snap_z = f32(base_cell_z) * spacing;
    let half_f = f32(half_i);

    var wx = snap_x + (f32(gx) - half_f) * spacing;
    var wz = snap_z + (f32(gz) - half_f) * spacing;

    let jitter = (fhash2(stable_seed) - 0.5) * spacing * 0.65;
    wx += jitter.x;
    wz += jitter.y;

    // Inner radius cull + outer density fade (far LOD only).
    if inner_radius_sq > 0.0 {
        let dx = wx - camera_x; let dz = wz - camera_z;
        let dist_sq = dx * dx + dz * dz;
        if dist_sq < inner_radius_sq {
            out.clip_pos = OFFSCREEN; return out;
        }
        // Fade density linearly from 1 at inner edge to 0 at outer edge.
        let dist      = sqrt(dist_sq);
        let inner_r   = sqrt(inner_radius_sq);
        let outer_r   = f32(grid_size) * 0.5 * spacing;
        let keep_prob = 1.0 - smoothstep(inner_r, outer_r, dist);
        if fhash1(stable_seed ^ 0xf4d301u) > keep_prob {
            out.clip_pos = OFFSCREEN; return out;
        }
    }

    // Height sample.
    let wy_raw = sample_height(vec2<f32>(wx, wz));
    if wy_raw < -1e8 { out.clip_pos = OFFSCREEN; return out; }
    let wy = wy_raw;

    if wy < alt_min || wy > alt_max { out.clip_pos = OFFSCREEN; return out; }

    // Slope filter with soft fade.
    if slope_max < 90.0 {
        let step_ws = params.clip_level.w * 2.0;
        let hx0 = sample_height(vec2<f32>(wx - step_ws, wz));
        let hx1 = sample_height(vec2<f32>(wx + step_ws, wz));
        let hz0 = sample_height(vec2<f32>(wx, wz - step_ws));
        let hz1 = sample_height(vec2<f32>(wx, wz + step_ws));
        let sx    = (hx1 - hx0) / (2.0 * step_ws);
        let sz    = (hz1 - hz0) / (2.0 * step_ws);
        let slope = length(vec2<f32>(sx, sz));
        if slope >= slope_max { out.clip_pos = OFFSCREEN; return out; }
        // Probabilistic thinning on steeper slopes — cull blades rather than squash them.
        let keep_prob = 1.0 - smoothstep(slope_max * 0.5, slope_max, slope);
        if fhash1(stable_seed ^ 0x9e3779b9u) > keep_prob { out.clip_pos = OFFSCREEN; return out; }
    }

    // Random Y rotation (world-stable).
    let rot_y = fhash1(stable_seed ^ 0xdeadbeefu) * 6.28318530718;
    let cos_r = cos(rot_y);
    let sin_r = sin(rot_y);

    // Texture variant (0, 1 or 2).
    let variant = uhash(stable_seed ^ 0x12345678u) % 3u;

    // Blade geometry — 2 crossed quads, 6 verts each.
    let quad_idx = local_vert / 6u;
    let tri_vert = local_vert % 6u;

    // Two crossed rectangular quads matching the grass_01.obj geometry:
    // bottom-left, bottom-right, top-right | bottom-left, top-right, top-left
    var uv_s = vec2<f32>(0.0, 0.0);
    switch tri_vert {
        case 0u: { uv_s = vec2<f32>(-0.5, 0.0); }
        case 1u: { uv_s = vec2<f32>( 0.5, 0.0); }
        case 2u: { uv_s = vec2<f32>( 0.5, 1.0); }
        case 3u: { uv_s = vec2<f32>(-0.5, 0.0); }
        case 4u: { uv_s = vec2<f32>( 0.5, 1.0); }
        case 5u: { uv_s = vec2<f32>(-0.5, 1.0); }
        default: {}
    }
    let v_height = uv_s.y;

    // Wind.
    let wind_wx = sin(wx * wind_scale + wind_time * 2.5) * cos(wz * wind_scale * 0.7 + wind_time * 1.8);
    let wind_wz = cos(wx * wind_scale * 0.6 + wind_time * 2.1) * sin(wz * wind_scale + wind_time * 2.9);
    let disp_x  = wind_wx * wind_strength * v_height * v_height;
    let disp_z  = wind_wz * wind_strength * 0.3 * v_height * v_height;

    var lx = uv_s.x * blade_width;
    var lz = 0.0;
    let ly = v_height * blade_height;
    if quad_idx == 1u { let t = lx; lx = lz; lz = t; }

    let rx = lx * cos_r - lz * sin_r + disp_x;
    let rz = lx * sin_r + lz * cos_r + disp_z;

    let world_pos = vec3<f32>(wx + rx, wy + ly, wz + rz);

    // TBN — tangent = horizontal axis of this quad, bitangent = world up.
    // Quad A extends along local X; Quad B extends along local Z.
    var local_tangent = vec3<f32>(1.0, 0.0, 0.0);
    if quad_idx == 1u { local_tangent = vec3<f32>(0.0, 0.0, 1.0); }
    let tangent = normalize(vec3<f32>(
        local_tangent.x * cos_r - local_tangent.z * sin_r, 0.0,
        local_tangent.x * sin_r + local_tangent.z * cos_r,
    ));
    let bitangent = vec3<f32>(0.0, 1.0, 0.0);
    // Face normal = cross(T, B): T=(tx,0,tz), B=(0,1,0)
    // cross = (Ty*Bz - Tz*By, Tz*Bx - Tx*Bz, Tx*By - Ty*Bx)
    //       = (0 - tz*1, tz*0 - tx*0, tx*1 - 0)
    //       = (-tz, 0, tx)
    // Tilt slightly upward (+Y) so blades self-light naturally from above.
    let world_n = normalize(vec3<f32>(-tangent.z, 0.3, tangent.x));

    out.clip_pos  = position_world_to_clip(world_pos);
    out.world_pos = world_pos;
    out.world_n   = world_n;
    // V is flipped: V=0 in image space is the top, so grass tips map to V=0.
    out.blade_uv  = vec2<f32>(uv_s.x + 0.5, 1.0 - v_height);
    out.tangent   = tangent;
    out.bitangent = bitangent;
    out.variant   = variant;
    return out;
}

// ── Lighting helpers ──────────────────────────────────────────────────────────

fn distance_attenuation(dist_sq: f32, inv_range_sq: f32) -> f32 {
    let factor        = dist_sq * inv_range_sq;
    let smooth_factor = saturate(1.0 - factor * factor);
    let falloff       = smooth_factor * smooth_factor;
    return falloff / max(dist_sq, 1e-4);
}

// ── Fragment shader ───────────────────────────────────────────────────────────
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv  = in.blade_uv;
    let vi  = i32(in.variant);

    // Opacity / alpha cutout first — cheapest discard.
    let opacity = textureSample(opacity_arr, opacity_samp, uv, vi).r;
    if opacity < 0.3 { discard; }

    // Diffuse colour.
    let diffuse = textureSample(diffuse_arr, diffuse_samp, uv, vi).rgb;

    // Tangent-space normal → world normal.
    let nm_ts = textureSample(normal_arr, normal_samp, uv, vi).xyz * 2.0 - 1.0;
    let T     = normalize(in.tangent);
    let B     = normalize(in.bitangent);
    // N_geo: geometry normal from vertex — used for shadow acne bias.
    let N_geo = normalize(in.world_n);
    // world_n: normal-mapped normal — used for all lighting calculations.
    let world_n = normalize(T * nm_ts.x + B * nm_ts.y + N_geo * nm_ts.z);

    // Debug mode: 1 = normals as colour (mirrors terrain F8 debug).
    let debug_mode = params.wind_color.w;
    if debug_mode >= 0.5 && debug_mode < 1.5 {
        return vec4<f32>(world_n * 0.5 + vec3<f32>(0.5), 1.0);
    }

    // Specular map value.
    let spec_val = textureSample(specular_arr, specular_samp, uv, vi).r;

    // view-space depth for shadow cascade selection and cluster z-slice lookup.
    let view_z = dot(
        vec4<f32>(view.view_from_world[0].z, view.view_from_world[1].z,
                  view.view_from_world[2].z, view.view_from_world[3].z),
        vec4<f32>(in.world_pos, 1.0),
    );
    let is_orthographic = view.clip_from_view[3].w == 1.0;
    let view_dir = normalize(view.world_position.xyz - in.world_pos);

    var direct = vec3<f32>(0.0);

    // --- Directional lights (sun etc.) ---
    for (var i: u32 = 0u; i < lights.n_directional_lights; i++) {
        let light = lights.directional_lights[i];
        let ndotl = max(dot(world_n, light.direction_to_light), 0.0);

        var shadow = 1.0;
        if (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            // Use geometry normal for shadow bias (more stable than normal-mapped).
            shadow = fetch_directional_shadow(i, vec4<f32>(in.world_pos, 1.0), N_geo, view_z);
        }

        let half_vec = normalize(light.direction_to_light + view_dir);
        let spec = pow(max(0.0, dot(world_n, half_vec)), 32.0) * spec_val * 0.3;
        direct += (diffuse * ndotl + vec3<f32>(spec)) * shadow * light.color.rgb;
    }

    // --- Clustered point lights ---
    let cluster_index = fragment_cluster_index(in.clip_pos.xy, view_z, is_orthographic);
    let ranges = unpack_clusterable_object_index_ranges(cluster_index);

    for (var i: u32 = ranges.first_point_light_index_offset;
             i < ranges.first_spot_light_index_offset; i++) {
        let light_id = get_clusterable_object_id(i);
        let light    = &clusterable_objects.data[light_id];
        let to_frag  = (*light).position_radius.xyz - in.world_pos;
        let dist_sq  = dot(to_frag, to_frag);
        let L        = normalize(to_frag);
        let ndotl    = max(dot(world_n, L), 0.0);
        let atten    = distance_attenuation(dist_sq, (*light).color_inverse_square_range.w);

        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_point_shadow(light_id, vec4<f32>(in.world_pos, 1.0), N_geo);
        }

        direct += diffuse * ndotl * atten * shadow * (*light).color_inverse_square_range.rgb;
    }

    // --- Spot lights ---
    for (var i: u32 = ranges.first_spot_light_index_offset;
             i < ranges.first_reflection_probe_index_offset; i++) {
        let light_id = get_clusterable_object_id(i);
        let light    = &clusterable_objects.data[light_id];
        let to_frag  = (*light).position_radius.xyz - in.world_pos;
        let dist_sq  = dot(to_frag, to_frag);
        let L        = normalize(to_frag);
        let ndotl    = max(dot(world_n, L), 0.0);
        let atten    = distance_attenuation(dist_sq, (*light).color_inverse_square_range.w);

        var spot_dir = vec3<f32>((*light).light_custom_data.x, 0.0, (*light).light_custom_data.y);
        spot_dir.y = sqrt(max(0.0, 1.0 - spot_dir.x * spot_dir.x - spot_dir.z * spot_dir.z));
        if ((*light).flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
            spot_dir.y = -spot_dir.y;
        }
        let cd         = dot(-spot_dir, L);
        let cone_atten = saturate(cd * (*light).light_custom_data.z + (*light).light_custom_data.w);
        let spot_atten = cone_atten * cone_atten;

        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_spot_shadow(
                light_id, vec4<f32>(in.world_pos, 1.0), N_geo, (*light).shadow_map_near_z,
            );
        }

        direct += diffuse * ndotl * atten * spot_atten * shadow
            * (*light).color_inverse_square_range.rgb;
    }

    // --- Ambient (hemisphere + scene ambient) ---
    let sky_t      = saturate(world_n.y * 0.5 + 0.5);
    let sky_col    = vec3<f32>(0.15, 0.25, 0.40);
    let gnd_col    = vec3<f32>(0.04, 0.03, 0.02);
    let hemisphere = mix(gnd_col, sky_col, sky_t);
    let ambient    = diffuse * (hemisphere / max(view.exposure, 1e-10) + lights.ambient_color.rgb);

    let color = (direct + ambient) * view.exposure;
    return vec4<f32>(color, opacity);
}

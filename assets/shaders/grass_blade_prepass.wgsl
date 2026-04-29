// grass_blade_prepass.wgsl — depth / shadow prepass for GPU grass blades.
//
// Reproduces the same blade geometry as grass_blade.wgsl so the depth written
// to shadow maps matches the lit geometry exactly.  The fragment stage samples
// the opacity array and discards transparent pixels so shadow edges respect the
// alpha cutout mask.

#import bevy_pbr::view_transformations::position_world_to_clip

// ── Constants ─────────────────────────────────────────────────────────────────
const VERTS_PER_BLADE: u32 = 12u;

// ── Uniforms & textures ───────────────────────────────────────────────────────
struct GrassParams {
    camera_grid:  vec4<f32>,  // xy=cam XZ, z=grid_size, w=spacing
    clip_level:   vec4<f32>,  // xy=ring_center XZ, z=inv_span, w=texel_ws
    blade:        vec4<f32>,  // x=inner_radius_sq, y=height, z=width, w=slope_max
    alt_wind:     vec4<f32>,  // x=alt_min, y=alt_max, z=wind_time, w=wind_strength
    wind_color:   vec4<f32>,  // x=wind_scale, yzw=fallback RGB
    world_bounds: vec4<f32>,  // xy=world_min XZ, zw=world_max XZ
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex:      texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: GrassParams;
// Opacity array — bindings 6 & 7 match the main grass_blade.wgsl layout.
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var opacity_arr:     texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var opacity_samp:    sampler;

// ── Vertex output ─────────────────────────────────────────────────────────────
struct VertexOutput {
    @builtin(position)              clip_pos: vec4<f32>,
    @location(0)                    blade_uv: vec2<f32>,
    @location(1) @interpolate(flat) variant:  u32,
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
    let blade_height    = params.blade.y;
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

    if inner_radius_sq > 0.0 {
        let dx = wx - camera_x; let dz = wz - camera_z;
        let dist_sq = dx * dx + dz * dz;
        if dist_sq < inner_radius_sq {
            out.clip_pos = OFFSCREEN; return out;
        }
        let dist      = sqrt(dist_sq);
        let inner_r   = sqrt(inner_radius_sq);
        let outer_r   = f32(grid_size) * 0.5 * spacing;
        let keep_prob = 1.0 - smoothstep(inner_r, outer_r, dist);
        if fhash1(stable_seed ^ 0xf4d301u) > keep_prob {
            out.clip_pos = OFFSCREEN; return out;
        }
    }

    let wy_raw = sample_height(vec2<f32>(wx, wz));
    if wy_raw < -1e8 { out.clip_pos = OFFSCREEN; return out; }
    let wy = wy_raw;

    if wy < alt_min || wy > alt_max { out.clip_pos = OFFSCREEN; return out; }

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
        let keep_prob = 1.0 - smoothstep(slope_max * 0.5, slope_max, slope);
        if fhash1(stable_seed ^ 0x9e3779b9u) > keep_prob { out.clip_pos = OFFSCREEN; return out; }
    }

    let rot_y = fhash1(stable_seed ^ 0xdeadbeefu) * 6.28318530718;
    let cos_r = cos(rot_y);
    let sin_r = sin(rot_y);

    let variant = uhash(stable_seed ^ 0x12345678u) % 3u;

    let quad_idx = local_vert / 6u;
    let tri_vert = local_vert % 6u;

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

    out.clip_pos = position_world_to_clip(world_pos);
    // V is flipped: V=0 in image space is the top.
    out.blade_uv = vec2<f32>(uv_s.x + 0.5, 1.0 - v_height);
    out.variant  = variant;
    return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
// Discard transparent pixels so shadow maps respect the alpha cutout mask.
// Void return works for both depth-only shadow passes and the depth prepass.
// Grass blades don't contribute normal or motion-vector prepass outputs.
@fragment
fn fragment(in: VertexOutput) {
    let opacity = textureSample(opacity_arr, opacity_samp, in.blade_uv, i32(in.variant)).r;
    if opacity < 0.3 { discard; }
}

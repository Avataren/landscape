// GPU-driven grass blade shader — no pre-baking, renders every frame.
//
// Each blade = 12 vertices (2 crossed quads × 2 triangles).
// The vertex shader decodes blade / intra-blade indices from
// @builtin(vertex_index) and places each blade by sampling LOD 0 of the
// clipmap height texture array (R32Float, world-space metres).
//
// Jitter and rotation are keyed on world-stable integer grid-cell coordinates
// so blades never shuffle as the camera moves.

#import bevy_pbr::view_transformations::position_world_to_clip

// ── Constants ─────────────────────────────────────────────────────────────────
const VERTS_PER_BLADE: u32 = 12u;
const BLADE_TAPER:     f32 = 0.15;   // tip width relative to base width

// ── Material bindings ─────────────────────────────────────────────────────────
// 6 × vec4 = 96 bytes; all fields are 4-byte aligned → no padding gaps.
struct GrassParams {
    // xy = camera XZ world pos,  z = grid_size (as f32),  w = cell spacing (m)
    camera_grid:  vec4<f32>,
    // LOD 0 clip level: xy = ring_center XZ,  z = inv_ring_span,  w = texel_world_size
    clip_level:   vec4<f32>,
    // x = UNUSED,  y = blade_height,  z = blade_width,  w = slope_max
    blade:        vec4<f32>,
    // x = alt_min (m),  y = alt_max (m),  z = wind_time (s),  w = wind_strength
    alt_wind:     vec4<f32>,
    // x = wind_scale (spatial freq),  yzw = base RGB colour (linear)
    wind_color:   vec4<f32>,
    // xy = world_min XZ,  zw = world_max XZ
    world_bounds: vec4<f32>,
}

// Clipmap height array (R32Float, world-space metres).  Only layer 0 is read.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var height_tex: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: GrassParams;

// ── Vertex output ─────────────────────────────────────────────────────────────
struct VertexOutput {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       normal:    vec3<f32>,
    @location(2)       blade_uv:  vec2<f32>,  // x: 0=left 1=right, y: 0=base 1=tip
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

// ── Toroidal LOD-0 height sample ──────────────────────────────────────────────
// Mirrors the terrain_vertex.wgsl height_at() function.
// Returns height in world-space metres, or -1e9 if outside world bounds.
fn sample_height(xz: vec2<f32>) -> f32 {
    let wmin = params.world_bounds.xy;
    let wmax = params.world_bounds.zw;
    if any(xz < wmin) || any(xz > wmax) {
        return -1e9;
    }

    let ring_center = params.clip_level.xy;
    let inv_span    = params.clip_level.z;
    let texel_ws    = params.clip_level.w;

    let dims_u = textureDimensions(height_tex, 0).xy;
    let dims_i = vec2<i32>(dims_u);
    let dims_f = vec2<f32>(dims_u);

    let sample_xz = clamp(xz, wmin, wmax - vec2<f32>(texel_ws));
    let uv        = fract((sample_xz + 0.5 * texel_ws) * inv_span);
    let coord     = uv * dims_f - vec2<f32>(0.5);
    let i0_f      = floor(coord);
    let f         = coord - i0_f;

    let i0 = vec2<i32>(i0_f);
    let x0 = ((i0.x % dims_i.x) + dims_i.x) % dims_i.x;
    let y0 = ((i0.y % dims_i.y) + dims_i.y) % dims_i.y;
    let x1 = (x0 + 1) % dims_i.x;
    let y1 = (y0 + 1) % dims_i.y;

    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), 0, 0).r;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), 0, 0).r;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y1), 0, 0).r;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y1), 0, 0).r;

    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// ── Vertex shader ─────────────────────────────────────────────────────────────
@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    // Off-screen position used to cull rejected blades.
    let OFFSCREEN = vec4<f32>(2.0, 2.0, 0.0, 1.0);

    let blade_idx  = vid / VERTS_PER_BLADE;
    let local_vert = vid % VERTS_PER_BLADE;

    // ── Unpack params ─────────────────────────────────────────────────────────
    let camera_x   = params.camera_grid.x;
    let camera_z   = params.camera_grid.y;
    let grid_size  = u32(params.camera_grid.z);
    let spacing    = params.camera_grid.w;

    let inner_radius_sq = params.blade.x;
    var blade_height    = params.blade.y;
    let blade_width     = params.blade.z;
    let slope_max       = params.blade.w;

    let alt_min       = params.alt_wind.x;
    let alt_max       = params.alt_wind.y;
    let wind_time     = params.alt_wind.z;
    let wind_strength = params.alt_wind.w;
    let wind_scale    = params.wind_color.x;

    // ── Grid cell XZ ──────────────────────────────────────────────────────────
    let gx = blade_idx % grid_size;
    let gz = blade_idx / grid_size;

    // Blades beyond the active grid (mesh is pre-allocated at max size).
    if gx >= grid_size || gz >= grid_size {
        out.clip_pos = OFFSCREEN;
        return out;
    }

    // Snap camera to grid so the entire patch shifts one cell at a time.
    let base_cell_x = i32(floor(camera_x / spacing));
    let base_cell_z = i32(floor(camera_z / spacing));
    let half_i      = i32(grid_size / 2u);

    // World-stable integer cell coordinates: same value for a given world cell
    // regardless of camera position.  Used to fix jitter and rotation so blades
    // don't shuffle when the camera moves.
    let wcx = u32(base_cell_x + i32(gx) - half_i);
    let wcz = u32(base_cell_z + i32(gz) - half_i);
    let stable_seed = (wcx * 2654435761u) ^ (wcz * 2246822519u + 1u);

    let snap_x = f32(base_cell_x) * spacing;
    let snap_z = f32(base_cell_z) * spacing;
    let half_f = f32(half_i);

    var wx = snap_x + (f32(gx) - half_f) * spacing;
    var wz = snap_z + (f32(gz) - half_f) * spacing;

    // Per-blade jitter keyed on world cell — blades stay put as camera moves.
    let jitter = (fhash2(stable_seed) - 0.5) * spacing * 0.65;
    wx += jitter.x;
    wz += jitter.y;

    // ── Inner radius cull (far LOD only) ──────────────────────────────────────
    // Discard blades inside the near-LOD coverage radius so the two passes
    // don't overlap. inner_radius_sq == 0 on the near entity → no cull.
    if inner_radius_sq > 0.0 {
        let dx = wx - camera_x;
        let dz = wz - camera_z;
        if (dx * dx + dz * dz) < inner_radius_sq {
            out.clip_pos = OFFSCREEN;
            return out;
        }
    }

    // ── Sample clipmap LOD 0 ──────────────────────────────────────────────────
    let wy_raw = sample_height(vec2<f32>(wx, wz));
    if wy_raw < -1e8 {
        // Outside world bounds.
        out.clip_pos = OFFSCREEN;
        return out;
    }
    let wy = wy_raw;   // already in world-space metres

    // ── Altitude filter ───────────────────────────────────────────────────────
    if wy < alt_min || wy > alt_max {
        out.clip_pos = OFFSCREEN;
        return out;
    }

    // ── Slope filter with soft fade ───────────────────────────────────────────
    // Hard discard above slope_max; soft scale of blade_height in the fade
    // zone [slope_max*0.5 .. slope_max] so grass tapers off naturally on slopes.
    if slope_max < 90.0 {
        let texel_ws = params.clip_level.w;
        let step_ws  = texel_ws * 2.0;
        let hx0 = sample_height(vec2<f32>(wx - step_ws, wz));
        let hx1 = sample_height(vec2<f32>(wx + step_ws, wz));
        let hz0 = sample_height(vec2<f32>(wx, wz - step_ws));
        let hz1 = sample_height(vec2<f32>(wx, wz + step_ws));
        let sx    = (hx1 - hx0) / (2.0 * step_ws);
        let sz    = (hz1 - hz0) / (2.0 * step_ws);
        let slope = length(vec2<f32>(sx, sz));
        if slope >= slope_max {
            out.clip_pos = OFFSCREEN;
            return out;
        }
        // Shrink blade height toward zero as slope approaches the threshold.
        let fade_start = slope_max * 0.5;
        let slope_fade = 1.0 - smoothstep(fade_start, slope_max, slope);
        blade_height *= slope_fade;
    }

    // ── Random Y rotation per blade (world-stable) ────────────────────────────
    let rot_y = fhash1(stable_seed ^ 0xdeadbeefu) * 6.28318530718;
    let cos_r = cos(rot_y);
    let sin_r = sin(rot_y);

    // ── Blade geometry ────────────────────────────────────────────────────────
    // 12 verts = 2 quads × 2 triangles × 3 verts
    // Quad A (local_vert 0-5): blade along local X  (faces ±Z)
    // Quad B (local_vert 6-11): blade along local Z  (faces ±X)
    //
    // Triangle winding per quad (6 verts, CCW front):
    //   [0]=BL [1]=BR [2]=TR   |  [3]=BL [4]=TR [5]=TL
    let quad_idx = local_vert / 6u;
    let tri_vert = local_vert % 6u;

    var uv_s = vec2<f32>(0.0, 0.0);
    switch tri_vert {
        case 0u: { uv_s = vec2<f32>(-0.5,                0.0); }   // BL
        case 1u: { uv_s = vec2<f32>( 0.5,                0.0); }   // BR
        case 2u: { uv_s = vec2<f32>( 0.5 * BLADE_TAPER,  1.0); }   // TR
        case 3u: { uv_s = vec2<f32>(-0.5,                0.0); }   // BL (repeat)
        case 4u: { uv_s = vec2<f32>( 0.5 * BLADE_TAPER,  1.0); }   // TR (repeat)
        case 5u: { uv_s = vec2<f32>(-0.5 * BLADE_TAPER,  1.0); }   // TL
        default: {}
    }

    let v_height = uv_s.y;

    // Wind: two-axis wave, scales with height² (tips sway more).
    let wind_wx = sin(wx * wind_scale       + wind_time * 2.5)
                * cos(wz * wind_scale * 0.7 + wind_time * 1.8);
    let wind_wz = cos(wx * wind_scale * 0.6 + wind_time * 2.1)
                * sin(wz * wind_scale       + wind_time * 2.9);
    let disp_x  = wind_wx * wind_strength * v_height * v_height;
    let disp_z  = wind_wz * wind_strength * 0.3 * v_height * v_height;

    var lx = uv_s.x * blade_width;
    var lz = 0.0;
    let ly = v_height * blade_height;

    if quad_idx == 1u {
        let tmp = lx;
        lx = lz;
        lz = tmp;
    }

    let rx = lx * cos_r - lz * sin_r + disp_x;
    let rz = lx * sin_r + lz * cos_r + disp_z;

    let world_pos = vec3<f32>(wx + rx, wy + ly, wz + rz);

    // ── Normal ────────────────────────────────────────────────────────────────
    var base_n = vec3<f32>(0.0, 0.0, 1.0);
    if quad_idx == 1u {
        base_n = vec3<f32>(1.0, 0.0, 0.0);
    }
    let nx = base_n.x * cos_r - base_n.z * sin_r;
    let nz = base_n.x * sin_r + base_n.z * cos_r;
    let world_normal = normalize(vec3<f32>(nx, 0.3, nz));

    out.clip_pos  = position_world_to_clip(world_pos);
    out.world_pos = world_pos;
    out.normal    = world_normal;
    out.blade_uv  = vec2<f32>(uv_s.x + 0.5, v_height);

    return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let tip_t  = in.blade_uv.y;
    let horiz  = in.blade_uv.x - 0.5;

    let half_w = mix(0.5, 0.5 * BLADE_TAPER, tip_t) + 0.04;
    if abs(horiz) > half_w { discard; }

    let base_col = params.wind_color.yzw;
    let tip_col  = base_col * vec3<f32>(1.35, 1.2, 0.65);
    var color    = mix(base_col * 0.5, tip_col, tip_t);

    let sun_dir  = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let ndotl    = max(0.0, dot(in.normal, sun_dir));
    color       *= 0.35 + 0.65 * ndotl;

    let alpha = 1.0 - smoothstep(0.82, 1.0, tip_t);

    return vec4<f32>(color, alpha);
}

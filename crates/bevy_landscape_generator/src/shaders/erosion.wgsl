// Hydraulic + thermal erosion compute shader.
// Based on Jakó & Tóth "Fast Hydraulic and Thermal Erosion on the GPU" (CESCG 2011)
// which extends Mei, Decaudin & Hu "Fast Hydraulic Erosion Simulation on GPU" (PG 2007).

struct ErosionParams {
    resolution:           vec2<u32>,   // offset  0
    dt:                   f32,         // offset  8
    gravity:              f32,         // offset 12
    pipe_length:          f32,         // offset 16
    rain_rate:            f32,         // offset 20
    evaporation_rate:     f32,         // offset 24
    sediment_capacity:    f32,         // offset 28
    erosion_rate:         f32,         // offset 32
    deposition_rate:      f32,         // offset 36
    min_slope:            f32,         // offset 40
    hardness_influence:   f32,         // offset 44
    repose_angle_radians: f32,         // offset 48
    talus_rate:           f32,         // offset 52
    num_particles:        u32,         // offset 56
    max_steps:            u32,         // offset 60
    inertia:              f32,         // offset 64
    frame_seed:           u32,         // offset 68
    pipe_area:            f32,         // offset 72
    erosion_depth_max:    f32,         // offset 76
    // Total: 80 bytes, 16-byte aligned, no padding needed
}

@group(0) @binding(0) var<uniform> params: ErosionParams;

@group(1) @binding(0) var height_a:     texture_storage_2d<r32float,    read_write>;
@group(1) @binding(1) var height_b:     texture_storage_2d<r32float,    read_write>;
@group(1) @binding(2) var water_tex:    texture_storage_2d<r32float,    read_write>;
@group(1) @binding(3) var sediment_tex: texture_storage_2d<r32float,    read_write>;
@group(1) @binding(4) var flux_tex:     texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(5) var velocity_tex: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(6) var hardness_tex: texture_storage_2d<r32float,    read_write>;
@group(1) @binding(7) var<storage, read_write> delta_height: array<atomic<i32>>;
@group(1) @binding(8) var raw_heights:  texture_storage_2d<r32float,    read_write>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rx() -> i32 { return i32(params.resolution.x); }
fn ry() -> i32 { return i32(params.resolution.y); }

fn cc(c: vec2<i32>) -> vec2<i32> {
    return clamp(c, vec2<i32>(0, 0), vec2<i32>(rx() - 1, ry() - 1));
}

fn load_ha(c: vec2<i32>) -> f32 {
    return textureLoad(height_a, cc(c)).r;
}

fn load_water(c: vec2<i32>) -> f32 {
    return textureLoad(water_tex, cc(c)).r;
}

fn surface_h(c: vec2<i32>) -> f32 {
    return load_ha(c) + load_water(c);
}

// For the flux update: out-of-bounds cells are open drains.
// Returns only the bed height (no water) so water always flows out at boundaries.
fn surface_h_open(c: vec2<i32>) -> f32 {
    if c.x < 0 || c.y < 0 || c.x >= rx() || c.y >= ry() {
        // Clamp to edge bed height so only the water layer drains, not a cliff.
        let ec = clamp(c, vec2<i32>(0), vec2<i32>(rx() - 1, ry() - 1));
        return load_ha(ec);
    }
    return surface_h(c);
}

fn load_flux(c: vec2<i32>) -> vec4<f32> {
    return textureLoad(flux_tex, cc(c));
}

// Resolution compensation factor: all slopes are measured in pixel-grid
// units (Δh / pixel).  At higher resolutions the same physical terrain has
// smaller per-pixel Δh.  Multiplying by res_scale converts pixel-gradient
// to a value that is consistent with the 512² reference resolution.
fn res_scale() -> f32 {
    return f32(params.resolution.x) / 512.0;
}

fn gradient2(c: vec2<i32>) -> vec2<f32> {
    let inv_2l = 0.5 / params.pipe_length;
    let gx = (load_ha(c + vec2<i32>(1, 0)) - load_ha(c + vec2<i32>(-1, 0))) * inv_2l;
    let gy = (load_ha(c + vec2<i32>(0, 1)) - load_ha(c + vec2<i32>(0, -1))) * inv_2l;
    return vec2<f32>(gx, gy) * res_scale();
}

// IQ-style gradient hash for hardness noise.
fn hash2n(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

fn grad_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(hash2n(i + vec2<f32>(0.0, 0.0)), f - vec2<f32>(0.0, 0.0));
    let b = dot(hash2n(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
    let c = dot(hash2n(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
    let d = dot(hash2n(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ---------------------------------------------------------------------------
// Init passes (run once per erosion session)
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn erosion_copy_in(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let h = textureLoad(raw_heights, coord).r;
    textureStore(height_a, coord, vec4<f32>(h, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn erosion_init_hardness(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let uv = vec2<f32>(f32(id.x), f32(id.y))
           / vec2<f32>(f32(params.resolution.x), f32(params.resolution.y));
    let seed_off = vec2<f32>(f32(params.frame_seed) * 0.2731,
                              f32(params.frame_seed) * 0.1571);
    let n0 = grad_noise(uv * 6.0 + seed_off);
    let n1 = grad_noise(uv * 12.0 + seed_off * 1.7);
    // Hardness: 0 = hard (resistant), 1 = soft (easily eroded).
    // Higher terrain → harder (matches paper: high-altitude material is usually harder).
    let h = textureLoad(raw_heights, coord).r;
    // Clamp h to [0,1] for the hardness formula so that deposited terrain
    // (h up to 1.5) does not invert the "higher = harder" gradient.
    let h01 = min(h, 1.0);
    let base = clamp(0.5 + n0 * 0.3 + n1 * 0.1, 0.0, 1.0);
    let hardness = clamp(base * (1.0 - h01 * 0.4), 0.0, 1.0);
    textureStore(hardness_tex, coord, vec4<f32>(hardness, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn erosion_clear_buffers(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord  = vec2<i32>(id.xy);
    let zero4  = vec4<f32>(0.0);
    textureStore(water_tex,    coord, zero4);
    textureStore(sediment_tex, coord, zero4);
    textureStore(flux_tex,     coord, zero4);
    textureStore(velocity_tex, coord, zero4);
    textureStore(height_b,     coord, zero4);
}

// ---------------------------------------------------------------------------
// Finalize — copy eroded height_a back to raw_heights for display
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn erosion_copy_out(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    // Write the full eroded height without clamping to 1.0.  Deposition can
    // raise height_a up to 1.5 (its internal ceiling); clipping that to 1.0
    // collapses variation in sediment-filled areas, causing subsequent runs
    // to start from a degenerate flat-1.0 terrain and the normalisation
    // pipeline to produce an all-white preview (range → 0, h → 1).
    // The preview_reduce_minmax / preview_normalize_display passes already
    // handle any positive float range via the global min/max reduction.
    // The CPU-side export scan_height_range also scans the actual range.
    let h = textureLoad(height_a, coord).r;
    textureStore(raw_heights, coord, vec4<f32>(h, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Hydraulic: Pass 1 — uniform rainfall  (eq. 1)
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn hydro_water_add(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let d = textureLoad(water_tex, coord).r + params.rain_rate * params.dt;
    textureStore(water_tex, coord, vec4<f32>(d, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Hydraulic: Pass 2 — virtual pipe outflow flux  (eq. 2-5)
// k = dt * A * g / l    (A = pipe_area, l = pipe_length)
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn hydro_flux_update(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let c = coord;

    let hd   = surface_h(c);
    let hd_L = surface_h_open(c + vec2<i32>(-1,  0));
    let hd_R = surface_h_open(c + vec2<i32>( 1,  0));
    let hd_T = surface_h_open(c + vec2<i32>( 0, -1));
    let hd_B = surface_h_open(c + vec2<i32>( 0,  1));

    let f4 = load_flux(c);
    // k = dt * A * g / l  (eq. 2 from paper)
    let k  = params.dt * params.pipe_area * params.gravity / params.pipe_length;
    let fL = max(0.0, f4.x + k * (hd - hd_L));
    let fR = max(0.0, f4.y + k * (hd - hd_R));
    let fT = max(0.0, f4.z + k * (hd - hd_T));
    let fB = max(0.0, f4.w + k * (hd - hd_B));

    // Scale flux down if total outflow would drain more than available water (eq. 4).
    let total = fL + fR + fT + fB;
    let d     = textureLoad(water_tex, coord).r;
    let l2    = params.pipe_length * params.pipe_length;
    let K     = min(1.0, d * l2 / (total * params.dt + 1e-8));

    textureStore(flux_tex, coord, vec4<f32>(fL, fR, fT, fB) * K);
}

// ---------------------------------------------------------------------------
// Hydraulic: Pass 3 — water depth update + velocity field  (eq. 6-8)
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn hydro_water_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);

    let f_self  = load_flux(coord);
    let in_L    = load_flux(coord + vec2<i32>(-1,  0)).y;  // right-of-left  → flows INTO us
    let in_R    = load_flux(coord + vec2<i32>( 1,  0)).x;  // left-of-right  → flows INTO us
    let in_T    = load_flux(coord + vec2<i32>( 0, -1)).w;  // bottom-of-top  → flows INTO us
    let in_B    = load_flux(coord + vec2<i32>( 0,  1)).z;  // top-of-bottom  → flows INTO us

    let delta_V = params.dt * (in_L + in_R + in_T + in_B
                               - f_self.x - f_self.y - f_self.z - f_self.w);
    let l2     = params.pipe_length * params.pipe_length;
    let d_prev = textureLoad(water_tex, coord).r;
    let d_new  = max(0.0, d_prev + delta_V / l2);
    textureStore(water_tex, coord, vec4<f32>(d_new, 0.0, 0.0, 0.0));

    // Velocity (eq. 8): v = ΔW / (d_bar * l)
    // Floor at 0.01 (not 1e-5): near-dry cells (avg_d < 0.01) would otherwise
    // produce velocities of 50 000×flux, blowing up the semi-Lagrangian sediment
    // advection lookback by 100+ pixels and violating mass conservation.
    let avg_d  = (d_prev + d_new) * 0.5;
    let inv_d  = 1.0 / (2.0 * params.pipe_length * max(avg_d, 0.01));
    let vx = (in_L - f_self.x + f_self.y - in_R) * inv_d;
    let vy = (in_T - f_self.z + f_self.w - in_B) * inv_d;
    textureStore(velocity_tex, coord, vec4<f32>(vx, vy, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Hydraulic: Pass 4 — erosion / deposition  (eq. 9-13 from Jakó & Tóth)
//
// Capacity:  C = Kc * sin(α) * |v| * lmax(d)
// lmax(d)  = clamp(d / Kdmax, 0, 1)  — limits erosion to areas with water
//
// Erosion:    b -= Δt * Ks * (C - s);  s += same;  d += same
// Deposition: b += Δt * Kd * (s - C);  s -= same;  d -= same
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn hydro_erode_deposit(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);

    let vel   = textureLoad(velocity_tex, coord).xy;
    let speed = length(vel);
    let grad  = gradient2(coord);
    let tilt  = max(length(grad), params.min_slope);

    // lmax: erosion ramp — zero when dry, full when d >= erosion_depth_max (eq. 10).
    let d = textureLoad(water_tex, coord).r;
    let lmax = clamp(d / max(params.erosion_depth_max, 1e-6), 0.0, 1.0);

    let C = params.sediment_capacity * tilt * speed * lmax;

    let hard     = textureLoad(hardness_tex, coord).r;
    // hard=0 → resistant (erodes less), hard=1 → soft (erodes fully)
    let softness = 1.0 - (1.0 - hard) * params.hardness_influence;

    var s    = textureLoad(sediment_tex, coord).r;
    var h    = textureLoad(height_a,     coord).r;
    var hard_w = hard;

    if s < C {
        // Erode: dissolve soil into suspended sediment (eq. 12 Jakó & Tóth)
        // Water depth unchanged — only bed height and sediment concentration change.
        let delta = params.erosion_rate * softness * (C - s) * params.dt;
        let clamped = min(delta, max(h, 0.0));
        h -= clamped;
        s += clamped;
    } else {
        // Deposit: settle sediment onto bed (eq. 13)
        let excess = s - C;
        let delta = params.deposition_rate * excess * params.dt;
        let clamped = min(delta, s);
        h += clamped;
        s -= clamped;
        // Eq. 14: freshly deposited material is less resistant.
        hard_w = max(0.05, hard - params.dt * params.hardness_influence * params.deposition_rate * excess);
    }

    textureStore(height_a,     coord, vec4<f32>(clamp(h, 0.0, 1.5),    0.0, 0.0, 0.0));
    // Upper-bound matches the copy_b_to_sediment clamp so hydro_sediment_transport
    // samples a consistently bounded value (Regression Skeptic Q5 follow-up).
    textureStore(sediment_tex, coord, vec4<f32>(clamp(s, 0.0, 2.0),    0.0, 0.0, 0.0));
    textureStore(hardness_tex, coord, vec4<f32>(clamp(hard_w, 0.0, 1.0), 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Hydraulic: Pass 5 — semi-Lagrangian sediment advection  (eq. 15)
// ---------------------------------------------------------------------------

fn sample_sediment_bl(pos: vec2<f32>) -> f32 {
    let x0 = i32(floor(pos.x));
    let y0 = i32(floor(pos.y));
    let fx = fract(pos.x);
    let fy = fract(pos.y);
    let s00 = textureLoad(sediment_tex, cc(vec2<i32>(x0,     y0    ))).r;
    let s10 = textureLoad(sediment_tex, cc(vec2<i32>(x0 + 1, y0    ))).r;
    let s01 = textureLoad(sediment_tex, cc(vec2<i32>(x0,     y0 + 1))).r;
    let s11 = textureLoad(sediment_tex, cc(vec2<i32>(x0 + 1, y0 + 1))).r;
    return mix(mix(s00, s10, fx), mix(s01, s11, fx), fy);
}

@compute @workgroup_size(8, 8, 1)
fn hydro_sediment_transport(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let vel   = textureLoad(velocity_tex, coord).xy;
    // Convert physical velocity to pixel displacement; clamp to ±1 pixel to
    // satisfy CFL and prevent long-range lookback artefacts.
    var disp  = vel * params.dt / params.pipe_length;
    let mag   = length(disp);
    if mag > 1.0 {
        disp = disp / mag;
    }
    let prev  = vec2<f32>(f32(id.x), f32(id.y)) - disp;
    textureStore(height_b, coord, vec4<f32>(sample_sediment_bl(prev), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn copy_b_to_sediment(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    // Clamp to prevent ghost sediment (created by clamped-edge lookbacks in
    // hydro_sediment_transport) from compounding across iterations.
    let s = clamp(textureLoad(height_b, coord).r, 0.0, 2.0);
    textureStore(sediment_tex, coord, vec4<f32>(s, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Hydraulic: Pass 6 — water evaporation  (eq. 16)
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn hydro_evaporate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let d = textureLoad(water_tex, coord).r * (1.0 - params.evaporation_rate * params.dt);
    textureStore(water_tex, coord, vec4<f32>(max(0.0, d), 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Thermal erosion — 8-neighbor virtual pipe model with fixed-point atomics
// Based on Jakó & Tóth eq. 17-18
// ---------------------------------------------------------------------------

const DELTA_SCALE: f32 = 65536.0;

@compute @workgroup_size(8, 8, 1)
fn clear_delta_height(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let idx = id.y * params.resolution.x + id.x;
    atomicStore(&delta_height[idx], 0);
}

fn atomic_add_delta(c: vec2<i32>, delta: f32) {
    let cx  = clamp(c.x, 0, rx() - 1);
    let cy  = clamp(c.y, 0, ry() - 1);
    let idx = u32(cy) * params.resolution.x + u32(cx);
    atomicAdd(&delta_height[idx], i32(round(clamp(delta * DELTA_SCALE, -1e8, 1e8))));
}

@compute @workgroup_size(8, 8, 1)
fn thermal_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord    = vec2<i32>(id.xy);
    let h_self   = load_ha(coord);
    let hard     = textureLoad(hardness_tex, coord).r;
    // hard=0 → resistant (erodes less), hard=1 → soft (erodes fully)
    let softness = 1.0 - (1.0 - hard) * params.hardness_influence;
    let threshold = tan(params.repose_angle_radians);
    let l = params.pipe_length;
    let l_diag = l * 1.41421356; // sqrt(2)
    // Scale pixel-height-differences to physical slope (consistent with 512²).
    let rs = res_scale();

    var total_out = 0.0;

    // Cardinal neighbors (4) — distance = l
    // slope = raw_dh * rs / l  (physical);  transfer uses raw_dh
    let rdh0 = h_self - load_ha(coord + vec2<i32>(-1,  0));
    let rdh1 = h_self - load_ha(coord + vec2<i32>( 1,  0));
    let rdh2 = h_self - load_ha(coord + vec2<i32>( 0, -1));
    let rdh3 = h_self - load_ha(coord + vec2<i32>( 0,  1));

    if rdh0 > 0.0 && rdh0 * rs / l > threshold {
        let t = (rdh0 * rs / l - threshold) * l * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>(-1,  0), t);
    }
    if rdh1 > 0.0 && rdh1 * rs / l > threshold {
        let t = (rdh1 * rs / l - threshold) * l * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>( 1,  0), t);
    }
    if rdh2 > 0.0 && rdh2 * rs / l > threshold {
        let t = (rdh2 * rs / l - threshold) * l * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>( 0, -1), t);
    }
    if rdh3 > 0.0 && rdh3 * rs / l > threshold {
        let t = (rdh3 * rs / l - threshold) * l * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>( 0,  1), t);
    }

    // Diagonal neighbors (4) — distance = l * sqrt(2)
    let rdh4 = h_self - load_ha(coord + vec2<i32>(-1, -1));
    let rdh5 = h_self - load_ha(coord + vec2<i32>( 1, -1));
    let rdh6 = h_self - load_ha(coord + vec2<i32>(-1,  1));
    let rdh7 = h_self - load_ha(coord + vec2<i32>( 1,  1));

    if rdh4 > 0.0 && rdh4 * rs / l_diag > threshold {
        let t = (rdh4 * rs / l_diag - threshold) * l_diag * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>(-1, -1), t);
    }
    if rdh5 > 0.0 && rdh5 * rs / l_diag > threshold {
        let t = (rdh5 * rs / l_diag - threshold) * l_diag * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>( 1, -1), t);
    }
    if rdh6 > 0.0 && rdh6 * rs / l_diag > threshold {
        let t = (rdh6 * rs / l_diag - threshold) * l_diag * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>(-1,  1), t);
    }
    if rdh7 > 0.0 && rdh7 * rs / l_diag > threshold {
        let t = (rdh7 * rs / l_diag - threshold) * l_diag * 0.5 * params.talus_rate * softness / rs;
        total_out += t;
        atomic_add_delta(coord + vec2<i32>( 1,  1), t);
    }

    if total_out > 0.0 {
        atomic_add_delta(coord, -total_out);
    }
}

@compute @workgroup_size(8, 8, 1)
fn thermal_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let coord = vec2<i32>(id.xy);
    let idx   = id.y * params.resolution.x + id.x;
    let fixed = atomicExchange(&delta_height[idx], 0);
    let delta = f32(fixed) / DELTA_SCALE;
    let h     = clamp(load_ha(coord) + delta, 0.0, 1.5);
    textureStore(height_a, coord, vec4<f32>(h, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Particle erosion (one GPU thread = one droplet)
// ---------------------------------------------------------------------------

fn hash_u(v: u32) -> u32 {
    var x = v;
    x ^= x >> 17u;
    x *= 0xbf324c81u;
    x ^= x >> 11u;
    x *= 0x68e31da4u;
    x ^= x >> 14u;
    return x;
}

fn rand_f(seed: ptr<function, u32>) -> f32 {
    *seed = hash_u(*seed);
    return f32(*seed) / 4294967296.0;
}

fn sample_ha_bl(pos: vec2<f32>) -> f32 {
    let x0 = i32(floor(pos.x));
    let y0 = i32(floor(pos.y));
    let fx = fract(pos.x);
    let fy = fract(pos.y);
    let h00 = load_ha(vec2<i32>(x0,     y0    ));
    let h10 = load_ha(vec2<i32>(x0 + 1, y0    ));
    let h01 = load_ha(vec2<i32>(x0,     y0 + 1));
    let h11 = load_ha(vec2<i32>(x0 + 1, y0 + 1));
    return mix(mix(h00, h10, fx), mix(h01, h11, fx), fy);
}

fn gradient_bl(pos: vec2<f32>) -> vec2<f32> {
    let e = 0.5;
    let gx = sample_ha_bl(pos + vec2<f32>(e, 0.0)) - sample_ha_bl(pos - vec2<f32>(e, 0.0));
    let gy = sample_ha_bl(pos + vec2<f32>(0.0, e)) - sample_ha_bl(pos - vec2<f32>(0.0, e));
    return vec2<f32>(gx, gy) / (2.0 * e);
}

fn splat_delta(pos: vec2<f32>, delta: f32) {
    let x0    = i32(floor(pos.x));
    let y0    = i32(floor(pos.y));
    let x1    = x0 + 1;
    let y1    = y0 + 1;
    let fx    = fract(pos.x);
    let fy    = fract(pos.y);
    let fixed = i32(round(delta * DELTA_SCALE));

    if x0 >= 0 && x0 < rx() && y0 >= 0 && y0 < ry() {
        let idx = u32(y0) * params.resolution.x + u32(x0);
        atomicAdd(&delta_height[idx], i32(round(f32(fixed) * (1.0 - fx) * (1.0 - fy))));
    }
    if x1 >= 0 && x1 < rx() && y0 >= 0 && y0 < ry() {
        let idx = u32(y0) * params.resolution.x + u32(x1);
        atomicAdd(&delta_height[idx], i32(round(f32(fixed) * fx * (1.0 - fy))));
    }
    if x0 >= 0 && x0 < rx() && y1 >= 0 && y1 < ry() {
        let idx = u32(y1) * params.resolution.x + u32(x0);
        atomicAdd(&delta_height[idx], i32(round(f32(fixed) * (1.0 - fx) * fy)));
    }
    if x1 >= 0 && x1 < rx() && y1 >= 0 && y1 < ry() {
        let idx = u32(y1) * params.resolution.x + u32(x1);
        atomicAdd(&delta_height[idx], i32(round(f32(fixed) * fx * fy)));
    }
}

@compute @workgroup_size(64, 1, 1)
fn particle_erode(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_particles { return; }

    var seed  = id.x ^ (params.frame_seed << 16u) ^ hash_u(id.x + 1u);
    let res_f = vec2<f32>(f32(params.resolution.x - 1u), f32(params.resolution.y - 1u));
    var pos   = vec2<f32>(rand_f(&seed) * res_f.x, rand_f(&seed) * res_f.y);
    var dir   = vec2<f32>(0.0, 0.0);
    var speed  = 0.0;
    var water  = 1.0;
    var carry  = 0.0;

    // Resolution compensation: each step covers 1 pixel = 1/rs physical units.
    // Scale per-step erosion/deposition by 1/rs so total material moved over the
    // same physical distance is consistent.  Slope is converted to physical units
    // (× rs) for capacity, matching the grid-based gradient2() convention.
    let rs = res_scale();
    let inv_rs = 1.0 / rs;

    for (var step = 0u; step < params.max_steps; step++) {
        let ipos = vec2<i32>(pos);
        if ipos.x < 0 || ipos.y < 0 || ipos.x >= rx() || ipos.y >= ry() { break; }

        let grad  = gradient_bl(pos);
        let g_len = length(grad);
        if g_len > 1e-6 {
            dir = normalize(dir * params.inertia - grad * (1.0 - params.inertia));
        } else {
            let a = rand_f(&seed) * 6.2831853;
            dir = vec2<f32>(cos(a), sin(a));
        }

        let new_pos  = pos + dir;
        let old_h    = sample_ha_bl(pos);
        let new_h    = sample_ha_bl(new_pos);
        let h_delta  = new_h - old_h;

        speed = sqrt(max(0.0, speed * speed - h_delta * params.gravity));
        // Physical slope: per-pixel h_delta × res_scale (consistent with gradient2).
        let slope    = max(-h_delta * rs, params.min_slope);
        let capacity = max(0.0, slope * speed * water * params.sediment_capacity);

        if carry > capacity || h_delta > 0.0 {
            let deposit = select(
                params.deposition_rate * (carry - capacity) * inv_rs,
                min(carry, h_delta),
                h_delta > 0.0
            );
            let deposit_clamped = max(0.0, deposit);
            carry -= deposit_clamped;
            splat_delta(pos, deposit_clamped);
        } else {
            // Scale erosion per step by 1/rs — more steps at higher res each
            // erode proportionally less, keeping total material consistent.
            // Cap by the per-pixel height drop (can't excavate below next cell).
            let erode = min(params.erosion_rate * (capacity - carry) * inv_rs, -h_delta);
            carry += erode;
            splat_delta(pos, -erode);
        }

        // Evaporation per physical distance: divide rate by rs so the same
        // physical path length evaporates the same fraction of water.
        water = max(0.0, water * (1.0 - params.evaporation_rate * inv_rs));
        pos    = new_pos;
        if water < 0.01 { break; }
    }
    // Deposit a fraction of remaining sediment instead of dumping it all
    // at the death position — prevents spike/boulder artifacts while
    // preserving material budget proportional to deposition_rate.
    let final_deposit = carry * params.deposition_rate;
    splat_delta(pos, final_deposit);
}

@compute @workgroup_size(8, 8, 1)
fn particle_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.resolution.x || id.y >= params.resolution.y { return; }
    let idx   = id.y * params.resolution.x + id.x;
    let fixed = atomicExchange(&delta_height[idx], 0);
    let delta = f32(fixed) / DELTA_SCALE;
    let coord = vec2<i32>(id.xy);
    let h     = clamp(load_ha(coord) + delta, 0.0, 1.5);
    textureStore(height_a, coord, vec4<f32>(h, 0.0, 0.0, 0.0));
}

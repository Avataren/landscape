// Tessendorf FFT ocean — multi-cascade GPU compute pipeline.
//
// All storage textures are 2-D arrays where the layer index is the cascade
// number.  Each frame the host dispatches:
//   1× animate                            → fills freq_a / freq_dz_a per cascade
//   2 × log2(N)  ifft_pass dispatches     → Stockham radix-2 IFFT, rows + cols
//   1× compose                            → packs (h, dx, dz, jacobian) per cascade
// The workgroup z dimension is the cascade index, so all cascades run in
// parallel within each dispatch.

const PI: f32 = 3.14159265358979323846;

struct PassParams {
    stage:        u32,
    direction:    u32,   // 0 = horizontal, 1 = vertical
    log_n:        u32,
    n:            u32,
    inverse:      u32,   // 0 = forward, 1 = inverse
    pingpong:     u32,   // 0 = read freq_a, write freq_b   1 = swapped
    choppy:       f32,
    time_seconds: f32,
    /// Per-cascade tile size in metres, indexed by gid.z (xy = cascade 0/1,
    /// zw reserved for future cascades up to 4).
    cascade_world_sizes: vec4<f32>,
};

@group(0) @binding(0) var<uniform> pass_params: PassParams;

// Init data — populated from CPU once per spectrum rebuild.
@group(0) @binding(1) var init_h0:         texture_storage_2d_array<rgba32float, read>;
@group(0) @binding(2) var init_omega_kvec: texture_storage_2d_array<rgba32float, read>;

// Frequency-domain ping-pong (H + Dx packed).
@group(0) @binding(3) var freq_a:    texture_storage_2d_array<rgba32float, read_write>;
@group(0) @binding(4) var freq_b:    texture_storage_2d_array<rgba32float, read_write>;

// Frequency-domain ping-pong (Dz alone, in xy; zw unused).
@group(0) @binding(5) var freq_dz_a: texture_storage_2d_array<rgba32float, read_write>;
@group(0) @binding(6) var freq_dz_b: texture_storage_2d_array<rgba32float, read_write>;

// Final water-sampled displacement texture array.
@group(0) @binding(7) var displacement: texture_storage_2d_array<rgba16float, write>;

// Complex-number helpers ----------------------------------------------------

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}
fn cexp(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}

// Per-cascade world size from the uniform vec4.
fn cascade_size(layer: i32) -> f32 {
    if layer == 0 { return pass_params.cascade_world_sizes.x; }
    if layer == 1 { return pass_params.cascade_world_sizes.y; }
    if layer == 2 { return pass_params.cascade_world_sizes.z; }
    return pass_params.cascade_world_sizes.w;
}

// ===========================================================================
// 1. ANIMATE — H(k,t) = H₀(k)·e^{iωt} + conj(H₀(-k))·e^{-iωt}
// ===========================================================================

@compute @workgroup_size(8, 8, 1)
fn animate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pass_params.n;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let layer = i32(gid.z);
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let h0_pair       = textureLoad(init_h0, coord, layer);
    let h0            = h0_pair.xy;
    let h0_conj_neg   = h0_pair.zw;

    let omega_kvec    = textureLoad(init_omega_kvec, coord, layer);
    let omega         = omega_kvec.x;
    let kx            = omega_kvec.y;
    let kz            = omega_kvec.z;
    let k_mag         = sqrt(kx * kx + kz * kz);

    let phase  = cexp(omega * pass_params.time_seconds);
    let h_freq = cmul(h0, phase) + cmul(h0_conj_neg, conj(phase));

    var dx_freq = vec2<f32>(0.0, 0.0);
    var dz_freq = vec2<f32>(0.0, 0.0);
    if k_mag > 1.0e-6 {
        let inv_k = 1.0 / k_mag;
        let nx = kx * inv_k;
        let nz = kz * inv_k;
        dx_freq = vec2<f32>(nx * h_freq.y, -nx * h_freq.x);
        dz_freq = vec2<f32>(nz * h_freq.y, -nz * h_freq.x);
    }

    textureStore(freq_a,    coord, layer, vec4<f32>(h_freq, dx_freq));
    textureStore(freq_dz_a, coord, layer, vec4<f32>(dz_freq, 0.0, 0.0));
}

// ===========================================================================
// 2. IFFT — Stockham OOP radix-2.
// ===========================================================================

fn read_freq(coord: vec2<i32>, layer: i32, from_a: bool) -> vec4<f32> {
    if from_a {
        return textureLoad(freq_a, coord, layer);
    } else {
        return textureLoad(freq_b, coord, layer);
    }
}
fn read_dz(coord: vec2<i32>, layer: i32, from_a: bool) -> vec2<f32> {
    if from_a {
        return textureLoad(freq_dz_a, coord, layer).xy;
    } else {
        return textureLoad(freq_dz_b, coord, layer).xy;
    }
}
fn write_freq(coord: vec2<i32>, layer: i32, val: vec4<f32>, to_a: bool) {
    if to_a {
        textureStore(freq_a, coord, layer, val);
    } else {
        textureStore(freq_b, coord, layer, val);
    }
}
fn write_dz(coord: vec2<i32>, layer: i32, val: vec2<f32>, to_a: bool) {
    if to_a {
        textureStore(freq_dz_a, coord, layer, vec4<f32>(val, 0.0, 0.0));
    } else {
        textureStore(freq_dz_b, coord, layer, vec4<f32>(val, 0.0, 0.0));
    }
}

@compute @workgroup_size(8, 8, 1)
fn ifft_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pass_params.n;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let layer = i32(gid.z);

    var x: u32;
    var perp: u32;
    if pass_params.direction == 0u {
        x = gid.x; perp = gid.y;
    } else {
        x = gid.y; perp = gid.x;
    }

    let s = pass_params.stage - 1u;
    let m_half = 1u << s;
    let m      = m_half << 1u;
    let n_over_m = n >> (s + 1u);

    let j = x / m_half;
    let k = x % m_half;
    let even_block = j >> 1u;
    let odd_offset = j & 1u;

    let in_lo = even_block * m_half + k;
    let in_hi = (even_block + n_over_m) * m_half + k;

    var theta = -2.0 * PI * f32(k) / f32(m);
    if pass_params.inverse == 1u {
        theta = -theta;
    }
    let w = cexp(theta);

    var lo_coord: vec2<i32>;
    var hi_coord: vec2<i32>;
    var out_coord: vec2<i32>;
    if pass_params.direction == 0u {
        lo_coord  = vec2<i32>(i32(in_lo), i32(perp));
        hi_coord  = vec2<i32>(i32(in_hi), i32(perp));
        out_coord = vec2<i32>(i32(x),     i32(perp));
    } else {
        lo_coord  = vec2<i32>(i32(perp), i32(in_lo));
        hi_coord  = vec2<i32>(i32(perp), i32(in_hi));
        out_coord = vec2<i32>(i32(perp), i32(x));
    }

    let from_a = pass_params.pingpong == 0u;

    // Packed (H, Dx).
    {
        let in_lo_v = read_freq(lo_coord, layer, from_a);
        let in_hi_v = read_freq(hi_coord, layer, from_a);
        let h_lo  = in_lo_v.xy;
        let h_hi  = in_hi_v.xy;
        let dx_lo = in_lo_v.zw;
        let dx_hi = in_hi_v.zw;
        let w_h_hi  = cmul(w, h_hi);
        let w_dx_hi = cmul(w, dx_hi);
        var out_h:  vec2<f32>;
        var out_dx: vec2<f32>;
        if odd_offset == 0u {
            out_h  = h_lo  + w_h_hi;
            out_dx = dx_lo + w_dx_hi;
        } else {
            out_h  = h_lo  - w_h_hi;
            out_dx = dx_lo - w_dx_hi;
        }
        write_freq(out_coord, layer, vec4<f32>(out_h, out_dx), !from_a);
    }
    // Dz.
    {
        let dz_lo = read_dz(lo_coord, layer, from_a);
        let dz_hi = read_dz(hi_coord, layer, from_a);
        let w_dz_hi = cmul(w, dz_hi);
        var out_dz: vec2<f32>;
        if odd_offset == 0u {
            out_dz = dz_lo + w_dz_hi;
        } else {
            out_dz = dz_lo - w_dz_hi;
        }
        write_dz(out_coord, layer, out_dz, !from_a);
    }
}

// ===========================================================================
// 3. COMPOSE — pack spatial fields + Jacobian into the sampled displacement.
// ===========================================================================

fn read_packed_either(layer: i32, from_a: bool, c: vec2<i32>) -> vec4<f32> {
    if from_a {
        return textureLoad(freq_a, c, layer);
    } else {
        return textureLoad(freq_b, c, layer);
    }
}
fn read_dz_either(layer: i32, from_a: bool, c: vec2<i32>) -> vec4<f32> {
    if from_a {
        return textureLoad(freq_dz_a, c, layer);
    } else {
        return textureLoad(freq_dz_b, c, layer);
    }
}

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pass_params.n;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let layer = i32(gid.z);
    let from_a = pass_params.pingpong == 0u;
    let coord  = vec2<i32>(i32(gid.x), i32(gid.y));

    let inv_n2 = 1.0 / f32(n * n);
    let choppy = pass_params.choppy;
    let world_size = cascade_size(layer);
    let world_per_texel = world_size / f32(n);
    let inv_2dx = 1.0 / (2.0 * world_per_texel);
    let scale_h  = inv_n2;
    let scale_dx = inv_n2 * choppy;

    let mask = n - 1u;
    let nx_c = vec2<i32>(i32((gid.x + 1u) & mask), i32(gid.y));
    let px_c = vec2<i32>(i32((gid.x + mask) & mask), i32(gid.y));
    let ny_c = vec2<i32>(i32(gid.x), i32((gid.y + 1u) & mask));
    let py_c = vec2<i32>(i32(gid.x), i32((gid.y + mask) & mask));

    let here_p = read_packed_either(layer, from_a, coord);
    let here_d = read_dz_either(layer, from_a, coord).x;
    let nx_p   = read_packed_either(layer, from_a, nx_c);
    let px_p   = read_packed_either(layer, from_a, px_c);
    let ny_p   = read_packed_either(layer, from_a, ny_c);
    let py_p   = read_packed_either(layer, from_a, py_c);
    let nx_d   = read_dz_either(layer, from_a, nx_c).x;
    let px_d   = read_dz_either(layer, from_a, px_c).x;
    let ny_d   = read_dz_either(layer, from_a, ny_c).x;
    let py_d   = read_dz_either(layer, from_a, py_c).x;

    let h_real  = here_p.x * scale_h;
    let dx_real = here_p.z * scale_dx;
    let dz_real = here_d   * scale_dx;

    let d_dx_dx = (nx_p.z - px_p.z) * scale_dx * inv_2dx;
    let d_dx_dz = (ny_p.z - py_p.z) * scale_dx * inv_2dx;
    let d_dz_dx = (nx_d   - px_d)   * scale_dx * inv_2dx;
    let d_dz_dz = (ny_d   - py_d)   * scale_dx * inv_2dx;

    let jacobian = (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dz_dx;

    textureStore(
        displacement,
        coord,
        layer,
        vec4<f32>(h_real, dx_real, dz_real, jacobian),
    );
}

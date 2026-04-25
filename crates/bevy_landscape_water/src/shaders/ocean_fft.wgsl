// Tessendorf FFT ocean — GPU compute pipeline.
//
// Per frame the host dispatches:
//   1× animate                           → fills freq_a / freq_dz_a from H₀, ω, t
//   2 × log2(N)  ifft_pass dispatches    → Stockham radix-2 IFFT (rows then cols)
//   1× compose                           → packs (h, dx, dz, jacobian) → displacement
//
// freq_a / freq_b store TWO complex signals per texel: H (xy) and Dx (zw),
// so the radix-2 butterfly does both in lockstep.  freq_dz_* store Dz alone
// (xy).  After 2·log2(N) butterfly passes the data is in spatial domain in
// natural order (no bit-reversal step needed thanks to Stockham OOP).

const PI: f32 = 3.14159265358979323846;

struct PassParams {
    // Stage 1..log2(N) for ifft_pass; ignored otherwise.
    stage:     u32,
    // 0 = horizontal (along X), 1 = vertical (along Y).  Ifft only.
    direction: u32,
    log_n:     u32,
    n:         u32,
    // 0 = forward, 1 = inverse.
    inverse:   u32,
    // 0 = read freq_a, write freq_b.   1 = read freq_b, write freq_a.
    pingpong:  u32,
    choppy:    f32,
    time_seconds: f32,
    // World-space tile size in metres (for Jacobian gradient scaling).
    world_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> pass_params: PassParams;

// Init data — populated from CPU once per spectrum rebuild.
@group(0) @binding(1) var init_h0:        texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var init_omega_kvec: texture_storage_2d<rgba32float, read>;

// Frequency-domain ping-pong (H + Dx packed).
@group(0) @binding(3) var freq_a:    texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var freq_b:    texture_storage_2d<rgba32float, read_write>;

// Frequency-domain ping-pong (Dz alone, in xy; zw unused).
@group(0) @binding(5) var freq_dz_a: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6) var freq_dz_b: texture_storage_2d<rgba32float, read_write>;

// Final water-sampled displacement texture.
@group(0) @binding(7) var displacement: texture_storage_2d<rgba16float, write>;

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

// ===========================================================================
// 1. ANIMATE — H(k,t) = H₀(k)·e^{iωt} + conj(H₀(-k))·e^{-iωt}
//    Then derive Dx = -i·(kx/|k|)·H,  Dz = -i·(kz/|k|)·H.
// ===========================================================================

@compute @workgroup_size(8, 8, 1)
fn animate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pass_params.n;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let h0_pair       = textureLoad(init_h0, coord);
    let h0            = h0_pair.xy;             // H₀(k)
    let h0_conj_neg   = h0_pair.zw;             // conj(H₀(-k))

    let omega_kvec    = textureLoad(init_omega_kvec, coord);
    let omega         = omega_kvec.x;
    let kx            = omega_kvec.y;
    let kz            = omega_kvec.z;
    let k_mag         = sqrt(kx * kx + kz * kz);

    let phase         = cexp(omega * pass_params.time_seconds);
    let h_freq        = cmul(h0, phase) + cmul(h0_conj_neg, conj(phase));

    var dx_freq = vec2<f32>(0.0, 0.0);
    var dz_freq = vec2<f32>(0.0, 0.0);
    if k_mag > 1.0e-6 {
        let inv_k = 1.0 / k_mag;
        // -i · scalar · H = (scalar · H_y, -scalar · H_x)
        let nx = kx * inv_k;
        let nz = kz * inv_k;
        dx_freq = vec2<f32>(nx * h_freq.y, -nx * h_freq.x);
        dz_freq = vec2<f32>(nz * h_freq.y, -nz * h_freq.x);
    }

    // Always write to "_a" buffers; subsequent ifft passes alternate.
    textureStore(freq_a,    coord, vec4<f32>(h_freq, dx_freq));
    textureStore(freq_dz_a, coord, vec4<f32>(dz_freq, 0.0, 0.0));
}

// ===========================================================================
// 2. IFFT — Stockham OOP radix-2.
//    Each thread computes one OUTPUT cell from two INPUT cells.
//    Done log2(N) times along X (horizontal pass) then log2(N) times along Y.
// ===========================================================================

// Read packed (H, Dx) at (x, y) from the chosen ping-pong slot.
fn read_freq(coord: vec2<i32>, from_a: bool) -> vec4<f32> {
    if from_a {
        return textureLoad(freq_a, coord);
    } else {
        return textureLoad(freq_b, coord);
    }
}
fn read_dz(coord: vec2<i32>, from_a: bool) -> vec2<f32> {
    if from_a {
        return textureLoad(freq_dz_a, coord).xy;
    } else {
        return textureLoad(freq_dz_b, coord).xy;
    }
}
fn write_freq(coord: vec2<i32>, val: vec4<f32>, to_a: bool) {
    if to_a {
        textureStore(freq_a, coord, val);
    } else {
        textureStore(freq_b, coord, val);
    }
}
fn write_dz(coord: vec2<i32>, val: vec2<f32>, to_a: bool) {
    if to_a {
        textureStore(freq_dz_a, coord, vec4<f32>(val, 0.0, 0.0));
    } else {
        textureStore(freq_dz_b, coord, vec4<f32>(val, 0.0, 0.0));
    }
}

@compute @workgroup_size(8, 8, 1)
fn ifft_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pass_params.n;
    if gid.x >= n || gid.y >= n {
        return;
    }

    // Determine the FFT axis coordinate `x` for this thread.
    var x: u32;
    var perp: u32;
    if pass_params.direction == 0u {
        x = gid.x; perp = gid.y;
    } else {
        x = gid.y; perp = gid.x;
    }

    // Stockham OOP radix-2: stage s ∈ [0, log2(N) - 1].
    // m = 2^(s+1), m_half = 2^s.  Each thread reads two values and writes one.
    let s = pass_params.stage - 1u;          // shader uses 0-indexed stages
    let m_half = 1u << s;
    let m      = m_half << 1u;
    let n_over_m = n >> (s + 1u);

    let j = x / m_half;
    let k = x % m_half;
    let even_block = j >> 1u;
    let odd_offset = j & 1u;

    // Indices in the input array of length N along the FFT axis.
    let in_lo = even_block * m_half + k;
    let in_hi = (even_block + n_over_m) * m_half + k;

    // Twiddle factor.  Forward: e^{-2πi k/m}.  Inverse: e^{+2πi k/m}.
    var theta = -2.0 * PI * f32(k) / f32(m);
    if pass_params.inverse == 1u {
        theta = -theta;
    }
    let w = cexp(theta);

    // Map (in_lo, in_hi, x) along FFT axis into 2-D texture coords.
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

    // --- Packed (H, Dx) -----------------------------------------------------
    {
        let in_lo_v = read_freq(lo_coord, from_a);
        let in_hi_v = read_freq(hi_coord, from_a);

        // The two complex signals in xy and zw share the same butterfly.
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
        write_freq(out_coord, vec4<f32>(out_h, out_dx), !from_a);
    }

    // --- Dz -----------------------------------------------------------------
    {
        let dz_lo = read_dz(lo_coord, from_a);
        let dz_hi = read_dz(hi_coord, from_a);
        let w_dz_hi = cmul(w, dz_hi);
        var out_dz: vec2<f32>;
        if odd_offset == 0u {
            out_dz = dz_lo + w_dz_hi;
        } else {
            out_dz = dz_lo - w_dz_hi;
        }
        write_dz(out_coord, out_dz, !from_a);
    }
}

// ===========================================================================
// 3. COMPOSE — pack spatial fields + Jacobian into the sampled displacement.
// ===========================================================================

// WGSL doesn't allow texture handles as fn parameters, hence helpers.
fn read_packed_either(from_a: bool, c: vec2<i32>) -> vec4<f32> {
    if from_a {
        return textureLoad(freq_a, c);
    } else {
        return textureLoad(freq_b, c);
    }
}
fn read_dz_either(from_a: bool, c: vec2<i32>) -> vec4<f32> {
    if from_a {
        return textureLoad(freq_dz_a, c);
    } else {
        return textureLoad(freq_dz_b, c);
    }
}

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = pass_params.n;
    if gid.x >= n || gid.y >= n {
        return;
    }
    let from_a = pass_params.pingpong == 0u;
    let coord  = vec2<i32>(i32(gid.x), i32(gid.y));

    let inv_n2 = 1.0 / f32(n * n);
    let choppy = pass_params.choppy;
    let world_per_texel = pass_params.world_size / f32(n);
    let inv_2dx = 1.0 / (2.0 * world_per_texel);
    let scale_h  = inv_n2;
    let scale_dx = inv_n2 * choppy;

    // Periodic neighbour offsets (N must be a power of two so AND wraps).
    let mask = n - 1u;
    let nx_c = vec2<i32>(i32((gid.x + 1u) & mask), i32(gid.y));
    let px_c = vec2<i32>(i32((gid.x + mask) & mask), i32(gid.y));
    let ny_c = vec2<i32>(i32(gid.x), i32((gid.y + 1u) & mask));
    let py_c = vec2<i32>(i32(gid.x), i32((gid.y + mask) & mask));

    // Centre + neighbours (un-normalised; .x = h.re, .z = dx.re for packed).
    let here_p = read_packed_either(from_a, coord);
    let here_d = read_dz_either(from_a, coord).x;
    let nx_p   = read_packed_either(from_a, nx_c);
    let px_p   = read_packed_either(from_a, px_c);
    let ny_p   = read_packed_either(from_a, ny_c);
    let py_p   = read_packed_either(from_a, py_c);
    let nx_d   = read_dz_either(from_a, nx_c).x;
    let px_d   = read_dz_either(from_a, px_c).x;
    let ny_d   = read_dz_either(from_a, ny_c).x;
    let py_d   = read_dz_either(from_a, py_c).x;

    // World-unit displacements (real parts; imaginary should be ~0).
    let h_real  = here_p.x * scale_h;
    let dx_real = here_p.z * scale_dx;
    let dz_real = here_d   * scale_dx;

    // Central-difference gradients of choppy displacements in world units.
    let d_dx_dx = (nx_p.z - px_p.z) * scale_dx * inv_2dx;
    let d_dx_dz = (ny_p.z - py_p.z) * scale_dx * inv_2dx;
    let d_dz_dx = (nx_d   - px_d)   * scale_dx * inv_2dx;
    let d_dz_dz = (ny_d   - py_d)   * scale_dx * inv_2dx;

    // Jacobian determinant of the 2-D displacement field.  J < 0 ⇒ foldover.
    let jacobian = (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dz_dx;

    textureStore(
        displacement,
        coord,
        vec4<f32>(h_real, dx_real, dz_real, jacobian),
    );
}

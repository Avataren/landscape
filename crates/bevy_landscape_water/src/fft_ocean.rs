//! Tessendorf FFT ocean — CPU prototype using rustfft.
//!
//! Builds a Phillips spectrum H₀(k) once on init / on parameter change, then
//! every frame:
//!   1. Animates H(k, t) = H₀(k)·e^{i ω t} + conj(H₀(-k))·e^{-i ω t}.
//!   2. Derives horizontal displacement frequency components
//!      D_x(k) = -i·(k_x/|k|)·H(k),   D_z(k) = -i·(k_z/|k|)·H(k).
//!   3. Runs three 2-D IFFTs (height, dx, dz) via separable 1-D rustfft.
//!   4. Computes the Jacobian determinant of the displacement field via
//!      central differences (for foldover foam in the fragment shader).
//!   5. Packs (h, dx, dz, jacobian) into an Rgba32Float image whose handle
//!      is fed to the water material every frame.
//!
//! GPU sampling tiles the texture in world space with period `world_size`
//! (e.g. 128 m).  Linear filtering smooths between samples; no mip chain is
//! generated yet — distance attenuation is handled in the shader.

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use half::f16;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

const G: f32 = 9.81;

// ---------------------------------------------------------------------------
// Public settings — exposed in the editor panel.
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct OceanFftSettings {
    /// Master toggle.  When false the GPU samples a flat (zero) texture and
    /// the Gerstner pipeline drives the surface as before.
    pub enabled: bool,
    /// Grid resolution N (must be a power of two).  128 is a good balance.
    pub size: u32,
    /// World-space tile size in metres (texture period when sampled).
    pub world_size: f32,
    /// Wind speed in m/s — drives the Phillips L = V²/g (largest wave).
    pub wind_speed: f32,
    /// Wind direction in world XZ.  Re-normalised every rebuild.
    pub wind_direction: Vec2,
    /// Phillips spectrum scalar amplitude.
    pub amplitude: f32,
    /// Horizontal-displacement (choppy) factor.  0 = pure heightfield, ~0.8
    /// gives realistic sharpened crests; ≥ 1 produces foldover (foam).
    pub choppy: f32,
    /// Hash seed for the Gaussian noise that randomises H₀ phases.
    pub seed: u32,
    /// Output strength applied in the shader (lets you cross-fade against
    /// the legacy Gerstner sum).  1.0 = full FFT, 0.0 = pure Gerstner.
    pub strength: f32,
}

impl Default for OceanFftSettings {
    fn default() -> Self {
        Self {
            // Default OFF so the FFT bindings/shader path don't break the
            // material on first run.  Toggle via the editor panel.
            enabled: false,
            size: 128,
            world_size: 128.0,
            wind_speed: 12.0,
            wind_direction: Vec2::new(1.0, 0.4),
            amplitude: 4.0e-3,
            choppy: 0.85,
            seed: 0xC0FFEE,
            strength: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state — stays alive across frames; rebuilt when settings change.
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct OceanFftState {
    n: usize,
    world_size: f32,
    /// Settings hash used to detect rebuild triggers cheaply.
    rebuild_key: u64,
    /// Initial Phillips amplitudes H₀(k), packed row-major in the rustfft
    /// natural layout (k=0 at index 0, negative-k mirrored at index ≥ N/2).
    h0: Vec<Complex32>,
    /// conj(H₀(-k)) precomputed so the per-frame animation is cheap.
    h0_conj_neg: Vec<Complex32>,
    /// Dispersion ω(k) = √(g·|k|) per cell.
    omega: Vec<f32>,
    /// (k_x, k_z) per cell.
    kvec: Vec<(f32, f32)>,
    fft: Arc<dyn Fft<f32>>,
    /// Reusable scratch buffers.
    h_freq: Vec<Complex32>,
    dx_freq: Vec<Complex32>,
    dz_freq: Vec<Complex32>,
    /// Reusable byte buffer for the texture upload.
    image_bytes: Vec<u8>,
    /// Public displacement texture handle (Rgba32Float).
    pub image_handle: Handle<Image>,
}

impl OceanFftState {
    fn rebuild_key(s: &OceanFftSettings) -> u64 {
        // Cheap hash of the parameters that affect H₀.  Animation parameters
        // (time) intentionally omitted.
        let mut x = 0u64;
        x = x.wrapping_mul(31).wrapping_add(s.size as u64);
        x = x.wrapping_mul(31).wrapping_add(s.world_size.to_bits() as u64);
        x = x.wrapping_mul(31).wrapping_add(s.wind_speed.to_bits() as u64);
        x = x.wrapping_mul(31).wrapping_add(s.wind_direction.x.to_bits() as u64);
        x = x.wrapping_mul(31).wrapping_add(s.wind_direction.y.to_bits() as u64);
        x = x.wrapping_mul(31).wrapping_add(s.amplitude.to_bits() as u64);
        x = x.wrapping_mul(31).wrapping_add(s.seed as u64);
        x
    }

    fn build(settings: &OceanFftSettings, images: &mut Assets<Image>) -> Self {
        let n = (settings.size.max(8)).next_power_of_two() as usize;
        let total = n * n;
        let l_world = settings.world_size.max(1.0);
        let wind_speed = settings.wind_speed.max(0.001);
        let wind_dir = settings.wind_direction.normalize_or_zero();
        let big_l = wind_speed * wind_speed / G;
        // Capillary suppression length: half a sample spacing.  Without this
        // the Phillips spectrum's k⁻⁴ tail produces unbounded high-frequency
        // noise that aliases at the texture's Nyquist limit.
        let small_l = (l_world / n as f32) * 0.5;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(n);

        let mut rng = SimpleLcg::new(settings.seed);
        let mut h0 = vec![Complex32::new(0.0, 0.0); total];
        let mut h0_conj_neg = vec![Complex32::new(0.0, 0.0); total];
        let mut omega = vec![0.0_f32; total];
        let mut kvec = vec![(0.0_f32, 0.0_f32); total];

        let dk = std::f32::consts::TAU / l_world;

        for j in 0..n {
            for i in 0..n {
                // Frequency index in [-N/2, N/2-1] using rustfft's natural
                // layout (i.e. np.fft.fftfreq * N).  k = freq_idx * dk.
                let fx = if i < n / 2 { i as f32 } else { i as f32 - n as f32 };
                let fz = if j < n / 2 { j as f32 } else { j as f32 - n as f32 };
                let kx = fx * dk;
                let kz = fz * dk;
                let k_mag = (kx * kx + kz * kz).sqrt();
                let idx = j * n + i;
                kvec[idx] = (kx, kz);

                if k_mag < 1.0e-6 {
                    h0[idx] = Complex32::new(0.0, 0.0);
                    omega[idx] = 0.0;
                    continue;
                }

                let k_hat_dot_w = (kx * wind_dir.x + kz * wind_dir.y) / k_mag;
                // Squared cosine — the classic Tessendorf directional term.
                let directional = k_hat_dot_w * k_hat_dot_w;

                let k_l = k_mag * big_l;
                let phillips = settings.amplitude * (-1.0 / (k_l * k_l)).exp()
                    / k_mag.powi(4)
                    * directional
                    * (-(k_mag * small_l).powi(2)).exp();

                // H₀ = (1/√2) · (ξ_r + i·ξ_i) · √(P(k)).
                let scale = (phillips.max(0.0) * 0.5).sqrt();
                let (xi_r, xi_i) = box_muller(&mut rng);
                h0[idx] = Complex32::new(xi_r * scale, xi_i * scale);
                omega[idx] = (G * k_mag).sqrt();
            }
        }

        // Precompute conj(H₀(-k)).  In rustfft's layout, -k corresponds to
        // wrap-around index ((N-i) mod N, (N-j) mod N).
        for j in 0..n {
            for i in 0..n {
                let ni = (n - i) % n;
                let nj = (n - j) % n;
                let idx = j * n + i;
                let nidx = nj * n + ni;
                h0_conj_neg[idx] = h0[nidx].conj();
            }
        }

        // Rgba16Float is linear-filterable on all wgpu adapters; Rgba32Float
        // is gated behind the FLOAT32_FILTERABLE feature and silently fails
        // pipeline validation when sampled with a Linear sampler on most
        // hardware.  f16 has more than enough precision for ±10 m heights
        // and choppy displacements.
        let mut img = Image::new(
            Extent3d {
                width: n as u32,
                height: n as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![0u8; total * 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            ..default()
        });
        let image_handle = images.add(img);

        Self {
            n,
            world_size: l_world,
            rebuild_key: Self::rebuild_key(settings),
            h0,
            h0_conj_neg,
            omega,
            kvec,
            fft,
            h_freq: vec![Complex32::new(0.0, 0.0); total],
            dx_freq: vec![Complex32::new(0.0, 0.0); total],
            dz_freq: vec![Complex32::new(0.0, 0.0); total],
            image_bytes: vec![0u8; total * 8],
            image_handle,
        }
    }

    pub fn world_size(&self) -> f32 {
        self.world_size
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct OceanFftPlugin;

impl Plugin for OceanFftPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<OceanFftSettings>()
            .init_resource::<OceanFftSettings>()
            .add_systems(PreStartup, setup_ocean_fft)
            .add_systems(Update, (rebuild_on_settings_change, tick_ocean_fft).chain());
    }
}

fn setup_ocean_fft(
    settings: Res<OceanFftSettings>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
) {
    let state = OceanFftState::build(&settings, &mut images);
    commands.insert_resource(state);
}

fn rebuild_on_settings_change(
    settings: Res<OceanFftSettings>,
    mut state: ResMut<OceanFftState>,
    mut images: ResMut<Assets<Image>>,
) {
    if !settings.is_changed() {
        return;
    }
    let key = OceanFftState::rebuild_key(&settings);
    if key == state.rebuild_key {
        return;
    }
    let new_state = OceanFftState::build(&settings, &mut images);
    *state = new_state;
}

// ---------------------------------------------------------------------------
// Per-frame animation + IFFT.
// ---------------------------------------------------------------------------

fn tick_ocean_fft(
    time: Res<Time>,
    settings: Res<OceanFftSettings>,
    mut state: ResMut<OceanFftState>,
    mut images: ResMut<Assets<Image>>,
) {
    if !settings.enabled {
        return;
    }
    let n = state.n;
    let total = n * n;
    let t = time.elapsed_secs();
    let choppy = settings.choppy;

    // 1. Animate H(k, t).
    {
        let OceanFftState {
            ref h0,
            ref h0_conj_neg,
            ref omega,
            ref kvec,
            ref mut h_freq,
            ref mut dx_freq,
            ref mut dz_freq,
            ..
        } = *state;
        for idx in 0..total {
            let phase = Complex32::from_polar(1.0, omega[idx] * t);
            let h = h0[idx] * phase + h0_conj_neg[idx] * phase.conj();
            h_freq[idx] = h;

            let (kx, kz) = kvec[idx];
            let k_mag = (kx * kx + kz * kz).sqrt();
            if k_mag < 1.0e-6 {
                dx_freq[idx] = Complex32::new(0.0, 0.0);
                dz_freq[idx] = Complex32::new(0.0, 0.0);
            } else {
                // -i · (k / |k|) · H(k).
                let i_unit = Complex32::new(0.0, 1.0);
                dx_freq[idx] = -i_unit * (kx / k_mag) * h;
                dz_freq[idx] = -i_unit * (kz / k_mag) * h;
            }
        }
    }

    // 2. Three 2-D IFFTs.
    {
        let OceanFftState {
            ref fft,
            ref mut h_freq,
            ref mut dx_freq,
            ref mut dz_freq,
            ..
        } = *state;
        fft_2d_inplace(h_freq, n, fft);
        fft_2d_inplace(dx_freq, n, fft);
        fft_2d_inplace(dz_freq, n, fft);
    }

    // rustfft's IFFT is unnormalised; divide by N² to recover unit scaling.
    let inv_total = 1.0 / total as f32;

    // 3. Pack (h, dx·choppy, dz·choppy, jacobian) into the image buffer.
    //    Jacobian is computed via central differences over the (now spatial)
    //    displacement fields.
    let dx_world = state.world_size / n as f32;
    let inv_2dx = 1.0 / (2.0 * dx_world);

    // Disjoint borrow split: take a mut ref to image_bytes and immut refs to
    // the freq buffers via the struct fields one-by-one.
    let OceanFftState {
        ref h_freq,
        ref dx_freq,
        ref dz_freq,
        ref mut image_bytes,
        ref image_handle,
        ..
    } = *state;

    for j in 0..n {
        for i in 0..n {
            let idx = j * n + i;
            let h_re = h_freq[idx].re * inv_total;
            let dx_re = dx_freq[idx].re * inv_total * choppy;
            let dz_re = dz_freq[idx].re * inv_total * choppy;

            // Periodic neighbours for finite differences.
            let il = (i + n - 1) % n;
            let ir = (i + 1) % n;
            let jl = (j + n - 1) % n;
            let jr = (j + 1) % n;

            let dx_e = dx_freq[j * n + ir].re * inv_total * choppy;
            let dx_w = dx_freq[j * n + il].re * inv_total * choppy;
            let dx_n = dx_freq[jr * n + i].re * inv_total * choppy;
            let dx_s = dx_freq[jl * n + i].re * inv_total * choppy;

            let dz_e = dz_freq[j * n + ir].re * inv_total * choppy;
            let dz_w = dz_freq[j * n + il].re * inv_total * choppy;
            let dz_n = dz_freq[jr * n + i].re * inv_total * choppy;
            let dz_s = dz_freq[jl * n + i].re * inv_total * choppy;

            let dxx = (dx_e - dx_w) * inv_2dx;
            let dxz = (dx_n - dx_s) * inv_2dx;
            let dzx = (dz_e - dz_w) * inv_2dx;
            let dzz = (dz_n - dz_s) * inv_2dx;

            let jacobian = (1.0 + dxx) * (1.0 + dzz) - dxz * dzx;

            // Rgba16Float: 4 × f16 = 8 bytes per texel.
            let off = idx * 8;
            image_bytes[off..off + 2]
                .copy_from_slice(&f16::from_f32(h_re).to_le_bytes());
            image_bytes[off + 2..off + 4]
                .copy_from_slice(&f16::from_f32(dx_re).to_le_bytes());
            image_bytes[off + 4..off + 6]
                .copy_from_slice(&f16::from_f32(dz_re).to_le_bytes());
            image_bytes[off + 6..off + 8]
                .copy_from_slice(&f16::from_f32(jacobian).to_le_bytes());
        }
    }

    // 4. Push to GPU.
    if let Some(img) = images.get_mut(image_handle) {
        img.data = Some(image_bytes.clone());
    }
}

// ---------------------------------------------------------------------------
// Helpers: separable 2-D IFFT using rustfft's 1-D plan.
// ---------------------------------------------------------------------------

fn fft_2d_inplace(buf: &mut [Complex32], n: usize, fft: &Arc<dyn Fft<f32>>) {
    // Rows.
    for row in 0..n {
        let start = row * n;
        fft.process(&mut buf[start..start + n]);
    }
    // Columns — rustfft is contiguous-only, so transpose into a scratch
    // column, transform, write back.
    let mut col = vec![Complex32::new(0.0, 0.0); n];
    for c in 0..n {
        for r in 0..n {
            col[r] = buf[r * n + c];
        }
        fft.process(&mut col);
        for r in 0..n {
            buf[r * n + c] = col[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Tiny LCG + Box–Muller for deterministic Gaussian noise.
// ---------------------------------------------------------------------------

struct SimpleLcg(u32);

impl SimpleLcg {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }
    fn next_unit(&mut self) -> f32 {
        // Numerical Recipes LCG.
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to (0, 1] avoiding exact zero (Box–Muller hates ln(0)).
        let f = (self.0 >> 8) as f32 / (1u32 << 24) as f32;
        f.max(1.0e-7)
    }
}

fn box_muller(rng: &mut SimpleLcg) -> (f32, f32) {
    let u1 = rng.next_unit();
    let u2 = rng.next_unit();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

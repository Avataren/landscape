//! Tessendorf FFT ocean — main-world settings + spectrum init.
//!
//! Phillips spectrum H₀(k) and dispersion ω(k) are computed on CPU once per
//! parameter change and uploaded into 2-D-array storage textures (one layer
//! per cascade).  The per-frame work — animation + IFFT + compose — runs on
//! the GPU and processes all cascades in parallel via the workgroup z axis.
//! See `fft_ocean_compute.rs`.

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
};

use crate::fft_ocean_compute::OceanFftComputePlugin;

const G: f32 = 9.81;

/// Number of FFT cascades.  Three cascades with non-harmonic period ratios
/// hide the per-tile periodicity at any view distance: cascade 0 carries the
/// long swell, cascade 1 the mid chop, cascade 2 the sharp wind chop.
pub const NUM_CASCADES: usize = 3;

/// World-size multipliers per cascade.  Cascade 0 = master tile size; the
/// other factors are deliberately non-power-of-two so the three cascades'
/// periods never re-align in world space — without this, a viewer looking
/// across the ocean still sees a regular grid even with N cascades.
pub const CASCADE_WORLD_FACTORS: [f32; NUM_CASCADES] = [1.0, 0.31, 0.085];

/// Phillips amplitude weight per cascade — Phillips-like rolloff so the
/// long-period swell dominates the silhouette and the fine cascades only
/// add detail.
pub const CASCADE_AMP_WEIGHTS: [f32; NUM_CASCADES] = [1.0, 0.55, 0.32];

/// Per-cascade rotation in radians.  Must match `fft_cascade_rotation()`
/// in `water_functions.wgsl`.  Currently identity; non-harmonic
/// `CASCADE_WORLD_FACTORS` already prevent the cascades' tile periods
/// from re-aligning, and any non-zero rotation here must be matched in
/// the shader plus a corresponding wind pre-rotation in the CPU bake to
/// avoid a diamond cross-hatch.
pub const CASCADE_ROTATIONS_RAD: [f32; NUM_CASCADES] = [0.0, 0.0, 0.0];

const STORAGE_USAGE: TextureUsages = TextureUsages::from_bits_retain(
    TextureUsages::COPY_SRC.bits()
        | TextureUsages::COPY_DST.bits()
        | TextureUsages::STORAGE_BINDING.bits()
        | TextureUsages::TEXTURE_BINDING.bits(),
);

// ---------------------------------------------------------------------------
// Public settings — exposed to the editor panel.
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct OceanFftSettings {
    /// Master toggle.  When false the GPU pipelines run no-ops and the
    /// legacy Gerstner pipeline drives the surface.
    pub enabled: bool,
    /// Grid resolution N (must be a power of two; we clamp to {64, 128, 256}).
    pub size: u32,
    /// Master cascade-0 tile size in metres.  Cascade k uses
    /// `world_size * CASCADE_WORLD_FACTORS[k]`.
    pub world_size: f32,
    /// Wind speed in m/s — drives Phillips L = V²/g.  Shared across cascades.
    pub wind_speed: f32,
    /// Wind direction in world XZ.  Shared across cascades.
    pub wind_direction: Vec2,
    /// Phillips spectrum scalar amplitude.  Distributed across cascades by
    /// `CASCADE_AMP_WEIGHTS`.
    pub amplitude: f32,
    /// Horizontal-displacement (choppy) factor.  ≥ 1 produces foldover (foam).
    pub choppy: f32,
    /// Hash seed.  Each cascade uses `seed XOR (k * GOLDEN_RATIO_INT)` so they
    /// don't share random phases.
    pub seed: u32,
    /// Output strength in the shader: 1 = full FFT, 0 = pure Gerstner.
    pub strength: f32,
}

impl Default for OceanFftSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 128,
            // Master tile size in metres.  At 1024 m the longest cascade
            // period is well past the natural distance the eye can resolve a
            // repeating tile, so the ocean reads as non-periodic from afar.
            world_size: 1024.0,
            wind_speed: 12.0,
            wind_direction: Vec2::new(1.0, 0.4),
            // Phillips total variance grows ~L² with tile size, so the
            // 1024 m default needs roughly 1/16 the amplitude that worked
            // for the original 256 m tile.  Tune via the panel slider.
            amplitude: 12.0,
            choppy: 0.85,
            seed: 0xC0FFEE,
            strength: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Resource holding all texture handles.  Extracted to the render world so the
// compute pipeline can build bind groups against them.
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, ExtractResource)]
pub struct OceanFftBuffers {
    pub n: u32,
    pub log_n: u32,
    /// Per-cascade tile size in metres.  Indexed by cascade.
    pub cascade_world_sizes: [f32; NUM_CASCADES],
    /// (H₀.re, H₀.im, conj(H₀(-k)).re, conj(H₀(-k)).im) per cascade layer.
    pub init_h0: Handle<Image>,
    /// (ω, kx, kz, _) per cascade layer.
    pub init_omega_kvec: Handle<Image>,
    /// (h.re, h.im, dx.re, dx.im) packed per cascade layer.
    pub freq_a: Handle<Image>,
    pub freq_b: Handle<Image>,
    /// (dz.re, dz.im, _, _) per cascade layer.
    pub freq_dz_a: Handle<Image>,
    pub freq_dz_b: Handle<Image>,
    /// Final (h, dx, dz, jacobian) per cascade layer.  Sampled by the water
    /// material via a `texture_2d_array` binding.
    pub displacement: Handle<Image>,
}

impl OceanFftBuffers {
    pub fn settings_changed_significantly(
        prev: &OceanFftSettings,
        next: &OceanFftSettings,
    ) -> bool {
        prev.size != next.size
            || prev.world_size.to_bits() != next.world_size.to_bits()
            || prev.wind_speed.to_bits() != next.wind_speed.to_bits()
            || prev.wind_direction.x.to_bits() != next.wind_direction.x.to_bits()
            || prev.wind_direction.y.to_bits() != next.wind_direction.y.to_bits()
            || prev.amplitude.to_bits() != next.amplitude.to_bits()
            || prev.seed != next.seed
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
            .add_plugins(ExtractResourcePlugin::<OceanFftBuffers>::default())
            .add_plugins(ExtractResourcePlugin::<OceanFftSettings>::default())
            .add_plugins(OceanFftComputePlugin)
            .add_systems(PreStartup, setup_ocean_fft)
            .add_systems(Update, rebuild_on_settings_change);
    }
}

impl ExtractResource for OceanFftSettings {
    type Source = OceanFftSettings;
    fn extract_resource(s: &Self::Source) -> Self {
        s.clone()
    }
}

// ---------------------------------------------------------------------------
// Init: build textures + fill H₀, ω.
// ---------------------------------------------------------------------------

fn setup_ocean_fft(
    settings: Res<OceanFftSettings>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
) {
    let buffers = build_buffers(&settings, &mut images);
    commands.insert_resource(buffers);
}

fn rebuild_on_settings_change(
    settings: Res<OceanFftSettings>,
    mut buffers: ResMut<OceanFftBuffers>,
    mut images: ResMut<Assets<Image>>,
    mut last: Local<Option<OceanFftSettings>>,
) {
    if !settings.is_changed() {
        return;
    }
    let needs_rebuild = match last.as_ref() {
        None => true,
        Some(prev) => OceanFftBuffers::settings_changed_significantly(prev, &settings),
    };
    *last = Some(settings.clone());
    if !needs_rebuild {
        return;
    }
    *buffers = build_buffers(&settings, &mut images);
}

fn build_buffers(settings: &OceanFftSettings, images: &mut Assets<Image>) -> OceanFftBuffers {
    let n_raw = settings.size.max(8).next_power_of_two();
    let n = n_raw.clamp(64, 1024);
    let log_n = n.trailing_zeros();
    let total_per_layer = (n * n) as usize;
    let master_size = settings.world_size.max(1.0);

    let mut cascade_world_sizes = [0.0_f32; NUM_CASCADES];
    for k in 0..NUM_CASCADES {
        cascade_world_sizes[k] = master_size * CASCADE_WORLD_FACTORS[k];
    }

    // Build H₀ + omega data for each cascade and concatenate the layers.
    let mut h0_bytes = Vec::with_capacity(total_per_layer * 16 * NUM_CASCADES);
    let mut omega_bytes = Vec::with_capacity(total_per_layer * 16 * NUM_CASCADES);
    for k in 0..NUM_CASCADES {
        let cascade_size = cascade_world_sizes[k];
        let amp = settings.amplitude * CASCADE_AMP_WEIGHTS[k];
        // Decorrelate cascades' random phases — golden-ratio multiplier so
        // each layer sees an unrelated noise stream.
        let seed = settings
            .seed
            .wrapping_add((k as u32).wrapping_mul(0x9E37_79B9));
        // Rotate the Phillips wind direction by -θ_k so that when the
        // shader samples this cascade through the matching +θ_k
        // rotation, the visible wave crests end up along the world
        // wind direction.  Without this, each cascade's wave train
        // points in a different world direction and the superposition
        // forms a visible cross-hatch / diamond weave.
        let theta = -CASCADE_ROTATIONS_RAD[k];
        let (cs, sn) = (theta.cos(), theta.sin());
        let world_wind = settings.wind_direction.normalize_or_zero();
        let cascade_wind = Vec2::new(
            cs * world_wind.x - sn * world_wind.y,
            sn * world_wind.x + cs * world_wind.y,
        );
        let (h0, omega) = build_spectrum_data(
            settings,
            n,
            cascade_size,
            amp,
            seed,
            cascade_wind,
        );
        h0_bytes.extend(h0);
        omega_bytes.extend(omega);
    }

    let init_h0 = images.add(make_storage_array(
        n,
        TextureFormat::Rgba32Float,
        h0_bytes,
        false,
    ));
    let init_omega_kvec = images.add(make_storage_array(
        n,
        TextureFormat::Rgba32Float,
        omega_bytes,
        false,
    ));
    let freq_a = images.add(make_storage_array(
        n,
        TextureFormat::Rgba32Float,
        zero_bytes(total_per_layer * 16 * NUM_CASCADES),
        false,
    ));
    let freq_b = images.add(make_storage_array(
        n,
        TextureFormat::Rgba32Float,
        zero_bytes(total_per_layer * 16 * NUM_CASCADES),
        false,
    ));
    let freq_dz_a = images.add(make_storage_array(
        n,
        TextureFormat::Rgba32Float,
        zero_bytes(total_per_layer * 16 * NUM_CASCADES),
        false,
    ));
    let freq_dz_b = images.add(make_storage_array(
        n,
        TextureFormat::Rgba32Float,
        zero_bytes(total_per_layer * 16 * NUM_CASCADES),
        false,
    ));
    let displacement = images.add(make_storage_array(
        n,
        TextureFormat::Rgba16Float,
        zero_bytes(total_per_layer * 8 * NUM_CASCADES),
        true,
    ));

    OceanFftBuffers {
        n,
        log_n,
        cascade_world_sizes,
        init_h0,
        init_omega_kvec,
        freq_a,
        freq_b,
        freq_dz_a,
        freq_dz_b,
        displacement,
    }
}

fn zero_bytes(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

fn make_storage_array(
    n: u32,
    format: TextureFormat,
    data: Vec<u8>,
    sampled_by_water: bool,
) -> Image {
    let mut img = Image::new(
        Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: NUM_CASCADES as u32,
        },
        TextureDimension::D2,
        data,
        format,
        // Storage textures live in the render world only; the data we ship
        // through Image::new is the initial GPU upload.
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.usage = STORAGE_USAGE;
    if sampled_by_water {
        img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            ..default()
        });
    }
    img
}

// ---------------------------------------------------------------------------
// Phillips spectrum + Box–Muller noise generation.
// ---------------------------------------------------------------------------

fn build_spectrum_data(
    settings: &OceanFftSettings,
    n: u32,
    world_size: f32,
    amplitude: f32,
    seed: u32,
    wind_dir: Vec2,
) -> (Vec<u8>, Vec<u8>) {
    let total = (n * n) as usize;
    let wind_speed = settings.wind_speed.max(0.001);
    let wind_dir = wind_dir.normalize_or_zero();
    let big_l = wind_speed * wind_speed / G;
    let small_l = (world_size / n as f32) * 0.5;
    let dk = std::f32::consts::TAU / world_size;

    let mut rng = SimpleLcg::new(seed);

    let mut h0_re = vec![0.0_f32; total];
    let mut h0_im = vec![0.0_f32; total];
    let mut omega = vec![0.0_f32; total];
    let mut kx_arr = vec![0.0_f32; total];
    let mut kz_arr = vec![0.0_f32; total];

    for j in 0..n {
        for i in 0..n {
            let fx = if i < n / 2 {
                i as f32
            } else {
                i as f32 - n as f32
            };
            let fz = if j < n / 2 {
                j as f32
            } else {
                j as f32 - n as f32
            };
            let kx = fx * dk;
            let kz = fz * dk;
            let k_mag = (kx * kx + kz * kz).sqrt();
            let idx = (j * n + i) as usize;
            kx_arr[idx] = kx;
            kz_arr[idx] = kz;

            if k_mag < 1.0e-6 {
                continue;
            }
            let k_hat_dot_w = (kx * wind_dir.x + kz * wind_dir.y) / k_mag;
            let directional = k_hat_dot_w * k_hat_dot_w;
            let k_l = k_mag * big_l;
            let phillips = amplitude * (-1.0 / (k_l * k_l)).exp() / k_mag.powi(4)
                * directional
                * (-(k_mag * small_l).powi(2)).exp();
            let scale = (phillips.max(0.0) * 0.5).sqrt();
            let (xi_r, xi_i) = box_muller(&mut rng);
            h0_re[idx] = xi_r * scale;
            h0_im[idx] = xi_i * scale;
            omega[idx] = (G * k_mag).sqrt();
        }
    }

    // Pack: (H₀.re, H₀.im, conj(H₀(-k)).re, conj(H₀(-k)).im) | (ω, kx, kz, 0).
    let mut h0_packed = Vec::with_capacity(total * 16);
    let mut omega_packed = Vec::with_capacity(total * 16);
    for j in 0..n {
        for i in 0..n {
            let idx = (j * n + i) as usize;
            let ni = (n - i) % n;
            let nj = (n - j) % n;
            let nidx = (nj * n + ni) as usize;
            let conj_re = h0_re[nidx];
            let conj_im = -h0_im[nidx];
            for v in [h0_re[idx], h0_im[idx], conj_re, conj_im] {
                h0_packed.extend_from_slice(&v.to_le_bytes());
            }
            for v in [omega[idx], kx_arr[idx], kz_arr[idx], 0.0_f32] {
                omega_packed.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    (h0_packed, omega_packed)
}

struct SimpleLcg(u32);

impl SimpleLcg {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
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

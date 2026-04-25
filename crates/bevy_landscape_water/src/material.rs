use bevy::{
    asset::{load_internal_asset, uuid_handle},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
        MeshPipelineKey,
    },
    prelude::*,
    reflect::{std_traits::ReflectDefault, Reflect},
    render::{render_asset::*, render_resource::*, texture::GpuImage},
    shader::*,
};
use bevy_landscape::MAX_SUPPORTED_CLIPMAP_LEVELS;

pub type StandardWaterMaterial = ExtendedMaterial<StandardMaterial, WaterMaterial>;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
#[uniform(100, WaterMaterialUniform)]
#[bind_group_data(WaterMaterialKey)]
#[reflect(Default, Debug)]
pub struct WaterMaterial {
    /// Water clarity: 0.0 = fully opaque, 1.0 = crystal clear.
    pub clarity: f32,
    /// Water colour at depth.
    pub deep_color: Color,
    /// Water colour in very shallow water.
    pub shallow_color: Color,
    /// Colour of the edge/shoreline effect.
    pub edge_color: Color,
    /// Scale of the water edge effect.
    pub edge_scale: f32,
    /// Global Gerstner wave amplitude (world units).
    pub amplitude: f32,
    /// Global wave speed multiplier (1.0 = physically correct dispersion).
    pub wave_speed: f32,
    /// Quality level (drives the QUALITY shader def; 1–4).
    pub quality: u32,
    /// Screen-space pixel offset for refraction distortion (≈15 px).
    pub refraction_strength: f32,
    /// Wave height above which crest foam appears (in amplitude units).
    pub foam_threshold: f32,
    /// Colour of wave-crest and shoreline foam.
    pub foam_color: Color,
    /// Maximum water depth (metres) at which shoreline foam appears.
    /// Foam is full at 0 m depth and fades to zero at this depth.
    pub shoreline_foam_depth: f32,
    /// Dominant wave / wind direction in world XZ.
    pub wave_direction: Vec2,
    /// Undisplaced water surface height in world Y.
    pub water_height: f32,
    /// Water depth range over which shore displacement fades out.
    pub shore_wave_damp_width: f32,
    /// Multiplier on Jacobian foldover foam (0 = off, 1 = default).
    pub jacobian_foam_strength: f32,
    /// Multiplier on capillary noise normal layer (0 = off, 1 = default).
    pub capillary_strength: f32,
    /// Macro-noise vertical amplitude in metres (breaks Gerstner repetition).
    pub macro_noise_amplitude: f32,
    /// Macro-noise dominant wavelength in metres.
    pub macro_noise_scale: f32,
    /// Terrain height clipmap bound for shoreline damping.
    #[texture(101, visibility(vertex, fragment), dimension = "2d_array")]
    #[sampler(102, visibility(vertex, fragment))]
    #[reflect(ignore)]
    pub terrain_height_texture: Handle<Image>,
    /// Terrain world bounds: (min_x, min_z, max_x, max_z).
    #[reflect(ignore)]
    pub terrain_world_bounds: Vec4,
    /// Terrain height scale in world-space metres.
    #[reflect(ignore)]
    pub terrain_height_scale: f32,
    /// Number of active terrain clipmap levels.
    #[reflect(ignore)]
    pub terrain_num_levels: u32,
    /// Per-level terrain clipmap data mirrored from the terrain material.
    #[reflect(ignore)]
    pub terrain_clip_levels: [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS],

    /// Tessendorf FFT displacement texture: RGBA16Float (h, dx, dz, jacobian),
    /// 2-D array with one layer per cascade.  Each cascade tiles at its own
    /// `cascade_world_sizes[k]` period in world XZ.
    #[texture(103, visibility(vertex, fragment), dimension = "2d_array")]
    #[sampler(104, visibility(vertex, fragment))]
    #[reflect(ignore)]
    pub fft_displacement_texture: Handle<Image>,
    /// Per-cascade tile size in metres (xy = cascade 0/1, zw reserved).
    #[reflect(ignore)]
    pub fft_cascade_world_sizes: Vec4,
    /// Mix strength: 0 = ignore FFT (legacy Gerstner only), 1 = full FFT.
    #[reflect(ignore)]
    pub fft_strength: f32,
    /// FFT texture grid resolution N (used to derive per-texel UV step
    /// for finite-difference slope/Jacobian sampling).
    #[reflect(ignore)]
    pub fft_size: u32,
}

impl Default for WaterMaterial {
    fn default() -> Self {
        Self {
            clarity: 0.1,
            deep_color: Color::srgba(0.2, 0.41, 0.54, 0.92),
            shallow_color: Color::srgba(0.45, 0.78, 0.81, 1.0),
            edge_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
            edge_scale: 0.1,
            amplitude: 1.0,
            wave_speed: 1.0,
            quality: 4,
            refraction_strength: 15.0,
            foam_threshold: 0.6,
            foam_color: Color::srgba(1.0, 1.0, 1.0, 0.9),
            shoreline_foam_depth: 2.0,
            wave_direction: Vec2::X,
            water_height: 0.0,
            shore_wave_damp_width: 5.5,
            jacobian_foam_strength: 1.0,
            capillary_strength: 1.0,
            macro_noise_amplitude: 2.0,
            macro_noise_scale: 110.0,
            terrain_height_texture: Handle::default(),
            terrain_world_bounds: Vec4::ZERO,
            terrain_height_scale: 0.0,
            terrain_num_levels: 0,
            terrain_clip_levels: [Vec4::ZERO; MAX_SUPPORTED_CLIPMAP_LEVELS],
            fft_displacement_texture: Handle::default(),
            fft_cascade_world_sizes: Vec4::new(256.0, 64.0, 0.0, 0.0),
            fft_strength: 0.0,
            fft_size: 128,
        }
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct WaterMaterialKey {
    quality: u32,
    fft_enabled: bool,
}

impl From<&WaterMaterial> for WaterMaterialKey {
    fn from(m: &WaterMaterial) -> Self {
        Self {
            quality: m.quality,
            fft_enabled: m.fft_strength > 0.0,
        }
    }
}

// Field order MUST match the WGSL WaterMaterial struct in water_bindings.wgsl.
#[derive(Clone, Default, ShaderType)]
pub struct WaterMaterialUniform {
    pub deep_color: Vec4,
    pub shallow_color: Vec4,
    pub edge_color: Vec4,
    pub foam_color: Vec4,
    pub wave_direction: Vec4,
    pub terrain_world_bounds: Vec4,
    pub wave_params: Vec4,
    pub optical_params: Vec4,
    pub extra_params: Vec4,
    pub terrain_params: Vec4,
    /// x = fft_strength, y = 1/N (UV per texel), zw = reserved.
    pub fft_params: Vec4,
    /// Per-cascade tile size in metres (xy = cascade 0/1, zw reserved).
    pub fft_cascade_world_sizes: Vec4,
    /// 1.0 / fft_cascade_world_sizes (matched components).  Pre-divided to
    /// save divides on the fragment shader hot path.
    pub fft_cascade_inv_world_sizes: Vec4,
    pub terrain_clip_levels: [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS],
}

impl AsBindGroupShaderType<WaterMaterialUniform> for WaterMaterial {
    fn as_bind_group_shader_type(&self, _images: &RenderAssets<GpuImage>) -> WaterMaterialUniform {
        WaterMaterialUniform {
            deep_color: self.deep_color.to_linear().to_vec4(),
            shallow_color: self.shallow_color.to_linear().to_vec4(),
            edge_color: self.edge_color.to_linear().to_vec4(),
            foam_color: self.foam_color.to_linear().to_vec4(),
            wave_direction: self.wave_direction.extend(0.0).extend(0.0),
            terrain_world_bounds: self.terrain_world_bounds,
            wave_params: Vec4::new(
                self.amplitude,
                self.clarity,
                self.edge_scale,
                self.wave_speed,
            ),
            optical_params: Vec4::new(
                self.refraction_strength,
                self.foam_threshold,
                self.shoreline_foam_depth,
                self.shore_wave_damp_width,
            ),
            extra_params: Vec4::new(
                self.jacobian_foam_strength,
                self.capillary_strength,
                self.macro_noise_amplitude,
                self.macro_noise_scale.max(1.0),
            ),
            terrain_params: Vec4::new(
                self.water_height,
                self.terrain_height_scale,
                self.terrain_num_levels as f32,
                0.0,
            ),
            fft_params: Vec4::new(
                self.fft_strength,
                1.0 / self.fft_size.max(1) as f32,
                0.0,
                0.0,
            ),
            fft_cascade_world_sizes: self.fft_cascade_world_sizes,
            fft_cascade_inv_world_sizes: Vec4::new(
                if self.fft_cascade_world_sizes.x > 0.0 { 1.0 / self.fft_cascade_world_sizes.x } else { 0.0 },
                if self.fft_cascade_world_sizes.y > 0.0 { 1.0 / self.fft_cascade_world_sizes.y } else { 0.0 },
                if self.fft_cascade_world_sizes.z > 0.0 { 1.0 / self.fft_cascade_world_sizes.z } else { 0.0 },
                if self.fft_cascade_world_sizes.w > 0.0 { 1.0 / self.fft_cascade_world_sizes.w } else { 0.0 },
            ),
            terrain_clip_levels: self.terrain_clip_levels,
        }
    }
}

// ---------------------------------------------------------------------------
// Shader handles — unique UUIDs, no collision with bevy_water handles.
// ---------------------------------------------------------------------------

pub const WATER_BINDINGS_HANDLE: Handle<Shader> =
    uuid_handle!("d3f8a1c2-4e56-7890-abcd-ef1234567890");

pub const WATER_FUNCTIONS_HANDLE: Handle<Shader> =
    uuid_handle!("d3f8a1c2-4e56-7890-abcd-ef1234567891");

pub const WATER_VERTEX_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("d3f8a1c2-4e56-7890-abcd-ef1234567892");

pub const WATER_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("d3f8a1c2-4e56-7890-abcd-ef1234567893");

impl MaterialExtension for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        WATER_VERTEX_SHADER_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        WATER_FRAGMENT_SHADER_HANDLE.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        WATER_VERTEX_SHADER_HANDLE.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        WATER_FRAGMENT_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let quality = ShaderDefVal::UInt("QUALITY".into(), key.bind_group_data.quality);
        let fft_def: Option<ShaderDefVal> = if key.bind_group_data.fft_enabled {
            Some("OCEAN_FFT_ENABLED".into())
        } else {
            None
        };
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push(quality.clone());
            if key.mesh_key.contains(MeshPipelineKey::DEPTH_PREPASS) {
                fragment.shader_defs.push("DEPTH_PREPASS".into());
            }
            if let Some(d) = fft_def.clone() {
                fragment.shader_defs.push(d);
            }
        }
        descriptor.vertex.shader_defs.push(quality);
        if let Some(d) = fft_def {
            descriptor.vertex.shader_defs.push(d);
        }
        Ok(())
    }
}

#[derive(Default, Clone, Debug)]
pub struct WaterMaterialPlugin;

impl Plugin for WaterMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            WATER_BINDINGS_HANDLE,
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/shaders/water_bindings.wgsl"
            ),
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            WATER_FUNCTIONS_HANDLE,
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/shaders/water_functions.wgsl"
            ),
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            WATER_VERTEX_SHADER_HANDLE,
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/shaders/water_vertex.wgsl"
            ),
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            WATER_FRAGMENT_SHADER_HANDLE,
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/shaders/water_fragment.wgsl"
            ),
            Shader::from_wgsl
        );

        app.add_plugins(MaterialPlugin::<StandardWaterMaterial>::default())
            .register_asset_reflect::<StandardWaterMaterial>();
    }
}

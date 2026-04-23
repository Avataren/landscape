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
        }
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct WaterMaterialKey {
    quality: u32,
}

impl From<&WaterMaterial> for WaterMaterialKey {
    fn from(m: &WaterMaterial) -> Self {
        Self { quality: m.quality }
    }
}

// Field order MUST match the WGSL WaterMaterial struct in water_bindings.wgsl.
// Layout (encase):
//   vec4 × 5   offsets   0–79
//   f32  × 7   offsets  80–107  (4-byte implicit end-padding → 112 bytes)
#[derive(Clone, Default, ShaderType)]
pub struct WaterMaterialUniform {
    pub deep_color: Vec4,
    pub shallow_color: Vec4,
    pub edge_color: Vec4,
    pub foam_color: Vec4,
    pub wave_direction: Vec4,
    pub amplitude: f32,
    pub clarity: f32,
    pub edge_scale: f32,
    pub wave_speed: f32,
    pub refraction_strength: f32,
    pub foam_threshold: f32,
    pub shoreline_foam_depth: f32,
}

impl AsBindGroupShaderType<WaterMaterialUniform> for WaterMaterial {
    fn as_bind_group_shader_type(&self, _images: &RenderAssets<GpuImage>) -> WaterMaterialUniform {
        WaterMaterialUniform {
            deep_color: self.deep_color.to_linear().to_vec4(),
            shallow_color: self.shallow_color.to_linear().to_vec4(),
            edge_color: self.edge_color.to_linear().to_vec4(),
            foam_color: self.foam_color.to_linear().to_vec4(),
            wave_direction: self.wave_direction.extend(0.0).extend(0.0),
            amplitude: self.amplitude,
            clarity: self.clarity,
            edge_scale: self.edge_scale,
            wave_speed: self.wave_speed,
            refraction_strength: self.refraction_strength,
            foam_threshold: self.foam_threshold,
            shoreline_foam_depth: self.shoreline_foam_depth,
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
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push(quality.clone());
            if key.mesh_key.contains(MeshPipelineKey::DEPTH_PREPASS) {
                fragment.shader_defs.push("DEPTH_PREPASS".into());
            }
        }
        descriptor.vertex.shader_defs.push(quality);
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

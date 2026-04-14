use bevy::{
    asset::RenderAssetUsages,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, ShaderType, Extent3d, RenderPipelineDescriptor,
        SpecializedMeshPipelineError, TextureDimension, TextureFormat,
    },
    shader::ShaderRef,
};

// ---------------------------------------------------------------------------
// Uniform struct (must match TerrainParams in terrain_common.wgsl exactly)
// ---------------------------------------------------------------------------

/// Per-material terrain parameters uploaded to the GPU each frame.
#[derive(Clone, Debug, Default, ShaderType)]
pub struct TerrainMaterialUniforms {
    /// World-space Y multiplier for the [0,1] heightmap values.
    pub height_scale: f32,
    /// World-space XZ extent covered by the height texture (square).
    pub world_size: f32,
    /// World-space X coordinate of the height texture's minimum corner.
    pub world_offset_x: f32,
    /// World-space Z coordinate of the height texture's minimum corner.
    pub world_offset_z: f32,
    /// Number of patches per ring edge (used for LOD morph band calculation).
    pub ring_patches: f32,
    /// Fraction of the ring at which LOD morphing starts (0..1).
    pub morph_start_ratio: f32,
    /// Number of vertices per patch edge (for coarse-grid snap in vertex shader).
    pub patch_resolution: f32,
    /// Alignment padding (unused).
    pub pad0: f32,
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Custom terrain material: height texture + slope/altitude fragment shading.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TerrainMaterial {
    /// Height texture. Sampled in the vertex stage to displace Y.
    /// Visibility includes vertex so the shader can read it during VS.
    #[texture(0, visibility(vertex, fragment))]
    #[sampler(1, visibility(vertex, fragment))]
    pub height_texture: Handle<Image>,

    /// World-scale uniforms. Bound in both stages.
    #[uniform(2)]
    pub params: TerrainMaterialUniforms,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/terrain_vertex.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_fragment.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Disable backface culling so steep terrain faces remain visible.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Height texture generation
// ---------------------------------------------------------------------------

/// Generates a `size × size` procedural height texture using multi-octave
/// sine waves.  Returns a normalized [0,1] Rgba8Unorm image (R channel = height).
pub fn generate_height_image(size: u32) -> Image {
    let texels = (size * size) as usize;
    let mut data = Vec::with_capacity(texels * 4);

    use std::f32::consts::TAU;
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            // Multi-octave hills (sums to [-1, 1] range)
            let h = 0.50 * (u * TAU * 2.0).sin() * (v * TAU * 2.0).cos()
                  + 0.25 * (u * TAU * 4.0 + 1.3).cos() * (v * TAU * 4.0 + 0.7).sin()
                  + 0.12 * (u * TAU * 8.0 + 0.5).sin() * (v * TAU * 8.0 + 2.1).cos()
                  + 0.06 * (u * TAU * 16.0).cos() * (v * TAU * 13.0).sin()
                  + 0.03 * (u * TAU * 32.0 + 0.9).sin() * (v * TAU * 27.0 + 1.5).cos();

            // Map [-1,1] → [0,1], clamp, encode as u8.
            let norm = ((h + 1.0) * 0.5).clamp(0.0, 1.0);
            let byte = (norm * 255.0) as u8;
            data.extend_from_slice(&[byte, byte, byte, 255]);
        }
    }

    Image::new(
        Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    )
}

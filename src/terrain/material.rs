use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, ShaderType,
        SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};
use crate::terrain::config::MAX_SUPPORTED_CLIPMAP_LEVELS;

// ---------------------------------------------------------------------------
// Uniform struct — must match TerrainParams in both WGSL shaders exactly.
//
// Memory layout (WGSL std140 rules):
//   offset  0 – height_scale      f32
//   offset  4 – base_patch_size   f32
//   offset  8 – morph_start_ratio f32
//   offset 12 – ring_patches      f32
//   offset 16 – num_lod_levels    f32   (= clipmap_levels, used to clamp coarse index)
//   offset 20 – patch_resolution  f32
//   offset 24 – world_bounds      vec4<f32>   (min_x, min_z, max_x, max_z)
//   offset 40 – bounds_fade       vec4<f32>   (fade_distance, use_macro_color_map, unused, unused)
//   offset 56 – clip_levels[0]    vec4<f32>
//   offset 72 – clip_levels[1]    vec4<f32>
//   …
//   offset296 – clip_levels[15]   vec4<f32>
//   Total: 312 bytes
//
// Each clip_levels entry: (origin_x, origin_z, inv_ring_span, texel_world_size)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, ShaderType)]
pub struct TerrainMaterialUniforms {
    /// World-space Y multiplier for [0,1] heightmap values.
    pub height_scale: f32,
    /// World-space size of one LOD-0 patch side (patch_resolution × world_scale).
    pub base_patch_size: f32,
    /// Fraction of the ring at which LOD morphing starts (0..1).
    pub morph_start_ratio: f32,
    /// Number of patches per ring edge side.
    pub ring_patches: f32,
    /// Number of active LOD levels (= clipmap_levels).  Used to clamp the coarse
    /// LOD index so we never read beyond the texture array bounds.
    pub num_lod_levels: f32,
    /// Vertex resolution per patch edge (number of quads).
    pub patch_resolution: f32,
    /// Terrain footprint in world space: (min_x, min_z, max_x, max_z).
    pub world_bounds: Vec4,
    /// Bounds fade params: x = fade distance beyond the footprint.
    pub bounds_fade: Vec4,
    /// Per-LOD clipmap data: (origin_x, origin_z, inv_ring_span, texel_world_size).
    /// Indexed by LOD level (0 = finest).  Unused entries are zero.
    pub clip_levels: [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS],
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Custom terrain material: per-level clipmap height array + slope/altitude shading.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct TerrainMaterial {
    /// R8Unorm texture array — one 2-D layer per LOD level.
    /// Sampled in the vertex stage to displace Y.
    #[texture(0, visibility(vertex, fragment), dimension = "2d_array")]
    #[sampler(1, visibility(vertex, fragment))]
    pub height_texture: Handle<Image>,

    /// World-scale uniforms shared between vertex and fragment stages.
    #[uniform(2)]
    pub params: TerrainMaterialUniforms,

    /// World-aligned macro/diffuse color map used for terrain albedo.
    #[texture(3, visibility(fragment))]
    #[sampler(4, visibility(fragment))]
    pub macro_color_texture: Handle<Image>,

    /// RG8Snorm texture array containing baked XZ normals per LOD level.
    #[texture(5, visibility(vertex), dimension = "2d_array")]
    #[sampler(6, visibility(vertex))]
    pub normal_texture: Handle<Image>,
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

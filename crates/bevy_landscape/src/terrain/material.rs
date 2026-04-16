use crate::terrain::config::MAX_SUPPORTED_CLIPMAP_LEVELS;
use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        render_resource::{
            AsBindGroup, Face, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError,
        },
        storage::ShaderStorageBuffer,
    },
    shader::ShaderRef,
};

/// Maximum material slots exposed to the shader.  Matches the size of
/// `TerrainParams.slots` in the WGSL shaders and the practical cap called out
/// in material_system.md §1 ("8 slots = 2 splatmaps").
pub const MAX_SHADER_MATERIAL_SLOTS: usize = 8;

/// Per-slot data uploaded to the fragment shader for the procedural baseline
/// blend.  No texture handles yet — that comes once the texture arrays are
/// introduced (§12 step 1 "texture arrays" phase).
///
/// Layout (std140 uniform):
///   offset  0 – tint_vis  vec4   (rgb = tint, a = visibility 0/1)
///   offset 16 – ranges    vec4   (x = alt_min, y = alt_max, z = slope_min°, w = slope_max°)
///   Total: 32 bytes, naturally 16-aligned.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct MaterialSlotGpu {
    pub tint_vis: Vec4,
    pub ranges: Vec4,
}

// ---------------------------------------------------------------------------
// Uniform struct — must match TerrainParams in all three WGSL shaders exactly.
//
// Memory layout (WGSL std140 rules):
//   offset  0 – height_scale      f32
//   offset  4 – base_patch_size   f32
//   offset  8 – morph_start_ratio f32
//   offset 12 – ring_patches      f32
//   offset 16 – num_lod_levels    f32   (= clipmap_levels, used to clamp coarse index)
//   offset 20 – patch_resolution  f32
//   offset 24 – world_bounds      vec4<f32>   (min_x, min_z, max_x, max_z)
//   offset 40 – bounds_fade       vec4<f32>   (fade_distance, use_macro_color_map, flip_v, show_wireframe)
//   offset 56 – debug_flags       vec4<f32>   (show_normals_only, _, _, _)
//   offset 72 – clip_levels[0]    vec4<f32>
//   offset 88 – clip_levels[1]    vec4<f32>
//   …
//   offset312 – clip_levels[15]   vec4<f32>
//   Total: 328 bytes
//
// Each clip_levels entry: (origin_x, origin_z, inv_ring_span, texel_world_size)
// Note: lighting data is NOT stored here — the fragment shader uses Bevy's
// pbr_functions::apply_pbr_lighting which reads from mesh_view_bindings::lights.
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
    /// Bounds/debug params:
    /// x = fade distance, y = use_macro_color, z = flip_v, w = show_wireframe.
    pub bounds_fade: Vec4,
    /// Per-pixel debug toggles applied in the fragment shader.
    /// x = show_normals_only (0 = normal shading, 1 = output `n*0.5+0.5` as colour).
    /// y = use_baked_normals (1 = sample baked RG8Snorm normal array for shading,
    ///     0 = recompute via finite-differences on the height clipmap).
    /// z, w = reserved.
    pub debug_flags: Vec4,
    /// Per-LOD clipmap data: (origin_x, origin_z, inv_ring_span, texel_world_size).
    /// Indexed by LOD level (0 = finest).  Unused entries are zero.
    pub clip_levels: [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS],
    /// Material slot header: x = active slot count, yzw = reserved.
    ///
    /// Packed as a `Vec4` so the following `slots` array sits on a natural
    /// 16-byte boundary — matches the WGSL std140 layout without manual
    /// padding tricks.
    pub slot_header: Vec4,
    /// Per-slot procedural blend data.  Unused entries have `visibility = 0`.
    pub slots: [MaterialSlotGpu; MAX_SHADER_MATERIAL_SLOTS],
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
    ///
    /// Visible to both stages: the vertex stage used to derive geometry
    /// normals from finite differences (kept as a fallback), and the fragment
    /// stage samples this directly when `debug_flags.y = 1.0` so shading is
    /// driven by the high-precision f32 normals produced at bake time instead
    /// of the R16Unorm-quantized height clipmap.
    #[texture(5, visibility(vertex, fragment), dimension = "2d_array")]
    #[sampler(6, visibility(vertex, fragment))]
    pub normal_texture: Handle<Image>,

    /// One `PatchDescriptorGpu` per patch instance, indexed by `instance_index`
    /// in the vertex shader.  Replaces the Transform-matrix decode and provides
    /// the explicit `lod_level` needed by the storage-buffer path.
    #[storage(7, read_only, visibility(vertex))]
    pub patch_buffer: Handle<ShaderStorageBuffer>,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/terrain_vertex.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_fragment.wgsl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/terrain_prepass.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Terrain is a heightfield — all triangles have consistent CCW winding
        // when viewed from above, so hardware backface culling is safe and halves
        // the rasterization cost for steep slopes viewed from the correct side.
        descriptor.primitive.cull_mode = Some(Face::Back);
        Ok(())
    }
}

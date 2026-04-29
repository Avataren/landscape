// terrain_common.wgsl
// Canonical WGSL struct definitions for the terrain material uniform.
// Must stay in sync with TerrainMaterialUniforms / MaterialSlotGpu in
// crates/bevy_landscape/src/terrain/material.rs.
//
// This file is NOT currently auto-imported — the three terrain shaders
// (terrain_vertex.wgsl, terrain_fragment.wgsl, terrain_prepass.wgsl) each
// embed a copy.  When a change is needed, update all four files together.
//
// Memory layout (encase/WGSL uniform rules):
//   offset   0 – height_scale      f32
//   offset   4 – base_patch_size   f32
//   offset   8 – morph_start_ratio f32
//   offset  12 – ring_patches      f32
//   offset  16 – num_lod_levels    f32
//   offset  20 – patch_resolution  f32
//   offset  24 – (8 bytes padding — next field is vec4, align 16)
//   offset  32 – world_bounds      vec4<f32>  (min_x, min_z, max_x, max_z)
//   offset  48 – bounds_fade       vec4<f32>  (fade_dist, use_macro_color, flip_v, show_wireframe)
//   offset  64 – debug_flags       vec4<f32>  (x = fragment_debug_mode, y = use_baked_normals, zw reserved)
//   offset  80 – clip_levels[0]    vec4<f32>  (origin_x, origin_z, inv_ring_span, texel_world_size)
//     ...
//   offset 592 – clip_levels[31]
//   offset 608 – slot_header       vec4<f32>  (x = active slot count)
//   offset 624 – slots[0]          MaterialSlotGpu (48 bytes)
//     ...
//   offset 960 – slots[7]
//   offset 1008 – synthesis_norm   vec4<f32>  (seed_x, seed_z, base_freq, octave_count)
//   offset 1024 – synthesis_norm2  vec4<f32>  (lacunarity, gain, erosion, normal_strength)
//   offset 1040 – source_meta      vec4<f32>  (origin_x, origin_z, extent_x, extent_z)
//   Total: 1056 bytes

// Per-material-slot procedural blend parameters.
// Layout: 3 × vec4 = 48 bytes, align 16.
struct MaterialSlotGpu {
    tint_vis: vec4<f32>,  // rgb = tint, a = visibility (0/1)
    ranges:   vec4<f32>,  // x = alt_min, y = alt_max, z = slope_min°, w = slope_max°
    uv_scale: vec4<f32>,  // x = fine_scale_m, y = coarse_scale_mul, z = has_tex (0/1), w = reserved
}

struct TerrainParams {
    height_scale:       f32,
    base_patch_size:    f32,
    morph_start_ratio:  f32,
    ring_patches:       f32,
    num_lod_levels:     f32,
    patch_resolution:   f32,
    world_bounds:       vec4<f32>, // (min_x, min_z, max_x, max_z)
    bounds_fade:        vec4<f32>, // x = fade_dist, y = use_macro_color, z = flip_v, w = show_wireframe
    debug_flags:        vec4<f32>, // x = fragment_debug_mode, y = use_baked_normals, zw reserved
    clip_levels:        array<vec4<f32>, 32>,
    slot_header:        vec4<f32>, // x = active slot count
    slots:              array<MaterialSlotGpu, 8>,
    synthesis_norm:     vec4<f32>, // x=seed_x, y=seed_z, z=base_freq, w=octave_count
    synthesis_norm2:    vec4<f32>, // x=lacunarity, y=gain, z=erosion, w=normal_strength
    source_meta:        vec4<f32>, // xy=world_origin, zw=world_extent
}

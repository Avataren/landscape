#define_import_path bevy_landscape_water::water_bindings

// Uniform layout (must match WaterMaterialUniform in material.rs exactly).
struct WaterMaterial {
    deep_color:           vec4<f32>,
    shallow_color:        vec4<f32>,
    edge_color:           vec4<f32>,
    foam_color:           vec4<f32>,
    // xy = normalised wind / dominant wave direction in world XZ.
    wave_direction:       vec4<f32>,
    terrain_world_bounds: vec4<f32>,
    // x = amplitude, y = clarity, z = edge_scale, w = wave_speed
    wave_params:          vec4<f32>,
    // x = refraction_strength, y = foam_threshold,
    // z = shoreline_foam_depth, w = shore_wave_damp_width
    optical_params:       vec4<f32>,
    // x = water_height, y = terrain_height_scale,
    // z = terrain_num_levels, w = reserved
    terrain_params:       vec4<f32>,
    terrain_clip_levels:  array<vec4<f32>, 16>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> material: WaterMaterial;

@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var terrain_height_tex: texture_2d_array<f32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var terrain_height_samp: sampler;

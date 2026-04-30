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
    // x = jacobian_foam_strength, y = capillary_strength,
    // z = macro_noise_amplitude (m), w = macro_noise_scale (m)
    extra_params:         vec4<f32>,
    // x = water_height, y = terrain_height_scale,
    // z = terrain_num_levels, w = reserved
    terrain_params:       vec4<f32>,
    // x = fft_strength, y = 1/fft_grid_size (UV per texel), zw = reserved.
    fft_params:                  vec4<f32>,
    // Per-cascade tile size in metres (xy = cascade 0/1, zw reserved).
    fft_cascade_world_sizes:     vec4<f32>,
    // 1 / fft_cascade_world_sizes per component (cascade UV scale).
    fft_cascade_inv_world_sizes: vec4<f32>,
    // x = ssr_enabled (0=off 1=on), y = ssr_steps, z = ssr_max_distance (m), w = ssr_thickness (m)
    ssr_params:                  vec4<f32>,
    terrain_clip_levels:         array<vec4<f32>, 16>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> material: WaterMaterial;

@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var terrain_height_tex: texture_2d_array<f32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var terrain_height_samp: sampler;

#ifdef OCEAN_FFT_ENABLED
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var fft_displacement_tex: texture_2d_array<f32>;

@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var fft_displacement_samp: sampler;
#endif

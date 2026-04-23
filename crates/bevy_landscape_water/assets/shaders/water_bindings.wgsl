#define_import_path bevy_landscape_water::water_bindings

// Uniform layout (must match WaterMaterialUniform in material.rs exactly):
// vec4 × 5  = 80 bytes  (offsets  0–79)
// f32  × 7  = 28 bytes  (offsets 80–107)
// 4-byte implicit padding → 112 bytes total
struct WaterMaterial {
    deep_color:           vec4<f32>,
    shallow_color:        vec4<f32>,
    edge_color:           vec4<f32>,
    foam_color:           vec4<f32>,
    // xy = normalised wind / dominant wave direction in world XZ.
    wave_direction:       vec4<f32>,
    amplitude:            f32,
    clarity:              f32,
    edge_scale:           f32,
    wave_speed:           f32,
    refraction_strength:  f32,
    foam_threshold:       f32,
    // Maximum water depth (in metres) that receives shoreline foam.
    // Foam fades from full at depth=0 to zero at depth=shoreline_foam_depth.
    shoreline_foam_depth: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> material: WaterMaterial;

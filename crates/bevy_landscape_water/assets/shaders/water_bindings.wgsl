#define_import_path bevy_landscape_water::water_bindings

// Uniform layout (must match WaterMaterialUniform in material.rs exactly):
// vec4 × 4  = 64 bytes  (offsets  0–63)
// f32  × 7  = 28 bytes  (offsets 64–91)
// 4-byte implicit padding → 96 bytes total
struct WaterMaterial {
    deep_color:           vec4<f32>,
    shallow_color:        vec4<f32>,
    edge_color:           vec4<f32>,
    foam_color:           vec4<f32>,
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

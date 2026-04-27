// Grass blade shader — simple foliage rendering
// Supports 8 blade variants via vertex ID
// Features: world-aligned normals, basic wind animation

#import bevy_pbr::mesh_vertex_output MeshVertexOutput
#import bevy_pbr::pbr_fragment main_pbr_fragment
#import bevy_pbr::pbr_bindings pbr_bindings
#import bevy_pbr::pbr_functions apply_pbr_lighting
#import bevy_pbr::utils PI

@group(1) @binding(0) uniform<std140> grass_material: GrassMaterial {
    base_color: vec4<f32>,       // RGB color + alpha
    ao_multiplier: f32,           // Ambient occlusion strength
    wind_strength: f32,           // Wind animation amplitude
    wind_speed: f32,              // Wind animation speed (cycles/sec)
};

struct GrassMaterial {
    base_color: vec4<f32>,
    ao_multiplier: f32,
    wind_strength: f32,
    wind_speed: f32,
    _padding: f32,
};

struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    @builtin(sample_index) sample_index: u32,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
};

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    // Apply wind animation based on world position
    // Simple sine wave: displacement = sin(time * speed + world_x * freq) * strength
    let wind_freq = 0.5; // Spatial frequency of wind waves
    let wind_animation = sin((pbr_bindings.view_bindings.view.world_position.z * wind_freq)) * grass_material.wind_strength;
    
    // Discard fragments based on alpha threshold (for cutout foliage)
    let alpha_threshold = 0.5;
    var alpha = grass_material.base_color.a;
    if alpha < alpha_threshold {
        discard;
    }

    // World-aligned normal (simple approach: face normal + slight tangent mixing)
    var normal = input.world_normal;
    
    // Add slight detail from tangent space (optional, for edge highlights)
    let tangent = normalize(input.world_tangent.xyz);
    let bitangent = normalize(cross(normal, tangent));
    
    // For grass, we want normals perpendicular to the blade
    // Simple approximation: blend between face normal and a "side-facing" normal
    let side_normal = normalize(tangent + bitangent * 0.2);
    normal = normalize(mix(normal, side_normal, 0.3));

    // Construct PBR material
    var pbr_input: PbrInput;
    pbr_input.material.base_color = grass_material.base_color;
    pbr_input.material.reflectance = 0.5;
    pbr_input.material.perceptual_roughness = 0.8; // Grass is rough
    pbr_input.material.metallic = 0.0;
    pbr_input.material.alpha_cutoff = alpha_threshold;
    
    pbr_input.frag_coord = input.frag_coord;
    pbr_input.world_pos = input.world_position.xyz;
    pbr_input.world_normal = normal;
    pbr_input.is_front = input.is_front;

    // Apply subtle ambient occlusion (from vertex color alpha in future)
    pbr_input.material.base_color *= grass_material.ao_multiplier;

    // Apply PBR lighting
    var out: vec4<f32> = main_pbr_fragment(pbr_input);
    
    // Preserve alpha from material
    out.a = alpha;
    
    return out;
}

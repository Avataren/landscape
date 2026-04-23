#import bevy_pbr::{
  mesh_functions,
  skinning,
  view_transformations::position_world_to_clip,
}

#ifdef PREPASS_PIPELINE
#import bevy_render::globals::Globals
@group(0) @binding(1) var<uniform> globals: Globals;
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#else
#import bevy_pbr::mesh_view_bindings::globals
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#endif

struct WaterMaterial {
  deep_color: vec4<f32>,
  shallow_color: vec4<f32>,
  edge_color: vec4<f32>,
  foam_color: vec4<f32>,
  amplitude: f32,
  clarity: f32,
  edge_scale: f32,
  wave_speed: f32,
  refraction_strength: f32,
  foam_threshold: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> material: WaterMaterial;

const PI: f32 = 3.14159265358979323846;
const G: f32 = 9.81;
const NUM_WAVES: f32 = 8.0;
const AMP_RATIO: f32 = 0.006;

const WAVE_0: vec4<f32> = vec4(1.000, 0.000, 83.7, 0.70);
const WAVE_1: vec4<f32> = vec4(0.921, 0.391, 61.3, 0.65);
const WAVE_2: vec4<f32> = vec4(0.707, 0.707, 47.1, 0.60);
const WAVE_3: vec4<f32> = vec4(0.966, -0.259, 53.9, 0.60);
const WAVE_4: vec4<f32> = vec4(0.809, 0.588, 31.7, 0.50);
const WAVE_5: vec4<f32> = vec4(0.891, -0.454, 23.3, 0.45);
const WAVE_6: vec4<f32> = vec4(0.978, 0.208, 17.1, 0.40);
const WAVE_7: vec4<f32> = vec4(0.743, -0.669, 11.9, 0.35);

struct WaveContribution {
  displacement: vec3<f32>,
  normal_terms: vec3<f32>,
}

struct WaveResult {
  displacement: vec3<f32>,
  normal: vec3<f32>,
}

fn gerstner_wave(p: vec2<f32>, params: vec4<f32>) -> WaveContribution {
  let dir = params.xy;
  let wavelength_m = params.z;
  let steepness = params.w;

  let omega = 2.0 * PI / wavelength_m;
  let amplitude = material.amplitude * AMP_RATIO * wavelength_m;
  let phase_speed = material.wave_speed * sqrt(G * omega);

  let phase = omega * dot(dir, p) + phase_speed * globals.time;
  let sin_phase = sin(phase);
  let cos_phase = cos(phase);

  let q = steepness / (omega * amplitude * NUM_WAVES);
  let wa = omega * amplitude;

  return WaveContribution(
    vec3(q * amplitude * dir.x * cos_phase, amplitude * sin_phase, q * amplitude * dir.y * cos_phase),
    vec3(q * wa * dir.x * sin_phase, wa * cos_phase, q * wa * dir.y * sin_phase),
  );
}

fn accumulate_wave(accum: WaveResult, wave: WaveContribution) -> WaveResult {
  return WaveResult(accum.displacement + wave.displacement, accum.normal + wave.normal_terms);
}

fn get_wave_result(p: vec2<f32>) -> WaveResult {
  var accum = WaveResult(vec3(0.0), vec3(0.0));

  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_0));
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_1));
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_2));
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_3));
#if QUALITY > 1
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_4));
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_5));
#endif
#if QUALITY > 2
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_6));
  accum = accumulate_wave(accum, gerstner_wave(p, WAVE_7));
#endif

  return WaveResult(
    accum.displacement,
    normalize(vec3(-accum.normal.x, 1.0 - accum.normal.y, -accum.normal.z)),
  );
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
  var out: VertexOutput;

#ifdef SKINNED
  let model = skinning::skin_model(vertex.joint_indices, vertex.joint_weights);
#else
  let model = mesh_functions::get_world_from_local(vertex.instance_index);
#endif

  let base_world_position = mesh_functions::mesh_position_local_to_world(
    model,
    vec4<f32>(vertex.position, 1.0),
  );
  let wave = get_wave_result(base_world_position.xz);

  out.world_position = base_world_position + vec4<f32>(wave.displacement, 0.0);
  out.world_normal = wave.normal;
  out.position = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_UVS
  out.uv = vertex.uv;
#endif

#ifdef VERTEX_TANGENTS
  out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
    model,
    vertex.tangent,
    vertex.instance_index,
  );
#endif

#ifdef VERTEX_COLORS
  out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
  out.instance_index = vertex.instance_index;
#endif

  return out;
}

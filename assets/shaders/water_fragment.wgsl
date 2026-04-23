#import bevy_pbr::{
  pbr_fragment::pbr_input_from_standard_material,
  pbr_functions::alpha_discard,
  mesh_view_bindings::globals,
  mesh_view_bindings::view,
}

#ifdef DEPTH_PREPASS
#import bevy_pbr::{
  prepass_utils,
  view_transformations::{frag_coord_to_ndc, position_ndc_to_world},
}
#endif

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
  prepass_io::{FragmentOutput, VertexOutput},
  pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
  forward_io::{FragmentOutput, VertexOutput},
  pbr_functions,
  pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
  pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}
#endif

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output
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

fn saturate(x: f32) -> f32 {
  return clamp(x, 0.0, 1.0);
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

@fragment
fn fragment(
#ifdef MESHLET_MESH_MATERIAL_PASS
  @builtin(position) frag_coord: vec4<f32>,
#else
  p_in: VertexOutput,
  @builtin(front_facing) is_front: bool,
#endif
) -> FragmentOutput {
#ifdef MESHLET_MESH_MATERIAL_PASS
  let p_in = resolve_vertex_output(frag_coord);
  let is_front = true;
#endif

  var in = p_in;
  let wave = get_wave_result(in.world_position.xz);
  in.world_normal = wave.normal;

  let view_dir = normalize(view.world_position.xyz - in.world_position.xyz);

  var terrain_depth_m = mix(35.0, 180.0, pow(1.0 - saturate(abs(view_dir.y)), 2.0));

#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
  let raw_depth = prepass_utils::prepass_depth(in.position, 0u);
  if raw_depth > 0.0 {
    let terrain_ndc = frag_coord_to_ndc(vec4(in.position.xy, raw_depth, 1.0));
    let terrain_world_y = position_ndc_to_world(terrain_ndc).y;
    if terrain_world_y > in.world_position.y + 0.05 {
      discard;
    }
    terrain_depth_m = max(in.world_position.y - terrain_world_y, 0.0);
  }
#endif
#endif

#ifdef VISIBILITY_RANGE_DITHER
  pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif

  var pbr_input = pbr_input_from_standard_material(in, is_front);

  let clarity = saturate(material.clarity);
  let optical_depth_m = terrain_depth_m / max(abs(view_dir.y), 0.12);

  let sigma_abs_clear = vec3<f32>(0.035, 0.014, 0.007);
  let sigma_abs_murky = vec3<f32>(0.160, 0.080, 0.040);
  let sigma_scat_clear = vec3<f32>(0.004, 0.009, 0.018);
  let sigma_scat_murky = vec3<f32>(0.020, 0.028, 0.036);

  let sigma_abs = mix(sigma_abs_murky, sigma_abs_clear, clarity);
  let sigma_scat = mix(sigma_scat_murky, sigma_scat_clear, clarity);
  let extinction = sigma_abs + sigma_scat;
  let transmittance = exp(-optical_depth_m * extinction);

  let depth_mix = saturate(1.0 - exp(-terrain_depth_m * mix(0.18, 0.035, clarity)));
  let scatter_color = mix(material.shallow_color.rgb, material.deep_color.rgb, depth_mix);
  let opacity = material.deep_color.a * (1.0 - dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722)));
  var water_color = vec4<f32>(scatter_color, max(opacity, 0.02));

  let shore_foam = exp(-terrain_depth_m * 1.2);
  let crest_foam = saturate((1.0 - in.world_normal.y) * 3.0 - material.foam_threshold * 0.35);
  let foam_weight = max(shore_foam * shore_foam, crest_foam * crest_foam);
  water_color = mix(water_color, material.foam_color, foam_weight);

  pbr_input.material.base_color *= water_color;
  pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
  return deferred_output(in, pbr_input);
#else
  var out: FragmentOutput;
  if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
    out.color = apply_pbr_lighting(pbr_input);
  } else {
    out.color = pbr_input.material.base_color;
  }
  out.color = main_pass_post_lighting_processing(pbr_input, out.color);
  return out;
#endif
}

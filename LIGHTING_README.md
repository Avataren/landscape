# Terrain Lighting and Shadow Pipeline

## Overview

The terrain uses a hand-written WGSL lighting pipeline rather than Bevy's standard PBR material. This gives direct control over how the directional sun light and cascade shadow maps are applied to the clipmap terrain.

---

## Scene Light

A single `DirectionalLight` is spawned in `src/main.rs`:

```rust
DirectionalLight {
    illuminance: lux::RAW_SUNLIGHT,
    shadows_enabled: true,
    ..default()
}
Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.25, 0.8, 0.0))
```

- **Illuminance** is set to raw sunlight so the atmosphere plugin handles scattering.
- The X rotation of −0.25 rad places the sun ~14° above the horizon, casting long shadows that reveal terrain relief.
- Y rotation of 0.8 rad sets the azimuth.

### Cascade Shadow Configuration

```rust
CascadeShadowConfigBuilder {
    num_cascades: 4,
    minimum_distance: 1.0,
    first_cascade_far_bound: 500.0,
    maximum_distance: 20_000.0,
    overlap_proportion: 0.2,
}
```

Four cascades cover the range from 1 wu to 20 000 wu. The first cascade boundary at 500 wu keeps close-up shadow detail sharp; the remaining three stretch logarithmically out to 20 km. A 20 % overlap blend zone prevents hard seams between cascades.

---

## Sun Direction Uniform

Because `TerrainMaterial` is a custom material, Bevy's lighting system does not automatically inject light parameters. A dedicated ECS system (`sync_sun_direction` in `src/terrain/mod.rs`) runs every frame:

```rust
let sun_dir = Vec3::from(-transform.forward());
mat.params.sun_direction = sun_dir.normalize_or_zero().extend(0.0);
```

`forward()` is the direction light rays travel (local −Z in world space). Negating it gives the **toward-sun** vector that the dot-product lighting formula expects. The result is stored in `TerrainMaterialUniforms::sun_direction` and uploaded to the GPU as part of the material uniform buffer at binding 2.

---

## Vertex Stage — Normal Derivation

Normals are not read from the pre-baked normal tiles. Instead, `terrain_vertex.wgsl` derives them analytically from height finite-differences at each frame:

```wgsl
fn normal_at(lod: u32, xz: vec2<f32>) -> vec3<f32> {
    let eps = terrain.clip_levels[lod].w;   // texel world size at this LOD
    let h   = height_at(lod, xz);
    let h_r = height_at(lod, xz + vec2<f32>(eps, 0.0));
    let h_u = height_at(lod, xz + vec2<f32>(0.0, eps));
    // tangent_x × tangent_z = (h−h_r, eps, h−h_u)
    return normalize(vec3<f32>(h - h_r, eps, h - h_u));
}
```

The fine and coarse normals are then blended by the same `morph_alpha` used for vertex position morphing, so normal transitions are smooth across LOD ring boundaries.

The vertex shader outputs `world_normal: vec3<f32>` (location 1) and `world_pos: vec4<f32>` (location 0) to the fragment stage.

---

## Shadow Casting — Custom Prepass Shader

Bevy's shadow map pass uses the material's **prepass vertex shader**, not the main vertex shader. Without a custom prepass, the shadow geometry would be a flat slab at Y = 0 (the raw mesh vertex positions before any height displacement), causing the displaced terrain surface to receive no self-shadows.

`assets/shaders/terrain_prepass.wgsl` replicates the full height sampling and CDLOD morphing from the main vertex shader. It only outputs clip position and world position — no normal, no albedo — since the shadow pass only writes depth.

```wgsl
fn vertex(v: Vertex) -> VertexOutput {
    // ... same LOD derivation, morph_alpha, coarse grid snap ...
    let h   = mix(height_at(lod_level, world_xz), height_at(coarse_lod, world_xz), morph_alpha);
    let pos = vec3<f32>(world_xz.x, h, world_xz.y);
    out.world_position = vec4<f32>(pos, 1.0);
    out.position       = position_world_to_clip(pos);
    return out;
}
```

This shader is registered in `src/terrain/material.rs`:

```rust
fn prepass_vertex_shader() -> ShaderRef {
    "shaders/terrain_prepass.wgsl".into()
}
```

---

## Fragment Stage — Lighting and Shadow Sampling

`assets/shaders/terrain_fragment.wgsl` computes the final lit colour in three steps.

### 1. Albedo

If a macro colour map (diffuse EXR) is loaded, it is sampled using world-space XZ projected into `[0, 1]` UV coordinates. Otherwise a procedural slope-and-altitude blend is used (grass → dirt → rock → snow).

### 2. Cascade Shadow

```wgsl
// Depth in view space — used by Bevy to select the correct cascade.
let view_z = dot(vec4<f32>(
    view_bindings::view.view_from_world[0].z,
    view_bindings::view.view_from_world[1].z,
    view_bindings::view.view_from_world[2].z,
    view_bindings::view.view_from_world[3].z,
), in.world_pos);

var shadow = 1.0;
if view_bindings::lights.n_directional_lights > 0u {
    let flags = view_bindings::lights.directional_lights[0].flags;
    if (flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
        shadow = shadows::fetch_directional_shadow(0u, in.world_pos, n, view_z);
    }
}
```

`view_z` is the view-space depth used by `fetch_directional_shadow` to select which cascade covers this fragment. The function handles PCF filtering and cascade blending internally.

Light index `0u` is used because there is exactly one directional light in the scene. The guard on `n_directional_lights` and the shadows-enabled flag ensures no shadow sampling occurs if the light is removed or shadows are toggled off at runtime.

### 3. Final Luminance

```wgsl
let ndotl = max(dot(n, sun), 0.0);
let lit   = 0.18 + ndotl * 0.82 * shadow;
return vec4<f32>(albedo * lit, 1.0);
```

- **0.18** — constant ambient term, unaffected by shadows. Ensures fully-shadowed terrain is dark but not black.
- **0.82** — direct sun contribution, attenuated by `shadow` (0 = fully shadowed, 1 = fully lit).
- Tonemapping (ACES Fitted) and bloom are applied by Bevy's post-process pipeline after the terrain is composited into the HDR framebuffer.

---

## Tuning

| What to change | Where |
|---|---|
| Sun angle / azimuth | `Transform::from_rotation` in `src/main.rs` |
| Shadow range and cascade splits | `CascadeShadowConfigBuilder` in `src/main.rs` |
| Shadow acne (dark stippling on slopes) | Increase `shadow_normal_bias` on the `DirectionalLight` in `src/main.rs` |
| Ambient level | Constant `0.18` in `terrain_fragment.wgsl` |
| Diffuse map V-flip | `macro_color_flip_v = true` in `landscape.toml` |

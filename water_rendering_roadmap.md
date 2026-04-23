# Water Rendering Roadmap

Date: 2026-04-23

## Goal

Replace the current brute-force Gerstner mesh path with a modern ocean pipeline that is:

- Stable in world space with no phase shifts when the camera moves
- Anti-aliased in the distance, with no moire or noisy far-field wave breakup
- Fast enough for large worlds
- Capable of convincing near-camera displacement, foam, reflection, and underwater scattering
- Able to support clear water with visibility on the order of 100 m below the surface

## Current Problems

- Too much wave detail is being pushed directly into geometry, which explodes vertex cost.
- Far-field waves alias into noise and moire because high frequencies remain visible beyond the mesh and pixel Nyquist limits.
- Foam is too sensitive to mesh/facet structure when driven by low-resolution geometry normals.
- The current path does not separate geometric swell from shading-frequency detail.
- The current path is not structured for scalable LOD.

## Core Decision

Do not keep iterating on a single dense Gerstner mesh.

The replacement should be:

1. World-space ocean clipmaps or ring LOD meshes around the camera
2. Compute-generated displacement/normal/foam data
3. Geometry displacement only for low-frequency swell bands
4. Mid/high-frequency waves represented in filtered textures, not raw vertex motion

This is the practical replacement for "GPU tessellation" in the current stack.

## Important Constraint

This Bevy `0.18.1` / `wgpu 27` stack does not expose classic fixed-function hull/domain tessellation stages in the normal render pipeline.

That means the correct path here is:

- GPU-driven clipmap meshes
- Compute-generated wave atlases
- Band-limited sampling by distance

Not:

- Direct DX11-style hardware tessellation shaders

If we later move to mesh shaders or a custom render path, the clipmap/compute design still remains valid.

## Target Architecture

### 1. Geometry: Ocean Clipmaps

Use a small set of reusable ring meshes centered on the camera.

Properties:

- Meshes are camera-centered for coverage only
- All wave sampling remains in absolute world space
- Ring origins snap only for topology alignment, never for wave phase
- Seams are handled with morph regions or stitch triangles
- Near rings are denser, far rings are coarse

Recommended setup:

- 5 to 7 LOD rings
- Constant per-ring resolution, e.g. `128x128` or `192x192`
- Ring scale doubles each level
- One inner patch plus outer rings

This gives bounded vertex cost independent of world size.

### 2. Wave Bands

Split the ocean into frequency bands.

#### Geometry Band

Low-frequency swell only:

- Long wavelengths
- Large amplitude
- Drives true vertex displacement
- Stable silhouette near the horizon

#### Shading Bands

Medium and high frequencies:

- Stored in compute-generated height/slope/normal textures
- Sampled in fragment shader
- Filtered by mip and distance
- Never pushed directly into far-field geometry

This is the key anti-aliasing and performance win.

### 3. Compute Passes

Add compute-driven wave generation for each active clip level.

Per level output textures:

- `height_low` or displacement field for geometric swell
- `normal_mid_high`
- `foam_mask`
- Optional `flow/velocity`

Two viable sources:

#### Option A: Banded Gerstner

Good first implementation.

- Keep existing Gerstner model
- Evaluate it in compute into textures
- Split waves by wavelength band
- Easy migration from current shader code

#### Option B: Spectral Ocean

Better long term.

- Phillips/JONSWAP style frequency spectrum
- FFT or tiled inverse transform
- More realistic directional energy and filtering
- More work

Recommended sequence:

- Start with banded Gerstner in compute
- Move to spectral later if needed

### 4. Anti-Aliasing Strategy

This must be built into the pipeline, not added at the end.

#### Distance-Based Band Culling

Do not render short wavelengths in the far field.

- Near camera: full band stack
- Mid distance: drop highest bands
- Far distance: only long swell survives

#### Mipped Wave Textures

Normals/foam must sample the correct mip level.

- Use derivatives to bias mip selection
- Prevent sub-pixel detail from shimmering

#### Temporal Filtering

Apply temporal accumulation where helpful:

- Foam evolution
- High-frequency normal detail
- Reflections if SSR is used

#### Horizon Simplification

Near the horizon:

- Reduce steepness
- Fade foam
- Favor long swell bands only

This avoids noisy sparkling lines and moire.

### 5. Foam Model

Foam should come from wave behavior, not from mesh faces.

Inputs:

- Curvature
- Slope / breaking threshold
- Crest compression
- Shore depth / contact

Outputs:

- Crest foam
- Persistent trailing foam
- Shore/intersection foam

Requirements:

- Generated from analytic or compute wave derivatives
- Stable over time
- Filtered in the distance

### 6. Water Shading

The shading model should combine:

- Reflections
- Refraction / transmission
- Depth-based absorption
- In-scattering
- Foam overlay

#### Above Water

- Fresnel-driven reflection
- SSR if available, environment fallback otherwise
- Refraction distorted by filtered normal field

#### Below Surface / Transmission

- Beer-Lambert absorption by path length
- Wavelength-dependent extinction
- Forward scattering tint
- Visibility target around 100 m in clear water presets

#### Shoreline / Depth

- Depth prepass driven
- Smooth transmittance changes near shore
- Shore foam and turbidity boost in shallow water

## Performance Plan

### Geometry Budget

Keep geometry cost bounded with clipmaps.

Target:

- No world-size-dependent vertex explosion
- Fixed ring count
- Reused meshes

### Compute Budget

Keep compute frequency adaptive.

Possible policy:

- Update near levels every frame
- Update mid levels every 2nd frame
- Update far levels every 4th frame

### Texture Budget

Use per-level texture resolutions based on perceptual need.

Example:

- LOD0: `1024x1024`
- LOD1: `1024x1024`
- LOD2: `512x512`
- LOD3+: `256x256`

### Shader Budget

Do not evaluate full wave stacks everywhere.

- Vertex shader: low-frequency geometric bands only
- Fragment shader: sampled normal/foam fields
- Avoid repeated raw Gerstner evaluation in both stages once compute textures exist

## Implementation Phases

## Phase 0: Reset the Current Water Path

Goal:

- Get back to one stable world-space rendering path

Tasks:

- Remove temporary experimental dual-layer logic
- Keep depth-based transparency/scattering path
- Keep world-space wave sampling only

Exit criteria:

- No camera-motion phase artifacts
- One stable baseline path to replace

## Phase 1: Introduce Clipmap Geometry

Goal:

- Replace world-covering mesh/chunks with reusable clipmap rings

Tasks:

- Add ring mesh assets
- Add per-level transforms and scales
- Add seam stitching or morph bands
- Sample displacement in absolute world coordinates only

Files/modules to add:

- `crates/bevy_landscape_water/src/clipmap.rs`
- `crates/bevy_landscape_water/src/mesh_cache.rs`

Exit criteria:

- Stable ocean coverage around camera
- Fixed vertex budget

## Phase 2: Compute-Generated Wave Atlases

Goal:

- Move wave evaluation out of brute-force vertex/fragment recomputation

Tasks:

- Add compute pipeline for banded Gerstner fields
- Generate height/slope/normal/foam textures
- Bind textures per clip level
- Add mip generation or explicit downsample pass

Files/modules to add:

- `crates/bevy_landscape_water/src/compute.rs`
- `crates/bevy_landscape_water/src/uniforms.rs`
- `crates/bevy_landscape_water/src/render.rs`
- `assets/shaders/water_compute.wgsl`

Exit criteria:

- Fragment shader consumes normal/foam textures
- Far field no longer aliases from raw high-frequency evaluation

## Phase 3: Distance-Band Filtering

Goal:

- Remove far-field shimmer and moire

Tasks:

- Assign wavelength bands to LOD levels
- Fade out short wavelengths with distance
- Bias mip selection from derivatives
- Fade foam intensity with distance

Exit criteria:

- Distant water reads as coherent swell, not noise

## Phase 4: Geometric Swell Displacement

Goal:

- Restore strong near-camera vertex motion without high-frequency instability

Tasks:

- Use only low-frequency bands for geometry
- Sample displacement textures or analytic swell in vertex stage
- Keep medium/high bands in shading only

Exit criteria:

- Strong near silhouette motion
- No spiky high-frequency geometry

## Phase 5: Underwater / Transmission Upgrade

Goal:

- Improve clarity and underwater look

Tasks:

- Tune extinction/scattering coefficients
- Add shallow/deep presets
- Improve shoreline turbidity
- Add underwater fog color and path-length blending

Exit criteria:

- Clear-water visibility near target range
- Smooth shore-to-deep transition

## Phase 6: Reflections and Polish

Goal:

- Bring visual quality closer to modern games

Tasks:

- SSR or reflection hierarchy
- Sun glint tuning
- Foam persistence and breakup
- Optional rain/wind state coupling

Exit criteria:

- Water reads as a finished surface, not only a displaced mesh

## Proposed File Layout

Recommended structure:

```text
crates/bevy_landscape_water/src/
  clipmap.rs
  compute.rs
  render.rs
  uniforms.rs
  resources.rs
  lib.rs

assets/shaders/
  water_compute.wgsl
  water_vertex.wgsl
  water_fragment.wgsl
  water_downsample.wgsl
  water_common.wgsl
```

## Debug Views We Should Build Early

Required debug modes:

- Clipmap level visualization
- Wave band visualization
- Foam mask only
- Geometric displacement only
- Far-field alias mask
- Mip level visualization
- Underwater transmittance view

These will prevent blind tuning.

## Success Criteria

The replacement is good enough when:

- Camera movement causes zero visible wave phase shifts
- Near water has convincing displaced swell and crest motion
- Far water remains smooth and stable, with no moire shimmer
- Foam no longer reveals mesh faces
- Frame cost is bounded and reasonable for large scenes
- Water clarity and scattering support deep clear water

## Immediate Next Step

Do not keep polishing the current brute-force mesh path.

Start Phase 1:

- Replace the current ocean mesh with clipmap/ring geometry
- Keep absolute world-space wave sampling
- Then move wave bands into compute textures before further visual tuning

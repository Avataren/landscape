# Loose Ends & Future Improvements

A running document of known gaps, deferred decisions, and improvement
opportunities.  Items are grouped by subsystem.  Tick them off or promote
them to a proper roadmap entry when they get picked up.

---

## Texturing

### Detail / tiling textures (missing)

**What**: The macro colour map is a single world-aligned texture capped at
`macro_color_resolution` (default 4 096 px).  For a 4 096 m world that is 1 px/m.
At walking speed (camera 1–2 m off the ground) individual pixels are visible,
and linear filtering makes them look blurry rather than sharp.

**Why it matters**: The "low-res and interpolated" close-up look is a fundamental
consequence of this architecture, not a bug.  Vertex density (1 m at LOD 0) and
height-clipmap resolution (1 texel/m at LOD 0) are fine; the bottleneck is colour
data per unit area.

**Fix**: Layer a high-frequency tiling texture (grass, rock, dirt, snow) that
repeats every N metres, blended by slope/altitude into the macro colour.  A
typical setup:
- Tile the detail texture at 2–4 m/repeat.
- Sample both macro and detail in the fragment shader.
- Blend: `albedo = mix(macro, detail, detail_weight)` where `detail_weight`
  falls off with view distance (use `fwidth(world_xz)` or an explicit LOD ramp).
- Four detail layers (grass, dirt, rock, snow) can be indexed by the same slope/
  altitude thresholds already used for procedural albedo.

This requires new texture assets and a small shader extension; no Rust changes
needed beyond binding the extra textures.

---

### `normal_tex` binding is dead weight

**What**: `TerrainMaterial` has a `normal_texture: Handle<Image>` bound at
slot 5 (`visibility(vertex)`).  The vertex shader declares `normal_tex` but
never samples it — `normal_at()` recomputes the normal from height finite
differences instead.  The baked RG8Snorm normal array is uploaded to the GPU
every frame for nothing.

**Why**: The original intent was to use baked per-pixel normals for higher
shading quality.  The fragment shader now uses `pixel_normal()` (height FD in
the fragment stage) which is always correct.  The baked normal map is only valid
when `procedural_fallback = true` or baked normal tiles are loaded; with neither,
the texture is all-zeros → flat (0,1,0) normals.

**Options**:
1. Remove the binding entirely and stop uploading the normal texture.  Saves a
   texture slot and eliminates dead clipmap writes.
2. Keep it but switch the vertex shader's `normal_at()` to sample it instead of
   recomputing — faster vertex stage, less bandwidth per vertex.
3. Expose it to the fragment shader once the normal tiles are reliably populated
   (baked pipeline) and use it to recover higher-frequency normals than height
   FD allows at 1 texel/m resolution.

**Prerequisite for option 3**: the tile baking pipeline (`bake_tiles`) must
produce normal tiles, and `TerrainSourceDesc::normal_root` must be configured.
When those conditions are met, removing the `procedural_fallback` guard in the
normal sampling path would enable the richer approach.

---

### `textureSampleLevel` vs `textureSample` in vertex stage

**What**: `height_at()` in the vertex shader calls
`textureSampleLevel(height_tex, height_samp, uv, i32(lod), 0.0)`.  The
explicit mip level 0.0 is correct because vertex shaders cannot use implicit
LOD (no screen-space derivatives).  This is fine and intentional — noted here
to avoid a future "fix" that would break it.

---

## Lighting & Normals

### Hemisphere ambient is artist-driven, not atmosphere-driven

**What**: `SKY_AMBIENT` and `GROUND_BOUNCE` in `terrain_fragment.wgsl` are
hard-coded display-space fractions.  They do not react to scene atmosphere
settings (time of day, weather).

**Fix**: Drive them from the actual atmosphere environment map that
`AtmosphereEnvironmentMapLight` generates.  The diffuse irradiance cubemap is
available in the view bindings (`diffuse_environment_map` at `@group(1)
@binding(0)`) and can be sampled with the surface normal to get physically
correct sky ambient per-fragment.  This would also make sunset/sunrise/cloudy
atmospheres automatically affect terrain shading.

**Complexity**: Moderate — requires importing and sampling the environment map
bindings, understanding the bind group layout for the current Bevy version.

---

### No specular on terrain

**What**: The terrain uses Lambertian diffuse only.  Wet rock, snow, and water
puddles would benefit from specular highlights.

**Note**: Intentionally omitted for now — terrain reads well as a diffuse
surface and specular requires the full PBR BRDF setup.  Add when there is
specific art direction for it.

---

### Shadows are soft but flat at the cascade boundary

The fourth cascade (2–8 km) still has coarse texels relative to the terrain
detail in that range.  `CascadeShadowConfigBuilder::overlap_proportion = 0.2`
gives some blending, but distant mountain shadows can look blocky.

**Fix options**:
- Increase `num_cascades` to 5 with tighter far bounds.
- Add PCSS (percentage-closer soft shadows) for the outer cascades.

---

## Physics

### Tile eviction removes fine colliders for distant objects

**What**: `sync_tile_colliders` despawns a tile's `Collider::heightfield` when
the tile is evicted from `resident_cpu`.  Physics objects resting on that tile
(AI, vehicles, props) fall to the coarse 32 m/cell global heightfield.

**Fix**: Add a second contributor to `required_now` that keeps tiles resident
wherever dynamic physics bodies exist, regardless of camera distance.  A system
running before `update_required_tiles` inserts tile keys for any body whose
AABB overlaps the tile footprint.

---

### Phase 4 (per-tile collision LOD) not implemented

The `physics_roadmap.md` describes using coarser LOD tile data as a collision
fallback when LOD-0 tiles are not yet resident.  Currently the gap is covered
by the global 32 m/cell coarse heightfield, which is coarse enough to cause
floating/sinking artefacts on steep terrain during streaming.

---

## Rendering

### Storage-buffer frustum cull is disabled

`update_patch_transforms` removed the per-patch frustum cull because
`intersects_obb_identity` gave wrong results with Bevy 0.18's reversed-Z
frustum planes when the camera was above ground.  All 352+ patches are always
submitted to the GPU.

**Fix**: Revisit once the frustum plane convention is understood.  Alternatively,
implement GPU-side culling via a compute shader that reads the patch buffer and
writes a compacted draw list — this is the "v2" noted in `render/queue.rs`.

---

### `render/queue.rs`, `render/pipelines.rs`, `render/prepare.rs` are stubs

The custom render pipeline (indirect draws, GPU culling, per-frame uniform
upload) is partially scaffolded but not wired into the render graph.
`TerrainRenderPlugin` currently handles only texture uploads via
`write_texture`.

---

### Far clip and depth precision

Far clip was reduced from 10 M to 100 000 m (P6).  Depth precision is still
limited by the 1 000 000:1 range.  Reversed-Z projection
(`near = ∞, far = 0`) would distribute precision toward the near plane, which
matters at terrain scale.  Requires pipeline descriptor changes and a
`DepthBiasState` re-tuning pass.

---

## Clipmap / Streaming

### `residency.evict_to_budget` removes tiles currently visible in the clipmap

When the tile budget is hit, eviction can remove a tile that is still inside
the current clipmap window.  `clipmap_needs_rebuild` is set, triggering a full
layer regeneration on the next frame.  The rebuild is O(res²) per level.

**Fix**: Weight the LRU eviction to prefer tiles outside the current clipmap
window — they are no longer contributing to rendering.

---

### Tile streaming does not prioritise by screen coverage

`request_tile_loads` sorts requests by LOD (coarse first) but does not account
for which tiles are most visible on screen.  A tile directly in front of the
camera should load before one at the edge of the ring.

---

### Procedural clipmap normals recompute height three times per texel

`generate_normal_clipmap_layer` calls `height_at_world` three times per texel
(centre, +X, +Z) to get the normal.  For a 512×512 layer with 5-octave sine
noise per sample, this is 3 × 512² × 5 = ~3.9 M trig evaluations per layer.
With 12 LOD levels that is expensive when a full rebuild is triggered.

**Fix**: Compute normals from the already-generated height buffer in a second
pass rather than resampling the procedural function.

---

## Configuration & Tools

### ✅ `height_scale` is now a single source of truth in `landscape.toml`

Previously `bake_tiles --height-scale` defaulted to 2048.0 while `landscape.toml`
and `TerrainConfig::height_scale` defaulted to 1024.0, and the bake default was
only in the CLI help text — easy to miss.  A mismatch makes baked normals appear
too flat or too steep in lighting.

Fixed: `bake_tiles` now reads `[terrain_config] height_scale` from `landscape.toml`
automatically when `--height-scale` is not passed.  The CLI flag still overrides.
The fallback default is 1024.0 (matching `config.rs`).  The `landscape.toml`
comment and `TerrainConfig::height_scale` doc comment both explain the relationship.

---

### `macro_color_resolution` has no LOD-based variant

The macro colour texture is loaded at a single fixed resolution.  A 16 km world
at 4 096 px gives 0.25 px/m, which is very blurry up close.  The resolution
cap was designed for memory, but the same constraint hurts quality uniformly.

**Fix**: Tile the macro colour (multiple image files at different zoom levels)
and blend between them based on distance — essentially a second clipmap but for
colour rather than height.

---

### `max_resident_tiles` is a fixed budget

The tile LRU does not account for system memory pressure.  On devices with less
RAM, the current default (256 tiles × 256 × 256 × 2 B = 32 MB) may be too
large; on high-end machines it could be raised to reduce streaming artefacts.

**Fix**: Query available system memory at startup and scale the budget
accordingly.

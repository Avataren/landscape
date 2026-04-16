# Procedural Heightmap System

## Goals

A GPU-driven heightmap generation and erosion system that:
- Produces geologically plausible terrain from noise primitives and layer compositing
- Runs hydraulic, thermal, and wind erosion as compute shaders
- Generates at interactive speeds for authoring feedback (~seconds, not minutes)
- Integrates with the existing CDLOD tile streaming pipeline
- Supports both offline baking (write to tile PNGs) and live in-editor preview
- Is fully controlled through `bevy_landscape_editor` with real-time parameter feedback

---

## 1. Heightmap Representation on GPU

All generation and erosion work against a single `StorageTexture2d<f32>` (R32Float) of
the full terrain resolution.  A second identical texture serves as a ping-pong target for
erosion passes that cannot read and write the same cell in the same dispatch.

```
generation_buf   : StorageTexture2d<R32Float, ReadWrite>  // primary
scratch_buf      : StorageTexture2d<R32Float, ReadWrite>  // ping-pong
sediment_buf     : StorageTexture2d<R32Float, ReadWrite>  // hydraulic erosion sediment
water_buf        : StorageTexture2d<R32Float, ReadWrite>  // hydraulic water depth
hardness_buf     : StorageTexture2d<R32Float, ReadWrite>  // per-cell rock hardness
```

At the end of a generation pipeline the result is either:
- **Written to tile PNGs** and loaded by the existing tile streaming system (offline bake), or
- **Directly uploaded into the CDLOD height clipmap** for immediate preview (live mode).

Tile resolution and world scale are read from `TerrainSourceDesc` so the procedural
system produces tiles that are drop-in compatible with hand-sculpted terrain data.

---

## 2. Noise Primitives (Compute Shaders)

All noise is evaluated per-texel in a compute workgroup of 16×16 threads.

### 2a. Gradient Noise (Simplex / Value)
Standard value noise and simplex noise implementations in WGSL.  Used as the base layer
for most generation presets.

### 2b. Fractional Brownian Motion (fBm)
Sum of noise octaves with increasing frequency and decreasing amplitude:

```wgsl
fn fbm(p: vec2<f32>, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    for (var i = 0; i < octaves; i++) {
        value     += amplitude * noise(p * frequency);
        frequency *= lacunarity;   // e.g. 2.0
        amplitude *= gain;         // e.g. 0.5
    }
    return value;
}
```

Controls: `octaves` (2–12), `lacunarity` (1.5–3.0), `gain` (0.3–0.7),
`frequency` (base scale in world units), `offset` (translation, for seed variation).

### 2c. Domain Warping
Distorts the input coordinates with a second fBm evaluation before sampling the primary
noise.  Produces highly natural-looking terrain with characteristic "S-bends" and
irregular valley floors (Inigo Quilez technique):

```wgsl
fn domain_warp(p: vec2<f32>, strength: f32) -> f32 {
    let q = vec2(fbm(p, 5, 2.0, 0.5), fbm(p + vec2(5.2, 1.3), 5, 2.0, 0.5));
    return fbm(p + strength * q, 5, 2.0, 0.5);
}
```

Controls: `warp_strength` (0–3), `warp_octaves`, `warp_scale`.

### 2d. Ridged / Billowed Noise
Useful for mountain ridgelines and rolling hills respectively:

```wgsl
fn ridged(p: vec2<f32>) -> f32 { return 1.0 - abs(noise(p)); }   // sharp ridges
fn billow(p: vec2<f32>) -> f32 { return abs(noise(p)) * 2.0 - 1.0; }  // rounded mounds
```

### 2e. Voronoi / Cellular Noise
Generates mesa plateaus, crater fields, and cell-structured rocky terrain.
Controls: `jitter` (cell irregularity), `mode` (F1 | F2 | F2-F1).

### 2f. Slope-Aware Noise Masking
Any noise layer can be masked by the current slope angle (derived from the current
heightmap state), so e.g. detail only appears on flat areas and not on cliff faces.

---

## 3. Layer Compositor

Heightmaps are built by stacking **generation layers** in order.  Each layer has:

| Field         | Type           | Description                                       |
|---------------|----------------|---------------------------------------------------|
| `kind`        | `LayerKind`    | Noise type (fBm, Ridged, Voronoi, …)              |
| `blend_mode`  | `BlendMode`    | Add \| Multiply \| Max \| Min \| Overlay \| Set   |
| `strength`    | `f32`          | Scale factor applied before blending              |
| `mask`        | `MaskSource`   | None \| Slope \| Altitude \| Painted \| Texture   |
| `parameters`  | layer-specific | Octaves, lacunarity, gain, frequency, offset, … |

Layers are evaluated top-to-bottom in a single dispatch per layer.  Because each layer
depends on the result of the previous, this is inherently sequential — but each layer
dispatch is still fully parallel across texels.

The compositor state is serialised to a `.toml` node graph alongside `landscape.toml`
so generation is fully reproducible from a seed + layer stack.

---

## 4. Erosion Algorithms

All erosion runs as compute shaders after the initial height is generated.  Erosion
passes are scheduled in a sequence: **thermal → hydraulic → wind → smooth**, though the
order and number of iterations of each is user-configurable.

### 4a. Thermal Erosion

Simulates material collapsing down steep slopes until the slope is below the **angle of
repose** (typically 30–45° for rock/soil).  Produces talus fans and scree slopes.

**Algorithm** (per-cell, checkerboard iteration to avoid read/write races):
```
for each neighbour n of cell (x,y):
    delta = height(x,y) - height(n)
    if delta / cell_size > tan(repose_angle):
        transfer = (delta - tan(repose_angle) * cell_size) * 0.5 * talus_rate
        height(x,y) -= transfer
        height(n)   += transfer
```

Checkerboard pattern: dispatch even cells on pass A, odd cells on pass B (two dispatches
per iteration).  Alternatively, use `atomicAdd` on a fixed-point scratch buffer.

Controls: `repose_angle` (degrees), `talus_rate` (0–1), `iterations` (10–500).

### 4b. Hydraulic Erosion (Particle-Based)

The most visually impactful erosion type.  Each simulated water droplet:
1. Spawns at a random position with zero velocity and zero sediment.
2. Accelerates downhill according to the local gradient.
3. Picks up sediment proportional to speed and slope (capacity formula).
4. Deposits sediment when slowing or on flat ground.
5. Evaporates after a fixed number of steps.

This is embarrassingly parallel: each thread simulates one droplet independently.
Writes to the heightmap use `atomicAdd` on a fixed-point buffer (R32 → int32).

```wgsl
@compute @workgroup_size(64, 1, 1)
fn hydraulic_erosion(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = id.x + params.frame_seed;
    var pos  = rand_pos(seed);           // random start
    var vel  = vec2(0.0);
    var water = 1.0;
    var sediment = 0.0;
    for (var step = 0u; step < params.max_steps; step++) {
        let grad   = gradient(pos);
        vel        = vel * params.inertia + grad * (1.0 - params.inertia);
        // ... capacity, erosion, deposition, evaporation
    }
}
```

Controls: `num_particles` (10k–1M per dispatch), `inertia` (0–1), `erosion_rate`,
`deposition_rate`, `evaporation_rate`, `min_slope`, `max_steps` (30–200),
`sediment_capacity_factor`.

Multiple dispatch rounds are queued to achieve the desired total particle count.
A typical high-quality erosion uses 500k–2M particles dispatched in batches of 64k.

### 4c. Thermal Erosion — Soft Rock / Layered
An extended thermal pass that respects a per-cell **hardness** value (from `hardness_buf`).
Hard rock erodes slower, soft rock faster.  Hardness can be seeded from a noise layer
or painted manually.  Produces cliff bands where hard rock sits above soft rock.

### 4d. Wind Erosion (Aeolian)

Simulates saltation: sand-sized particles hop along the surface, eroding exposed ridges
and depositing in sheltered hollows.  Produces:
- Yardangs (wind-scoured ridges)
- Shadow dunes on the lee side of obstacles

Wind is specified as a direction vector + speed.  Each particle steps in wind direction,
checking if it can erode (slope facing wind) or must deposit (slope sheltered).

Controls: `wind_direction` (degrees), `wind_speed`, `saltation_length`, `iterations`.

### 4e. Smoothing / Gaussian Blur Pass

A separable Gaussian blur applied after erosion to remove single-texel spikes introduced
by particle quantisation.  Radius is 1–4 texels; fully configurable.

---

## 5. Compute Pipeline Architecture

### Scheduling
All compute passes are submitted through a dedicated `ProceduralHeightmapPipeline` Bevy
render-world resource.  Generation is triggered by a `GenerateHeightmap` event from the
editor; each pass writes its result to `generation_buf` (or the ping-pong target).

```rust
pub enum ProceduralPass {
    Generate(GenerationParams),
    ThermalErode(ThermalParams),
    HydraulicErode(HydraulicParams),
    WindErode(WindParams),
    Smooth(SmoothParams),
    WriteTiles,       // readback → tile PNGs
    UploadClipmaps,   // directly patch the CDLOD clipmap
}
```

A pipeline run is a `Vec<ProceduralPass>` that is consumed in order by the render graph.

### Progress Reporting
Because GPU work is async, progress is reported back to the ECS via a shared
`Arc<AtomicUsize>` counter incremented after each pass.  The editor polls this each
frame to update a progress bar without blocking.

### Iteration Count vs. Quality
| Quality preset | Thermal iters | Hydraulic particles | Wind iters | Time (4090) |
|----------------|---------------|---------------------|------------|-------------|
| Preview        | 50            | 100k                | 0          | ~0.2 s      |
| Medium         | 200           | 500k                | 50         | ~1.5 s      |
| High           | 500           | 2M                  | 200        | ~8 s        |
| Ultra          | 1000          | 8M                  | 500        | ~40 s       |

Preview quality runs live during parameter adjustment (re-triggers on slider release);
High/Ultra are triggered explicitly with a **Generate** button.

---

## 6. Region-Based Generation

Generation does not have to cover the full terrain.  A **region mask** (RGBA8 texture
the same size as the heightmap) controls which cells are affected by each pass:

- `R` channel: generation influence (0 = keep existing height, 255 = full replace)
- `G` channel: erosion influence (can erode more/less in specific zones)
- `B` / `A`: reserved for future use

The region mask is painted in the editor (same brush system as material painting) and
persisted alongside the splatmap tiles.

This allows:
- Preserving hand-sculpted areas while running procedural erosion elsewhere.
- Adding a procedurally generated mountain range without affecting the coastline.
- Running wind erosion only on a desert biome region.

---

## 7. Integration with the Tile Pipeline

### Offline Bake (Default)
After generation completes, `WriteTiles` reads back `generation_buf` to CPU and
subdivides it into tile PNGs matching the existing `tile_root` layout.  Normal tiles are
also regenerated from the new height data.  The bake tool is exposed as:

- A menu item in the editor: **Generate → Bake to Tiles**
- The existing `bake_tiles` binary (extended with a `--procedural` flag)

### Live Preview (Editor Mode)
When the editor is open, `UploadClipmaps` bypasses tile I/O entirely and directly patches
the CDLOD height clipmap textures for the affected region.  The terrain renders the new
heights immediately.  Normal vectors are recomputed in the fragment shader from the
updated clipmap (already the case for the existing rendering path).

Live preview runs at **Preview** quality to keep latency below 0.5 s on typical hardware.

---

## 8. Presets and Serialisation

Generation pipelines are saved as `.toml` files in `assets/presets/`:

```toml
[preset]
name = "Alpine Mountains"
seed = 42
world_min = [−8192, −200]
world_max = [8192, 2400]

[[layers]]
kind = "fbm"
blend_mode = "set"
strength = 1.0
octaves = 8
lacunarity = 2.1
gain = 0.5
frequency = 0.0003

[[layers]]
kind = "ridged"
blend_mode = "add"
strength = 0.3
frequency = 0.001

[[erosion]]
kind = "thermal"
iterations = 300
repose_angle = 38.0

[[erosion]]
kind = "hydraulic"
particles = 1_000_000
inertia = 0.3
```

Presets are listed in the editor and can be saved, duplicated, and shared.

---

## 9. UI Integration (`bevy_landscape_editor`)

### 9a. Panel Layout

The editor gains a **Generate** tab in the left panel (alongside Materials and Layers):

```
┌──────────────────┐
│ ▼ Presets        │  [dropdown]  [Load]  [Save As]
│                  │
│ ▼ Seed           │  [int drag]  [🎲 Randomise]
│                  │
│ ▼ World Bounds   │  min Y / max Y sliders
│                  │
│ ▼ Layer Stack    │  [list of generation layers]
│   + Add Layer    │
│                  │
│ ▼ Erosion        │
│   Thermal  [▶]   │  iterations, repose angle
│   Hydraulic [▶]  │  particles, inertia, …
│   Wind     [▶]   │  direction, speed, …
│   Smooth   [▶]   │  radius
│                  │
│ ▼ Region Mask    │  [paint / clear / invert]
│                  │
│ [Preview]  [Generate ▸ Bake]    │
│ ████████░░  62%  │  (progress bar)
└──────────────────┘
```

### 9b. Layer Stack Panel

Each generation layer row shows:
- Drag handle for reordering
- Layer type icon + name (editable)
- Blend mode dropdown
- Strength slider (0–2)
- Eye toggle (disable layer without deleting)
- ✕ delete button

**Expanded layer detail** shows all parameters for that noise type:
- Frequency (world-space scale, shown as "tile size in meters" for clarity)
- Octaves, lacunarity, gain (for fBm / ridged / billow)
- Warp strength and scale (domain warp)
- Jitter (Voronoi)
- Mask source and mask parameters

A **mini preview** (64×64 greyscale thumbnail, updated on slider release) shows what this
layer alone would produce at the current seed.

### 9c. Erosion Controls

Each erosion type has a collapsible section with a **Run** (▶) button to apply just that
pass interactively, and an **Undo** button to revert it.  Parameter sliders show tooltips
explaining the real-world analogy (e.g. "Repose angle: angle at which loose rock starts
sliding, typically 30–45°").

A **Preset Curve** widget (small editable graph) controls per-iteration strength falloff
for thermal erosion — useful for a burst of heavy erosion early, tapering to polish.

### 9d. Region Mask Panel

Switches the viewport brush to **Mask Paint** mode:
- Paint (white): full generation influence
- Erase (black): no generation influence
- Smooth: blur mask edges

Shows a viewport overlay of the current region mask as a transparent colour tint.

### 9e. Generation Progress

When **Preview** is clicked, the terrain updates live in the viewport within ~0.5 s.
When **Generate → Bake** is clicked:
1. A modal dialog appears with a progress bar (polled from the `Arc<AtomicUsize>` counter).
2. Pass names ("Thermal erosion: 300/300 iterations…") are shown as subtitles.
3. A **Cancel** button sends an `AbortGeneration` event; the heightmap reverts to its
   pre-generation state via the undo stack.
4. On completion, a **View Changes** button switches to the diff overlay (new vs old).

### 9f. Diff Overlay (Review Mode)

After baking, a toggle shows a false-colour overlay in the viewport:
- **Blue**: terrain was lowered (eroded)
- **Orange**: terrain was raised (deposition / added noise)
- **Grey**: unchanged

This helps the user understand what the erosion did and whether it's desirable before
committing to disk.

### 9g. Integration with Height Painting

The procedural generator and the manual height brush share the same CPU heightmap
buffer and dirty-tile upload mechanism (planned in the terrain editing system).
**Procedural generation is just a large-area brush stroke** from the undo/redo system's
perspective — it pushes one snapshot onto the undo stack before running and the user
can Ctrl-Z to revert the entire generation in one step.

### 9h. Plugin Structure

```rust
// Inside LandscapeEditorPlugin::build():
app.add_plugins(ProceduralPanelPlugin);   // the Generate tab panel
app.add_plugins(RegionMaskBrushPlugin);   // mask painting mode
app.add_plugins(GenerationProgressPlugin); // progress bar + cancel
app.add_plugins(DiffOverlayPlugin);       // post-generation diff visualisation
```

---

## 10. Open Questions / Decisions Deferred

- **Hydraulic erosion concurrency model**: atomicAdd on fixed-point vs. per-particle
  independent heightmap copies merged at the end.  AtomicAdd is simpler; the merged
  approach avoids integer quantisation artefacts on very fine detail.
- **River simulation**: full flow-accumulation river carving is a separate, more complex
  system (requires graph traversal, not just per-cell compute).  Deferred to a later pass.
- **Texture synthesis on hardness map**: whether hardness is purely noise-driven or can
  reference geological layer data (e.g. imported from a real DEM).
- **Tectonic uplift simulation**: very slow but produces the most realistic large-scale
  forms.  Could be a long-running offline preset, not a real-time tool.
- **16k vs. region-based generation**: generating the full 16k×16k map in one shot
  requires ~4 GB of GPU memory for the five 32-bit buffers.  Region-based tiling
  (generate in 4k×4k chunks with overlap) limits peak GPU memory to ~256 MB per chunk.
- **Normal map regeneration**: after baking new heights, normal tiles must also be
  regenerated and written.  This can run as an additional compute pass immediately after
  generation (`bake_normals` pass in the pipeline).
- **Undo stack memory for large terrains**: a single 16k×16k R32Float snapshot is 1 GB.
  Options: compress snapshots (zstd, ~10:1 on terrain data), store only delta regions,
  or limit undo depth for full-map operations to 1–3 steps.

---

## 11. Suggested Implementation Order

1. **Single-pass fBm generation compute shader** — generate a heightmap, write to tiles, load in terrain.
2. **Domain warp + ridged noise variants** — expand the noise primitive library.
3. **Layer compositor** — stack multiple layers with blend modes; verify reproducibility from seed.
4. **Thermal erosion** — checkerboard dispatch, tune angle of repose, visual comparison.
5. **Hydraulic erosion (particle)** — 64k-particle dispatch, verify valleys and fans form.
6. **Hardness-aware thermal erosion** — layer hardness from noise, produce cliff bands.
7. **Wind erosion** — directional saltation, tune for desert vs. alpine presets.
8. **Gaussian smooth pass** — clean up quantisation noise after erosion.
9. **Live clipmap upload** (preview mode) — bypass tile I/O for sub-second feedback.
10. **Region mask** — paint influence zones, tie into existing brush system.
11. **Editor Generate panel** — full UI: layer stack, erosion controls, progress bar, diff overlay.
12. **Preset save/load** — `.toml` serialisation, preset browser in editor.
13. **Normal tile regeneration** — compute pass to re-bake normals after height changes.

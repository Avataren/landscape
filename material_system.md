# Procedural Landscape Material System

## Goals

A material system that:
- Blends multiple physical material types (rock, soil, grass, snow, etc.) seamlessly
- Eliminates visible tiling through stochastic / multi-scale sampling
- Drives instanced vegetation and scatter objects (grass, bushes, trees, rocks)
- Supports runtime user painting of both material weights and instance density
- Integrates cleanly with the existing CDLOD clipmap streaming architecture

---

## 1. Material Layer Definitions

Each **material slot** is a standalone physical surface and contains:

| Map        | Format  | Notes                                              |
|------------|---------|----------------------------------------------------|
| Albedo     | sRGB    | Base color                                         |
| Normal     | RG8     | Tangent-space, Z reconstructed                     |
| ORM        | RGB8    | Occlusion / Roughness / Metallic packed            |
| Height     | R8      | Used for parallax and height-blend transitions     |

Up to **16 material slots** are supported (4 splatmaps × 4 channels).  
Initial implementation can target 8 (2 splatmaps) as a practical limit.

All material textures live in a `Texture2DArray` per map type, indexed by slot.  
This means a single shader bind group services all materials — no permutations.

---

## 2. Splatmap Architecture

### Layout
Each splatmap is an **RGBA8 texture** whose channels store blend weights for four material slots.

```
splatmap_0.r = weight[material 0]   (e.g. bare rock)
splatmap_0.g = weight[material 1]   (e.g. dry soil)
splatmap_0.b = weight[material 2]   (e.g. grass)
splatmap_0.a = weight[material 3]   (e.g. wet mud)
splatmap_1.r = weight[material 4]   (e.g. snow)
...
```

Weights across all active slots must sum to 1.0 (normalised on write).

### Resolution
- Splatmaps use the **same tile grid** as the heightmap (`assets/tiles/`) for streaming coherence.
- Target resolution: **4× the heightmap texel density** (one splatmap texel per ~0.5 world units at max LOD).
- Stored as PNG tiles alongside height tiles: `assets/splatmap_0/z_x.png` etc.

### Clipmap Streaming
Splatmap tiles are loaded into a `Texture2DArray` clipmap (same ring-buffer scheme as the height clipmap).  
One clipmap per splatmap layer; uploaded/evicted in lockstep with height tiles.

---

## 3. Non-Repeating Texture Sampling

Visible tiling is eliminated by combining three complementary techniques:

### 3a. Distance-Adaptive Multi-Scale Blending
Each material is sampled at **two scales** (fine and coarse) and blended by a factor
derived from camera distance — not fixed weights:

```wgsl
// Blend factor: 0.0 = fine dominant (close), 1.0 = coarse dominant (far)
let dist        = distance(camera_pos.xz, world_pos.xz);
let dist_blend  = smoothstep(NEAR_DIST, FAR_DIST, dist);  // e.g. 4m → 32m

let c_fine   = sample_material(slot, uv / SCALE_FINE);    // e.g. 1 m per tile
let c_coarse = sample_material(slot, uv / SCALE_COARSE);  // e.g. 6 m per tile
let c        = mix(c_fine, c_coarse, dist_blend);
```

**Why distance-driven, not fixed weights:**
Fixed weights (e.g. 50/50) produce visible double-frequency shimmer at all distances.
Distance-driven blending ensures the fine layer is never visible where it would alias,
and the coarse layer never blurs what the eye can resolve up close.

**LOD coherence:** tie `dist_blend` to `lod_level + morph_alpha` rather than raw
camera distance.  This anchors the UV scale transitions to the geometry LOD ring
boundaries, making them effectively invisible — the texture scale hops at exactly the
same point the geometry morphs, with `morph_alpha` providing the smooth cross-fade
between adjacent scale tiers just as it does for vertex positions.

### 3b. Stochastic Tiling (Hash-Based UV Jitter)
Based on the Blending In Detail / histogram-preserving blending technique:
- Divide UV space into an irregular Voronoi-like grid using a hash function.
- Offset and rotate UVs per cell; blend at cell boundaries with a smooth weight.
- Preserves histogram statistics so the result doesn't wash out.
- Implemented in WGSL; no extra textures required.

### 3c. Triplanar Projection for Steep Slopes
When surface slope exceeds a threshold (~45°), blend from UV-based sampling to triplanar:
```wgsl
let triplanar_weight = smoothstep(0.3, 0.7, abs(normal.y));  // 0 = cliff, 1 = flat
let c_uv  = sample_uv(slot, world_pos.xz);
let c_tri = sample_triplanar(slot, world_pos, normal);
let c = mix(c_tri, c_uv, triplanar_weight);
```
This prevents UV-stretched textures on rock faces and cliff edges.

### 3d. Height-Based Blend Transitions
Instead of linear alpha blending between materials, use the material height maps to create
natural contact edges (e.g. snow pooling in crevices, soil showing through worn grass):
```wgsl
// height_blend: shifts the linear weight using the surface height map
fn height_blend(w_a: f32, h_a: f32, w_b: f32, h_b: f32, sharpness: f32) -> f32 {
    let a = h_a + w_a;
    let b = h_b + w_b;
    let t = clamp((a - b) / sharpness + 0.5, 0.0, 1.0);
    return t;  // blend factor: 1.0 = fully A, 0.0 = fully B
}
```

---

## 4. Close-up Texture Quality

Terrain materials must hold up at every viewing distance — from a character crouching on
the ground (~0.5 m eye height) to a bird's-eye view kilometres away.  The multi-scale
blending in §3a handles the mid-to-far transition; this section addresses the **sub-5 m
close range** where individual surface micro-structure must be legible.

### 4a. The Core Problem

A material texture tiled at 1 m per repeat looks sharp up close but repeats visibly at
distance.  One tiled at 6 m looks fine at 30 m but blurry at 1 m.  GPU mipmapping
handles blur gracefully but cannot invent detail that was never in the texture.  No
single tiling scale is adequate across the full depth range of a terrain camera.

The techniques below are layered: each addresses a different frequency band and distance
range, and they compound without conflicting.

### 4b. Micro-Detail Normal Map (primary, always on)

A **universal micro-detail normal map** — a single tileable RG8 texture at ~0.3 m per
tile — is overlaid on every material's normal at close range and fades out with distance:

```wgsl
let detail_uv    = world_pos.xz / DETAIL_TILE_SIZE;          // e.g. 0.3 m
let detail_n     = sample_normal(detail_normal_map, detail_uv);
let detail_blend = 1.0 - smoothstep(DETAIL_NEAR, DETAIL_FAR, dist); // e.g. 2–12 m
let n            = normalize(blend_normals(material_n, detail_n, detail_blend));
```

This single texture (shared across all material slots) adds micro-surface structure —
pores in soil, grain in rock, fibre in grass — that the per-material textures cannot
hold at their coarser tile size.  Cost: **one extra sample per fragment**, only at close
range (early-exit when `detail_blend == 0`).

The detail normal map is authoring-neutral: it does not contain material-specific colour
or roughness, so it blends correctly regardless of which material slot dominates.

### 4c. LOD-Coherent Scale Transitions

Tying the UV scale blend directly to `lod_level + morph_alpha` rather than raw camera
distance prevents **texture swimming** — the sliding/shimmering that occurs when the
scale blend factor changes continuously as the camera moves through a LOD ring.

```wgsl
// Map LOD level to a 0–1 scale blend: LOD 0 = fine, LOD N = coarse.
let lod_t       = f32(lod_level) / f32(terrain.num_lod_levels - 1u);
let scale_blend = mix(lod_t, min(lod_t + lod_step, 1.0), morph_alpha);
```

With this approach, the UV scale changes only at LOD ring boundaries, morphing smoothly
over the morph zone — exactly like the geometry.  The viewer never sees a texture scale
transition that is not already masked by geometry morphing.

### 4d. Parallax Offset (optional, close range only)

When the camera is within `PARALLAX_MAX_DIST` (default 4 m), the per-material height
map can offset UVs to simulate depth on rocky faces and uneven soil.  This turns flat
texture into something that reads as genuinely 3-D.

**Simple parallax** (1 sample, fast, suitable for low-relief surfaces):
```wgsl
let h       = sample_height(slot, uv) * parallax_scale;  // e.g. scale = 0.02
let view_ts = world_to_tangent(normalize(camera_pos - world_pos));
let uv_off  = view_ts.xy * h;
```

**Parallax Occlusion Mapping** (8–16 samples, high quality, steep faces / rocks):
- Ray-march along the view vector in tangent space until the height field is intersected.
- Produces correct self-occlusion in deep crevices.
- Gate behind a `TerrainConfig::parallax_mode` enum: `Off | Simple | POM`.
- Only evaluate when `dist < PARALLAX_MAX_DIST`; skip entirely at LOD ≥ 2.

### 4e. Macro-to-Micro Normal Compositing

Three normal contributions exist at any surface point, each at a different frequency:

| Layer           | Source                          | Tile size  | Distance active |
|-----------------|---------------------------------|------------|-----------------|
| Macro normal    | Height clipmap finite diff      | ~2 m       | All distances   |
| Material normal | Per-slot normal map, coarse UV  | 4–8 m      | All distances   |
| Detail normal   | Universal micro-detail map      | 0.2–0.5 m  | < 12 m          |

Compositing order (surface → camera):

```
n_macro    = pixel_normal(lod, world_xz)          // already in fragment shader
n_material = sample_normal(slot, uv_coarse)
n_detail   = sample_normal(detail_map, uv_micro) * detail_blend

// Reoriented Normal Blending (RNB) — preserves both macro curvature and micro detail
n_out = reoriented_blend(reoriented_blend(n_macro, n_material), n_detail)
```

**Reoriented Normal Blending (RNB)** is preferred over simple `normalize(a + b)` because
it correctly handles large-angle macro normals (cliffs) without flipping detail normals.

### 4f. Distance Fade Budget

| Technique           | Active range  | Cost when inactive |
|---------------------|---------------|--------------------|
| POM                 | 0 – 4 m       | Zero (skipped)     |
| Simple parallax     | 0 – 6 m       | Zero (skipped)     |
| Micro-detail normal | 0 – 12 m      | Zero (early exit)  |
| Fine UV scale       | 0 – 32 m      | Coarse only        |
| Material normals    | All           | Always on          |
| Macro normal        | All           | Always on          |

All thresholds are `TerrainConfig` fields so they can be tuned per project without
recompiling shaders (passed as push constants / uniform buffer).

### 4g. Authoring Guidance

- **Micro-detail normal map**: bake from a high-poly scan or sculpt of generic ground
  surface at ~5–10 cm texel density.  Should be neutral in hue and contrast — it adds
  bump, not colour.  One map serves all material slots.
- **Per-material normal maps**: author at the coarse tile scale (4–8 m).  Close-up
  micro-detail comes from the universal layer, so the per-material normal does not need
  to contain very high frequency information.
- **Height maps**: keep range modest (0–1 mapped to ~0–5 cm real depth) unless POM is
  enabled, in which case 0–15 cm is appropriate.  Deeper values cause UV skewing at
  grazing angles.
- **UV scale per material**: recommended starting points:
  - Fine scale: 0.5–1 m per tile (rock), 1–2 m per tile (grass/soil)
  - Coarse scale: 4–6× the fine scale
  - Detail normal: 0.25–0.4 m per tile (independent of material, global setting)

---



Before any user painting, weights are computed procedurally from terrain properties:

| Property         | Drives                                                              |
|------------------|---------------------------------------------------------------------|
| Altitude         | Snow above threshold, rock near summits, vegetation in valleys      |
| Slope (degrees)  | Rock/cliff on steep faces, soil/grass on gentle slopes              |
| Curvature        | Wet/mud in concave hollows, dry/bare on convex ridges               |
| Noise octaves    | Fine variation within altitude/slope bands, breaks up uniformity    |

These are evaluated in the fragment shader from available data (world pos, normal, clipmap height).

Painted weights **additively override** the procedural baseline per channel.

---

## 5. Instance / Vegetation Layers

### Density Map
Each instance type has a dedicated **R8 density map** tiled identically to the splatmap.
- `0` = no instances, `255` = maximum density.
- Procedural baseline fills the density map; painting adjusts it.

### Instance Types

| Type   | Rendering          | LOD Strategy                        | Typical density      |
|--------|--------------------|-------------------------------------|----------------------|
| Grass  | GPU billboard batch| Fades out beyond ~100 m             | High (per m²)        |
| Ferns  | GPU mesh instancing| 2 LOD levels, ~150 m cutoff         | Medium               |
| Bushes | Mesh instancing    | 3 LOD levels, ~300 m cutoff         | Low-medium           |
| Trees  | Mesh instancing    | Full LOD chain, impostors ~1 km     | Low                  |
| Rocks  | Mesh instancing    | 2–3 LOD levels, ~500 m cutoff       | Sparse               |

### Placement Rules (per layer, configured at authoring time)
- Altitude min/max
- Slope min/max
- Density falloff curve
- Random scale range (min, max)
- Random Y-axis rotation
- Align to surface normal (yes/no)
- Cluster radius / Poisson disk radius

### GPU Instancing Pipeline
- Instance buffers generated in a compute shader or CPU system from density map + rules.
- Re-generated when density map tiles are loaded/painted (dirty region tracking).
- Grass uses a dedicated billboard shader with wind animation.

---

## 6. Painting System

### Brush Stroke Event
```rust
pub struct MaterialBrushStroke {
    pub center:   Vec2,           // world XZ
    pub radius:   f32,
    pub strength: f32,
    pub op:       MaterialBrushOp,
}

pub enum MaterialBrushOp {
    PaintMaterial { slot: usize },    // raise weight of this slot
    EraseMaterial { slot: usize },    // lower weight of this slot
    SmoothMaterial,                   // gaussian blur weights in area
    PaintDensity   { layer: usize },  // raise instance density
    EraseDensity   { layer: usize },  // lower instance density
    SampleMaterial,                   // read dominant slot at cursor (eyedropper)
}
```

### CPU-Side Buffers
- `SplatmapBuffer`: `Vec<[u8; 4]>` per splatmap, organised as tiles matching the tile grid.
- `DensityBuffer`: `Vec<u8>` per instance layer.
- On stroke: modify buffer in affected region → mark tiles dirty → re-upload clipmap tiles.

### Dirty Region Tracking
- Each tile has a `dirty: bool` flag.
- Upload system flushes dirty tiles to GPU each frame (capped to N tiles/frame to bound cost).
- Undo/redo stack stores pre-stroke tile snapshots (memory-bounded ring buffer).

---

## 7. Shader Architecture

### Bind Group Layout (material pass)
```
group 0: terrain globals (same as current)
group 1: height clipmap array (existing)
group 2: splatmap clipmap array (new, N splatmaps)
group 3: material texture arrays (albedo[], normal[], orm[], height[])
group 4: density maps (one per instance layer, for debug viz)
```

### Fragment Shader Flow
```
1. Sample splatmap(s) → weight[0..N]
2. For each weight > threshold:
   a. Compute UV (world XZ / material scale)
   b. Apply stochastic jitter
   c. Sample albedo, normal, orm, height
   d. Apply triplanar blend if steep
3. Height-blend all sampled materials using weights + height maps
4. Reconstruct final normal, roughness, metallic, occlusion
5. Feed into existing PBR lighting model
```

---

## 8. Data Pipeline / Asset Layout

```
assets/
  tiles/            ← existing heightmap tiles
  normals/          ← existing normal tiles
  macro_color/      ← existing macro color
  splatmap_0/       ← new: material weight tiles (RGBA8 PNG), layer 0
  splatmap_1/       ← new: material weight tiles (RGBA8 PNG), layer 1
  density/
    grass/          ← R8 PNG tiles
    trees/
    rocks/
  materials/
    rock/           ← albedo.png, normal.png, orm.png, height.png
    soil/
    grass/
    snow/
    ...
```

`landscape.toml` gains a `[materials]` section listing active slots and their asset paths,
scale factors, and triplanar settings.

---

## 9. Integration Points in `bevy_landscape`

| Module                         | Change                                                          |
|--------------------------------|-----------------------------------------------------------------|
| `world_desc.rs`                | Add splatmap/density root paths                                 |
| `clipmap_texture.rs`           | Add splatmap clipmap (mirrors height clipmap)                   |
| `mod.rs`                       | Add material setup system, load material arrays                 |
| `terrain_fragment.wgsl`        | Replace current macro-color lookup with full material pipeline  |
| `terrain_vertex.wgsl`          | No changes needed (world pos / normal already available)        |
| New: `material.rs`             | `TerrainMaterial` resource, material slot definitions           |
| New: `editing/material_paint.rs` | `MaterialBrushStroke` event, CPU buffer, dirty-tile upload    |
| New: `instancing/`             | Density maps, placement rules, instance buffer generation       |

---

## 10. Open Questions / Decisions Deferred

- **Splatmap resolution vs heightmap resolution ratio**: 4× is a good default but should be configurable.
- **Maximum material slot count**: 8 (2 splatmaps) is a pragmatic first target; 16 is possible but doubles bind group cost.
- **Stochastic tiling algorithm**: Wang tiles vs hash-based jitter vs histogram-preserving — benchmark needed.
- **Grass wind**: simple sine wave in vertex shader vs GPU simulation.
- **Impostor generation for trees**: pre-baked vs runtime.
- **Serialisation of painted data**: paint tiles saved as PNG (lossless, existing tooling) or custom binary format (smaller).
- **Undo stack memory budget**: configurable, defaults to ~50 strokes.
- **Whether procedural baseline is baked into splatmap tiles or evaluated at runtime**: baking is faster at runtime but less flexible for iterative authoring.

---

## 11. Editor UI Integration (`bevy_landscape_editor`)

The editor crate provides all authoring controls for the material system as egui panels.
None of this UI code belongs in `bevy_landscape` itself — the lib crate only exposes
events and resources; the editor crate reads and writes them.

### 11a. Panel Layout

The editor uses a left-side **docked panel** (`egui::SidePanel::left`) subdivided into
collapsible sections. The toolbar (already present) gains a **Tools** menu for toggling
individual panels.

```
┌──────────────────────────────────────────────────────┐
│ File   Tools                       [toolbar]          │
├───────────────┬──────────────────────────────────────┤
│ ▼ Materials   │                                      │
│   [slot list] │                                      │
│               │        3-D viewport                  │
│ ▼ Brush       │                                      │
│   [controls]  │                                      │
│               │                                      │
│ ▼ Layers      │                                      │
│   [inst list] │                                      │
└───────────────┴──────────────────────────────────────┘
```

### 11b. Materials Panel

Displays the full list of active material slots.  Each row shows:
- Thumbnail (16×16 preview sampled from the albedo texture)
- Slot name (editable inline)
- Visibility toggle (exclude from blend without deleting)
- Drag handle for reordering

**Slot detail** (expands on click):
- Albedo / Normal / ORM / Height — each shows a file path with a **Browse…** button that
  opens a native file dialog (`rfd` crate) and hot-reloads the texture via `AssetServer`.
- Scale factor (drag-float, affects all UV sampling for this slot)
- Triplanar threshold (angle in degrees at which triplanar kicks in; per-slot override)
- Multi-scale blend ratios (three floats for fine/mid/coarse, summing to 1)
- Height-blend sharpness

**Slot management buttons** at the bottom of the panel:
- **+ Add Slot** — appends a new empty slot (up to the configured maximum)
- **Duplicate** — copies selected slot's settings to a new slot
- **Delete** — removes slot, redistributes its splatmap weight to a fallback slot

**Procedural rules sub-section** (collapsible per slot):
- Altitude range (min/max sliders mapped to world-space Y)
- Slope range (degrees)
- Curvature bias (concave / convex / both)
- Noise scale and strength
- A small real-time preview curve showing how weight varies with altitude

### 11c. Brush Panel

Active whenever the viewport is in **Paint Mode** (toggled from the Tools menu or a
keyboard shortcut).

```
Mode:   ○ Paint   ○ Erase   ○ Smooth   ○ Sample
Slot:   [dropdown of material slot names]

Radius:   ──●───────  64 m
Strength: ───●──────  0.4
Falloff:  [Smooth ▼]   (Smooth | Linear | Square | Constant)

[ Hold Shift = Erase ]   [ Alt-click = Sample ]
```

In **Density** sub-mode (switched via a tab at the top of the panel):
```
Layer:    [dropdown: Grass | Trees | Rocks | …]
Mode:     ○ Add   ○ Erase   ○ Smooth

Density:  ──────●──  0.8
Radius:   ──●───────  32 m
```

The brush cursor is rendered in the 3-D viewport as a projected circle decal (a
screenspace overlay drawn via an egui paint callback or a dedicated Bevy `Mesh2d`).

Ray-casting from the cursor to the terrain surface is handled by the editor crate using
the existing `TerrainCollisionCache` (already public from `bevy_landscape`).

### 11d. Layers Panel (Vegetation / Scatter)

Lists all configured instance layers.  Each row:
- Layer name + icon (🌿 grass, 🌲 tree, 🪨 rock …)
- Visibility toggle (hide from viewport without deleting)
- Lock toggle (prevent accidental painting)
- Density map thumbnail (R8, rendered as greyscale)

**Layer detail** (expands on click):

*Mesh / Asset*
- Mesh path (Browse…)
- LOD chain paths (LOD 0 … LOD N, each with a distance threshold)
- Impostor path (optional, for trees)

*Placement Rules*
- Altitude range
- Slope range
- Align to normal (checkbox)
- Scale: min / max (two drag-floats)
- Random Y rotation (checkbox)
- Cluster mode: None | Poisson (radius drag-float)

*Wind* (grass/foliage only)
- Wind speed, direction, turbulence strength

**Layer management** at panel bottom:
- **+ Add Layer** (opens a type picker: Grass | Bush | Tree | Rock | Custom)
- **Duplicate**
- **Delete**

### 11e. Undo / Redo

All paint operations push a snapshot onto a undo stack (managed inside `bevy_landscape`).
The editor crate sends `UndoStroke` / `RedoStroke` events (Ctrl-Z / Ctrl-Y) and displays
the stack depth in the status bar.

### 11f. Save / Export

- **Save** (Ctrl-S) — flushes all dirty CPU splatmap and density tiles to disk as PNG,
  matching the existing tile-directory layout.  Triggered via a `SaveTerrainEdits` event.
- **Export Splatmap** — writes a full-resolution merged splatmap (all tiles stitched) for
  use in external tools.
- The **File → Exit** flow already implemented checks for unsaved changes and shows a
  confirmation dialog before sending `AppExit`.

### 11g. Plugin Structure in `bevy_landscape_editor`

```rust
pub struct LandscapeEditorPlugin;   // top-level, already exists

// Internal sub-plugins registered by LandscapeEditorPlugin:
struct ToolbarPlugin;               // File / Tools menu bar
struct MaterialPanelPlugin;         // slot list, texture pickers, proc rules
struct BrushPlugin;                 // brush state, ray-cast, viewport cursor
struct LayerPanelPlugin;            // instance layer list and detail
struct UndoPlugin;                  // undo/redo stack integration
```

Each sub-plugin is a private `struct` inside the editor crate and registers only its own
systems into the `EguiPrimaryContextPass` schedule (or `Update` for non-UI logic like
ray-casting).

---

## 12. Suggested Implementation Order

1. **Material texture arrays + static weights** — hardcode a blend between 2 materials by slope, verify no tiling.
2. **Distance-adaptive UV scale blending** — LOD-coherent fine/coarse scale blend; verify no swimming at LOD transitions.
3. **Micro-detail normal map** — universal detail layer, fade with distance; verify close-up quality.
4. **Reoriented Normal Blending** — composite macro + material + detail normals correctly.
5. **Stochastic tiling** — implement hash-based UV jitter in WGSL, compare visual quality.
6. **Triplanar on steep faces** — blend in triplanar, tune threshold.
7. **Height-based blend transitions** — replace lerp with height-blend, tune sharpness.
8. **Parallax (optional)** — add Simple then POM modes, gate behind distance + LOD check.
9. **Splatmap streaming** — add splatmap clipmap alongside height clipmap.
10. **Procedural baseline shader** — altitude/slope/curvature → splatmap weights at runtime.
11. **CPU splatmap buffers + dirty upload** — foundation for painting.
12. **Material painting in editor** — wire `MaterialBrushStroke` through `bevy_landscape_editor`.
13. **Density maps + grass instancing** — first instance layer as proof of concept.
14. **Additional instance layers** — bushes, trees, rocks, each with LOD.
15. **Density painting in editor** — brush for erasing/adding vegetation.

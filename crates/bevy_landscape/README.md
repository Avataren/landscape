# bevy_landscape

A large-scale outdoor terrain renderer for [Bevy](https://bevyengine.org/), built on
Continuous Distance-Dependent Level of Detail (CDLOD) and GPU clipmap streaming.

---

## High-Level Overview

### The problem it solves

A real outdoor landscape might span dozens of kilometres with centimetre-scale surface
detail near the camera.  No GPU can hold that much geometry in memory at once, and no
rasteriser can shade billions of polygons per frame.  The renderer's job is to present
*just enough* detail *exactly where it matters* — right in front of the viewer — while
gracefully simplifying everything else.

### How it looks from the outside

From above, the visible terrain looks like a series of concentric square rings centred
on the camera.  The innermost ring is drawn with the smallest, most detailed polygons.
Each ring outward is drawn with polygons twice as large as the previous ring.  Far rings
cover much more ground but with less geometric detail — which is fine because at that
distance the eye cannot resolve it anyway.

```
  ┌───────────────────────────────┐
  │  LOD 5  (largest patches)     │
  │  ┌─────────────────────┐      │
  │  │  LOD 4              │      │
  │  │  ┌─────────────┐    │      │
  │  │  │  LOD 3      │    │      │
  │  │  │  ┌───────┐  │    │      │
  │  │  │  │ LOD 2 │  │    │      │
  │  │  │  │ ┌───┐ │  │    │      │
  │  │  │  │ │ 1 │ │  │    │      │
  │  │  │  │ │ 0 │ │  │    │      │  ← camera is here
  │  │  │  │ └───┘ │  │    │      │
  │  │  │  └───────┘  │    │      │
  │  │  └─────────────┘    │      │
  │  └─────────────────────┘      │
  └───────────────────────────────┘
```

### Invisible transitions

The tricky part is that the rings move with the camera.  As the camera advances, the
rings shift forward — but shifting a ring means abruptly changing the polygon size, which
would produce a visible "pop".  The renderer eliminates this with **geomorphing**: in the
transition zone of each ring, vertex positions smoothly slide from the fine grid to the
coarse grid over a short distance.  The change is gradual enough that it is never
noticeable in motion.

### Where the terrain heights come from

The surface shape is stored as a pyramid of pre-baked image tiles on disk — think of it
like a map at several zoom levels.  The finest zoom level has one height sample every
metre; each coarser level covers twice the area per sample.  As the camera moves, a
background thread loads the tiles for nearby areas and discards tiles that are no longer
needed.  The GPU always has exactly the data it needs for the visible rings, nothing more.

### Lighting and surface detail

Because the polygons at distant LOD levels are large, per-vertex normals would look blocky
on large slopes.  Instead, the fragment shader re-derives the surface normal from the
height data at every pixel, giving sharp, accurate lighting regardless of polygon count.
At LOD boundaries, the normal computation is smoothly blended across the morph zone so
that the lighting never shows a hard seam even though the geometry resolution is changing.

---

## Technical Description

### Terminology

| Term | Meaning |
|------|---------|
| **LOD level** | One ring in the clipmap hierarchy. LOD 0 is closest, LOD N−1 is farthest. |
| **Patch** | A single `P×P`-vertex quad mesh drawn once per instance. |
| **Ring** | An `R×R` grid of patches (minus the inner quarter covered by the finer level). |
| **Clipmap** | A texture array where each layer stores height data for one LOD level's ring. |
| **Tile** | A `T×T`-texel on-disk chunk of the height/normal pyramid at a given mip level. |
| **Geomorphing** | Smooth interpolation of vertex positions from fine to coarse grid near LOD boundaries. |
| **Morph alpha** | A scalar [0,1] that drives geomorphing: 0 = fully fine, 1 = fully coarse. |

---

### Geometry: Patches and Rings

Every LOD level is drawn from **one shared `P×P` patch mesh** (default `P = 64`
vertices per edge, so 63×63 quads).  All patches in all levels use this identical mesh.
The GPU draws thousands of instances from a single draw call; each instance's world
position, LOD level, and morph range are stored in a **GPU storage buffer**
(`PatchDescriptorGpu`, one entry per instance).

In the vertex shader, the flat-grid mesh UVs are converted to world-space XZ using the
per-instance `origin_ws` and `patch_size_ws`.  The height is then sampled from the
clipmap texture for that LOD level.

Each ring at level `L` is an `R×R` grid of patches (default `R = 8`) with the inner
`(R/2)×(R/2)` centre removed (that region is covered by the finer level `L-1`).
Level 0 has no hole — it fills the entire centre.

The patch grid is **snapped** to the level's texel grid every frame.  The snap granularity
doubles with each level, so the camera must travel `2^L` world units before level `L`
re-centres.  This prevents the clipmap from scrolling continuously and bounds the amount
of texture data that must be re-uploaded per frame.

---

### LOD Scale

Each level's texel spacing is:

```
level_scale(L) = world_scale × 2^L
```

With the default runtime `world_scale = 1.0`:
- LOD 0: 1 m per patch vertex, 64 m patch side
- LOD 1: 2 m, 128 m patch side
- LOD 2: 4 m, 256 m patch side
- …
- LOD 11: 2048 m, 131 072 m patch side

The full 12-level default configuration covers ~1 000 km of visible terrain from any
camera position.

---

### Geomorphing

Within each ring there is a **morph band** from `morph_start` to `morph_end` (the outer
edge of the ring).  In this band, `morph_alpha` ramps linearly from 0 → 1.

In the vertex shader:
```wgsl
// Snap the fine-grid position to the coarse grid.
let coarse_xz = floor(world_xz / coarse_step) * coarse_step;
// Blend between fine and coarse position.
let morphed_xz = mix(world_xz, coarse_xz, morph_alpha);
```

At `morph_alpha = 1` (the ring boundary), the fine-level vertex is exactly at a
coarse-level vertex position — so adjacent rings share identical edge vertices and no
crack can form.

The height and normal at the morphed position are also blended:
```wgsl
let h = mix(height_at(lod, world_xz), height_at(coarse_lod, coarse_xz), morph_alpha);
let n = mix(normal_at(lod, world_xz), normal_at(coarse_lod, coarse_xz), morph_alpha);
```

`morph_alpha` is passed through to the fragment stage so the per-pixel normal blend
mirrors the geometry blend exactly.

---

### Clipmap Textures

The height data is stored in a `Texture2DArray`:
- **Format**: `R16Unorm` (65 535 discrete levels over `height_scale` world units, giving
  ~0.008 m precision over a 512 m range — enough to eliminate staircase artefacts in normals).
- **Array layers**: one per LOD level (up to 16).
- **Resolution per layer**: `ring_patches × patch_resolution` texels square (default 512×512).
- **Addressing**: `ClampToEdge`, `Linear` filtering.

Each layer covers the camera's current ring for that LOD level.  When the camera crosses
a snapped grid boundary for level `L`, the affected row or column of texels is
overwritten with freshly loaded (or procedurally generated) data.

A parallel `Texture2DArray` stores pre-baked surface normals in `RG8Snorm` (X and Z
components; Y is reconstructed as `sqrt(1 - x² - z²)`).

The per-layer clip origin and texel-to-world scale are uploaded to the shader as a
`vec4` array in a uniform buffer (`clip_levels[L]`), allowing the vertex and fragment
shaders to convert between world XZ and clipmap UV for any level in O(1).

---

### Tile Streaming

The on-disk dataset is a mip pyramid of tiles:
```
assets/tiles/
  height/L0/  tx_ty.bin     R16Unorm, 256×256 texels
  height/L1/  …
  …
  normal/L0/  tx_ty.bin     RG8Snorm, 256×256 texels
  …
```

Each frame, `update_required_tiles` computes which tiles are needed for each active LOD
ring.  Tiles not already resident are enqueued in `TerrainStreamQueue`.
`request_tile_loads` dispatches background load jobs via a `std::sync::mpsc` channel;
each job either reads a binary tile file or falls back to procedural generation.
Completed tiles are received by `poll_tile_stream_jobs` and written into the clipmap
texture array via `apply_tiles_to_clipmap`.

If a tile for the precise LOD level is not available (the baked hierarchy has fewer mip
levels than the geometry has rings), the loader resamples from the coarsest available
mip rather than leaving the region empty.  This prevents distant terrain from appearing
as a flat floor.

The tile residency manager (`TerrainResidency`) tracks which tiles are loaded and evicts
the least-recently-needed tiles when `max_resident_tiles` is exceeded.

---

### Fragment Shader: Per-Pixel Normals

Because polygon size grows exponentially with LOD level, vertex normals would look faceted
on large distant patches.  The fragment shader discards the interpolated vertex normal for
surface lighting purposes and recomputes the normal from the height clipmap using finite
differences:

```wgsl
fn pixel_normal(lod: u32, xz: vec2<f32>) -> vec3<f32> {
    let eps   = clip_levels[lod].w;   // texel world size for this level
    let h_c   = sample_height(lod, xz);
    let h_r   = sample_height(lod, xz + vec2(eps, 0.0));
    let h_u   = sample_height(lod, xz + vec2(0.0, eps));
    return normalize(vec3(h_c - h_r, eps, h_c - h_u));
}
```

At LOD boundaries the fine and coarse normals are blended using `morph_alpha` (passed
from the vertex stage) so there is no lighting discontinuity at ring edges:

```wgsl
let coarse_lod = min(lod_level + 1u, max_lod);
let n_fine     = pixel_normal(lod_level, world_pos.xz);
let n_coarse   = pixel_normal(coarse_lod, world_pos.xz);
let n          = normalize(mix(n_fine, n_coarse, morph_alpha));
```

Both sides of the ring boundary sample `pixel_normal` with `morph_alpha = 1`, which
forces them to use the same coarse-level height data at the same world position —
guaranteeing continuous lighting across the seam.

---

### Macro Color

A world-aligned image (EXR or PNG) provides a base albedo that covers the entire terrain
at coarse resolution.  It is loaded once at startup, downsampled to
`macro_color_resolution` (default 4096 px), and uploaded as a standard `Texture2d`.

The fragment shader blends between the macro color and a procedural slope/altitude
shading based on camera distance and a `bounds_fade` parameter.  At large distances the
macro color dominates, giving artistically controlled coloration of mountains, valleys,
and coastlines without per-texel material data.

---

### Physics Colliders (Avian3D)

The crate integrates with [avian3d](https://github.com/Jondolf/avian) for collision.
A coarse global heightfield collider covers the entire terrain footprint at a reduced
resolution for broad character physics.  As the camera moves, per-tile heightfield
colliders are added and removed at fine resolution for the loaded region immediately
around the player, allowing accurate contact with small terrain features.

---

### Plugin API

```rust
use bevy_landscape::{TerrainPlugin, TerrainConfig, TerrainSourceDesc, TerrainCamera};

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(TerrainPlugin {
        config: TerrainConfig {
            clipmap_levels: 12,
            height_scale: 1024.0,
            // … see TerrainConfig fields
            ..default()
        },
        source: TerrainSourceDesc {
            tile_root: Some("assets/tiles".into()),
            world_min: Vec2::splat(-8192.0),
            world_max: Vec2::splat(8192.0),
            max_mip_level: 8,
            ..default()
        },
    })
    .run();
```

Mark the camera entity with the `TerrainCamera` component so the renderer knows which
view to centre the clipmap on.

`TerrainPlugin` automatically registers `WireframePlugin` if it is not already present
(required for the debug wireframe overlay).  It is safe to add your own `WireframePlugin`
before `TerrainPlugin`; the duplicate is suppressed.

---

### Key Configuration Fields

| Field | Default | Effect |
|-------|---------|--------|
| `clipmap_levels` | 12 | Number of LOD rings. Each adds 2× view range. |
| `patch_resolution` | 64 | Vertices per patch edge. Higher = smoother slopes, more GPU cost. |
| `ring_patches` | 8 | Patches per ring edge. Higher = wider rings, more draw calls. |
| `world_scale` | 1.0 | Uniform runtime terrain scale. `2.0` makes the same landscape 2× larger in X, Y, and Z. In the app binary this is usually loaded from `landscape.toml`. |
| `height_scale` | 1024.0 | Base height range before the uniform `world_scale` multiplier is applied. |
| `morph_start_ratio` | 0.6 | Fraction of ring width where morphing begins. |
| `max_resident_tiles` | 256 | GPU tile cache size. Increase for lower pop-in. |
| `use_macro_color_map` | true | Use the world-aligned EXR for distant albedo. |
| `procedural_fallback` | false | Fill missing tiles with sine-wave stub heights. |

---

### Module Map

| Module | Responsibility |
|--------|---------------|
| `config.rs` | `TerrainConfig` resource |
| `world_desc.rs` | `TerrainSourceDesc` resource (tile paths, world bounds) |
| `clipmap.rs` | CPU-side ring layout: builds `PatchInstanceCpu` list each frame |
| `clipmap_texture.rs` | Clipmap `Texture2DArray` management; layer upload logic |
| `math.rs` | `level_scale`, `snap_camera_to_level_grid`, `build_ring_patch_origins`, `morph_factor` |
| `streamer.rs` | Background tile loader; `mpsc` channel; procedural fallback |
| `residency.rs` | Tile residency tracking; LRU eviction |
| `resources.rs` | `TerrainResidency`, `TerrainViewState`, `TileKey`, `TileState` |
| `patch_mesh.rs` | Builds the shared `P×P` patch mesh |
| `material.rs` | `TerrainMaterial` and `TerrainMaterialUniforms` |
| `macro_color.rs` | Loads and downsamples the world-aligned diffuse map |
| `collision.rs` | Per-tile heightfield colliders (avian3d) |
| `physics_colliders.rs` | Coarse global heightfield; tile collider sync |
| `components.rs` | `TerrainCamera` marker component |
| `render/` | GPU types, draw commands, render plugin |
| `debug.rs` | `TerrainDebugPlugin` (wireframe toggle, stats overlay) |

Shaders live in `assets/shaders/`:
- `terrain_vertex.wgsl` — CDLOD vertex displacement and geomorphing
- `terrain_fragment.wgsl` — per-pixel normals, PBR lighting, macro color
- `terrain_prepass.wgsl` — depth prepass (no normal computation)

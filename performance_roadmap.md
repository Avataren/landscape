# Terrain Performance Roadmap

## Methodology

Before touching any code, establish a baseline and a repeatable measurement harness.

### Instrumentation

Add Bevy's built-in diagnostic plugins for a first pass:

```rust
// main.rs
.add_plugins(bevy::diagnostic::SystemInformationDiagnosticsPlugin)
.add_plugins(bevy::diagnostic::EntityCountDiagnosticsPlugin)
```

For per-system CPU timing, enable the `trace` feature and attach `tracy` or use
`bevy_diagnostic`'s `LogDiagnosticsPlugin`.  The most important spans are the
terrain update systems: `update_clipmap_textures`, `apply_tiles_to_clipmap`,
`update_patch_transforms`, and `sync_tile_colliders`.

For GPU profiling: use a **RenderDoc** frame capture to inspect GPU draw time,
texture upload bandwidth, and the shadow map passes.  For finer-grained GPU
timings, `wgpu::Features::TIMESTAMP_QUERY` can be added to the `WgpuSettings`
to get per-pass GPU timestamps.

### Criterion microbenchmarks

Create `benches/terrain.rs` targeting the pure-CPU hot paths:

```rust
criterion_group!(benches,
    bench_build_patch_instances,   // P2
    bench_write_new_strip,         // P1
    bench_apply_tiles_to_clipmap,  // P1
);
```

Run with `cargo bench --bench terrain` before and after each fix.

### Benchmark scenarios

Run each of the following for 60 seconds and capture min/avg/max frame time,
GPU texture upload bandwidth (RenderDoc), and per-system CPU time (tracy):

| # | Scenario | What it stresses |
|---|---|---|
| S1 | Camera perfectly still | Idle cost, should be near zero streaming work |
| S2 | Camera pan at 500 m/s | Normal play, incremental strip updates |
| S3 | Camera pan at 5 000 m/s (shift) | Peak streaming load, full clipmap rebuilds |
| S4 | Freecam teleport across world | Burst tile eviction + re-load |

---

## Bottlenecks (ordered by estimated impact)

---

### ✅ P0 — Full clipmap texture GPU re-upload every camera-move frame

**Status: FIXED** (`8945d2d`, `16a5793`)

**Original impact: HIGH — ~12 MB/frame GPU upload when moving**

**What was done**

Implemented a custom `TerrainRenderPlugin` with a `RenderQueue::write_texture`
path that uploads only the dirty rectangles to the GPU each frame:

1. **`8945d2d`** — Removed `MAIN_WORLD | RENDER_WORLD` as the mutation target.
   Introduced `TerrainClipmapUploads` (an `ExtractResource`) to carry dirty-level
   descriptors from the main world to the render world.  `TerrainRenderPlugin`
   now runs `apply_terrain_texture_uploads` in `PrepareResources`, calling
   `RenderQueue::write_texture` once per dirty level.  Repeated uploads to the
   same level within a frame are deduplicated before submission.

2. **`16a5793`** — Refined further: instead of uploading an entire layer per
   dirty level, the system now slices only the bytes corresponding to the L-shaped
   strip that actually changed (the same region `write_new_strip` computed on the
   CPU side).  Toroidal seam splits are handled — strips that wrap around the
   texture edge are submitted as two separate `write_texture` calls.

**Result**: normal camera motion uploads only the exposed border strips (a thin
band of texels per level per frame) rather than full 512×512 layers.  Full-layer
uploads are retained as a fallback for rebuilds and large teleports.

---

### ✅ P1 — `apply_tiles_to_clipmap`: O(tiles × tile_texels) CPU writes per clip-center shift

**Status: FIXED** (`352d2d1`, `16a5793`)

**Original impact: HIGH — up to ~33 MB of CPU memory writes per frame when moving**

**What was done**

`352d2d1` rewrote `apply_tiles_to_clipmap` with two separate paths:

- **New tiles / full rebuild**: write the full intersection of the tile with the
  current clipmap window (unchanged from before, only triggered when a new tile
  arrives or an eviction forces a rebuild).

- **Resident tiles during camera movement**: compute the newly exposed strip for
  each resident tile (the intersection of the tile with the delta between old and
  new window) and write only those texels.  Tiles that have not moved relative to
  the current window contribute zero bytes.

`16a5793` then replaced the direct CPU-buffer memcpy with enqueuing dirty rects
into `TerrainClipmapUploads`, which the render plugin submits via
`RenderQueue::write_texture` (see P0 above).

**Result**: during steady panning, CPU tile writes drop from O(tiles × tile²) to
O(newly_visible_strip_area), typically a thin band of texels per tile per axis of
motion rather than the full 256 × 256 tile.

---

### ✅ P2 — `build_patch_instances_for_view` called twice per frame

**Status: FIXED** (`b8a695a`)

**Original impact: MEDIUM — double CPU allocation and computation every frame**

**What was done**

Removed `extract_terrain_frame` and `ExtractedTerrainFrame` entirely — the
result was never consumed by anything.  `update_patch_transforms` was already
computing the patch list independently; that single call remains.

Additionally, `update_patch_transforms` now returns early when clip centers
haven't changed (`!positions_changed`), so `build_patch_instances_for_view` is
not called at all on static-camera frames.

---

### ✅ P3 — Storage buffer re-upload every frame

**Status: FIXED** (as part of `b8a695a`, frustum-cull removal)

**Original impact: MEDIUM — ~18 KB GPU upload per frame even when nothing changed**

**What was done**

The storage buffer frustum-cull path (which unconditionally rebuilt and
re-uploaded the buffer every frame to zero-out culled patches) was removed after
it was found to incorrectly cull all patches from altitude in Bevy 0.18
(reversed-Z frustum planes).  `update_patch_transforms` now:

- Returns early if `!positions_changed`.
- Rebuilds and uploads the storage buffer only when the clip grid shifts.

Entities carry `NoFrustumCulling`, so Bevy's visibility system never hides them;
the GPU hardware rasteriser discards out-of-view triangles at near-zero cost.

**Remaining note**: a correct GPU-side frustum cull (driven by the shader or by
compute) is still a future optimisation opportunity, but the correctness bug must
be understood before reintroducing it.

---

### ✅ P5 — `sync_sun_direction` mutates material unconditionally every frame

**Status: FIXED** (`b8a695a`)

**Original impact: LOW — spurious material re-upload each frame**

**What was done**

`sync_sun_direction` was deleted entirely.  The terrain fragment shader now reads
all light directions directly from `mesh_view_bindings::lights` (Bevy's clustered
forward lighting bindings), so no per-frame CPU-side sync is needed.  As a side
effect the `sun_direction` field was removed from `TerrainMaterialUniforms`,
shrinking the uniform buffer by 16 bytes.

---

### ✅ P4 — Backface culling disabled on terrain mesh

**Status: FIXED**

**Original impact: MEDIUM — up to 2× triangle throughput on steep terrain**

**What was done**

Enabled `cull_mode = Some(Face::Back)` in `TerrainMaterial::specialize`.
The original concern (steep faces becoming invisible) does not apply to a
heightfield: all mesh triangles have consistent CCW winding when viewed from
above, regardless of steepness.  Backfaces are only visible from underground,
which is the correct thing to cull.  The fragment-shader discard workaround
was not needed.

---

### ✅ P6 — Cascade shadow range and far clip plane

**Status: FIXED**

**Original impact: MEDIUM — GPU shadow map and depth buffer precision costs**

**What was done**

- Far clip reduced from 10 000 000 to **100 000** (100 km), cutting the
  depth range from 100 000 000:1 to 1 000 000:1 and eliminating Z-fighting
  at normal terrain viewing distances.
- Shadow cascade `maximum_distance` reduced from 20 000 to **8 000** m.
  The last cascade now covers 2–8 km instead of 5–20 km, giving meaningfully
  denser shadow-map texels over the visible terrain range.

---

## Status summary

| Item | Description | Status |
|---|---|---|
| P0 | Partial GPU texture upload via `write_texture` | ✅ Fixed (`8945d2d`, `16a5793`) |
| P1 | Incremental tile stamping in `apply_tiles_to_clipmap` | ✅ Fixed (`352d2d1`, `16a5793`) |
| P2 | Deduplicate `build_patch_instances_for_view` | ✅ Fixed (`b8a695a`) |
| P3 | Storage buffer upload gated on clip-center change | ✅ Fixed (`b8a695a`) |
| P4 | Backface culling on terrain mesh | ✅ Fixed |
| P5 | Remove `sync_sun_direction` | ✅ Fixed (`b8a695a`) |
| P6 | Cascade shadow range and far clip plane | ✅ Fixed |

All seven items are now resolved.

---

## Expected gains (rough estimates before profiling)

| Fix | S1 (static) | S2 (normal pan) | S3 (fast pan) |
|---|---|---|---|
| P2 + P3 ✅ | −0.5 ms CPU | −0.5 ms CPU | −0.5 ms CPU |
| P5 ✅ | −0.1 ms GPU | −0.1 ms GPU | −0.1 ms GPU |
| P6 ✅ | −0.1 ms GPU | −0.1 ms GPU | −0.1 ms GPU |
| P4 ✅ | — | −1–3 ms GPU | −1–3 ms GPU |
| P1 ✅ | — | −2–5 ms CPU | −5–15 ms CPU |
| P0 ✅ | — | −2–8 ms GPU BW | −5–20 ms GPU BW |

All estimates are speculative until measured.  Profile first, implement second.

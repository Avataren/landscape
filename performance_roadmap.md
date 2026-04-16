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

### P0 — Full clipmap texture GPU re-upload every camera-move frame

**Estimated impact: HIGH — ~12 MB/frame GPU upload when moving**

**Root cause**

Both clipmap images are created with:
```rust
// clipmap_texture.rs:177
RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD
```

When any system calls `images.get_mut(&handle)` and writes even one byte, Bevy
marks the entire `Image` asset as changed and re-uploads the full texture to the
GPU on the next render frame.

Texture sizes:
- Height: 512 × 512 × 2 B (R16Unorm) × 12 levels = **6.3 MB**
- Normal: 512 × 512 × 2 B (RG8Snorm) × 12 levels = **6.3 MB**
- **Total: ~12.5 MB uploaded to GPU every frame the camera moves**

`update_clipmap_textures` (which calls `get_mut`) fires whenever any clip center
changes — i.e., on every frame the camera crosses a texel boundary, which in
practice is most frames during movement.

**Fix: partial texture writes via a staging buffer**

Stop using `MAIN_WORLD | RENDER_WORLD` images as the mutation target.  Instead:

1. Keep a CPU-side `Vec<u8>` for each level layer (already done — it's
   `image.data`).
2. In a custom render extract system, diff the dirty rect against a
   `last_uploaded_center` and call `wgpu::Queue::write_texture` targeting only
   the sub-region that changed — the same L-shaped strip computed by
   `write_new_strip`.

Alternatively, as a lower-effort interim fix, split the texture into per-level
separate `Image` handles (12 images instead of 1 array) and only call `get_mut`
on the levels that actually moved their clip center that frame.  This still
re-uploads the full 512 KB per dirty level, but only the levels that moved
(usually 1–3 per frame at walking speed) instead of all 12.

**Implementation detail**

The `write_new_strip` function already computes the correct dirty rect (gx range,
gz range).  Translate that to texel coordinates and call:

```rust
queue.write_texture(
    wgpu::TexelCopyTextureInfo {
        texture: &gpu_texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x: tx_lo, y: tz_lo, z: level as u32 },
        aspect: wgpu::TextureAspect::All,
    },
    &strip_bytes,
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(strip_width_texels * 2),
        rows_per_image: None,
    },
    wgpu::Extent3d { width: strip_width_texels, height: strip_height_texels, depth_or_array_layers: 1 },
);
```

This requires moving to a custom `RenderPlugin` (Phase 2+ of the render
roadmap), which is already planned (`TerrainRenderPlugin` is a stub).

---

### P1 — `apply_tiles_to_clipmap`: O(tiles × tile_texels) CPU writes per clip-center shift

**Estimated impact: HIGH — up to ~33 MB of CPU memory writes per frame when moving**

**Root cause**

`apply_tiles_to_clipmap` (`clipmap_texture.rs:634`) re-stamps every resident CPU
tile into the clipmap image buffer whenever clip centers change *or* new tiles
arrive.  With 256 resident tiles at 256 × 256 texels each:

```
256 tiles × 256 × 256 texels × 2 bytes = 33.6 MB of CPU memcpy per frame
```

The full re-stamp is necessary today because `update_clipmap_textures` runs
first and may regenerate entire layers from the procedural fallback, overwriting
tile data.  But this coupling creates an O(tiles × tile_texels) cost on every
frame the camera moves.

**Fix: tile-level incremental stamping**

Maintain a per-tile "last write clip center" cache alongside `TileColliders`:

```rust
// In apply_tiles_to_clipmap, for each tile:
let tile_min_grid = IVec2::new(key.x * ts as i32, key.y * ts as i32);
let tile_max_grid = tile_min_grid + IVec2::splat(ts as i32);

// Clip center window: [center - half, center + half)
let win_min = new_center - IVec2::splat(half);
let win_max = new_center + IVec2::splat(half);

// Intersection of tile and new window that wasn't in old window:
let dirty_x = compute_new_strip_range(win_min.x, win_max.x, old_win_min.x, old_win_max.x);
let dirty_z = compute_new_strip_range(win_min.y, win_max.y, old_win_min.y, old_win_max.y);
// Only write the L-shaped strip of texels that newly entered the window.
```

When `procedural_fallback = false` (the production path), `update_clipmap_textures`
only writes zeros into new strips and tile data is the ground truth.  In this
case, tile re-stamping can be skipped entirely for tiles that haven't moved in
or out of the window.

**Additional fix: decouple tile stamping from procedural regeneration**

Instead of procedural regeneration zeroing tiles and then tile stamping
re-applying them, separate the paths entirely:

- Procedural path: `update_clipmap_textures` generates the full layer.
- Tile path: `apply_tiles_to_clipmap` incrementally stamps new tiles only.
- Tiles never need to be re-stamped when the clip center shifts — only the
  newly exposed strip for each tile needs writing, and only if that strip
  intersects the tile's bounds.

---

### P2 — `build_patch_instances_for_view` called twice per frame

**Estimated impact: MEDIUM — double CPU allocation and computation every frame**

**Root cause**

`build_patch_instances_for_view` is called independently in two systems:

```rust
// extract.rs:37
let patches = build_patch_instances_for_view(&config, &view);

// mod.rs:533 (update_patch_transforms)
let patches = build_patch_instances_for_view(&config, &view);
```

Both systems run every `Update` frame.  The function iterates all LOD levels,
calls `build_ring_patch_origins` per level (which allocates a `Vec<Vec2>`), and
collects into a final `Vec<PatchInstanceCpu>`.  With the default config
(12 levels, 8 ring patches): ~592 patches.

**Fix: compute once, share via `TerrainViewState`**

Add a `patches` field to `TerrainViewState` (or a dedicated `PatchLayout`
resource) and populate it in `update_patch_transforms`, which already does
the computation.  Remove the call from `extract_terrain_frame` and read from the
shared resource instead:

```rust
// In TerrainViewState or a new PatchLayout resource:
pub patches: Vec<PatchInstanceCpu>,

// update_patch_transforms computes patches once and stores them:
let patches = build_patch_instances_for_view(&config, &view);
view.patches = patches.clone();  // or store in PatchLayout

// extract_terrain_frame reads without recomputing:
extracted.patches = view.patches.clone();
```

A further optimisation: skip `build_patch_instances_for_view` entirely on frames
where clip centers haven't changed — the layout is identical.  Use the
`positions_changed` flag already computed in `update_patch_transforms`:

```rust
if !positions_changed && !view.patches.is_empty() {
    // Frustum cull only — reuse existing patch layout.
}
```

---

### P3 — Storage buffer re-upload every frame (unconditional frustum cull path)

**Estimated impact: MEDIUM — ~18 KB GPU upload per frame even when nothing changed**

**Root cause**

`update_patch_transforms` (`mod.rs:550`) unconditionally rebuilds and re-uploads
the `ShaderStorageBuffer` every frame:

```rust
// Always runs, every frame:
if let Some(ssb) = storage_buffers.get_mut(&patch_entities.patch_buffer_handle) {
    let descs: Vec<PatchDescriptorGpu> = patches.iter().map(|p| { ... }).collect();
    ssb.data = Some(bytemuck::cast_slice(&descs).to_vec());
}
```

Calling `storage_buffers.get_mut()` marks the buffer as changed and triggers a
GPU upload every frame (592 patches × 32 bytes = ~18 KB).  Frustum culling is
the reason this runs every frame — the `patch_size_ws` field is zeroed for
culled patches, which changes with camera rotation.

**Fix: cache the last frustum and skip re-upload when nothing changed**

```rust
#[derive(Default)]
struct FrustumCache {
    last_planes: [Vec4; 6],
    last_clip_centers: Vec<IVec2>,
}

fn update_patch_transforms(
    mut frustum_cache: Local<FrustumCache>,
    ...
) {
    let frustum_planes = extract_frustum_planes(&frustum);
    let frustum_changed = frustum_planes != frustum_cache.last_planes;
    let centers_changed = view.clip_centers != patch_entities.last_clip_centers;

    if !frustum_changed && !centers_changed { return; }

    // rebuild + upload
    frustum_cache.last_planes = frustum_planes;
    ...
}
```

For the static camera case (S1), this eliminates the storage buffer upload
entirely.

---

### P4 — Backface culling disabled on terrain mesh

**Estimated impact: MEDIUM — up to 2× triangle throughput on steep terrain**

**Root cause**

```rust
// material.rs:118
descriptor.primitive.cull_mode = None;
```

Backface culling is disabled because steep cliff faces would be invisible if the
winding order is inconsistent.  The flat mesh generates CCW winding when viewed
from above, but a steep face (nearly vertical) can appear from below, making the
winding appear CW to the camera — which backface culling would discard.

**Fix: conditionally enable backface culling via shader discard on steep faces**

Keep `cull_mode = Some(Face::Back)` and prevent the issue at the mesh level:

1. Compute per-vertex normals in the vertex shader (already done via the
   `normal_at` function).
2. In the fragment shader, discard fragments where `dot(normal, view_dir) < 0`.
   This avoids the cost of double-sided rasterization while still allowing steep
   faces to be visible from adjacent lower geometry.

Alternatively, keep `cull_mode = None` but split the geometry: use backface
culling for LOD levels 3+ (coarse, low slope range), and only disable it for
LOD 0–2 where steep geometry is densest.

---

### P5 — `sync_sun_direction` mutates material unconditionally every frame

**Estimated impact: LOW — spurious material re-upload each frame**

**Root cause**

```rust
// mod.rs (sync_sun_direction) — runs every Update frame:
let Some(mat) = materials.get_mut(&state.material_handle) else { return };
mat.params.sun_direction = ...;
```

`materials.get_mut()` marks the material as changed regardless of whether the
sun direction actually changed.  This triggers Bevy to re-upload the 312-byte
`TerrainMaterialUniforms` uniform buffer every frame, even when the
`DirectionalLight` is perfectly still.

**Fix: compare before mutating**

```rust
fn sync_sun_direction(...) {
    let Ok(transform) = lights.single() else { return };
    let new_dir = Vec3::from(-transform.forward()).normalize_or_zero().extend(0.0);

    let Some(mat) = materials.get(&state.material_handle) else { return };
    if mat.params.sun_direction == new_dir { return; }  // skip if unchanged

    let Some(mat) = materials.get_mut(&state.material_handle) else { return };
    mat.params.sun_direction = new_dir;
}
```

---

### P6 — Cascade shadow range and far clip plane

**Estimated impact: MEDIUM — GPU shadow map and depth buffer precision costs**

**Current values**

```rust
// main.rs
maximum_distance: 20_000.0,   // 20 km shadow range
far: 10_000_000.0,            // 10 000 km far clip plane
```

A 20 km shadow range with 4 cascades covers a large area.  The last cascade's
shadow map must cover a frustum slice from ~5 km to 20 km, at which point texels
are extremely sparse and shadow quality collapses.  This also forces large shadow
map texture allocations.

The 10 million unit far clip plane combined with near=0.1 gives a 100 000 000:1
depth range.  Even with a 32-bit depth buffer this causes Z-fighting at
distances beyond a few hundred metres.

**Fix**

```rust
// Reduce shadow range to what's visually meaningful:
maximum_distance: 8_000.0,

// Reduce far plane to something sane for the world size:
far: 100_000.0,   // 100 km — covers the full terrain footprint with margin
```

For depth precision at large scales, consider a reversed-Z projection
(`camera_near = f32::MAX, camera_far = 0`) combined with
`DepthBiasState` tweaks in the terrain pipeline — reversed-Z distributes
depth precision toward the near plane where it matters most.

---

## Implementation order

| Priority | Bottleneck | Effort | Risk |
|---|---|---|---|
| P2 | Deduplicate `build_patch_instances_for_view` | Low | Low |
| P5 | Guard `sync_sun_direction` | Trivial | None |
| P6 | Reduce shadow range and far clip | Trivial | Visual regression check |
| P3 | Cache frustum for storage buffer skip | Low | Low |
| P4 | Backface culling | Medium | Visual regression check |
| P1 | Tile incremental stamping | Medium | Clipmap correctness |
| P0 | Partial texture upload via staging buffer | High | Requires render plugin |

P2, P5, P6 are safe wins achievable in a single session.  P0 and P1 are the
largest gains but require either the custom render pipeline (P0) or careful
correctness work on the tile stamping logic (P1).

---

## Expected gains (rough estimates before profiling)

| Fix | S1 (static) | S2 (normal pan) | S3 (fast pan) |
|---|---|---|---|
| P2 + P3 | −0.5 ms CPU | −0.5 ms CPU | −0.5 ms CPU |
| P5 + P6 | −0.1 ms GPU | −0.1 ms GPU | −0.1 ms GPU |
| P4 | — | −1–3 ms GPU | −1–3 ms GPU |
| P1 | — | −2–5 ms CPU | −5–15 ms CPU |
| P0 | — | −2–8 ms GPU BW | −5–20 ms GPU BW |

All estimates are speculative until measured.  Profile first, implement second.

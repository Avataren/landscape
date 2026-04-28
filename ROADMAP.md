# Landscape Editor: Foliage Implementation Roadmap

**Last updated:** 2026-04-27  
**Current phase:** GPU Grass (instant rendering) ✅  
**Progress:** Phases 1-8a ✅, GPU grass ✅, Pre-bake pipeline ⏳ (on hold)

---

## Executive Summary

The foliage system provides **GPU-driven instant grass rendering** for the Bevy landscape editor — no pre-baking step, blades appear immediately when a level loads. The vertex shader places each blade on the terrain by sampling the heightmap directly on the GPU, with slope and altitude filtering.

**Architecture (active):**
- `GpuGrassPlugin` — vertex-shader-driven grass, zero CPU placement cost per frame
- `GpuGrassMaterial` — `AsBindGroup` material; binds heightmap + grass params
- `grass_blade.wgsl` — 12-vert blade geometry generated from `@builtin(vertex_index)`
- `update_grass_material` — pushes camera XZ + wind time each frame (small uniform update)
- `GpuGrassConfig` — live-tweakable: grid size, spacing, blade dimensions, slope filter, color

**Legacy pipeline (Phases 1-8a, kept for reference / future painter tool):**
- Pre-bake pipeline: generate → disk tiles → stream → GPU
- Editor UI panel with parameter inspection and "Generate" button
- landscape.toml integration, hot-reload, LRU streaming infrastructure

**Current state:**
- GPU grass renders instantly at startup — no generation step needed
- 128×128 = 16,384 blades visible at once, centred on camera, follow-cam seamlessly
- Blades placed by heightmap sample in vertex shader (slope + altitude filtering)
- Two-axis wind animation, tapered blade shape, half-lambert lighting
- All 110+ existing tests passing

---

## Completed Phases (1-8a)

### Phase 1: Data Model & Storage ✅
**Files:** `foliage.rs`  
**Description:** Defined core types and binary serialization format.

**Deliverables:**
- `FoliageInstance` (position: Vec3, rotation: Quat, scale: Vec3, variant_id: u32)
- `FoliageConfig` (LOD distances, density multipliers, procedural parameters)
- `FoliageSourceDesc` (foliage_root path reference)
- Binary tile format matching terrain layout: `tiles/foliage/LOD{n}/L{level}/{tx}_{ty}.bin`
- Extended `LevelDesc` JSON with `foliage_root` and `FoliageConfig`
- Extended `ReloadTerrainRequest` for hot-reload

**Key decisions:**
- Binary format (not JSON) — efficient serialization, no bloat
- Same L0-Ln mip hierarchy as terrain for aligned streaming
- 48 bytes per instance (Vec3 + Quat + Vec3 + u32 variant + padding)

**Testing:** 9 unit tests validating serialization, parsing, edge cases

---

### Phase 2: Procedural Density Mask Generation ✅
**Files:** `procedural_mask_gen.rs`  
**Description:** Compute density masks from terrain properties.

**Deliverables:**
- `compute_slope_density(normal)` — steeper = lower density
- `compute_altitude_density(height, alt_min, alt_max)` — band-based
- `compute_water_proximity_density(dist_to_water)` — sparse near shorelines
- Combined mask per tile: `mask[cell] = slope × altitude × water_dist`
- Baked into `foliage_root/procedural_masks/L{level}/{tx}_{ty}.bin`

**Key parameters (tunable in FoliageConfig):**
- `slope_threshold: f32` (0-1, default 0.3) — max gradient before falloff
- `altitude_min/max: f32` (world units, default -∞ to +∞) — altitude band
- `water_proximity_radius: f32` (world units, default 10.0)

**Testing:** 8 unit tests for each density function

---

### Phase 3: Instance Generation Pipeline ✅
**Files:** `foliage_instance_gen.rs`  
**Description:** Blend painted splatmap and procedural mask, spawn instances per cell.

**Deliverables:**
- `generate_foliage_instances()` system:
  1. Load painted splatmap (if exists) from `foliage_root/painted/`
  2. Load procedural mask from `foliage_root/procedural_masks/`
  3. Blend: `final_density = painted[cell] × procedural_mask[cell] / 255`
  4. Spawn N instances per cell (N = density × instances_per_cell / 255)
  5. Assign positions (jitter, Poisson disk sampling)
  6. Assign rotations (random Y axis)
  7. Assign scales (random ±10%)
  8. Assign variant_id (0-7, random)
- Bake per LOD tier:
  - **LOD0:** full instance set (100% density)
  - **LOD1:** 50% subsampling (50% density)
  - **LOD2:** 10% subsampling (10% density)
- Write to `foliage_root/LOD{n}/L{level}/{tx}_{ty}.bin`

**Key parameters:**
- `instances_per_cell: u32` (default 64, range 1-256)
- `random_seed: u64` (ensures deterministic generation across runs)

**Testing:** 7 unit tests for blending, subsampling, placement

---

### Phase 4: Grass Mesh Generation ✅
**Files:** `grass_mesh.rs`  
**Description:** Generate 8 procedural grass blade variants.

**Deliverables:**
- 8 mesh assets: `grass_blade_v0.mesh` through `grass_blade_v7.mesh`
- Each variant: 2-sided quad, simple geometry
- Per-variant parameters:
  - `bend_forward: f32` (angle, 0-45°)
  - `tip_curl: f32` (curl amount, 0-1)
  - `asymmetry: f32` (offset, 0-0.5)
- All variants use shared grass material

**Mesh spec:**
- Vertices: 8 per blade (quad + wind deformation points)
- Indices: 12 (2 triangles × 2 sides)
- Bounds: automatically computed per variant
- UV: top-to-bottom gradient (fade alpha at tip)

**Key decisions:**
- Minimal geometry — performance priority
- 2-sided quads — reduces vertex count vs. full LOD models
- Variant generation at startup — no runtime overhead

**Testing:** 6 unit tests for mesh generation, bounds, UV layout

---

### Phase 5: Grass Material Shader ✅
**Files:** `grass_material.rs`  
**Description:** PBR material for grass rendering with world-aligned normals.

**Deliverables:**
- Custom Bevy `Material3d` implementation
- World-aligned normal generation (slope-based)
- Subtle ambient occlusion
- Per-instance color variation
- Placeholder wind animation (TODO: Phase 11)

**Shader uniforms:**
- `world_bounds: Vec4` (world_min.xy, world_max.xy)
- `height_scale: f32` (terrain Y range)
- `num_lod_levels: u32` (clamped to max_mip_level + 1)
- `lod_densities: Vec4` (LOD0, LOD1, LOD2 multipliers)

**Key design:**
- Instance data passed via vertex attributes (no per-instance uniforms)
- Variant ID determines which mesh is rendered (via 24 separate entities)
- Color variation from instance position hash + time

**Testing:** 5 unit tests for shader compilation, uniform binding

---

### Phase 6: Hot-Reload Integration ✅
**Files:** `foliage_reload.rs`  
**Description:** Coordinate foliage hot-reload with terrain reload.

**Deliverables:**
- `reload_foliage_system` (subscribes to `ReloadTerrainRequest`)
- Generation counter on all 3 LOD tiers (prevents stale tile corruption)
- Atomic clearing of:
  - Tile residency (LRU cache)
  - GPU state (vertex buffers)
  - Collision cache
- GPU sync request flag (signals render-graph barrier insertion)

**Hot-reload flow:**
```
ReloadTerrainRequest fired
  ↓
reload_foliage_system (early in frame):
  1. Bump generation counter (all LOD tiers)
  2. Clear FoliageResidency cache
  3. Clear FoliageGpuState buffers
  4. Set GPU sync request flag
  ↓
next frame: update_terrain_view_state:
  5. Repopulate LOD selection based on camera
  ↓
subsequent frames:
  6. poll_tile_stream_jobs: discard old tiles (generation check)
  7. New tiles stream in from fresh tile set
```

**Key design:**
- No blocking waits — all async tile loading continues in background
- Generation counter prevents in-flight tiles from corrupting new set
- Preserves existing terrain reload infrastructure

**Testing:** 6 unit tests for generation counter, clearing logic, edge cases

---

### Phase 7: Runtime GPU Instancing Setup ✅
**Files:** `foliage_stream_queue.rs`, `foliage_gpu.rs`, `foliage_entities.rs`, `foliage_render.rs`  
**Description:** Full GPU infrastructure for rendering foliage.

**Deliverables:**

#### 7a: Background Tile Streaming
- `FoliageStreamQueue` resource (per LOD tier)
- `FoliageTileKey` uniquely identifies tiles by level + position
- `FoliageTileCpu` holds decoded instance data (Vec<FoliageInstance>)
- `FoliageResidency` with LRU eviction (160k instances/LOD max)
- Background loader spawning tiles into memory on secondary threads
- Memory budgeting: ~23 MB total (3 LODs × 160k instances × 48 bytes)

#### 7b: GPU Buffer Management
- `FoliageGpuState` resource (global)
- Per-variant buffer offsets and resident counts
- `FoliageStagingQueue` for CPU→GPU batch transfers
- `FoliageGpuSyncRequest` for render-graph coordination

#### 7c: Entity System
- 24 independent foliage entities (8 variants × 3 LODs)
- Each entity: Transform, Visibility, FoliageVariantComponent, Name
- Selective LOD spawning (can disable LOD2 if memory constrained)
- No shader variant logic — separate entities eliminate GPU branching

#### 7d: Render Infrastructure
- `IndirectDrawCommand` metadata for multi-draw
- LOD distance-based culling (XZ only, Y ignored for vertical movement)
  - **LOD0:** 0-50m (full density)
  - **LOD1:** 50-200m (50% density)
  - **LOD2:** 200m+ (10% density)
- `update_foliage_view_state` system for per-frame LOD selection
- Placeholder systems for Phase 8+ (extract, prepare, queue phases)

**Key design:**
- Rubber-duck critique recommendation: **24 entities instead of per-instance shader logic**
  - Simpler to implement ✅
  - More robust ✅
  - Better GPU utilization ✅
- Memory: Fixed allocation (not dynamic) prevents OOM crashes
- LRU eviction when cache fills

**Testing:** 38 unit tests across all 4 sub-phases (all passing)

---

### Phase 8a: Editor UI ✅
**Files:** `foliage_panel.rs` (editor), `landscape.toml`, `main.rs`, `app_config.rs`  
**Description:** UI for foliage configuration and generation control.

**Deliverables:**
- `FoliagePanelState` resource (UI-local state)
- Floating "Foliage" window in Tools menu
- Display:
  - Root path from `foliage_root`
  - Instances/cell count
  - Procedural parameters (slope, altitude, LOD distances/densities)
  - Preview toggle
  - **Generate / Regenerate Foliage** button (backend deferred to Phase 9)
- landscape.toml integration:
  - `[foliage]` section with `foliage_root` path
  - Loaded at startup when no level JSON provided
  - Creates `FoliageSourceDesc` and `FoliageConfig` resources automatically

**Current state:**
- Panel displays correctly on startup
- Button ready (click handler set, no backend yet)
- All parameters readable from UI

**Testing:** Integration tested with 139 total tests

---

## Active: GPU Grass System ✅

### GpuGrassPlugin (completed 2026-04-27)
**Files:** `foliage_gpu_grass.rs`, `assets/shaders/grass_blade.wgsl`  
**Approach:** Vertex-shader-driven placement, zero pre-baking.

**How it works:**
1. A single mesh with `grid_size² × 12` dummy vertices is spawned at startup
2. `GpuGrassMaterial` binds the terrain heightmap + `GrassParamsGpu` uniform
3. Vertex shader uses `@builtin(vertex_index)` to:
   - Decode blade index and intra-blade vertex
   - Snap camera position to grid, add per-blade jitter
   - Sample heightmap at blade XZ with `textureSampleLevel`
   - Reject blade if slope > max or altitude out of range
   - Generate two crossed-quad geometry (12 verts per blade)
   - Apply wind displacement at tip
4. `update_grass_material` runs every frame to push camera XZ, wind_time, and heightmap handle

**Configuration (GpuGrassConfig):**
| Field | Default | Notes |
|-------|---------|-------|
| `grid_size` | 128 | N×N blades centred on camera |
| `spacing` | 1.5 m | Cell size (jitter within ±0.65×spacing) |
| `blade_height` | 0.7 m | World-space blade height |
| `blade_width` | 0.12 m | Base width |
| `slope_max` | 0.8 | Gradient magnitude cutoff |
| `altitude_min/max` | −100/3000 m | Altitude band |
| `wind_strength` | 0.15 m | Tip displacement amplitude |
| `wind_scale` | 0.035 | Spatial frequency |
| `base_color` | dark green | RGB in linear space |

**Performance characteristics:**
- GPU draw: 16,384 blades × 12 verts = 196,608 vertex shader invocations
- CPU cost per frame: ~1 uniform write (80 bytes)
- Degenerate triangles (blades outside filters) → zero rasterizer cost
- Draw distance = grid_size × spacing / 2 = ~96 m radius at defaults

**Tuning for larger draw distance:**
- `grid_size = 256, spacing = 2.0` → 256m radius, 786K verts (~3ms GPU)
- `grid_size = 512, spacing = 1.0` → 256m radius, 3.1M verts (dense close, sparse far)

---

## Remaining Phases

### Pre-bake pipeline (Phases 8b-11) — On hold
**Reason:** Superseded by `GpuGrassPlugin` for the primary grass use case.  
The pre-bake infrastructure (Phases 1-7) remains in the codebase for potential future use:
- Painted splatmap support (artist control of density)
- Tree / rock instancing (non-grass foliage types)
- Very high instance counts beyond what the vertex-shader grid can handle

### Phase 8b: Render-Graph GPU Integration ⏳ (deferred)
**Estimated scope:** 2-3 sessions  
**Dependencies:** Phase 7 complete (✅)  
**Blocks:** Phase 8c (rendering)

**Goal:** Upload instance buffers to GPU and set up indirect draw infrastructure.

**Deliverables:**

#### Extract Phase
- `extract_foliage_staging` system (runs once per frame)
- Copies CPU staging data into render world
- Reads `FoliageStagingQueue`, writes `RenderFoliageData`

#### Prepare Phase
- `prepare_foliage_gpu_buffers` system
- Allocates GPU buffers via `RenderQueue::write_buffer`
- Uploads to VRAM (vertex buffer, instance buffer, indirect draw buffer)
- Tracks buffer offsets for each LOD tier

#### Queue Phase
- `queue_foliage_indirect_draws` system
- Builds `IndirectDrawCommand` (per-variant, per-LOD)
- Inserts draw calls into render graph
- One multi-draw per variant (all LOD instances in one call)

#### GPU Sync Barrier
- `foliage_gpu_sync_barrier` system
- Runs after prepare phase
- Inserts GPU synchronization point when `FoliageGpuSyncRequest` is set
- Prevents race condition: CPU staging writes must complete before GPU reads

**Technical notes:**
- Use `RenderQueue::write_buffer` to stream instance data
- One vertex buffer per variant (mesh geometry)
- One instance buffer per LOD tier (packed instance data)
- One indirect buffer (draw command metadata)
- Total GPU memory: ~50 MB (vertex + instance + indirect)

**Key challenges:**
1. **GPU memory alignment** — ensure instance offsets are cache-line aligned
2. **Sync barrier timing** — ensure CPU→GPU writes complete before draw calls execute
3. **Frame pacing** — may need double-buffering if GPU writes stall CPU

**Testing strategy:**
- Unit tests for buffer offset calculation
- Integration test: render frame with foliage, verify no GPU errors
- Performance test: measure frame time with 1M instances

**Estimated effort:** 
- Extract phase: 100 lines, 2 tests
- Prepare phase: 150 lines, 4 tests
- Queue phase: 200 lines, 5 tests
- Sync barrier: 50 lines, 1 test
- Total: ~500 lines of code, ~12 tests

---

### Phase 9: Generation Backend & Splatmap Painter 🎨 HIGH PRIORITY
**Estimated scope:** 2-3 sessions  
**Dependencies:** Phase 8b complete (renders but no data yet)  
**Blocks:** Phase 10 (polish)

**Goal:** Let users generate and paint foliage data.

**Deliverables:**

#### 9a: Instance Generation Backend
- `generate_foliage_from_button` system (triggered by UI button click)
- Execute instance generation pipeline:
  1. Scan terrain heightmap at L0
  2. Read painted splatmap (if exists)
  3. Read procedural masks (or generate on-the-fly)
  4. Generate instances per cell (Phase 3 logic)
  5. Bake to foliage_root/LOD{n}/L{level}/{tx}_{ty}.bin
  6. Trigger hot-reload (ReloadTerrainRequest)
- Show progress bar: "⏳ Generating foliage (50% complete)..."
- On completion: hot-reload tiles into viewport

**Implementation:**
- Spawn background task (async) to avoid blocking main thread
- Use same instance generation logic from Phase 3 (foliage_instance_gen.rs)
- Write binary tiles using existing serialization (Phase 1)

#### 9b: Splatmap Painter Tool
- New brush tool in foliage panel: "Paint Grass Density"
- Brush parameters:
  - **Strength:** 0-1 (paint intensity)
  - **Radius:** 1-100 world units (brush size)
  - **Opacity:** 0-1 (fade at edges)
- Paint onto `foliage_root/painted/L{level}/{tx}_{ty}.bin`
- Real-time preview: show density heatmap in viewport
- Undo/redo support (track paint history)
- Clear / Reset buttons

**Implementation:**
- Raycast from camera to terrain at mouse position
- Find tile + cell containing the hit point
- Paint to CPU splatmap buffer
- Write incrementally to disk (auto-save)
- Trigger partial regeneration (only affected tiles)

#### 9c: Procedural Parameter Tuning
- UI sliders in foliage panel:
  - `slope_threshold` (0-1, default 0.3)
  - `altitude_min / altitude_max` (world units)
  - `instances_per_cell` (1-256, default 64)
  - `lod0_density / lod1_density / lod2_density` (0-1)
- Live preview: show density mask overlay
- Apply button: regenerate with new parameters

**Testing strategy:**
- Unit tests for painter (raycasting, cell lookup, paint blending)
- Integration test: paint, regenerate, verify tiles written
- UX test: smooth brush, no stutter, responsive feedback

**Estimated effort:**
- Generation backend: 200 lines, 6 tests
- Splatmap painter: 400 lines, 8 tests
- Parameter tuning UI: 150 lines, 2 tests
- Total: ~750 lines, ~16 tests

**Risk:** Painting can be slow if splatmap tiles are large (256×256 cells).  
**Mitigation:** Consider sparse tile format or compression for future iteration.

---

### Phase 10: Testing & Performance Polish ✓ QUALITY
**Estimated scope:** 1-2 sessions  
**Dependencies:** Phase 9 complete  
**Blocks:** Phase 11 (advanced features)

**Goal:** Achieve target performance and visual quality.

**Deliverables:**

#### 10a: Performance Profiling
- Measure frame time with varying foliage densities:
  - 100k instances: target <5ms GPU time
  - 1M instances: target <20ms GPU time
  - 10M instances: target <100ms GPU time (LOD2 only)
- Memory usage profiling:
  - Resident set size (RSS) vs. foliage LOD count
  - GPU VRAM usage vs. instance count
  - Cache hit rates on LRU eviction
- Profile on target hardware (if available)

#### 10b: Visual Quality Pass
- Variant blending — verify 8 variants appear distinct
- Scale variety — check that ±10% scale is noticeable
- Density uniformity — ensure no obvious patterns or clustering
- LOD transitions — smooth pop-less LOD switching at 50m/200m
- Normal/lighting — grass faces camera correctly, receives shadows

#### 10c: Edge Case Testing
- Tile boundaries — instances don't cluster or gap at transitions
- Memory limits — LRU eviction doesn't cause stutter when cache fills
- Hot-reload stress — reload many times in a row, check for leaks
- Extreme parameters — paint 100% density, 0% density, mixed patterns
- Very large worlds — 100×100 tile grid (25M instances LOD0)

#### 10d: Regression Testing
- Ensure foliage changes don't affect terrain rendering
- Ensure foliage changes don't affect water/sky/clouds
- Ensure hot-reload of foliage doesn't interfere with terrain reload
- Ensure saving/loading level JSON preserves foliage config

**Testing strategy:**
- Automated performance benchmarks (benchmark suite)
- Visual regression tests (screenshot comparison)
- Stress tests (10 consecutive reloads, memory stable?)
- User acceptance criteria:
  - Frame rate 60+ FPS with 500k LOD0 instances in view
  - <100 MB GPU VRAM on dense scenes
  - Zero memory leaks after 30 min continuous play

**Estimated effort:**
- Profiling harness: 200 lines
- Regression test suite: 300 lines, 10 tests
- Visual inspection guide (documentation)
- Total: ~500 lines, ~10 tests

---

### Phase 11: Advanced Features 🚀 FUTURE
**Estimated scope:** 2-3 sessions (deferred, not critical for v1)  
**Dependencies:** Phase 10 complete  
**Blocks:** Nothing (final polish)

**Goal:** Enhanced visuals and interactivity.

**Deliverables:**

#### 11a: Wind Animation
- Per-instance wind response in shader
- Vertex displacement based on:
  - Sine wave (global wind direction)
  - Time offset (per-instance for phase variation)
  - Height along blade (tip moves more than base)
- Wind parameters in shader:
  - `wind_strength: f32` (0-1)
  - `wind_frequency: f32` (Hz)
  - `wind_direction: Vec2` (normalized XZ)

#### 11b: Dynamic Destruction
- Per-instance tombstone in visibility buffer
- Footstep detection (player collision capsule)
- Mark instances as "trampled" (low visibility)
- Fade back in over time (recovery)
- Non-destructive (no permanent tile changes)

#### 11c: Tree Instancing
- LOD-aware tree mesh streaming
- Separate from grass (different material, larger scale)
- Collision capsules for trees (walkthrough prevention)
- Density mask extension for tree placement

#### 11d: Water Interaction
- Grass underwater fades to translucent
- Shoreline foam blending
- Wave influence on grass (based on water FFT)
- Procedural mask: reduce density near water edges

#### 11e: Seasonal Variation
- Color cycling (dry/wet/snow) based on climate simulation
- Fade in/out of foliage based on altitude + temperature
- Snow layer on top of grass (white shader variant)

**Testing strategy:**
- Visual inspection only (no automated tests)
- User feedback iteration
- Performance profiling for wind/destruction overhead

**Estimated effort per feature:**
- Wind animation: 150 lines shader + 50 lines CPU = 200 lines, 3 tests
- Destruction: 300 lines, 5 tests
- Trees: 400 lines, 6 tests
- Water interaction: 200 lines, 3 tests
- Seasonal: 300 lines, 4 tests
- **Total Phase 11:** ~1400 lines, ~21 tests

---

## Architecture Overview

### Data Flow

```
[landscape.toml / level.json]
          ↓
      [FoliageConfig]  ←→  [UI Panel] (Phase 8a)
      [FoliageSourceDesc]
          ↓
    [Instance Gen] (Phase 3, 9a)
          ↓
   [Binary Tiles on Disk]
          ↓
  [Background Loader] (Phase 7a)
          ↓
   [CPU Instance Cache] (LRU)
          ↓
    [GPU Staging Queue]
          ↓
      [GPU Buffers] (Phase 8b)
          ↓
   [Indirect Draw Calls]
          ↓
    [Screen Output]
```

### Memory Hierarchy

| Level | Size | Notes |
|-------|------|-------|
| **Disk** | ∞ (as much as you want) | Foliage tiles in `foliage_root/LOD{n}/...` |
| **RAM Cache** | ~23 MB (160k/LOD) | FoliageResidency with LRU eviction |
| **GPU VRAM** | ~50 MB | Vertex + instance + indirect buffers |

### Thread Usage

| System | Thread | Phase |
|--------|--------|-------|
| Tile loading | Background (async) | 7a |
| Instance generation | Background (blocking UI) | 9a |
| Painting | Main (maybe threaded later) | 9b |
| GPU upload | Main (GPU side) | 8b |
| Rendering | GPU (compute + draw) | 8b+ |

---

## Critical Design Decisions

### ✅ 24 Separate Entities (not per-instance shader variants)
**Decision:** Each of 8 variants × 3 LODs = 24 independent rendering entities.  
**Why:**
- No GPU branching per instance (simpler shader)
- Each entity gets its own mesh (variant geometry)
- Material per entity (future: per-variant textures)
- Scales better than `if (variant_id == X)` for 8 variants

**Alternative rejected:**
- Per-instance variant in shader — would require 8 mesh variants in GPU memory, complex branching

### ✅ Binary Format (not JSON)
**Decision:** Foliage instances stored as compact binary (not JSON).  
**Why:**
- 48 bytes per instance (binary) vs. ~200 bytes (JSON string)
- No parsing overhead at load time
- Deterministic serialization (no floating-point precision issues)
- Matches terrain tile format (L0-Ln hierarchy)

**Alternative rejected:**
- JSON — human-readable but bloated and slow to parse

### ✅ LRU Eviction (not dynamic allocation)
**Decision:** Fixed cache size (160k/LOD), LRU eviction when full.  
**Why:**
- Prevents OOM crashes (bounded memory use)
- Predictable performance (no GC pauses)
- Simple to implement and test

**Alternative rejected:**
- Unlimited growth — can crash if world is too large
- Dynamic resizing — complex memory management, unpredictable

### ✅ Generation Counter (not full clear on hot-reload)
**Decision:** Bump generation counter on reload, discard stale tiles by generation check.  
**Why:**
- Background threads can continue loading old tiles; they're just discarded
- No blocking waits for in-flight I/O
- New tiles stream in immediately without restart

**Alternative rejected:**
- Full cache clear + blocking wait — slower hot-reload

---

## Known Limitations & Future Work

### v1 (Current)
- ❌ **No wind animation** — deferred to Phase 11
- ❌ **No destruction** — deferred to Phase 11
- ❌ **Grass only** — trees/rocks deferred to Phase 11
- ❌ **No collision** — grass is visual only
- ❌ **8 variants max** — could extend to 16 with more memory
- ⚠️ **No undo/redo** — painter is destructive to disk files

### Future Enhancements
- **Wind simulation** — integrate with global wind system (Phase 11)
- **Per-instance physics** — trampling, crushed grass recovery
- **Seasonal systems** — color cycling, snowfall
- **Tree LOD** — full 3D models at distance, billboards far away
- **Foliage sprites** — 2D cards for extreme LOD2 distances
- **Deferred decals** — footprints, blood, scorch marks

---

## Success Criteria

### Phase 8b Complete ✓
- [ ] GPU buffers allocated and uploaded
- [ ] Instance data visible in RenderDoc/GPU debugger
- [ ] No GPU validation errors
- [ ] Frame time <5ms for 100k instances

### Phase 9 Complete ✓
- [ ] "Generate" button works (creates foliage tiles)
- [ ] Painter tool functional (paint, save, regenerate)
- [ ] Parameters tunable and live-preview works
- [ ] 50k instances rendering with proper LOD switching

### Phase 10 Complete ✓
- [ ] Frame rate stable 60+ FPS (500k LOD0 in view)
- [ ] GPU memory <100 MB
- [ ] Visual quality approved (variants distinct, smooth LOD)
- [ ] No memory leaks (30 min test)

### Phase 11 Complete ✓
- [ ] Wind looks natural
- [ ] Destruction responds to player movement
- [ ] Trees render correctly with collision
- [ ] Water interaction blends smoothly

---

## Testing Strategy

### Unit Tests (Phases 8b-11)
- Buffer offset calculations (8b)
- Raycasting and cell lookup (9b)
- Generation parameters (9a)
- Memory eviction (already in Phase 7)
- Shader compilation (already in Phase 5)

### Integration Tests
- Render frame with 100k instances (8b)
- Paint tile, regenerate, verify new tiles on disk (9)
- Hot-reload with new parameters (9a)
- Visual regression (screenshot comparison, Phase 10)

### Performance Benchmarks
- Frame time vs. instance count (10a)
- GPU memory vs. LOD count (10a)
- Paint performance vs. brush radius (9b)

### Stress Tests
- Generate 100×100 tile grid (25M instances LOD0, 10)
- Reload 100 times consecutively, memory stable (10)
- Paint entire map at max strength, no stutter (9b)

---

## Build & Test Commands

```bash
# Build all crates
cargo build --quiet

# Run all tests
cargo test --workspace --quiet

# Run with verbose output
cargo test --workspace -- --nocapture

# Run specific test
cargo test foliage_instance_gen -- --nocapture

# Build with optimizations
cargo build --release

# Run editor
cargo run

# Run editor with specific level
cargo run -- --level my_level.json
```

---

## File Structure

```
landscape/
├── src/
│   ├── main.rs                  # Startup, resource creation, app setup
│   ├── app_config.rs            # landscape.toml loading
│   └── player.rs                # Camera controller
├── crates/
│   ├── bevy_landscape/
│   │   └── src/
│   │       ├── foliage.rs                    # Phase 1: Data model
│   │       ├── procedural_mask_gen.rs        # Phase 2: Density masks
│   │       ├── foliage_instance_gen.rs       # Phase 3: Instance generation
│   │       ├── grass_mesh.rs                 # Phase 4: Grass variants
│   │       ├── grass_material.rs             # Phase 5: Material shader
│   │       ├── foliage_reload.rs             # Phase 6: Hot-reload
│   │       ├── foliage_stream_queue.rs       # Phase 7a: Tile streaming
│   │       ├── foliage_gpu.rs                # Phase 7b: GPU buffers
│   │       ├── foliage_entities.rs           # Phase 7c: Entity system
│   │       ├── foliage_render.rs             # Phase 7d: Render infrastructure
│   │       └── lib.rs
│   └── bevy_landscape_editor/
│       └── src/
│           ├── foliage_panel.rs              # Phase 8a: Editor UI
│           └── lib.rs
├── assets/
│   ├── tiles/                   # Terrain heightmap tiles
│   └── foliage/                 # Foliage tiles (generated at runtime)
│       ├── painted/             # Painted splatmaps
│       ├── procedural_masks/    # Procedural density masks
│       ├── LOD0/                # Full density instances
│       ├── LOD1/                # 50% density instances
│       └── LOD2/                # 10% density instances
├── landscape.toml               # Configuration
├── level.json                   # Current level (optional)
├── ROADMAP.md                   # This file
└── Cargo.toml
```

---

## Timeline Estimate

| Phase | Effort | Status |
|-------|--------|--------|
| 1-7 | Complete | ✅ Done |
| 8a | ✅ 1 session | ✅ Done (2026-04-27) |
| **8b** | **2-3 sessions** | 🔄 Next |
| **9** | **2-3 sessions** | ⏳ Queued |
| **10** | **1-2 sessions** | ⏳ Queued |
| **11** | **2-3 sessions** | 📅 Deferred (optional) |
| **Total** | ~10-12 sessions | 50% complete |

**Current session count:** ~6 sessions invested (Phases 1-8a)  
**Remaining to full feature:** ~8-10 sessions  
**Timeline to v1 (Phases 1-10):** End of this sprint  
**Timeline to v2 (all phases):** 2-3 sprints total

---

## How to Contribute

### For Next Developer
1. **Start with Phase 8b** — GPU rendering infrastructure (highest priority)
2. Read Phase 7 code (foliage_render.rs) to understand LOD selection
3. Implement `extract_foliage_staging`, `prepare_foliage_gpu_buffers`, `queue_foliage_indirect_draws`
4. Test with simple hardcoded instance data before connecting to real streaming pipeline
5. **Then Phase 9** — generation backend (UI needs this to be useful)

### Testing Workflow
```bash
# Make a change to phase 8b code
vim crates/bevy_landscape/src/foliage_render.rs

# Build and test
cargo build --quiet && cargo test --workspace --quiet

# If tests pass, run the editor
cargo run

# Check foliage panel in Tools → Foliage
# Verify GPU data is being uploaded (use RenderDoc to debug)
```

### Debugging
- **RenderDoc:** Capture GPU frame to see vertex/index/indirect buffers
- **Cargo expand:** `cargo expand --lib foliage_render` to see macro expansions
- **Flame graph:** `cargo flamegraph --bin landscape` for perf profiling

---

## References

- **Bevy 0.18 Docs:** https://docs.rs/bevy/0.18
- **CDLOD Paper:** "Continuous Distance-Dependent Level of Detail for Rendering Heightmaps" (Strugar 2009)
- **GPU Instancing:** https://learnopengl.com/Advanced-OpenGL/Instancing
- **Foliage Systems:** "Efficient Real-Time Rendering of Massive Particle Systems" (GPU Gems 3, Ch 23)
- **Project Brief:** `/home/avataren/src/landscape/claude.md`

---

**Last updated:** 2026-04-27  
**Status:** GPU grass rendering live ✅

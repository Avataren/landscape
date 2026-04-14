# Bevy Terrain Renderer Handoff

## Purpose

This document is the final implementation handoff for a coding agent building a **large-scale realistic heightfield terrain renderer in Bevy**.

The target system is:

- heightfield terrain only
- huge outdoor landscapes
- modern GPU-oriented rendering
- streamed terrain data
- stable level-of-detail transitions
- clean separation between terrain base rendering and landscape assets such as cliffs, rocks, and foliage

This is **not** a voxel terrain plan and **not** a meshlet-first terrain plan.

---

## Final recommendation

Build the terrain system as a **GPU-sampled heightfield renderer** using:

- **nested clipmap-style LOD rings**
- **reusable grid patch meshes**
- **vertex-shader height sampling**
- **streamed height/material tiles**
- **LOD morphing for transition stability**
- later: **GPU culling + indirect draws**

Use Bevy’s normal rendering path for most landscape assets on top of the ground surface.

### Why this is the right base architecture

The terrain base is a **regular heightfield problem**, not an irregular triangle-mesh problem.

That means the renderer should exploit:

- regular patch topology
- predictable streaming
- camera-centered LOD structure
- compact terrain data formats
- cheap height queries for gameplay and collision

A clipmap/CDLOD-style terrain renderer matches those requirements better than chunked CPU-generated meshes or a meshlet-first geometry pipeline.

---

## Why not use meshlets for the terrain base?

Meshlets are not bad. They are just the wrong first tool for the base landscape surface.

### Meshlets are best for:

- dense static meshes
- irregular high-poly geometry
- cliffs
- hero rocks
- ruins
- photogrammetry assets
- non-heightfield landscape supplements

### Terrain base is different

The base terrain surface is naturally:

- regular
- continuous
- heightfield-driven
- easy to sample procedurally in the vertex shader
- easier to stream as textures than as geometry clusters

Using meshlets for the terrain base usually means converting a problem that is naturally represented as sampled height data into a much heavier preprocessed geometry problem.

That is backwards.

### Final split

Use:

- **custom clipmap/CDLOD-style terrain renderer** for the base ground
- **meshlets later** for cliffs, rocks, ruins, and other dense static landscape assets

---

## Scope

### In scope for v1

- terrain plugin suite
- large continuous heightfield terrain
- nested LOD rings
- reusable grid patch mesh
- streamed height tiles
- optional streamed material/mask tiles
- shader-based height sampling
- crack reduction via morphing
- low-resolution CPU collision cache
- agent-friendly modular code layout

### Out of scope for v1

- caves
- tunnels
- overhangs as part of base terrain
- voxel terrain
- terrain editing
- erosion simulation at runtime
- full virtual texturing
- meshlet-based base terrain
- full occlusion culling for terrain
- procedural foliage system
- road authoring system

---

## Project structure

```text
src/
  main.rs

  terrain/
    mod.rs
    config.rs
    components.rs
    resources.rs
    world_desc.rs
    math.rs
    patch_mesh.rs
    clipmap.rs
    streamer.rs
    residency.rs
    collision.rs
    debug.rs

    render/
      mod.rs
      extract.rs
      prepare.rs
      queue.rs
      graph.rs
      pipelines.rs
      bind_groups.rs
      gpu_types.rs

    shaders/
      terrain_common.wgsl
      terrain_vertex.wgsl
      terrain_fragment.wgsl
      terrain_shadow.wgsl
      terrain_cull.wgsl
      terrain_indirect.wgsl
```

---

## Architecture overview

### CPU side responsibilities

- track terrain camera position
- snap clipmap centers to level grids
- compute visible/required LOD patches
- compute required terrain tiles
- manage streaming requests
- decode and stage tile payloads
- maintain a low-resolution collision cache
- expose terrain query functions to gameplay systems

### GPU side responsibilities

- hold clipmap textures or per-level textures
- read patch descriptors
- sample heights in vertex shader
- reconstruct normals or sample normal data
- shade terrain by slope/altitude/material masks
- later: cull visible patches and generate indirect draws

---

## Core design rules

These are hard rules for the coding agent.

1. **Do not generate unique terrain chunk meshes every frame.**
2. **Do not rebuild terrain topology during camera movement.**
3. **All clipmap/ring origins must be snapped to their level grid.**
4. **LOD boundaries must align exactly in world space.**
5. **Streaming must be separate from rendering.**
6. **Collision data must be separate from render data.**
7. **Use one shared patch mesh, not one mesh per patch.**
8. **Prefer one draw per level with instancing in v1.**
9. **Move to GPU culling and indirect draws in v2.**
10. **Do not make meshlets the foundation of the terrain renderer.**

---

## Phased implementation plan

## Phase 0 — plugin skeleton and debugability

### Goal

Establish a stable plugin architecture and debugging foundation before building terrain logic.

### Deliverables

- `TerrainPlugin`
- `TerrainRenderPlugin`
- `TerrainDebugPlugin`
- terrain config resources
- terrain source description
- terrain camera marker
- debug drawing hooks
- stats overlay

### Required debug views

- patch bounds
- ring bounds
- resident tile bounds
- camera-centered clip origins
- LOD color mode
- resident tile count
- pending load count
- patch count by level

---

## Phase 1 — flat patch renderer

### Goal

Render a reusable instanced patch mesh before adding terrain sampling.

### Deliverables

- grid patch mesh generator
- patch descriptor structure
- instanced draw path
- one draw per level or a minimal CPU fallback path
- shader can place patches in world space

### Acceptance criteria

- multiple patches render correctly
- patch transforms are stable
- patch ordering is stable between frames

---

## Phase 2 — single heightmap terrain

### Goal

Displace patches from a single height texture in the vertex shader.

### Deliverables

- one height texture bound to terrain shader
- world-space height sampling
- normal reconstruction in shader
- basic terrain shading

### Acceptance criteria

- camera can fly over displaced terrain
- geometry follows height data correctly
- normals are stable enough for simple lighting

---

## Phase 3 — multi-ring clipmap layout

### Goal

Move from one terrain patch set to nested clipmap-style LOD rings.

### Deliverables

- per-level world scale computation
- camera snapping per level
- ring patch generation
- outer rings with inner hole
- per-patch morph band parameters

### Acceptance criteria

- 4 to 6 levels render correctly
- patch layout shifts only when camera crosses snapped boundaries
- outer levels exclude inner region cleanly

---

## Phase 4 — LOD transition stability

### Goal

Reduce cracks and popping across LOD levels.

### Deliverables

- morph factor computation in shader
- snap-to-coarser-grid morph target
- aligned patch boundaries
- optional trim/fixup patches if needed

### Acceptance criteria

- no visible cracks in normal use
- reduced popping during traversal
- transitions remain stable on steep terrain

---

## Phase 5 — tile streaming and residency

### Goal

Replace the single heightmap with streamed terrain tiles.

### Deliverables

- `TileKey`
- tile state tracking
- required tile set computation
- background loading
- CPU tile decode
- GPU texture upload path
- LRU-style eviction
- coarse fallback when detailed tile not resident

### Acceptance criteria

- moving camera triggers loads predictably
- memory use remains bounded
- no frame stalls from sync file IO
- terrain still renders while waiting for fine tiles

---

## Phase 6 — terrain materials and macro shading

### Goal

Make the terrain visually credible.

### Deliverables

- slope-based blending
- altitude-based blending
- macro variation
- base material layer set
- optional streamed material masks
- fog integration

### Material layers

At minimum:

- grass
- dirt
- rock
- scree
- snow

### Acceptance criteria

- mid/far field no longer looks obviously tiled
- cliffs and slopes read correctly
- terrain remains readable from both ground and aerial views

---

## Phase 7 — collision and terrain queries

### Goal

Support gameplay systems without mirroring full render detail on CPU.

### Deliverables

- low-resolution height tile cache
- height queries
- normal queries
- terrain raycasts
- stable near-player collision representation

### Acceptance criteria

- terrain sampling APIs are deterministic
- collision updates independently from render residency
- player movement/queries remain stable near the camera

---

## Phase 8 — GPU-driven culling and indirect draws

### Goal

Reduce CPU submission cost and prepare for large-scale rendering.

### Deliverables

- patch descriptor storage buffer
- compute frustum culling
- visible patch list
- indirect args buffer
- indirect indexed draw path

### Acceptance criteria

- visible patch count is produced on GPU
- render submission scales better than CPU patch iteration
- terrain draw path no longer depends on many ECS entities

---

## Phase 9 — landscape assets integration

### Goal

Layer other world geometry on top of the terrain base.

### Deliverables

- terrain query support for asset placement
- cliffs as separate meshes
- rock placement hooks
- foliage/prop systems can query height/slope/material

### Important note

This is where meshlets may become useful.

Use meshlets for:

- cliffs
- hero rocks
- dense ruins
- photogrammetry-like environment assets

Do not replace the terrain base with a meshlet path.

---

## File-by-file breakdown

## `terrain/mod.rs`

Registers plugins and app systems.

### Responsibilities

- register resources
- register update systems
- register render plugin
- register debug plugin

### Expected systems

- `setup_terrain`
- `update_terrain_view_state`
- `update_required_tiles`
- `request_tile_loads`
- `poll_tile_stream_jobs`
- `apply_loaded_tiles_to_cpu_cache`
- `update_collision_tiles`
- `update_patch_instances_cpu_fallback` (temporary if needed)

---

## `terrain/config.rs`

Defines terrain configuration resource.

```rust
#[derive(Resource, Clone)]
pub struct TerrainConfig {
    pub clipmap_levels: u32,
    pub patch_resolution: u32,
    pub ring_patches: u32,
    pub tile_size: u32,
    pub clipmap_resolution: u32,
    pub world_scale: f32,
    pub height_scale: f32,
    pub morph_start_ratio: f32,
    pub max_resident_tiles: usize,
    pub max_view_distance: f32,
}
```

Recommended defaults:

- `clipmap_levels = 6`
- `patch_resolution = 64`
- `ring_patches = 8`
- `tile_size = 256`
- `clipmap_resolution = 2048`

---

## `terrain/world_desc.rs`

Describes terrain sources and world bounds.

```rust
#[derive(Resource, Clone, Default)]
pub struct TerrainSourceDesc {
    pub height_root: String,
    pub normal_root: Option<String>,
    pub material_root: Option<String>,
    pub macro_color_root: Option<String>,
    pub world_min: Vec2,
    pub world_max: Vec2,
    pub max_mip_level: u8,
}
```

---

## `terrain/components.rs`

Defines ECS markers and optional patch components.

```rust
#[derive(Component)]
pub struct TerrainCamera;

#[derive(Component, Clone, Copy)]
pub struct TerrainPatchInstance {
    pub lod_level: u32,
    pub patch_kind: u32,
    pub patch_origin_ws: Vec2,
    pub patch_scale_ws: f32,
}
```

---

## `terrain/resources.rs`

Holds runtime CPU-side terrain state.

```rust
#[derive(Resource, Default)]
pub struct TerrainViewState {
    pub camera_pos_ws: Vec3,
    pub clip_centers: Vec<IVec2>,
    pub level_scales: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Debug)]
pub enum TileState {
    Unloaded,
    Requested,
    LoadedCpu,
    ResidentGpu { slot: u32 },
    Evicting,
}
```

Also define:

- `TerrainResidency`
- `TerrainStreamQueue`
- `HeightTileCpu`
- `MaterialTileCpu`

---

## `terrain/math.rs`

This should contain pure functions and unit tests.

### Required functions

- `level_scale(base_sample_spacing, level) -> f32`
- `snap_camera_to_level_grid(camera_xz, level_scale) -> IVec2`
- `build_ring_patch_origins(...) -> Vec<Vec2>`
- `compute_needed_tiles_for_level(...) -> Vec<TileKey>`
- `morph_factor(distance, band_start, band_end) -> f32`

### Notes

This file is critical. Bugs here will look like “rendering bugs” later.

---

## `terrain/patch_mesh.rs`

Creates the reusable grid patch mesh.

### Requirements

- one square grid patch
- positions from `[0, 1]` in local xz
- regular UVs
- regular indexed triangles

Do not overcomplicate this file.

---

## `terrain/clipmap.rs`

Builds patch instances for the current terrain view.

### Main output type

```rust
pub struct PatchInstanceCpu {
    pub lod_level: u32,
    pub patch_kind: u32,
    pub origin_ws: Vec2,
    pub patch_size_ws: f32,
    pub morph_start: f32,
    pub morph_end: f32,
}
```

### Responsibilities

- compute per-level centers
- compute per-level scales
- generate ring patch origins
- assign morph ranges
- keep ordering stable

---

## `terrain/streamer.rs`

Runs background tile loading and decoding.

### Responsibilities

- request missing tiles
- spawn jobs
- receive finished tile payloads
- update CPU-side tile state

### Rules

- no blocking disk IO in update/render systems
- decode off-thread
- return compact CPU payloads

---

## `terrain/residency.rs`

Tracks what tiles are required and what can be evicted.

### Responsibilities

- compute `required_now`
- update tile states
- maintain LRU
- evict tiles beyond memory budget
- avoid evicting needed tiles

---

## `terrain/collision.rs`

Maintains a low-resolution CPU collision cache.

### Required API

```rust
fn sample_height(world_xz: Vec2) -> Option<f32>;
fn sample_normal(world_xz: Vec2) -> Option<Vec3>;
fn raycast_terrain(ray: Ray3d, max_dist: f32) -> Option<TerrainHit>;
```

### Important note

Do not read terrain data back from GPU. Maintain a separate CPU collision representation.

---

## `terrain/debug.rs`

Adds debug drawing and stats.

### Required views

- patch bounds
- ring bounds
- resident tiles
- collision tile bounds
- current clip centers
- LOD debug coloring

---

## Render module breakdown

## `terrain/render/mod.rs`

Registers render-app resources and systems.

### Responsibilities

- initialize GPU state
- register extract system
- register prepare system
- register queue system
- optionally register render graph node

---

## `terrain/render/gpu_types.rs`

Contains `#[repr(C)]` shared CPU/GPU data.

### Required structs

- `TerrainFrameUniform`
- `PatchDescriptorGpu`
- later: indirect draw argument struct

Example:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TerrainFrameUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos_ws: [f32; 4],
    pub clip_centers: [[i32; 4]; 8],
    pub level_scales: [f32; 8],
    pub height_scale: f32,
    pub morph_start_ratio: f32,
    pub clipmap_levels: u32,
    pub _pad0: u32,
}
```

---

## `terrain/render/extract.rs`

Extracts minimal frame data into render world.

### Extracted resource

- camera position
- clip centers
- level scales
- patch list for v1 CPU-submitted path

### Rule

Extract only what the render world actually needs.

---

## `terrain/render/prepare.rs`

Uploads and prepares GPU resources.

### Responsibilities

- update uniform buffer
- upload patch descriptor buffer
- upload changed terrain tiles into textures
- create/update bind groups
- track upload counts

### Important rule

Do not upload full terrain textures every frame. Only upload changed regions.

---

## `terrain/render/pipelines.rs`

Creates and specializes terrain render pipeline.

### Binding groups should include

- frame uniform
- patch descriptor storage buffer
- height textures
- material/mask textures
- samplers

### Draw path recommendation for v1

Use **one draw per level with instancing** rather than one draw per patch.

That is the best compromise before moving to indirect draws.

---

## `terrain/render/queue.rs`

Queues terrain draw work.

### v1

- queue one instanced draw per level

### v2

- queue one or a few indirect draws driven by compute output

---

## `terrain/render/graph.rs`

Optional custom render graph integration.

### Recommendation

Keep terrain in an opaque-style pass first.

Only add a custom node if the pass integration clearly demands it.

---

## WGSL breakdown

## `terrain_common.wgsl`

Put helpers here.

### Required content

- frame uniform definition
- patch descriptor definition
- height sampling helper
- morph helper
- coarser-grid snap helper
- normal reconstruction helper
- material blending helpers

---

## `terrain_vertex.wgsl`

### Main responsibilities

- read patch descriptor by `instance_index`
- map local patch coordinates to world xz
- compute morph factor
- blend toward coarser snap target
- sample height
- reconstruct normal
- output world position and clip position

### Expected logic outline

1. load patch descriptor
2. compute local xz inside patch
3. compute fine world xz
4. compute camera distance
5. compute morph factor
6. compute coarser-grid xz target
7. mix fine/coarse world xz
8. sample height at final xz
9. reconstruct or sample normal
10. output clip-space position and varying data

---

## `terrain_fragment.wgsl`

### Main responsibilities

- blend terrain layers by slope and altitude
- apply macro variation
- compute simple lighting
- integrate fog/atmosphere hook

### Layer suggestions

- grass for low slope
- rock for high slope
- snow for high altitude
- dirt/scree for transitions

---

## `terrain_shadow.wgsl`

### Main responsibilities

- same terrain displacement logic as vertex path
- no full material logic
- output correct shadow caster position

---

## `terrain_cull.wgsl` (v2)

### Main responsibilities

- conservative frustum test per patch
- write visible patch IDs
- increment visible count
- later: support Hi-Z occlusion test

Start with frustum culling only.

---

## `terrain_indirect.wgsl` (v2)

### Main responsibilities

- write indirect draw args from visible patch count
- optionally compact visible patch descriptors

---

## Execution order for main systems

1. update terrain camera state
2. compute clip centers/scales
3. compute required tiles
4. request missing tile loads
5. poll finished jobs
6. update CPU collision tiles
7. extract frame data into render world
8. upload buffers/textures
9. queue terrain draws
10. render terrain

---

## Execution order for build milestones

### Milestone 1

- math/config/resources compile
- patch mesh renders flat
- shader can place patch instances

### Milestone 2

- one height texture displaces terrain
- normals reconstruct
- camera can fly over terrain

### Milestone 3

- 4–6 clipmap levels render
- ring layout stable
- no obvious overlap bugs

### Milestone 4

- LOD morphing added
- transitions improved
- no major cracks

### Milestone 5

- streamed tiles replace single baked texture
- coarse fallback works
- loading is asynchronous

### Milestone 6

- material layering added
- macro variation added
- fog/atmosphere hook added

### Milestone 7

- collision cache works
- terrain query API exposed

### Milestone 8

- GPU culling added
- indirect draws added
- CPU patch submission reduced

### Milestone 9

- landscape asset placement hooks added
- cliffs/rocks integrate on top of terrain

---

## Minimal pseudocode by file

## `update_terrain_view_state`

```rust
fn update_terrain_view_state(
    config: Res<TerrainConfig>,
    camera_q: Query<&GlobalTransform, With<TerrainCamera>>,
    mut view: ResMut<TerrainViewState>,
) {
    let cam = camera_q.single();
    let cam_pos = cam.translation();
    view.camera_pos_ws = cam_pos;

    view.clip_centers.clear();
    view.level_scales.clear();

    let base_spacing = config.world_scale;

    for level in 0..config.clipmap_levels {
        let scale = level_scale(base_spacing, level);
        let center = snap_camera_to_level_grid(cam_pos.xz(), scale);

        view.level_scales.push(scale);
        view.clip_centers.push(center);
    }
}
```

---

## `build_patch_instances_for_view`

```rust
pub fn build_patch_instances_for_view(
    config: &TerrainConfig,
    view: &TerrainViewState,
) -> Vec<PatchInstanceCpu> {
    let mut out = Vec::new();

    for level in 0..config.clipmap_levels {
        let level_scale = view.level_scales[level as usize];
        let center = view.clip_centers[level as usize];

        let origins = build_ring_patch_origins(
            center,
            level_scale,
            config.patch_resolution,
            config.ring_patches,
            level > 0,
        );

        let patch_world_size = config.patch_resolution as f32 * level_scale;
        let band_near = patch_world_size * config.ring_patches as f32 * config.morph_start_ratio;
        let band_far = patch_world_size * config.ring_patches as f32;

        for origin in origins {
            out.push(PatchInstanceCpu {
                lod_level: level,
                patch_kind: 0,
                origin_ws: origin,
                patch_size_ws: patch_world_size,
                morph_start: band_near,
                morph_end: band_far,
            });
        }
    }

    out
}
```

---

## `request_tile_loads`

```rust
fn request_tile_loads(
    mut queue: ResMut<TerrainStreamQueue>,
    mut residency: ResMut<TerrainResidency>,
) {
    for key in residency.required_now.iter().copied() {
        if !queue.pending_requests.contains(&key) {
            if matches!(residency.tiles.get(&key), None | Some(TileState::Unloaded)) {
                queue.pending_requests.insert(key);
                residency.tiles.insert(key, TileState::Requested);
                spawn_background_height_job(key);
            }
        }
    }
}
```

---

## `extract_terrain_frame`

```rust
#[derive(Resource, Clone, Default)]
pub struct ExtractedTerrainFrame {
    pub camera_pos_ws: Vec3,
    pub clip_centers: Vec<IVec2>,
    pub level_scales: Vec<f32>,
    pub patches: Vec<PatchInstanceCpu>,
}

pub fn extract_terrain_frame(
    mut commands: Commands,
    view: Extract<Res<TerrainViewState>>,
    config: Extract<Res<TerrainConfig>>,
) {
    let patches = build_patch_instances_for_view(&config, &view);

    commands.insert_resource(ExtractedTerrainFrame {
        camera_pos_ws: view.camera_pos_ws,
        clip_centers: view.clip_centers.clone(),
        level_scales: view.level_scales.clone(),
        patches,
    });
}
```

---

## `prepare_terrain_gpu`

```rust
pub fn prepare_terrain_gpu(
    mut gpu: ResMut<TerrainGpuState>,
    extracted: Res<ExtractedTerrainFrame>,
) {
    // 1. Upload TerrainFrameUniform
    // 2. Upload patch descriptor storage buffer
    // 3. Upload changed terrain tiles into clipmap textures
    // 4. Create or refresh bind groups as needed
}
```

---

## `terrain_vertex.wgsl` outline

```wgsl
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  let patch = patch_descs[in.instance_idx];

  let local_xz = in.local_pos.xz * patch.patch_size_ws;
  let world_xz_fine = patch.origin_ws + local_xz;

  let lod = patch.lod_level;
  let fine_spacing = frame.level_scales[lod];

  let camera_dist = distance(world_xz_fine, frame.camera_pos_ws.xz);
  let morph_t = compute_morph_factor(camera_dist, patch.morph_start, patch.morph_end);

  let world_xz_coarse = snap_to_coarser_grid(world_xz_fine, fine_spacing);
  let world_xz = mix(world_xz_fine, world_xz_coarse, morph_t);

  let h = sample_height_ws(world_xz, lod);
  let normal_ws = reconstruct_normal(world_xz, lod, fine_spacing);

  var out: VertexOutput;
  let world_pos = vec3<f32>(world_xz.x, h, world_xz.y);
  out.world_pos = world_pos;
  out.world_normal = normal_ws;
  out.clip_pos = frame.view_proj * vec4<f32>(world_pos, 1.0);
  return out;
}
```

---

## `terrain_cull.wgsl` outline

```wgsl
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&patch_descs)) {
    return;
  }

  let patch = patch_descs[idx];

  // conservative patch bounds
  let visible = frustum_test_patch(patch);

  if (visible) {
    let out_idx = atomicAdd(&visible_count, 1u);
    visible_patch_ids[out_idx] = idx;
  }
}
```

---

## Testing plan

## Unit tests

### `math.rs`

- level scale doubles correctly
- snapped centers update only on grid crossing
- ring layout counts are stable
- inner hole removal works

### `residency.rs`

- required tiles computed correctly
- non-required tiles evict first
- required tiles are preserved

### `clipmap.rs`

- patch ordering remains stable
- level patch sizes are correct
- per-level centers match expected snapped cells

---

## Visual tests

- slow movement across LOD boundaries
- fast movement across terrain
- steep slope flyovers
- high-altitude world overview
- low-angle sunlight test
- LOD color visualization
- wireframe debug mode

---

## Performance test goals

- traversal across large world coordinates
- stable frame time during camera movement
- bounded memory under tile streaming
- low CPU overhead in v1 instanced path
- better submission scaling in v2 indirect path

---

## What the agent must not do

Do not let the implementation drift into any of these:

- one mesh per terrain chunk
- dynamic topology rebuilds during movement
- heavy CPU meshing as the main terrain path
- reading collision from GPU output
- mixing foliage generation into terrain core milestone
- trying to solve cliffs by abandoning heightfields too early
- replacing terrain with meshlets before the base renderer is proven

---

## Final handoff summary

### Build this first

- flat instanced patch renderer
- single-heightmap terrain
- multi-ring clipmap layout
- morphing and crack control
- tile residency/streaming
- terrain materials
- collision/query layer
- compute culling + indirect draws

### Use meshlets later for

- cliffs
- rocks
- ruins
- dense static landscape supplements

### Do not use meshlets first for

- the base terrain surface
- streamed heightfield ground
- regular terrain LOD rings

---

## Short agent brief

```text
Implement a Bevy terrain renderer for huge heightfield landscapes.

Core approach:
- clipmap/CDLOD-style nested LOD rings
- reusable grid patch mesh
- vertex-shader height sampling
- streamed height/material tiles
- LOD morphing
- later: compute culling + indirect draws

Do not:
- generate one mesh per terrain chunk
- rebuild terrain topology every frame
- use meshlets for the base terrain
- couple collision and render data tightly

Use meshlets later only for cliffs/rocks/ruins and other dense irregular landscape assets.
```

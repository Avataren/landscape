# Hydraulic & Thermal Erosion — Implementation Plan

> **Scope**: Replace `erosion_shaped_fbm` (fake gradient-attenuation erosion) and the
> `channel_mask` hack in `terrain_height_for` with proper GPU-simulated hydraulic and
> thermal erosion running as compute passes in the existing generator pipeline.
>
> **Context**: The broader procedural generation vision is in `procgen_heightmap.md`.
> This document focuses exclusively on what is needed to implement realistic erosion,
> with concrete binding indices, buffer layouts, and WGSL pseudocode tied to the
> current codebase state.

---

## 1. Current State and What to Remove

`generator.wgsl` currently fakes erosion in two places inside `terrain_height_for`:

```wgsl
// Fake 1: gradient-accumulation fBm attenuation
let detail = erosion_shaped_fbm(detail_pos, octaves, lacunarity, gain, erosion);

// Fake 2: ridged-fbm channel carving
height -= erosion * channel_mask_pow4 * highlands * (0.03 + 0.11 * continent);
```

Both should be **removed**. The `erosion_shaped_fbm` function and the channel-carving
block are deleted from `terrain_height_for`. The `erosion_strength` field in
`GeneratorParams` / `GeneratorUniform` is removed (or repurposed; see §10).

After removal, `terrain_height_for` only produces a clean noise-based heightfield that
the erosion passes will then sculpt.

---

## 2. Pipeline Overview

The generator produces results through this sequence of compute passes:

```
[preview_generate_raw]          existing — writes noise height → raw_heights (R32Float)
        │
        ▼
[erosion_init]                  NEW — copy raw_heights → height_a; clear sim buffers
        │
        ▼
  ╔═══════════════════╗
  ║  N erosion ticks  ║
  ║                   ║
  ║ [hydraulic_*]     ║  ~6 passes/tick (see §4)
  ║ [thermal_*]       ║  ~2 passes/tick (see §5)
  ╚═══════════════════╝
        │
        ▼
[erosion_finalize]              NEW — copy height_a → raw_heights
        │
        ▼
[preview_reduce_minmax]         existing — find min/max for normalisation
        │
        ▼
[preview_normalize_display]     existing — normalise + palette → preview_output
```

For **export**, the same erosion passes run on the export-resolution height texture
before tiles are baked — the export pipeline gains an `erosion_iterations` parameter.

---

## 3. GPU Simulation Buffers

All erosion textures share the same resolution as the raw-height texture (preview or
export resolution).  All are `R32Float` or `RG32Float` stored as `StorageTexture2d`.
Each is created alongside `NormalizationImage` on startup and on resolution change.

| Resource name        | Format       | Content                                              |
|----------------------|--------------|------------------------------------------------------|
| `height_a`           | R32Float     | Primary height buffer (ping)                         |
| `height_b`           | R32Float     | Scratch/ping-pong height buffer (pong)               |
| `water`              | R32Float     | Water column depth `d`                               |
| `sediment`           | R32Float     | Suspended sediment load `s`                          |
| `flux`               | Rgba32Float  | Outflow flux (Left, Right, Top, Bottom)              |
| `velocity`           | Rg32Float    | 2-D velocity vector derived from flux                |
| `hardness`           | R32Float     | Per-cell erosion resistance (seeded from noise)      |
| `delta_height`       | R32Sint      | Fixed-point height delta for particle atomics        |

All are `STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC | COPY_DST`.

A new Rust resource `ErosionBuffers` holds all handles.  `ErosionParams` (a new
`Resource`) carries the simulation parameters (see §8).

---

## 4. Hydraulic Erosion — Virtual Pipe Model (Grid-Based)

### Why grid-based?

The virtual pipe model (D'Alesio et al. 2007) is:
- Naturally data-parallel: every cell is independent within a pass
- Numerically stable for large iteration counts
- Accurate at producing wide valleys, alluvial fans, and drainage networks
- Directly composable with the ping-pong textures we already manage

Particle-based erosion (§6) complements this by adding fine river channels and
high-frequency gullies that the grid model smooths over.

### 4a. State variables per cell

```
h  = terrain height (height_a)
d  = water depth    (water)
s  = sediment       (sediment)
f  = [fL, fR, fT, fB] outflow flux (flux)
v  = [vx, vy]       velocity (velocity)
```

### 4b. Pass sequence per erosion tick

One "tick" consists of the following passes (all separate compute dispatches so that
wgpu pass boundaries act as memory barriers between dependent reads/writes):

#### Pass 1: `hydro_water_add`
```wgsl
// Rain: uniformly increment water depth
d[x,y] += params.rain_rate * dt;
```
Entry point: `hydro_water_add`
Reads: `water`    Writes: `water`

#### Pass 2: `hydro_flux_update`
The virtual pipe model: compute outflow flux to each of the 4 orthogonal neighbours.

```wgsl
fn cross_section_area() -> f32 { return params.pipe_length * params.pipe_length; }

// For cell (x,y), outflow to neighbour n at direction d:
// delta_h = (h[x,y] + d[x,y]) - (h[n] + d[n])   // virtual water surface difference
// new_f = max(0, old_f + dt * A * (g * delta_h / l))
//   where A = pipe cross-section, g = gravity, l = pipe length

let hd   = h + d;        // virtual surface at (x,y)
let hd_L = h_left  + d_left;
let hd_R = h_right + d_right;
let hd_T = h_top   + d_top;
let hd_B = h_bottom + d_bottom;

var fL = max(0.0, f.x + dt * A * g * (hd - hd_L) / l);
var fR = max(0.0, f.y + dt * A * g * (hd - hd_R) / l);
var fT = max(0.0, f.z + dt * A * g * (hd - hd_T) / l);
var fB = max(0.0, f.w + dt * A * g * (hd - hd_B) / l);

// Scale down if total outflow would drain more water than available
let total = fL + fR + fT + fB;
let K = min(1.0, d * l * l / (total * dt + 1e-8));
flux[x,y] = vec4(fL, fR, fT, fB) * K;
```
Entry: `hydro_flux_update`
Reads: `height_a`, `water`, `flux`    Writes: `flux`

#### Pass 3: `hydro_water_velocity`
Update water depth and velocity from flux conservation:

```wgsl
// Net volume change from flux in/out
let delta_V = dt * (
    flux_right_of_left_neighbour    // inflow from left
  + flux_left_of_right_neighbour    // inflow from right
  + flux_bottom_of_top_neighbour    // inflow from top
  + flux_top_of_bottom_neighbour    // inflow from bottom
  - f.x - f.y - f.z - f.w           // outflow from this cell
);

d[x,y] = max(0.0, d[x,y] + delta_V / (l * l));

// Velocity from flux differences (semi-Lagrangian source)
let avg_water = (d_prev + d[x,y]) * 0.5;
vx = (flux_right_of_left - f.x + f.y - flux_left_of_right) / (2.0 * l * max(avg_water, 1e-4));
vy = (flux_bottom_of_top  - f.z + f.w - flux_top_of_bottom ) / (2.0 * l * max(avg_water, 1e-4));
velocity[x,y] = vec2(vx, vy);
```
Entry: `hydro_water_velocity`
Reads: `water`, `flux` (self + neighbours)    Writes: `water`, `velocity`

#### Pass 4: `hydro_erode_deposit`
Erosion / deposition driven by transport capacity:

```wgsl
let speed  = length(velocity[x,y]);
let slope  = length(gradient_at(height_a, x, y));   // finite differences
let tilt   = max(slope, params.min_slope);

// Sediment transport capacity
let C = params.sediment_capacity * tilt * speed;

let hardness_factor = 1.0 - hardness[x,y] * params.hardness_influence;

if s < C {
    // Under capacity: erode terrain
    let delta = params.erosion_rate * hardness_factor * (C - s);
    height_a[x,y] -= delta;
    sediment[x,y]  = s + delta;
} else {
    // Over capacity: deposit sediment
    let delta = params.deposition_rate * (s - C);
    height_a[x,y] += delta;
    sediment[x,y]  = s - delta;
}
```
Entry: `hydro_erode_deposit`
Reads: `height_a`, `velocity`, `sediment`, `hardness`    Writes: `height_a`, `sediment`

#### Pass 5: `hydro_sediment_transport`
Semi-Lagrangian advection: trace each cell back along velocity to find where
its sediment came from.

```wgsl
// Back-trace position at previous timestep
let prev = vec2(f32(x), f32(y)) - velocity[x,y] * dt * inv_cell_size;

// Bilinear sample of sediment at back-traced position
sediment[x,y] = bilinear_sample(sediment, prev);
```
Entry: `hydro_sediment_transport`
Reads: `sediment`, `velocity`    Writes: `height_b` (result sediment)
Then swap `sediment` ← `height_b` ping-pong.

#### Pass 6: `hydro_evaporate`
```wgsl
water[x,y] *= 1.0 - params.evaporation_rate * dt;
```
Entry: `hydro_evaporate`
Reads: `water`    Writes: `water`

---

## 5. Thermal Erosion

Simulates material sliding downhill until the slope is below the angle of repose.
Produces talus fans, scree slopes, and softened cliff edges.

### 5a. Algorithm
```wgsl
// Checkerboard: pass A operates on (x+y) % 2 == 0, pass B on == 1
// This avoids read/write races without atomics.

for each orthogonal neighbour n of (x,y):
    let delta = height_a[x,y] - height_a[n];
    let slope = delta / cell_size;   // = tan(actual_angle)
    let threshold = tan(params.repose_angle_radians);

    if slope > threshold:
        let transfer = (slope - threshold) * cell_size * 0.5 * params.talus_rate;
        height_a[x,y] -= transfer;
        // neighbour write would race: accumulate in height_b instead
        height_b[n]   += transfer;
```

Two-pass approach per tick:
1. `thermal_compute`: compute transfers into `height_b` (accumulation buffer, zero-init each tick)
2. `thermal_apply`: `height_a[x,y] -= outflow_from_step1; height_a[x,y] += height_b[x,y]`

Alternatively, use `atomicAdd` on a fixed-point `delta_height` (i32) buffer to collapse
into one dispatch, at the cost of integer quantisation (acceptable for rock-size features).

Entry points: `thermal_compute`, `thermal_apply`
Reads/writes: `height_a`, `height_b`

### 5b. Hardness influence
Multiply transfer by `(1.0 - hardness[x,y])` so hard-rock cells resist sliding.
Hard rock above soft rock naturally creates cliff bands.

---

## 6. Particle-Based Hydraulic Erosion (Fine Detail)

Run after the grid-based simulation to carve sharp river channels and gullies that the
virtual pipe model diffuses.  This matches the "millions of particles" model.

### 6a. Algorithm (one thread = one particle)

```wgsl
@compute @workgroup_size(64, 1, 1)
fn particle_erode(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = id.x ^ (params.frame_seed << 16u);
    var pos  = vec2<f32>(rand(seed) * f32(res.x), rand(seed+1u) * f32(res.y));
    var dir  = vec2<f32>(0.0);
    var speed   = 0.0;
    var water   = 1.0;
    var sediment_carry = 0.0;

    for (var step = 0u; step < params.max_steps; step++) {
        let ipos = vec2<i32>(pos);
        if any(ipos < vec2(0)) || any(ipos >= vec2<i32>(res)) { break; }

        // Bilinear gradient (central differences across the 4 surrounding texels)
        let grad = gradient_bilinear(height_a, pos);

        // Update direction with inertia
        dir = normalize(dir * params.inertia - grad * (1.0 - params.inertia));

        // Advance particle
        let new_pos = pos + dir;
        let new_h   = sample_bilinear(height_a, new_pos);
        let old_h   = sample_bilinear(height_a, pos);
        let height_delta = new_h - old_h;  // positive = moving uphill

        // Sediment capacity proportional to speed × slope
        let slope = max(-height_delta, params.min_slope);
        speed = sqrt(max(0.0, speed * speed + height_delta * params.gravity));
        let capacity = max(0.0, slope * speed * water * params.sediment_capacity);

        if sediment_carry > capacity || height_delta > 0.0 {
            // Deposit
            let deposit = select(
                params.deposition_rate * (sediment_carry - capacity),
                min(sediment_carry, -height_delta),   // filling uphill step
                height_delta > 0.0,
            );
            sediment_carry -= deposit;
            // Bilinear splat deposit to the 4 surrounding integer cells
            splat_delta_fixed(delta_height, pos, deposit);
        } else {
            // Erode
            let erode = min(
                params.erosion_rate * (capacity - sediment_carry),
                -height_delta * 0.1,   // don't erode more than the local dip
            );
            sediment_carry += erode;
            splat_delta_fixed(delta_height, pos, -erode);
        }

        water   *= 1.0 - params.evaporation_rate;
        pos      = new_pos;
        if water < 0.01 { break; }
    }
    // Final deposit of remaining sediment
    splat_delta_fixed(delta_height, pos, sediment_carry);
}

// Bilinear splat using atomicAdd on fixed-point i32 buffer
// delta_fixed = round(delta * SCALE) where SCALE = 1 << 16
fn splat_delta_fixed(buf: ptr<..>, pos: vec2<f32>, delta: f32) {
    let x0 = i32(floor(pos.x));   let x1 = x0 + 1;
    let y0 = i32(floor(pos.y));   let y1 = y0 + 1;
    let fx = fract(pos.x);         let fy = fract(pos.y);
    let d_fixed = i32(round(delta * 65536.0));
    atomicAdd(&buf[y0*stride + x0], i32(round(f32(d_fixed) * (1.0-fx) * (1.0-fy))));
    atomicAdd(&buf[y0*stride + x1], i32(round(f32(d_fixed) * fx       * (1.0-fy))));
    atomicAdd(&buf[y1*stride + x0], i32(round(f32(d_fixed) * (1.0-fx) * fy      )));
    atomicAdd(&buf[y1*stride + x1], i32(round(f32(d_fixed) * fx       * fy      )));
}
```

Entry: `particle_erode`
Reads: `height_a`    Writes: `delta_height` (i32 storage buffer, via atomicAdd)

### 6b. Apply pass
```wgsl
@compute @workgroup_size(8, 8, 1)
fn particle_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * res.x + id.x;
    let delta = f32(atomicExchange(&delta_height[idx], 0)) / 65536.0;
    height_a[id.xy] = saturate(height_a[id.xy] + delta);
}
```
Entry: `particle_apply`
Reads/writes: `height_a`, `delta_height`

### 6c. Dispatch sizing
- Workgroup size: `(64, 1, 1)` — each thread is one particle
- Dispatch count: `num_particles / 64` workgroups in X
- Typical: 200 000–2 000 000 particles per invocation
- The `delta_height` buffer is an `R32Sint` texture (or storage buffer of `array<atomic<i32>>`)

---

## 7. Wind Erosion (Aeolian, Stretch Goal)

Wind saltation: particles hop along wind direction, eroding exposed ridges and depositing
in lee shadows.

```wgsl
@compute @workgroup_size(64, 1, 1)
fn wind_erode(@builtin(global_invocation_id) id: vec3<u32>) {
    var pos = rand_pos(id.x ^ params.frame_seed);
    let wind_step = params.wind_dir * params.saltation_length;

    for (var i = 0u; i < params.saltation_hops; i++) {
        let surface_h = sample_bilinear(height_a, pos);
        let next      = pos + wind_step;
        let next_h    = sample_bilinear(height_a, next);

        if next_h < surface_h {
            // Exposed to wind — erode current cell, carry material
            splat_delta_fixed(delta_height, pos,  -params.wind_erosion_rate);
            splat_delta_fixed(delta_height, next, +params.wind_erosion_rate);
        }
        pos = next;
    }
}
```

---

## 8. Shader Binding Layout

All erosion entry points live in a new shader file: `shaders/erosion.wgsl`.

### Group 0 — Uniforms

`@group(0) @binding(0) var<uniform> params: ErosionParams`

```wgsl
struct ErosionParams {
    resolution:              vec2<u32>,
    dt:                      f32,    // simulation timestep
    gravity:                 f32,
    pipe_length:             f32,    // cell size in world units
    rain_rate:               f32,
    evaporation_rate:        f32,
    sediment_capacity:       f32,
    erosion_rate:            f32,
    deposition_rate:         f32,
    min_slope:               f32,
    hardness_influence:      f32,
    repose_angle_radians:    f32,    // thermal
    talus_rate:              f32,    // thermal
    num_particles:           u32,
    max_steps:               u32,
    inertia:                 f32,
    frame_seed:              u32,
    _pad:                    vec3<u32>,
}
```

### Group 1 — Textures and Buffers

```wgsl
@group(1) @binding(0) var height_a:     texture_storage_2d<r32float,    read_write>;
@group(1) @binding(1) var height_b:     texture_storage_2d<r32float,    read_write>;
@group(1) @binding(2) var water:        texture_storage_2d<r32float,    read_write>;
@group(1) @binding(3) var sediment_tex: texture_storage_2d<r32float,    read_write>;
@group(1) @binding(4) var flux_tex:     texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(5) var velocity_tex: texture_storage_2d<rg32float,   read_write>;
@group(1) @binding(6) var hardness_tex: texture_storage_2d<r32float,    read_write>;
@group(1) @binding(7) var<storage, read_write> delta_height: array<atomic<i32>>;
```

The `delta_height` buffer is `(resolution.x * resolution.y) * 4` bytes.

---

## 9. Rust-Side Architecture

### New resources

```rust
/// All erosion simulation textures.  Rebuilt on resolution change.
#[derive(Resource)]
pub struct ErosionBuffers {
    pub height_a:    Handle<Image>,
    pub height_b:    Handle<Image>,
    pub water:       Handle<Image>,
    pub sediment:    Handle<Image>,
    pub flux:        Handle<Image>,
    pub velocity:    Handle<Image>,
    pub hardness:    Handle<Image>,
    pub delta_height: Buffer,        // array<atomic<i32>>, STORAGE | COPY_DST
}

/// Parameters forwarded to the GPU via a UniformBuffer.
#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct ErosionParams {
    pub enabled:              bool,
    pub iterations:           u32,   // ticks per preview regeneration
    pub dt:                   f32,
    pub gravity:              f32,
    pub rain_rate:            f32,
    pub evaporation_rate:     f32,
    pub sediment_capacity:    f32,
    pub erosion_rate:         f32,
    pub deposition_rate:      f32,
    pub min_slope:            f32,
    pub hardness_influence:   f32,
    pub repose_angle:         f32,   // degrees; converted to radians for GPU
    pub talus_rate:           f32,
    pub thermal_iterations:   u32,   // thermal passes per tick
    pub num_particles:        u32,
    pub particle_max_steps:   u32,
    pub particle_inertia:     f32,
}
```

### New plugin

```rust
pub(crate) struct ErosionComputePlugin;

impl Plugin for ErosionComputePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/erosion.wgsl");
        app.init_resource::<ErosionParams>()
           .add_plugins(ExtractResourcePlugin::<ErosionParams>::default())
           // ... extract ErosionBuffers handles
           .add_systems(Startup, setup_erosion_buffers)
           .add_systems(PostUpdate, sync_erosion_buffers_on_resize);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ErosionPipeline>();
        // create delta_height buffer
    }
}
```

### Render graph node

`ErosionNode` runs after `GeneratorNode` (which fills `raw_heights`) and before the
existing normalisation passes.  It:

1. Copies `raw_heights` → `height_a` (one `copy_texture_to_texture` command)
2. Clears `water`, `sediment`, `flux`, `velocity` to zero  
3. Optionally initialises `hardness` from a noise dispatch
4. Runs `N` erosion ticks (each tick = passes 1–8 above)
5. Copies `height_a` → `raw_heights`

The `copy_texture_to_texture` and buffer clears are done via wgpu `CommandEncoder`
methods, not compute dispatches.

---

## 10. Progressive Preview Strategy

Regenerating all erosion iterations on every parameter tweak is expensive.  Strategy:

```
On GeneratorParams change:
  → set dirty = true
  → reset erosion_accumulated_ticks = 0

Each frame (in ErosionNode.run):
  → if dirty:
       run TICKS_PER_FRAME erosion ticks (e.g. 10)
       erosion_accumulated_ticks += TICKS_PER_FRAME
       if erosion_accumulated_ticks >= params.iterations:
           dirty = false
```

This gives a progressive, animated preview that converges within a few seconds
depending on `params.iterations` and `TICKS_PER_FRAME`.

The dirty flag is communicated via an `Arc<AtomicBool>` extracted into the render world.

For **export**, the full `params.iterations` ticks always run to completion before
readback (same as the existing export blocking model).

---

## 11. Parameter Migration

| Old `GeneratorParams` field | New home                              | Notes                      |
|-----------------------------|---------------------------------------|----------------------------|
| `erosion_strength`          | **Remove from generator**             | Was `erosion_shaped_fbm`   |
| —                           | `ErosionParams::erosion_rate`         | Real hydraulic erosion     |
| —                           | `ErosionParams::sediment_capacity`    | Transport capacity K_c     |
| —                           | `ErosionParams::deposition_rate`      | Deposition rate K_d        |
| —                           | `ErosionParams::repose_angle`         | Thermal talus angle        |
| —                           | `ErosionParams::iterations`           | Total sim ticks            |
| —                           | `ErosionParams::rain_rate`            | Uniform rainfall           |
| —                           | `ErosionParams::evaporation_rate`     | Water evaporation K_e      |

The `erosion_shaped_fbm` function remains in the shader for backward compatibility
until the new erosion is confirmed working, then is deleted along with `erosion_strength`
from `GeneratorUniform`.

---

## 12. Export Integration

`GeneratorExportPlugin` gains an `erosion_iterations_export` field (default: same as
preview but 5× — e.g. 250).  After `generate_height` fills the full-resolution height
texture, before the downsample passes run:

```rust
// In GeneratorExportNode::run (new section):
if erosion_params.enabled {
    for tick in 0..erosion_params.export_iterations {
        run_erosion_tick(&mut pass, &resources, tick);
    }
}
// Then existing downsample + normal passes...
```

The `ErosionBuffers` at export resolution are separate allocations from the preview
buffers (different `Handle<Image>` set, created when the export job starts and freed
when it completes).

---

## 13. Quality Presets

| Preset     | Grid iters | Particles  | Thermal iters | Frame time (RTX 4090, 1024²) |
|------------|-----------|------------|----------------|-------------------------------|
| Draft      | 30        | 0          | 20             | ~30 ms                        |
| Preview    | 80        | 100 000    | 50             | ~120 ms                       |
| Quality    | 200       | 500 000    | 150            | ~600 ms                       |
| Export     | 500       | 2 000 000  | 300            | ~3 s                          |
| Ultra      | 1 000     | 8 000 000  | 600            | ~15 s                         |

Frame times are per **full regeneration**, not per frame.  With progressive preview,
each frame only runs `TICKS_PER_FRAME = 10` ticks, so the UI stays responsive.

---

## 14. Implementation Phases

### Phase 1 — Grid-Based Hydraulic Erosion (Core)
- [ ] Add `ErosionBuffers` resource and `ErosionParams` with serialisation
- [ ] Write `erosion.wgsl` with passes 1–6 (water add, flux, water/vel, erode/deposit, sediment advect, evaporate)
- [ ] `ErosionComputePlugin` with render graph node
- [ ] `erosion_init` and `erosion_finalize` copy passes
- [ ] Progressive dirty-flag preview
- [ ] Editor panel: erosion section with sliders for all params
- [ ] Remove `erosion_shaped_fbm` and channel-mask hack from `generator.wgsl`

### Phase 2 — Thermal Erosion
- [ ] Add `thermal_compute` and `thermal_apply` passes to `erosion.wgsl`
- [ ] Initialise `hardness` from a noise dispatch (seeded from `GeneratorParams`)
- [ ] Hardness-aware erosion (harder rock resists carving)
- [ ] Editor: thermal sub-section with repose angle and iterations

### Phase 3 — Particle-Based Fine Detail
- [ ] `delta_height` `array<atomic<i32>>` storage buffer
- [ ] `particle_erode` entry point with bilinear splat
- [ ] `particle_apply` pass
- [ ] Integrate as post-processing step after grid-based erosion
- [ ] Performance tuning: batch size vs. quality trade-off

### Phase 4 — Export Integration
- [ ] Export-resolution `ErosionBuffers` (allocated per export job)
- [ ] `erosion_iterations_export` param in `GeneratorExportPlugin`
- [ ] Verify normal tile accuracy after erosion (normals computed from eroded heights)

### Phase 5 — Wind Erosion
- [ ] `wind_erode` + `particle_apply` entry points
- [ ] Wind direction vector in `ErosionParams`
- [ ] Editor: wind sub-section

---

## 15. Key References

- D'Alesio et al. (2007) — "Fast Hydraulic Erosion Simulation and Visualization on GPU"
  — the definitive virtual pipe model paper.
- Hans Theobald Beyer (2015) — "Implementation of a method for hydraulic erosion"
  — clean particle-based implementation widely used in games/tools.
- Sebastian Lague (2019) — "Coding Adventure: Hydraulic Erosion" — practical GPU
  particle implementation with bilinear splatting via atomics.
- Inigo Quilez — "Simulating Erosion" (iquilezles.org) — analytical approximations
  useful for the gradient-shaping fallback.

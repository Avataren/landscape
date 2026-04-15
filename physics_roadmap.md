# Terrain Physics Collision — Roadmap

## Why the Current Approach Falls Through

`PlayerPlugin` maintains a single 256 m × 256 m rolling heightfield centred on
the player body.  This has three structural weaknesses:

1. **No coverage outside 128 m of the last rebuild centre.**  Flying far away in
   freecam and switching to walking puts the body in empty space — the heightfield
   may be thousands of metres away.  (The freecam→walking teleport fix mitigates
   this particular case, but the underlying coverage gap remains.)

2. **One-frame rebuild latency at edges.**  When the player walks fast enough to
   cross the 64 m update threshold, the old heightfield disappears one physics
   tick before the new one appears (deferred commands).  Steep slopes at the
   2 m/cell resolution can send the capsule through the gap.

3. **Cannot support collision queries at arbitrary world positions.**  Any system
   that wants to fire a ray or overlap-test against the terrain (AI, projectiles,
   VFX) is limited to the 256 m window.

## Target Architecture

One **static `Collider::heightfield` per loaded LOD-0 tile**, spawned and
despawned in lockstep with the existing tile residency system.  The result is
that full collision coverage follows tile streaming automatically — wherever the
game has terrain data, physics has it too.

```
TerrainResidency::resident_cpu   ──sync──▶   TileColliders (HashMap<TileKey, Entity>)
    tile (0, 0)  ──────────────────────▶  RigidBody::Static + Collider::heightfield
    tile (0, 1)  ──────────────────────▶  RigidBody::Static + Collider::heightfield
    tile (1, 0)  ──────────────────────▶  RigidBody::Static + Collider::heightfield
    …                                     …
```

At 256 loaded tiles (full 4 096 m × 4 096 m terrain at LOD 0, each tile 256 m):
- 16 × 16 = 256 static bodies
- 64 × 64 samples per tile (resampled from 256 × 256 → 4 m/cell) = reasonable memory
- Avian handles hundreds of static bodies with no performance concern

---

## Phase 1 — Persistent Tile Heightfields  *(primary fix)*

### New resource

```rust
// src/terrain/physics_colliders.rs  (new file)

#[derive(Resource, Default)]
pub struct TileColliders {
    entities: HashMap<TileKey, Entity>,
}
```

### Tile heightfield constructor

Tile data layout: `HeightTileCpu::data[row * tile_size + col]`
where **row = Z axis, col = X axis**.

Avian3d heightfield layout: `heights[xi][zi]`
where **outer index = X, inner = Z**.

Therefore:

```rust
fn build_tile_heightfield(
    tile: &HeightTileCpu,
    height_scale: f32,
    resolution: u32,      // e.g. 64  (resample factor = tile_size / resolution)
) -> (Vec<Vec<f32>>, Vec3) {
    let ts = tile.tile_size as usize;
    let step = ts / resolution as usize;  // e.g. 256 / 64 = 4
    let n = (ts / step) as usize;         // number of sample points per axis

    let heights: Vec<Vec<f32>> = (0..n).map(|xi| {
        (0..n).map(|zi| {
            tile.data[(zi * step) * ts + (xi * step)] * height_scale
        }).collect()
    }).collect();

    let tile_world = (ts as f32);          // 256 m  (world_scale = 1.0)
    let scale = Vec3::new(tile_world, 1.0, tile_world);
    (heights, scale)
}
```

Tile (tx, ty) centre in world space:
```rust
let cx = tile.key.x as f32 * 256.0 + 128.0;
let cz = tile.key.y as f32 * 256.0 + 128.0;
Transform::from_xyz(cx, 0.0, cz)
```

### Sync system

```rust
// Runs in Update, after update_collision_tiles.
// Only processes LOD-0 tiles (the collision cache uses level = 0).
fn sync_tile_colliders(
    residency:  Res<TerrainResidency>,
    config:     Res<TerrainConfig>,
    mut colliders: ResMut<TileColliders>,
    mut commands:  Commands,
) {
    // Spawn colliders for newly resident tiles.
    for (key, tile) in &residency.resident_cpu {
        if key.level != 0 { continue; }
        if colliders.entities.contains_key(key) { continue; }

        let (heights, scale) = build_tile_heightfield(tile, config.height_scale, 64);
        let cx = key.x as f32 * 256.0 + 128.0;
        let cz = key.y as f32 * 256.0 + 128.0;

        let entity = commands.spawn((
            RigidBody::Static,
            Collider::heightfield(heights, scale),
            Transform::from_xyz(cx, 0.0, cz),
        )).id();
        colliders.entities.insert(*key, entity);
    }

    // Despawn colliders for evicted tiles.
    colliders.entities.retain(|key, entity| {
        if residency.resident_cpu.contains_key(key) {
            true
        } else {
            commands.entity(*entity).despawn();
            false
        }
    });
}
```

### Registration

```rust
// In TerrainPlugin::build():
app
    .init_resource::<TileColliders>()
    .add_systems(Update,
        sync_tile_colliders
            .after(update_collision_tiles)
            .after(update_terrain_view_state)
    );
```

### Remove rolling heightfield from PlayerPlugin

Once tile colliders are working, delete `HeightfieldState`, `spawn_heightfield`,
and `update_heightfield` from `player.rs`.  The player body just falls onto
whatever tile colliders are present.

---

## Phase 2 — Coarse Full-World Fallback  *(belt and suspenders)*

Tile colliders are only present for loaded tiles.  During the 1–3 seconds before
streaming fills in a freshly-visited area (e.g. right after a freecam→walking
switch to a distant location) the player would still fall.

Fix: build a single low-resolution heightfield covering the entire terrain
footprint during `PostStartup` from the preloaded tiles.

```
128 × 128 samples  →  4 096 m / 128 = 32 m/cell
Memory: 128 × 128 × 4 B = 65 KB
```

Procedure:
- Sample the full world bounds in 32 m steps using `TerrainCollisionCache::sample_height`
  (available after `preload_terrain_startup`)
- Spawn `RigidBody::Static + Collider::heightfield(...)` once in PostStartup
- Never despawn it — it acts as the floor of last resort

Fine tile colliders (Phase 1) will shadow this coarse one for all loaded areas
because they are more precise and avian processes all colliders in the scene.

---

## Phase 3 — Spatial Query Integration  *(optional)*

Once stable tile colliders exist, avian3d's `SpatialQuery` system parameter
provides raycasting, shape overlap, and nearest-point queries against all
physics geometry, including the terrain heightfields.

Typical uses:
- Player ground detection (grounded check for jump, footstep sounds)
- Projectile impact points
- Vegetation/prop placement at terrain surface
- AI pathfinding height queries

Replace `TerrainCollisionCache::raycast_terrain` (step-marching CPU loop) with:
```rust
fn example(spatial: SpatialQuery) {
    let hit = spatial.cast_ray(
        Vec3::new(x, 10_000.0, z),
        Dir3::NEG_Y,
        f32::MAX,
        true,
        &SpatialQueryFilter::default(),
    );
}
```

---

## Phase 4 — Per-tile Collision LOD  *(optional, low priority)*

When high-res LOD-0 tile data is not yet loaded, fall back to the coarser LOD
tile that IS resident for that area, instead of relying only on the Phase 2
coarse global heightfield.  This closes the gap in mountainous terrain where
32 m/cell is too coarse to land safely.

Implementation: in `sync_tile_colliders`, walk the LOD levels from 1 → max for
each region of the world not yet covered by a LOD-0 collider.  Tag each collider
with its LOD level so it can be superseded (despawned) when a finer tile arrives.

---

## Resolution Guidance

| Resolution | Cell size | Memory per tile | Use case |
|---|---|---|---|
| 256 × 256 | 1 m | 256 KB | Too heavy; matches GPU clipmap exactly but no gain |
| 128 × 128 | 2 m | 64 KB | Good for walking; 16 MB for full 256-tile world |
| **64 × 64** | **4 m** | **16 KB** | **Recommended — 4 MB total, imperceptible for walking** |
| 32 × 32 | 8 m | 4 KB | Acceptable for coarse vehicle/projectile collision |

---

## Implementation Order

1. **`src/terrain/physics_colliders.rs`** — `TileColliders` resource + `sync_tile_colliders` system
2. **`src/terrain/mod.rs`** — register `TileColliders` and `sync_tile_colliders`
3. **`src/player.rs`** — remove `HeightfieldState`, `spawn_heightfield`, `update_heightfield`
4. **(Phase 2)** `src/terrain/physics_colliders.rs` — coarse global heightfield in PostStartup
5. **(Phase 3)** Replace `TerrainCollisionCache::raycast_terrain` call sites with `SpatialQuery`

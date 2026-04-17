# Landscape Editor — Project Brief for AI Assistants

## Vision

This project is a **real-time landscape editor** built on Bevy 0.18 (Rust). The goal is a tool where an artist or developer can:

1. Import a raw heightmap (EXR / PNG / TIFF) or eventually generate one procedurally.
2. Assign diffuse colour maps and normal maps, or let the engine derive them automatically.
3. Paint and configure procedural surface materials (rock, soil, grass, snow, etc.) that blend by slope and altitude.
4. Instance foliage, trees, rocks, and other scatter objects across the landscape, with density driven by material weights and terrain slope.
5. Place and configure bodies of water — lakes, rivers — with realistic flow and shoreline blending.
6. Export the finished scene as a portable level file (JSON + baked tile data) that can be loaded at runtime or shared.

The editor must remain **intuitive**. The user should never need to understand internal parameters like `clipmap_levels`, `max_mip_level`, or tile resolution. These are derived automatically from the data. Where choices must be exposed, they should be presented as curated options with human-readable labels and sensible defaults inferred from the loaded heightmap.

---

## Core Principles

### 1. Parameters derive from data, not the other way around
- `max_mip_level` is scanned from the baked tile output directory — never typed by hand.
- `clipmap_levels` is computed as `max_mip_level + 1` after import so that every clipmap ring has real tile data. Rings without baked data cause a "terrain rising from the floor" artifact (geometry defaults to height 0 for out-of-bounds pixels).
- World bounds (`world_min` / `world_max`) are scanned from the tile grid at L0, not authored in a config file.
- The number of bake levels is derived from the heightmap resolution and tile size (`n = log2(resolution / tile_size)`).

### 2. Valid ranges, not raw numbers
When a user-facing control must expose a numeric parameter:
- Show a slider or combo box bounded to values that make physical sense for the current dataset.
- Example: height scale slider range computed from `world_bounds` and the heightmap's actual min/max values.
- Example: material altitude range bounded to `[world_min.y, world_max.y]`.

### 3. No required restart
The editor hot-reloads terrain without restarting the process. A `ReloadTerrainRequest` message drives all live terrain swaps — tile streaming, clipmap textures, material uniforms, and macro colour are all replaced in place.

### 4. Non-destructive workflow
- Import: bakes raw heightmap → tile hierarchy in an output directory.
- Save Landscape: writes a `LevelDesc` JSON referencing baked tile paths + material definitions.
- Load Landscape: reads JSON → hot-reloads terrain.
- Nothing is overwritten without an explicit save action.

---

## Architecture

### Workspace layout

```
landscape/           ← root binary (launches editor + terrain)
  src/
    main.rs          ← startup: --level arg, preferences, landscape.toml fallback
    app_config.rs    ← parse landscape.toml; scan_world_bounds from tile grid
    player.rs        ← first-person camera controller
  crates/
    bevy_landscape/  ← reusable terrain renderer library
      src/
        bake.rs      ← BakeConfig + bake_heightmap() — full mip pyramid bake
        level.rs     ← LevelDesc JSON format; save_level / load_level / into_runtime
        terrain/
          config.rs  ← TerrainConfig (clipmap_levels, world_scale, height_scale, …)
          mod.rs     ← TerrainPlugin; ReloadTerrainRequest + reload_terrain_system
          streamer.rs← background tile loader; clamps key.level to max_mip_level
          clipmap_texture.rs ← CPU/GPU clipmap ring management
          material_slots.rs  ← MaterialLibrary; altitude/slope-based slot blending
          world_desc.rs      ← TerrainSourceDesc (tile_root, world bounds, mip levels)
          material.rs        ← TerrainMaterial GPU uniforms
    bevy_landscape_editor/  ← egui editor UI plugin
      src/
        toolbar.rs   ← top menu bar: File / Tools
        import.rs    ← Import Heightmap wizard (pick → settings → bake → hot-reload)
        level_io.rs  ← File→Save Landscape / Load Landscape
        preferences.rs ← AppPreferences; default level file; persisted to preferences.json
        material_panel.rs  ← Tools→Materials: per-slot altitude/slope/colour tweaks
```

### Level file format (`LevelDesc` JSON)
```json
{
  "tile_root": "assets/tiles",
  "normal_root": null,
  "diffuse_path": "assets/diffuse.exr",
  "max_mip_level": 3,
  "collision_mip_level": 2,
  "world_scale": 1.0,
  "height_scale": 2048.0,
  "clipmap_levels": 4,
  "material_library": { ... }
}
```
`normal_root: null` means the streamer infers it as `tile_root/normal/`.  
`clipmap_levels` is always capped to `max_mip_level + 1` on load to prevent out-of-bounds sampling.

### Hot-reload path
```
UI action (import / load)
  → send ReloadTerrainRequest { config, source, material_library }
  → reload_terrain_system (runs before update_terrain_view_state):
      ① Bump stream_queue.reload_generation (old in-flight tiles are now stale)
      ② *config = new_config  (clipmap_levels, world_scale, height_scale updated)
      ③ *desc = new_source    (tile_root, world bounds, max_mip_level updated)
      ④ *material_library = new_library
      ⑤ clear TerrainResidency, TerrainStreamQueue (preserving new generation), TerrainCollisionCache, TerrainViewState
      ⑥ resize CPU clipmap buffers to MAX_SUPPORTED_CLIPMAP_LEVELS × bpl, fill with 0
      ⑦ zero GPU clipmap image data (triggers full re-upload)
      ⑧ set last_clip_centers / tile_apply_centers to IVec2::MAX sentinel (forces full rebuild)
      ⑨ update material uniforms: height_scale, num_lod_levels, world_bounds, bounds_fade
      ⑩ replace macro colour texture in place (no handle change)
  → next frame: update_terrain_view_state repopulates clip_centers
  → next frame: update_patch_transforms sees new config → respawns patch entities if count changed
  → subsequent frames: tiles load from NEW tile set only (stale tiles discarded by generation check)
```

### Generation counter (critical for correct hot-reload)
`TerrainStreamQueue.reload_generation: u64` is incremented on every hot-reload.  
`HeightTileCpu.generation` carries the generation the tile was requested for.  
`poll_tile_stream_jobs` discards any tile with `tile.generation != queue.reload_generation`.  
This prevents old tile data (from background threads dispatched before the reload) from merging into the new terrain clipmap.

### Mip levels and clipmap levels (critical)
- The bake produces `levels = floor(log2(image_size / tile_size))` mip levels: L0 (full res) through L(levels-1) (coarsest).
- Each baked level L has tile indices `[-(n/2), n/2)` where `n = image_size / (tile_size * 2^L)`.
- `max_mip_level = levels - 1` (0-indexed, scanned from directory after bake).
- `clipmap_levels = max_mip_level + 1`. **Never set this higher than the number of baked levels.** Rings beyond max_mip_level will be resampled from the coarsest available mip; but when the ring footprint far exceeds the heightmap bounds, most pixels default to height=0, producing the "terrain rising from floor" artifact.
- The GPU clipmap texture array is allocated at startup with `clipmap_levels` layers. Changing `clipmap_levels` at hot-reload is safe because (a) the texture has `MAX_SUPPORTED_CLIPMAP_LEVELS` (16) layers and (b) `mat.params.num_lod_levels` is updated in `reload_terrain_system` to clamp shader reads to the new count.

### Procedural material system
- Up to 16 material slots, each blended by slope and altitude.
- Altitude ranges are stored in **base units** (before `world_scale`). `sync_material_library_to_terrain_material` multiplies by `world_scale` before GPU upload. Never scale authored values or shader reads.
- `height_scale * world_scale` is the runtime Y range. At startup and in `reload_terrain_system`, `config.height_scale` already includes `world_scale` (i.e. `config.height_scale = authored_height_scale * world_scale`). To recover base: `base = config.height_scale / config.world_scale`.
- Procedural texturing (slope, altitude blending) must remain visually consistent regardless of `world_scale`. This is achieved by always computing normalised height (`h / height_scale`) in the shader.

---

## Bevy 0.18 API notes

- `EventWriter` / `EventReader` → **`MessageWriter`** / **`MessageReader`**.
- Register messages with `app.add_message::<T>()`.
- `#[derive(Message)]` on the event struct.
- `Receiver<T>` is `Send` but not `Sync` → wrap as `Mutex<Receiver<T>>` for Bevy `Resource` bounds.
- Egui borrow pattern: never use `.open(&mut bool)` when the bool is also written inside the closure. Use a local `do_cancel` bool, set inside the closure, acted on after.
- `EguiPrimaryContextPass` is the system set for egui UI systems.

---

## Startup flow
```
cargo run [--level <path.json>]

1. Parse CLI args for --level.
2. Load AppPreferences from preferences.json (CWD).
3. Level resolution order:
     --level arg  →  preferences.default_level  →  landscape.toml
4. If level JSON: load_level() → into_runtime() → TerrainPlugin
5. If landscape.toml: app_config::load_config() → scan_world_bounds() → TerrainPlugin
```

---

## What exists today

| Feature | Status |
|---|---|
| CDLOD clipmap terrain renderer | ✅ Working |
| Background tile streaming | ✅ Working |
| Rapier physics heightfield | ✅ Working |
| Procedural slope/altitude material blending | ✅ Working |
| World-aligned macro colour EXR | ✅ Working |
| Import Heightmap wizard (bake + hot-reload) | ✅ Working |
| File → Save / Load Landscape (JSON) | ✅ Working |
| File → Preferences (default level) | ✅ Working |
| Materials panel (per-slot tuning) | ✅ Working |
| Detail / tiling textures (grass, rock, etc.) | ❌ Not started |
| Splatmap painting | ❌ Not started |
| Foliage / tree instancing | ❌ Not started |
| Water bodies (lakes, rivers) | ❌ Not started |
| Procedural heightmap generation | ❌ Not started |
| GPU frustum culling for patches | ❌ Deferred (see loose_ends.md) |

---

## Roadmap (rough priority order)

1. **Detail textures** — tiling albedo/normal/ORM per material slot, blended with macro colour by distance. Defined in `material_system.md`.
2. **Splatmap painting** — RGBA8 tile hierarchy for material weights; in-editor brush tools.
3. **Foliage instancing** — GPU-instanced grass/shrubs driven by splatmap density and slope.
4. **Tree placement** — LOD-aware tree instances; collision capsules.
5. **Water** — lake planes with shore blending; river splines with flow maps.
6. **Procedural generation** — noise-based heightmap generation inside the editor (no external image required).

See `material_system.md`, `loose_ends.md`, `performance_roadmap.md`, and `physics_roadmap.md` for detailed technical notes on each area.

---

## Build and test

```bash
cargo build --quiet          # compile all crates
cargo test --workspace --quiet  # run all tests
cargo run                    # launch editor with default landscape.toml
cargo run -- --level my_level.json   # launch with a specific level file
```

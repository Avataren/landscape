# Landscape Editor

A real-time landscape editor built on [Bevy 0.18](https://bevyengine.org/) (Rust).

---

## What it does

- Import a raw heightmap (EXR / PNG / TIFF) and bake it into a streaming mip-tile hierarchy.
- Assign and paint procedural surface materials (rock, soil, grass, snow) that blend by slope and altitude.
- GPU-instanced grass rendered without any pre-bake step.
- Physics heightfield colliders via [avian3d](https://github.com/Jondolf/avian).
- Export the finished scene as a portable level JSON that can be loaded at runtime.
- Hot-reload terrain without restarting — swap heightmaps, materials, and foliage in-place.

---

## Getting started

```bash
# Build
cargo build --quiet

# Run with the default landscape.toml
cargo run

# Run with a specific level file
cargo run -- --level my_level.json

# Run all tests
cargo test --workspace --quiet
```

---

## Workspace layout

```
landscape/           ← root binary (editor + terrain)
  src/
    main.rs          ← startup: --level arg, preferences, landscape.toml fallback
    app_config.rs    ← parse landscape.toml; scan_world_bounds from tile grid
    player.rs        ← first-person camera controller
  crates/
    bevy_landscape/  ← reusable terrain renderer library
    bevy_landscape_editor/  ← egui editor UI plugin
```

See [`crates/bevy_landscape/README.md`](crates/bevy_landscape/README.md) for a full
technical description of the CDLOD clipmap renderer, tile streaming, GPU grass, and
the plugin API.

---

## Current feature status

| Feature | Status |
|---|---|
| CDLOD clipmap terrain renderer | ✅ Working |
| Background tile streaming | ✅ Working |
| avian3d physics heightfield | ✅ Working |
| Procedural slope/altitude material blending | ✅ Working |
| World-aligned macro colour EXR | ✅ Working |
| SSAO (screen-space ambient occlusion) | ✅ Working |
| GPU grass (vertex-shader instanced, no pre-bake) | ✅ Working |
| Foliage shadow toggle | ✅ Working |
| Import Heightmap wizard (bake + hot-reload) | ✅ Working |
| File → Save / Load Landscape (JSON) | ✅ Working |
| File → Preferences (default level) | ✅ Working |
| Materials panel (per-slot tuning) | ✅ Working |
| Detail / tiling textures (grass, rock, etc.) | ⚠️ In progress |
| Splatmap painting | ⚠️ Infrastructure exists, painter not wired |
| Tree / rock instancing | ❌ Not started |
| Water bodies (lakes, rivers) | ❌ Not started |
| Procedural heightmap generation | ❌ Not started |

---

## Architecture notes

See [`CLAUDE.md`](CLAUDE.md) for detailed architecture notes covering:
- How clipmap levels and mip levels are derived from the data
- The hot-reload path and generation counter
- The procedural material system and world-scale invariants
- Bevy 0.18 API specifics used in this codebase

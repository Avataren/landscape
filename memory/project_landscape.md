---
name: Landscape terrain renderer project
description: Bevy 0.18.1 CDLOD/clipmap terrain generator at /home/avataren/src/landscape — current state and architecture
type: project
---

Bevy 0.18.1 clipmap/CDLOD terrain at /home/avataren/src/landscape.

**Phase 0 complete**: GPU terrain rendering, procedural noise generator, export pipeline, egui editor.

**Erosion system implemented (2026-04-20)**:
- New files: `erosion_params.rs`, `erosion_images.rs`, `erosion_compute.rs`, `shaders/erosion.wgsl`
- `GeneratorNode` split into `GeneratorRawNode` + `GeneratorNormNode`; `ErosionNode` inserts between them
- Render graph: `GeneratorRawLabel → ErosionLabel → GeneratorNormLabel → CameraDriverLabel`
- Fake erosion (`erosion_shaped_fbm`, channel carving) removed from `generator.wgsl`
- `ErosionParams` resource (serializable, main world + render world via ExtractResource)
- `ErosionBuffers` resource: height_a/b, water, sediment, flux, velocity, hardness textures
- `delta_height` storage buffer (`array<atomic<i32>>`) for thermal + particle atomics
- All 3 erosion phases: hydraulic grid (6 passes/tick), thermal (atomic, N iters/tick), particle (64-thread droplets)
- Editor panel: full erosion controls in generator_panel.rs

**Why:** Per erosion_implementation.md — replace fake gradient-based erosion with proper GPU simulation.

**How to apply:** Erosion is disabled by default (`ErosionParams::enabled = false`). Enable via the Terrain Generator UI or by setting `ErosionParams` in code. Iterations default to 30 which is heavy; set lower (5-10) for fast live preview.

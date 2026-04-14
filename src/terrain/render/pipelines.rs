// Terrain render pipeline.
// Phase 0 skeleton — pipeline creation is stubbed until Phase 2 when the
// custom terrain WGSL shaders are wired in.
//
// Planned binding layout:
//   group 0, binding 0 — TerrainFrameUniform
//   group 0, binding 1 — PatchDescriptorGpu[] (storage buffer)
//   group 1, binding 0 — height clipmap texture array
//   group 1, binding 1 — height sampler
//   group 2, binding 0 — material mask texture
//   group 2, binding 1 — material sampler

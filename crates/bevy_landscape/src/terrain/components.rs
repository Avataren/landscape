use bevy::prelude::*;

/// Marker component: the camera that drives terrain LOD and streaming.
/// Exactly one entity should carry this in a scene.
#[derive(Component, Default)]
pub struct TerrainCamera;

/// Per-patch instance data placed by the clipmap builder.
/// In v1 this lives on ECS entities; in v2 it goes into a GPU storage buffer.
#[derive(Component, Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct TerrainPatchInstance {
    /// Clipmap LOD level (0 = finest).
    pub lod_level: u32,
    /// Patch type hint (0 = normal ring patch, non-zero reserved for trim/fill).
    pub patch_kind: u32,
    /// World-space XZ origin of this patch (its minimum corner).
    pub patch_origin_ws: Vec2,
    /// World-space size of one patch side.
    pub patch_scale_ws: f32,
}

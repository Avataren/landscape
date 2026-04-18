use bevy::prelude::*;

/// Marker component: the camera that drives terrain LOD and streaming.
/// Exactly one entity should carry this in a scene.
#[derive(Component)]
pub struct TerrainCamera {
    /// Fraction of the LOD-0 ring half-extent by which the clipmap centers are
    /// pushed along the camera's forward XZ direction.
    ///
    /// 0.0 = classic camera-centered behaviour (rings stay directly under the
    /// camera).  0.4–0.7 shifts the rings *in front of* the camera so the
    /// fine LOD rings cover the visible foreground for near-horizontal views
    /// rather than being wasted under the player's feet.
    ///
    /// The same bias is applied to every level, so strict ring nesting (and
    /// thus inner-hole alignment) is preserved.
    pub forward_bias_ratio: f32,
}

impl Default for TerrainCamera {
    fn default() -> Self {
        // Keep the finest clipmap centered on the camera by default. Forward
        // projection can improve foreground detail in some views, but it can
        // also make near visible terrain appear to "drop out" when the finest
        // coverage is pushed too far ahead of the player.
        Self {
            forward_bias_ratio: 0.0,
        }
    }
}

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

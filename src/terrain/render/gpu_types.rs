use crate::terrain::config::MAX_SUPPORTED_CLIPMAP_LEVELS;
use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Frame uniform (one per frame, bound to group 0, binding 0)
// ---------------------------------------------------------------------------

/// Per-frame uniform data uploaded to the GPU before terrain draws.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct TerrainFrameUniform {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Camera world-space position (w unused).
    pub camera_pos_ws: [f32; 4],
    /// Snapped clipmap center grid coords for up to 16 levels.
    pub clip_centers: [[i32; 4]; MAX_SUPPORTED_CLIPMAP_LEVELS],
    /// World-space texel spacing for up to 16 levels.
    pub level_scales: [f32; MAX_SUPPORTED_CLIPMAP_LEVELS],
    /// World-space height scale (maps [0,1] -> [0, height_scale]).
    pub height_scale: f32,
    /// Fraction of ring span at which morphing begins.
    pub morph_start_ratio: f32,
    /// Number of active clipmap levels.
    pub clipmap_levels: u32,
    pub _pad0: u32,
}

// ---------------------------------------------------------------------------
// Patch descriptor (one per patch instance, bound as storage buffer)
// ---------------------------------------------------------------------------

/// GPU representation of one terrain patch, matching PatchInstanceCpu.
/// Laid out as a storage buffer entry.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PatchDescriptorGpu {
    /// World-space XZ origin of this patch (minimum corner).
    pub origin_ws: [f32; 2],
    /// World-space size of one patch side.
    pub patch_size_ws: f32,
    /// LOD level (0 = finest).
    pub lod_level: u32,
    /// Camera distance at which morphing starts.
    pub morph_start: f32,
    /// Camera distance at which morphing ends (full blend).
    pub morph_end: f32,
    /// Patch kind (reserved, 0 = normal).
    pub patch_kind: u32,
    pub _pad0: u32,
}

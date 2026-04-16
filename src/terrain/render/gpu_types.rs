use bytemuck::{Pod, Zeroable};

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

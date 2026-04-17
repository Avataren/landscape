//! Material library — the authoring-side description of the procedural terrain
//! material slots.
//!
//! This module defines the `MaterialLibrary` resource, which holds the list of
//! active material slots (rock, soil, grass, snow, …) and their properties.
//! It is the single source of truth the editor UI reads from and writes to.
//!
//! The shader-side `TerrainMaterial` in `material.rs` holds the bind-group and
//! uniform layout actually sent to the GPU.  As the material system is
//! implemented (texture arrays, splatmap clipmaps, etc.) this resource will
//! grow and the sync path from `MaterialLibrary` → GPU will be filled in.
//!
//! Keeping the UI-facing data in a plain `Resource` (no render-asset handles)
//! lets the editor crate depend on `bevy_landscape` without pulling in any
//! rendering specifics.

use crate::terrain::config::TerrainConfig;
use crate::terrain::material::{MaterialSlotGpu, TerrainMaterial, MAX_SHADER_MATERIAL_SLOTS};
use crate::terrain::PatchEntities;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Practical initial cap on material slots (2 splatmaps × 4 channels).
/// The design allows 16 but 8 is the first implementation target.
pub const DEFAULT_MAX_MATERIAL_SLOTS: usize = 8;

/// Library of active material slots, mirrored by the editor panel.
///
/// The editor mutates this resource directly; downstream systems (not yet
/// implemented) will observe changes and upload texture-array contents and
/// per-slot uniforms to the GPU.
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct MaterialLibrary {
    pub slots: Vec<MaterialSlot>,
    /// Upper bound on slot count enforced by the editor UI.
    pub max_slots: usize,
    /// When `true`, the fragment shader replaces the library-blended albedo
    /// with the baked macro color / diffuse EXR (if loaded).  Default is
    /// `false` so the Materials panel is the authority while the procedural
    /// system is under active development — flip it back on to compare.
    pub use_macro_color_override: bool,
    /// Set once at startup by `setup_terrain` to `true` if the macro color
    /// EXR was successfully loaded.  When `false`, the macro color override
    /// toggle is a no-op (shader sampling falls back to a 1×1 white texture).
    /// Kept here so the Materials panel can grey out the toggle.
    /// Not persisted to level files — resolved at load time.
    #[serde(skip)]
    pub macro_color_loaded: bool,
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        // Seed slots and procedural rules roughly matching the previous
        // hardcoded shader palette so the viewport looks familiar when the
        // material pipeline is turned on for the first time.  Tints / rules
        // can be tweaked live from the editor.
        let mut grass = MaterialSlot::new("Grass");
        grass.tint = [0.28, 0.52, 0.18];
        grass.procedural.altitude_range_m = Vec2::new(-10_000.0, 10_000.0);
        grass.procedural.slope_range_deg = Vec2::new(0.0, 18.0);

        let mut soil = MaterialSlot::new("Soil");
        soil.tint = [0.50, 0.40, 0.28];
        soil.procedural.altitude_range_m = Vec2::new(-10_000.0, 10_000.0);
        soil.procedural.slope_range_deg = Vec2::new(12.0, 30.0);

        let mut rock = MaterialSlot::new("Rock");
        rock.tint = [0.44, 0.38, 0.32];
        rock.procedural.altitude_range_m = Vec2::new(-10_000.0, 10_000.0);
        rock.procedural.slope_range_deg = Vec2::new(28.0, 90.0);

        let mut snow = MaterialSlot::new("Snow");
        snow.tint = [0.90, 0.93, 0.98];
        snow.procedural.altitude_range_m = Vec2::new(600.0, 10_000.0);
        snow.procedural.slope_range_deg = Vec2::new(0.0, 55.0);

        Self {
            slots: vec![grass, soil, rock, snow],
            max_slots: DEFAULT_MAX_MATERIAL_SLOTS,
            use_macro_color_override: false,
            macro_color_loaded: false,
        }
    }
}

/// One physical surface type (rock, grass, …).  Mirrors the fields called out
/// in the design doc's "Materials Panel" section (§11b).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialSlot {
    pub name: String,
    /// Exclude from blend without deleting the slot.
    pub visible: bool,

    /// Debug / fallback tint used for shader colour until per-slot albedo
    /// textures are implemented.  Linear RGB in [0, 1].
    pub tint: [f32; 3],

    pub albedo_path: Option<PathBuf>,
    pub normal_path: Option<PathBuf>,
    pub orm_path: Option<PathBuf>,
    pub height_path: Option<PathBuf>,

    /// World-space tile size at the fine UV scale (metres per tile repeat).
    pub fine_scale_m: f32,
    /// Multiplier applied to `fine_scale_m` for the coarse UV scale.  Typical
    /// 4×–6× the fine scale; §3a recommends this range.
    pub coarse_scale_mul: f32,
    /// Surface angle (degrees from horizontal) at which UV sampling fades to
    /// triplanar.  §3c default: 45°.
    pub triplanar_threshold_deg: f32,
    /// Height-blend sharpness — larger = harder contact edges.  §3d.
    pub height_blend_sharpness: f32,

    pub procedural: ProceduralRules,
}

impl MaterialSlot {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            visible: true,
            tint: [0.5, 0.5, 0.5],
            albedo_path: None,
            normal_path: None,
            orm_path: None,
            height_path: None,
            fine_scale_m: 1.0,
            coarse_scale_mul: 5.0,
            triplanar_threshold_deg: 45.0,
            height_blend_sharpness: 0.1,
            procedural: ProceduralRules::default(),
        }
    }
}

/// Procedural placement rules used to compute the baseline splatmap weights
/// before any painting is applied.  §4 of the design doc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProceduralRules {
    /// World-space Y range in which this slot is active.
    pub altitude_range_m: Vec2,
    /// Surface slope range (degrees from horizontal).
    pub slope_range_deg: Vec2,
    /// -1 = concave hollows only, 0 = agnostic, 1 = convex ridges only.
    pub curvature_bias: f32,
    /// World-space scale of the modulating noise.
    pub noise_scale_m: f32,
    /// Strength (0 = off, 1 = fully noise-driven) of the modulating noise.
    pub noise_strength: f32,
}

impl Default for ProceduralRules {
    fn default() -> Self {
        Self {
            altitude_range_m: Vec2::new(-10_000.0, 10_000.0),
            slope_range_deg: Vec2::new(0.0, 90.0),
            curvature_bias: 0.0,
            noise_scale_m: 50.0,
            noise_strength: 0.25,
        }
    }
}

// ---------------------------------------------------------------------------
// Library → shader sync
//
// When the editor mutates `MaterialLibrary`, we copy the relevant per-slot
// fields into `TerrainMaterial.params.slots` so the fragment shader sees the
// changes on the next frame.  Bevy detects the `TerrainMaterial` write and
// re-uploads the uniform buffer automatically.
// ---------------------------------------------------------------------------

pub(crate) fn sync_material_library_to_terrain_material(
    library: Res<MaterialLibrary>,
    patch_entities: Res<PatchEntities>,
    config: Res<TerrainConfig>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
) {
    // Only write when the library changed — mutable access to the material
    // asset is a change signal Bevy picks up for GPU re-upload, and we do not
    // want to trigger that every frame.
    if !library.is_changed() {
        return;
    }
    let Some(mat) = materials.get_mut(&patch_entities.material_handle) else {
        return;
    };

    // Altitude bands are authored in "base world units" (world_scale = 1.0).
    // Scale them up so they map to the correct fraction of the scaled terrain.
    let alt_scale = config.world_scale;

    let mut slots = [MaterialSlotGpu::default(); MAX_SHADER_MATERIAL_SLOTS];
    let n = library.slots.len().min(MAX_SHADER_MATERIAL_SLOTS);
    for (i, slot) in library.slots.iter().take(n).enumerate() {
        slots[i] = MaterialSlotGpu {
            tint_vis: Vec4::new(
                slot.tint[0],
                slot.tint[1],
                slot.tint[2],
                if slot.visible { 1.0 } else { 0.0 },
            ),
            ranges: Vec4::new(
                slot.procedural.altitude_range_m.x * alt_scale,
                slot.procedural.altitude_range_m.y * alt_scale,
                slot.procedural.slope_range_deg.x,
                slot.procedural.slope_range_deg.y,
            ),
        };
    }
    mat.params.slots = slots;
    mat.params.slot_header = Vec4::new(n as f32, 0.0, 0.0, 0.0);

    // `bounds_fade.y` is the shader's "sample macro color instead of library"
    // flag.  Gate the user toggle on whether the EXR was actually loaded so
    // flipping the panel switch without a configured texture is a no-op.
    mat.params.bounds_fade.y = if library.use_macro_color_override && library.macro_color_loaded {
        1.0
    } else {
        0.0
    };
}

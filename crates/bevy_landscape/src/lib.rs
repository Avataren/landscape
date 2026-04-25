pub mod bake;
pub mod level;
pub mod metadata;
mod terrain;

pub use level::{load_level, save_level, LevelDesc};
pub use metadata::TerrainMetadata;
pub use terrain::clipmap_texture::{
    compute_clip_levels, compute_initial_clip_levels, TerrainClipmapState,
};
pub use terrain::collision::TerrainCollisionCache;
pub use terrain::components::{TerrainCamera, TerrainPatchInstance};
pub use terrain::config::{TerrainConfig, MAX_SUPPORTED_CLIPMAP_LEVELS};
pub use terrain::detail_synthesis::DetailSynthesisConfig;
pub use terrain::material_slots::{
    MaterialLibrary, MaterialSlot, ProceduralRules, DEFAULT_MAX_MATERIAL_SLOTS,
};
pub use terrain::pbr_textures::{PbrRebuildProgress, PbrTexturesDirty};
pub use terrain::physics_colliders::{LocalColliderState, ShowTerrainCollision};
pub use terrain::resources::TerrainViewState;
pub use terrain::source_heightmap::SourceHeightmapState;
pub use terrain::{ReloadTerrainRequest, TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc};

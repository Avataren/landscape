pub mod bake;
pub mod level;
pub mod metadata;
mod terrain;

pub use level::{load_level, save_level, LevelDesc};
pub use metadata::TerrainMetadata;
pub use terrain::collision::TerrainCollisionCache;
pub use terrain::components::TerrainCamera;
pub use terrain::config::{TerrainConfig, MAX_SUPPORTED_CLIPMAP_LEVELS};
pub use terrain::material_slots::{
    MaterialLibrary, MaterialSlot, ProceduralRules, DEFAULT_MAX_MATERIAL_SLOTS,
};
pub use terrain::pbr_textures::{PbrRebuildProgress, PbrTexturesDirty};
pub use terrain::{ReloadTerrainRequest, TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc};

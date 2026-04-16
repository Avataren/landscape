mod terrain;

pub use terrain::collision::TerrainCollisionCache;
pub use terrain::components::TerrainCamera;
pub use terrain::config::TerrainConfig;
pub use terrain::material_slots::{
    MaterialLibrary, MaterialSlot, ProceduralRules, DEFAULT_MAX_MATERIAL_SLOTS,
};
pub use terrain::{TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc};

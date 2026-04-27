pub mod bake;
pub mod foliage;
pub mod foliage_entities;
pub mod foliage_generation;
pub mod foliage_gpu;
pub mod foliage_instance_gen;
pub mod foliage_reload;
pub mod foliage_stream_queue;
pub mod foliage_tiles;
pub mod grass_material;
pub mod grass_mesh;
pub mod level;
pub mod metadata;
pub mod painted_splatmap;
mod terrain;

pub use foliage::{
    foliage_tile_path, painted_splatmap_path, procedural_mask_path, FoliageConfig, FoliageInstance,
    FoliageLodTier, FoliageSourceDesc,
};
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
pub use terrain::{
    ReloadTerrainRequest, TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc, TerrainSystemSet,
};

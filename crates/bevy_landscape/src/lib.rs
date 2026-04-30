pub mod bake;
pub mod foliage;
pub mod foliage_gpu_grass;
pub mod grass_material;
pub mod level;
pub mod metadata;
pub mod painted_splatmap;
mod terrain;
mod texture_arrays;

pub use foliage::{
    foliage_tile_path, painted_splatmap_path, procedural_mask_path, FoliageConfig, FoliageInstance,
    FoliageLodTier, FoliageSourceDesc,
};
pub use foliage::backend::FoliageGenerateRequest;
pub use foliage_gpu_grass::{GpuGrassConfig, GpuGrassPlugin, GRASS_MAX_GRID};
pub use foliage::reload::FoliageConfigResource;
pub use level::{load_level, save_level, scan_world_bounds, LevelDesc};
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

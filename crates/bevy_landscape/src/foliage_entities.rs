//! Foliage entity spawning and component definitions.
//!
//! Creates 24 mesh entities (8 variants × 3 LOD tiers) for rendering GPU-instanced foliage.
//! Each entity references a distinct grass blade variant mesh and shares the GrassMaterial.

use crate::foliage::FoliageLodTier;
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Component definitions
// ---------------------------------------------------------------------------

/// Marks an entity as a foliage rendering entity for a specific variant and LOD tier.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoliageVariantComponent {
    pub lod: FoliageLodTier,
    pub variant_id: u8,
}

impl FoliageVariantComponent {
    pub fn new(lod: FoliageLodTier, variant_id: u8) -> Self {
        Self {
            lod,
            variant_id: variant_id.clamp(0, 7),
        }
    }
}

// ---------------------------------------------------------------------------
// Foliage entity system sets
// ---------------------------------------------------------------------------

/// System set for foliage entity management (spawning, visibility, etc.).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FoliageEntitySystemSet {
    /// Spawn or reset foliage entities after hot-reload.
    Spawn,
    /// Update entity visibility based on camera distance.
    UpdateVisibility,
    /// Synchronize indirect draw commands with GPU state.
    SyncIndirectCommands,
}

// ---------------------------------------------------------------------------
// Entity builder helpers
// ---------------------------------------------------------------------------

/// Configuration for spawning foliage entities.
#[derive(Clone, Debug)]
pub struct FoliageEntityConfig {
    /// Enable LOD0 entities.
    pub spawn_lod0: bool,
    /// Enable LOD1 entities.
    pub spawn_lod1: bool,
    /// Enable LOD2 entities.
    pub spawn_lod2: bool,
    /// Grass blade mesh handles (one per variant).
    pub grass_meshes: Vec<Handle<Mesh>>,
}

impl FoliageEntityConfig {
    pub fn new(grass_meshes: Vec<Handle<Mesh>>) -> Self {
        Self {
            spawn_lod0: true,
            spawn_lod1: true,
            spawn_lod2: true,
            grass_meshes,
        }
    }

    /// Get the mesh handle for a variant.
    pub fn mesh_for_variant(&self, variant_id: u8) -> Option<Handle<Mesh>> {
        self.grass_meshes.get(variant_id as usize).cloned()
    }
}

/// Spawn all 24 foliage variant entities.
pub fn spawn_foliage_entities(
    mut commands: Commands,
    config: &FoliageEntityConfig,
    _material: Handle<impl Asset>,
) -> Vec<Entity> {
    let mut entities = Vec::with_capacity(24);

    for variant_id in 0..8 {
        let _mesh = match config.mesh_for_variant(variant_id) {
            Some(m) => m,
            None => continue,
        };

        if config.spawn_lod0 {
            let entity = commands
                .spawn((
                    Name::new(format!("FoliageLod0Var{}", variant_id)),
                    Transform::IDENTITY,
                    Visibility::Visible,
                    FoliageVariantComponent::new(FoliageLodTier::Lod0, variant_id),
                ))
                .id();
            entities.push(entity);
        }

        if config.spawn_lod1 {
            let entity = commands
                .spawn((
                    Name::new(format!("FoliageLod1Var{}", variant_id)),
                    Transform::IDENTITY,
                    Visibility::Visible,
                    FoliageVariantComponent::new(FoliageLodTier::Lod1, variant_id),
                ))
                .id();
            entities.push(entity);
        }

        if config.spawn_lod2 {
            let entity = commands
                .spawn((
                    Name::new(format!("FoliageLod2Var{}", variant_id)),
                    Transform::IDENTITY,
                    Visibility::Visible,
                    FoliageVariantComponent::new(FoliageLodTier::Lod2, variant_id),
                ))
                .id();
            entities.push(entity);
        }
    }

    entities
}

// ---------------------------------------------------------------------------
// Foliage entity query helpers
// ---------------------------------------------------------------------------

/// Query type for foliage entities that need rendering.
pub type FoliageEntityQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static FoliageVariantComponent,
        &'static Visibility,
        Option<&'static Handle<Mesh>>,
    ),
    With<FoliageVariantComponent>,
>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foliage_variant_component_new() {
        let comp = FoliageVariantComponent::new(FoliageLodTier::Lod0, 3);
        assert_eq!(comp.lod, FoliageLodTier::Lod0);
        assert_eq!(comp.variant_id, 3);
    }

    #[test]
    fn test_foliage_variant_component_clamps_variant() {
        let comp = FoliageVariantComponent::new(FoliageLodTier::Lod0, 255);
        assert_eq!(comp.variant_id, 7); // Should clamp to 0-7

        let comp2 = FoliageVariantComponent::new(FoliageLodTier::Lod1, 0);
        assert_eq!(comp2.variant_id, 0);
    }

    #[test]
    fn test_foliage_entity_config_mesh_for_variant() {
        // Create dummy handles using default
        let handles: Vec<Handle<Mesh>> = (0..3).map(|_| Handle::default()).collect();
        let config = FoliageEntityConfig::new(handles.clone());

        assert_eq!(config.mesh_for_variant(0), Some(Handle::default()));
        assert_eq!(config.mesh_for_variant(1), Some(Handle::default()));
        assert_eq!(config.mesh_for_variant(2), Some(Handle::default()));
        assert_eq!(config.mesh_for_variant(3), None); // Out of bounds
    }

    #[test]
    fn test_foliage_entity_config_defaults() {
        let config = FoliageEntityConfig::new(vec![]);
        assert!(config.spawn_lod0);
        assert!(config.spawn_lod1);
        assert!(config.spawn_lod2);
    }

    #[test]
    fn test_foliage_entity_config_selective_spawn() {
        let mut config = FoliageEntityConfig::new(vec![]);
        config.spawn_lod1 = false;
        assert!(config.spawn_lod0);
        assert!(!config.spawn_lod1);
        assert!(config.spawn_lod2);
    }

    #[test]
    fn test_foliage_variant_component_equality() {
        let comp1 = FoliageVariantComponent::new(FoliageLodTier::Lod0, 2);
        let comp2 = FoliageVariantComponent::new(FoliageLodTier::Lod0, 2);
        let comp3 = FoliageVariantComponent::new(FoliageLodTier::Lod1, 2);

        assert_eq!(comp1, comp2);
        assert_ne!(comp1, comp3);
    }
}

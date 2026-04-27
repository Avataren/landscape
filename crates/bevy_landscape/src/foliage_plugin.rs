//! Phase 8b: Foliage rendering plugin.
//!
//! Wires up all foliage resources and systems into Bevy's schedule.
//! Uses CPU-baked instancing: each (variant × LOD) entity has one `Mesh`
//! whose vertices are pre-transformed blade copies.  The mesh is rebuilt
//! whenever `FoliageStagingQueue` receives new batches.

use crate::{
    foliage::{FoliageInstance, FoliageLodTier},
    foliage_backend::{
        load_existing_foliage_tiles, poll_foliage_generation, start_foliage_generation,
        FoliageGenerateRequest, FoliageGenerationState,
    },
    foliage_entities::FoliageVariantComponent,
    foliage_gpu::{FoliageGpuState, FoliageGpuSyncRequest, FoliageStagingQueue},
    foliage_reload::{reload_foliage_system, FoliageConfigResource, FoliageLoadState},
    foliage_render::update_foliage_view_state,
    foliage_stream_queue::{FoliageResidency, FoliageStreamQueue, FoliageViewState},
    grass_mesh::{GrassBladeMeshConfig, generate_grass_blade_variants},
};
use bevy::{
    camera::visibility::NoFrustumCulling,
    mesh::VertexAttributeValues,
    prelude::*,
};

// ---------------------------------------------------------------------------
// FoliageMeshHandles resource
// ---------------------------------------------------------------------------

/// Holds the 24 mesh handles (8 variants × 3 LODs) used for baked-instance rendering.
///
/// Index: `lod as usize * 8 + variant_id as usize`
#[derive(Resource, Default)]
pub struct FoliageMeshHandles {
    /// 24 mesh handles.  Each mesh contains all baked blade instances for
    /// that (LOD, variant) pair.  Empty until `FoliageStagingQueue` is
    /// processed by `update_foliage_meshes`.
    pub mesh_handles: Vec<Handle<Mesh>>,
    /// Material shared by all 24 entities.
    pub material_handle: Handle<StandardMaterial>,
    /// Entity handles for the 24 entities, same indexing as mesh_handles.
    pub entities: Vec<Entity>,
    /// Base blade meshes per variant (used to bake instances).
    /// Stored so we don't regenerate the geometry each rebuild.
    pub base_blade_meshes: Vec<BladeMeshData>,
    /// True once test instances have been injected at startup.
    pub test_instances_seeded: bool,
}

impl FoliageMeshHandles {
    pub fn entity_idx(lod: FoliageLodTier, variant_id: u8) -> usize {
        (lod as usize) * 8 + (variant_id as usize)
    }

    pub fn get_entity(&self, lod: FoliageLodTier, variant_id: u8) -> Option<Entity> {
        self.entities.get(Self::entity_idx(lod, variant_id)).copied()
    }

    pub fn get_mesh_handle(&self, lod: FoliageLodTier, variant_id: u8) -> Option<&Handle<Mesh>> {
        self.mesh_handles.get(Self::entity_idx(lod, variant_id))
    }
}

// ---------------------------------------------------------------------------
// Blade mesh data (CPU-side, kept for fast rebuild)
// ---------------------------------------------------------------------------

/// CPU-side blade geometry for one variant.
#[derive(Clone)]
pub struct BladeMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl BladeMeshData {
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(v)) => v.clone(),
            _ => vec![],
        };
        let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(v)) => v.clone(),
            _ => vec![],
        };
        let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(v)) => v.clone(),
            _ => vec![],
        };
        let indices = match mesh.indices() {
            Some(bevy::mesh::Indices::U32(idx)) => idx.clone(),
            Some(bevy::mesh::Indices::U16(idx)) => idx.iter().map(|&i| i as u32).collect(),
            None => vec![],
        };
        Self { positions, normals, uvs, indices }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct FoliagePlugin;

impl Plugin for FoliagePlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<FoliageMeshHandles>()
            .init_resource::<FoliageGpuState>()
            .init_resource::<FoliageGpuSyncRequest>()
            .init_resource::<FoliageStagingQueue>()
            .init_resource::<FoliageStreamQueue>()
            .init_resource::<FoliageResidency>()
            .init_resource::<FoliageViewState>()
            .init_resource::<FoliageLoadState>()
            .init_resource::<FoliageConfigResource>()
            .init_resource::<FoliageGenerationState>()
            .init_resource::<crate::foliage::FoliageSourceDesc>()
            .add_message::<FoliageGenerateRequest>()
            .add_systems(
                Startup,
                (setup_foliage_rendering, load_existing_foliage_tiles).chain(),
            )
            .add_systems(
                Update,
                (
                    reload_foliage_system,
                    start_foliage_generation,
                    poll_foliage_generation,
                    update_foliage_view_state,
                    update_foliage_lod_visibility,
                    update_foliage_meshes,
                )
                    .chain(),
            );
    }
}

// ---------------------------------------------------------------------------
// Startup: create entities, meshes, material
// ---------------------------------------------------------------------------

pub fn setup_foliage_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut handles: ResMut<FoliageMeshHandles>,
) {
    let config = GrassBladeMeshConfig::default();
    let blade_meshes = generate_grass_blade_variants(&config);

    // Store CPU blade data for later mesh rebuilds
    handles.base_blade_meshes = blade_meshes
        .iter()
        .map(BladeMeshData::from_mesh)
        .collect();

    // Grass material: double-sided, opaque. We use Opaque rather than
    // AlphaMode::Mask to avoid the PBR prepass needing tangents that our
    // generated blades don't provide.
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.28, 0.48, 0.18),
        double_sided: true,
        cull_mode: None,
        alpha_mode: AlphaMode::Opaque,
        perceptual_roughness: 0.9,
        metallic: 0.0,
        ..default()
    });
    handles.material_handle = material.clone();

    // Build test instance data immediately (all LODs get subsampled versions)
    let lod0_instances = build_test_instances();
    // LOD1: every other instance; LOD2: every 4th instance
    let lod1_instances: [Vec<FoliageInstance>; 8] = std::array::from_fn(|v| {
        lod0_instances[v].iter().step_by(2).copied().collect()
    });
    let lod2_instances: [Vec<FoliageInstance>; 8] = std::array::from_fn(|v| {
        lod0_instances[v].iter().step_by(4).copied().collect()
    });

    // Spawn 24 entities (8 variants × 3 LODs)
    let lods = [
        FoliageLodTier::Lod0,
        FoliageLodTier::Lod1,
        FoliageLodTier::Lod2,
    ];
    for lod in lods {
        for variant_id in 0u8..8 {
            let instances: &[FoliageInstance] = match lod {
                FoliageLodTier::Lod0 => &lod0_instances[variant_id as usize],
                FoliageLodTier::Lod1 => &lod1_instances[variant_id as usize],
                FoliageLodTier::Lod2 => &lod2_instances[variant_id as usize],
            };

            // Build the mesh immediately so entities never hold an empty mesh.
            // An empty mesh causes the PBR prepass pipeline validation to fail
            // because the vertex layout doesn't match the shader requirements.
            let blade = &handles.base_blade_meshes[variant_id as usize];
            let mut mesh = Mesh::new(
                bevy::mesh::PrimitiveTopology::TriangleList,
                bevy::asset::RenderAssetUsages::default(),
            );
            build_instanced_mesh(&mut mesh, blade, instances);
            let mesh_handle = meshes.add(mesh);
            handles.mesh_handles.push(mesh_handle.clone());

            let entity = commands
                .spawn((
                    Name::new(format!("FoliageLod{}Var{}", lod as u8, variant_id)),
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(material.clone()),
                    Transform::IDENTITY,
                    Visibility::Visible,
                    NoFrustumCulling,
                    FoliageVariantComponent::new(lod, variant_id),
                ))
                .id();
            handles.entities.push(entity);
        }
    }

    handles.test_instances_seeded = true;
    info!(
        "Foliage: spawned 24 entities with pre-built meshes ({} blade mesh handles)",
        handles.mesh_handles.len(),
    );
}

// ---------------------------------------------------------------------------
// Test instance generation (Phase 8b demo data)
// ---------------------------------------------------------------------------

/// Generate a 40×40 grid of test grass instances distributed across 8 variants.
/// Returns one Vec<FoliageInstance> per variant (index 0-7).
fn build_test_instances() -> [Vec<FoliageInstance>; 8] {
    let mut per_variant: [Vec<FoliageInstance>; 8] = Default::default();

    let grid = 40i32;
    let spacing = 2.5f32;
    let half = grid as f32 * spacing * 0.5;

    for row in 0..grid {
        for col in 0..grid {
            let x = col as f32 * spacing - half;
            let z = row as f32 * spacing - half;

            let angle = ((row * 31 + col * 17) % 628) as f32 * 0.01;
            let rot = Quat::from_rotation_y(angle);

            let scale_f = 0.8 + ((row * 7 + col * 13) % 40) as f32 * 0.01;
            let scale = Vec3::splat(scale_f);

            let variant = ((row + col) as usize) % 8;
            per_variant[variant]
                .push(FoliageInstance::new(Vec3::new(x, 0.0, z), rot, scale, variant as u32));
        }
    }

    per_variant
}

// ---------------------------------------------------------------------------
// Update: rebuild meshes from staging queue
// ---------------------------------------------------------------------------

/// Each frame, process `FoliageStagingQueue` and rebuild GPU meshes.
///
/// Only rebuilds the (LOD, variant) meshes that have new batches; other
/// entities are untouched.  After processing, the queue is cleared.
pub fn update_foliage_meshes(
    handles: Res<FoliageMeshHandles>,
    mut staging_queue: ResMut<FoliageStagingQueue>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
) {
    if staging_queue.batches.is_empty() {
        return;
    }

    // Group batches by (lod, variant_id)
    let mut pending: std::collections::HashMap<(u8, u8), Vec<FoliageInstance>> =
        std::collections::HashMap::new();
    for batch in &staging_queue.batches {
        pending
            .entry((batch.lod as u8, batch.variant_id))
            .or_default()
            .extend_from_slice(&batch.instances);
    }
    staging_queue.clear();

    for ((lod_idx, variant_id), instances) in pending {
        let lod = match lod_idx {
            0 => FoliageLodTier::Lod0,
            1 => FoliageLodTier::Lod1,
            _ => FoliageLodTier::Lod2,
        };

        let Some(mesh_handle) = handles.get_mesh_handle(lod, variant_id) else {
            continue;
        };
        let Some(blade_data) = handles.base_blade_meshes.get(variant_id as usize) else {
            continue;
        };
        let Some(mesh) = mesh_assets.get_mut(mesh_handle) else {
            continue;
        };

        build_instanced_mesh(mesh, blade_data, &instances);
    }
}

// ---------------------------------------------------------------------------
// LOD visibility
// ---------------------------------------------------------------------------

/// Show only the LOD tier appropriate for the current camera distance.
///
/// LOD0: camera within lod0_distance
/// LOD1: lod0_distance – lod1_distance
/// LOD2: beyond lod1_distance
pub fn update_foliage_lod_visibility(
    foliage_config: Res<FoliageConfigResource>,
    view_state: Res<FoliageViewState>,
    mut query: Query<(&FoliageVariantComponent, &mut Visibility)>,
) {
    let (lod0_dist, lod1_dist) = match &foliage_config.0 {
        Some(c) => (c.lod0_distance, c.lod1_distance),
        None => (50.0f32, 200.0f32),
    };

    let cam = view_state.camera_pos_ws;
    let dist_xz = Vec2::new(cam.x, cam.z).length();

    let active_lod = if dist_xz < lod0_dist {
        FoliageLodTier::Lod0
    } else if dist_xz < lod1_dist {
        FoliageLodTier::Lod1
    } else {
        FoliageLodTier::Lod2
    };

    for (comp, mut vis) in &mut query {
        *vis = if comp.lod == active_lod {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

// ---------------------------------------------------------------------------
// Mesh builder: bake N blade instances into one world-space mesh
// ---------------------------------------------------------------------------

/// Rebuild `mesh` in-place with all blade instances baked into world-space.
///
/// For each `FoliageInstance`, every blade vertex is transformed by
/// `(position, rotation, scale)` → world position and normal.
pub fn build_instanced_mesh(
    mesh: &mut Mesh,
    blade: &BladeMeshData,
    instances: &[FoliageInstance],
) {
    if instances.is_empty() || blade.positions.is_empty() {
        // Clear the mesh
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
        mesh.insert_indices(bevy::mesh::Indices::U32(vec![]));
        return;
    }

    let verts_per_blade = blade.positions.len();
    let indices_per_blade = blade.indices.len();
    let total_verts = verts_per_blade * instances.len();
    let total_indices = indices_per_blade * instances.len();

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(total_verts);
    let mut indices: Vec<u32> = Vec::with_capacity(total_indices);

    for (inst_idx, inst) in instances.iter().enumerate() {
        let transform = Transform {
            translation: inst.position,
            rotation: inst.rotation,
            scale: inst.scale,
        };
        let mat = transform.to_matrix();
        // For uniform scales the normal matrix == the rotation matrix.
        // For non-uniform scales use the inverse-transpose of the upper 3×3.
        let normal_mat = Mat3::from_mat4(mat.inverse().transpose());

        // Transform each blade vertex
        let base_idx = (inst_idx * verts_per_blade) as u32;
        for i in 0..verts_per_blade {
            let local_pos = Vec3::from(blade.positions[i]);
            let world_pos = mat.transform_point3(local_pos);
            positions.push(world_pos.into());

            let local_norm = Vec3::from(blade.normals[i]);
            let world_norm = (normal_mat * local_norm).normalize_or_zero();
            normals.push(world_norm.into());

            uvs.push(blade.uvs[i]);
        }

        // Offset indices for this instance
        for &idx in &blade.indices {
            indices.push(base_idx + idx);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blade() -> BladeMeshData {
        BladeMeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn test_build_instanced_mesh_empty() {
        let blade = make_blade();
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        build_instanced_mesh(&mut mesh, &blade, &[]);

        // Should have empty attributes
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(v)) => assert!(v.is_empty()),
            _ => {}
        }
    }

    #[test]
    fn test_build_instanced_mesh_single_instance() {
        let blade = make_blade();
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        let instance = FoliageInstance::new(
            Vec3::new(10.0, 0.0, 5.0),
            Quat::IDENTITY,
            Vec3::ONE,
            0,
        );
        build_instanced_mesh(&mut mesh, &blade, &[instance]);

        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(v)) => {
                assert_eq!(v.len(), 3); // 3 blade vertices
                // First vertex should be at position + (0,0,0) = (10, 0, 5)
                assert!((v[0][0] - 10.0).abs() < 1e-5);
                assert!((v[0][1] - 0.0).abs() < 1e-5);
                assert!((v[0][2] - 5.0).abs() < 1e-5);
            }
            _ => panic!("Expected Float32x3 positions"),
        }

        match mesh.indices() {
            Some(bevy::mesh::Indices::U32(idx)) => assert_eq!(idx.len(), 3),
            _ => panic!("Expected u32 indices"),
        }
    }

    #[test]
    fn test_build_instanced_mesh_two_instances() {
        let blade = make_blade();
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        let instances = vec![
            FoliageInstance::new(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE, 0),
            FoliageInstance::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE, 1),
        ];
        build_instanced_mesh(&mut mesh, &blade, &instances);

        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(v)) => assert_eq!(v.len(), 6),
            _ => panic!("Expected Float32x3 positions"),
        }
        match mesh.indices() {
            Some(bevy::mesh::Indices::U32(idx)) => {
                assert_eq!(idx.len(), 6);
                // Second instance indices should be offset by 3
                assert_eq!(idx[3], 3);
                assert_eq!(idx[4], 4);
                assert_eq!(idx[5], 5);
            }
            _ => panic!("Expected u32 indices"),
        }
    }

    #[test]
    fn test_foliage_mesh_handles_entity_idx() {
        assert_eq!(FoliageMeshHandles::entity_idx(FoliageLodTier::Lod0, 0), 0);
        assert_eq!(FoliageMeshHandles::entity_idx(FoliageLodTier::Lod0, 7), 7);
        assert_eq!(FoliageMeshHandles::entity_idx(FoliageLodTier::Lod1, 0), 8);
        assert_eq!(FoliageMeshHandles::entity_idx(FoliageLodTier::Lod2, 7), 23);
    }
}

use avian3d::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    mesh::{Indices, PrimitiveTopology},
    pbr::wireframe::{Wireframe, WireframeColor},
    prelude::*,
};
use std::collections::HashMap;

use crate::terrain::{
    collision::TerrainCollisionCache,
    config::TerrainConfig,
    resources::{TerrainViewState, TileKey},
    streamer::load_tile_data,
    world_desc::TerrainSourceDesc,
};

type FallbackTileCache = HashMap<(i32, i32), Vec<f32>>;

/// Marker for the local sliding heightfield that tracks the camera.
#[derive(Component)]
pub struct LocalTerrainCollider;

/// Marker for the debug wireframe mesh that mirrors the physics trimesh.
#[derive(Component)]
pub struct TerrainCollisionDebugMesh;

/// When `true`, a green wireframe debug mesh is rendered on top of the
/// physics trimesh so you can visually verify collision accuracy.
#[derive(Resource, Default)]
pub struct ShowTerrainCollision(pub bool);

/// Tracks the center of the last local heightfield build so we can skip
/// frames where the camera hasn't moved far enough to warrant a rebuild.
#[derive(Resource)]
pub struct LocalColliderState {
    last_center: Vec2,
}

impl Default for LocalColliderState {
    fn default() -> Self {
        Self {
            last_center: Vec2::splat(f32::MAX / 2.0),
        }
    }
}

impl LocalColliderState {
    /// Force a rebuild on the next `update_local_terrain_collider` call.
    pub fn force_rebuild(&mut self) {
        self.last_center = Vec2::splat(f32::MAX / 2.0);
    }
}

// ---------------------------------------------------------------------------
// Local sliding terrain trimesh — exact match to the rendered geometry
// ---------------------------------------------------------------------------

/// Maintains a terrain trimesh collider centred on the camera.
///
/// Built at full LOD-0 resolution (`world_scale` metres per cell) covering a
/// ~256 m square.  Uses `Collider::trimesh` with the **same quad winding as
/// the render mesh** (triangle 1: i00→i01→i10; triangle 2: i10→i01→i11)
/// so physics contact normals and heights are identical to what is rendered.
///
/// `Collider::heightfield` uses Parry's internal diagonal (NW→SE), which is
/// opposite to the render mesh (SW→NE).  On saddle-shaped cells the two
/// surfaces differ by up to half the cell height range — metres of error on
/// steep terrain, causing objects to hover or sink.
///
/// Rebuilt whenever the camera moves more than 32 m from the last centre.
/// Cache misses are filled synchronously from disk to avoid height=0 holes.
pub fn update_local_terrain_collider(
    mut state: ResMut<LocalColliderState>,
    view: Res<TerrainViewState>,
    cache: Res<TerrainCollisionCache>,
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    collider_q: Query<Entity, With<LocalTerrainCollider>>,
    debug_q: Query<Entity, With<TerrainCollisionDebugMesh>>,
    show_debug: Res<ShowTerrainCollision>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    if cache.tile_size == 0 {
        return;
    }

    let camera_xz = view.camera_pos_ws.xz();
    let cell_size = config.world_scale;

    const COVERAGE: f32 = 256.0;
    const REBUILD_DIST: f32 = COVERAGE * 0.125; // 32 m
    // Compute n as the smallest ODD vertex count that spans at least COVERAGE.
    // Odd n guarantees half_n = (n-1)/2 is a whole integer, so every vertex
    // lands at an exact integer multiple of cell_size — the same grid used by
    // the render mesh.  Even n (e.g. n=10 for world_scale=30) puts all vertices
    // at half-integer positions, making ridges appear shifted by 0.5 cells.
    let n_half = (COVERAGE * 0.5 / cell_size).ceil() as usize;
    let n = (2 * n_half + 1).min(257).max(9);

    if (camera_xz - state.last_center).length() <= REBUILD_DIST {
        // If debug visibility changed without a position change, still
        // rebuild so the debug mesh appears/disappears immediately.
        let needs_debug_spawn = show_debug.0 && debug_q.is_empty();
        let needs_debug_despawn = !show_debug.0 && !debug_q.is_empty();
        if !needs_debug_spawn && !needs_debug_despawn {
            return;
        }
    }

    let cx = (camera_xz.x / cell_size).round() * cell_size;
    let cz = (camera_xz.y / cell_size).round() * cell_size;
    let center = Vec2::new(cx, cz);
    let half_n = (n as f32 - 1.0) * 0.5;

    let tile_world_size = config.tile_size as f32 * cell_size;
    let mut fallback: FallbackTileCache = HashMap::default();

    // Build a flat vertex list: vertices[zi * n + xi]  (row-major, Z outer).
    let mut vertices: Vec<Vec3> = Vec::with_capacity(n * n);
    for zi in 0..n {
        for xi in 0..n {
            let wx = center.x + (xi as f32 - half_n) * cell_size;
            let wz = center.y + (zi as f32 - half_n) * cell_size;
            let h = cache
                .sample_height(Vec2::new(wx, wz))
                .unwrap_or_else(|| {
                    let tx = (wx / tile_world_size).floor() as i32;
                    let tz = (wz / tile_world_size).floor() as i32;
                    let data = fallback.entry((tx, tz)).or_insert_with(|| {
                        load_tile_data(
                            TileKey { level: 0, x: tx, y: tz },
                            config.tile_size,
                            config.world_scale,
                            1.0,
                            desc.max_mip_level,
                            desc.tile_root.as_deref(),
                            None,
                            Some((desc.world_min, desc.world_max)),
                            false,
                        )
                        .data
                    });
                    let lx = ((wx - tx as f32 * tile_world_size) / cell_size)
                        .round()
                        .clamp(0.0, (config.tile_size - 1) as f32)
                        as usize;
                    let lz = ((wz - tz as f32 * tile_world_size) / cell_size)
                        .round()
                        .clamp(0.0, (config.tile_size - 1) as f32)
                        as usize;
                    data[lz * config.tile_size as usize + lx] * config.height_scale
                });
            vertices.push(Vec3::new(wx, h, wz));
        }
    }

    // Triangulate with the same winding as build_rect_mesh so physics surfaces
    // match the rendered geometry exactly.
    // Quad (xi, zi):  i00=(zi*n+xi)  i10=(zi*n+xi+1)
    //                 i01=((zi+1)*n+xi)  i11=((zi+1)*n+xi+1)
    // Tri 1: i00, i01, i10   Tri 2: i10, i01, i11
    let quads = (n - 1) * (n - 1);
    let mut indices: Vec<[u32; 3]> = Vec::with_capacity(quads * 2);
    for zi in 0..(n - 1) {
        for xi in 0..(n - 1) {
            let i00 = (zi * n + xi) as u32;
            let i10 = (zi * n + xi + 1) as u32;
            let i01 = ((zi + 1) * n + xi) as u32;
            let i11 = ((zi + 1) * n + xi + 1) as u32;
            indices.push([i00, i01, i10]);
            indices.push([i10, i01, i11]);
        }
    }

    for entity in collider_q.iter() {
        commands.entity(entity).despawn();
    }
    for entity in debug_q.iter() {
        commands.entity(entity).despawn();
    }

    commands.spawn((
        LocalTerrainCollider,
        RigidBody::Static,
        Collider::trimesh(vertices.clone(), indices.clone()),
        Transform::IDENTITY,
    ));

    if show_debug.0 {
        // Build a Bevy Mesh from the same vertices (+tiny Y offset to prevent z-fighting).
        let debug_verts: Vec<[f32; 3]> = vertices
            .iter()
            .map(|v| [v.x, v.y + 0.05, v.z])
            .collect();
        let flat_indices: Vec<u32> = indices.iter().flat_map(|tri| tri.iter().copied()).collect();

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, debug_verts);
        // Normals aren't needed for wireframe-only rendering, but Bevy requires them.
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            vec![[0.0_f32, 1.0, 0.0]; n * n],
        );
        mesh.insert_indices(Indices::U32(flat_indices));

        let mat = materials.add(StandardMaterial {
            base_color: Color::srgba(0.0, 1.0, 0.0, 0.0),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            cull_mode: None,
            ..default()
        });

        commands.spawn((
            TerrainCollisionDebugMesh,
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(mat),
            Transform::IDENTITY,
            Wireframe,
            WireframeColor {
                color: Color::srgb(0.0, 1.0, 0.2),
            },
            NoFrustumCulling,
        ));
    }

    let fallback_tiles = fallback.len();
    if fallback_tiles > 0 {
        info!(
            "[Terrain] Local trimesh rebuilt: {}×{} @ {:.1} m/cell, centre ({:.0},{:.0}) [{} tile(s) from disk]",
            n, n, cell_size, center.x, center.y, fallback_tiles
        );
    } else {
        info!(
            "[Terrain] Local trimesh rebuilt: {}×{} @ {:.1} m/cell, centre ({:.0},{:.0})",
            n, n, cell_size, center.x, center.y
        );
    }

    state.last_center = center;
}

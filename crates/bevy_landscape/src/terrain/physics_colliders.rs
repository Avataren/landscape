use avian3d::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    mesh::{Indices, PrimitiveTopology},
    pbr::wireframe::{Wireframe, WireframeColor},
    prelude::*,
};
use std::sync::mpsc::{self, Receiver};

use crate::terrain::{
    collision::TerrainCollisionCache,
    config::TerrainConfig,
    resources::{TerrainViewState, TileKey},
    streamer::load_tile_data,
    world_desc::TerrainSourceDesc,
    ReloadTerrainRequest,
};

// ---------------------------------------------------------------------------
// Public components / resources
// ---------------------------------------------------------------------------

/// Marker for the local sliding trimesh collider that tracks the camera.
#[derive(Component)]
pub struct LocalTerrainCollider;

/// Marker for the debug wireframe mesh that mirrors the physics trimesh.
#[derive(Component)]
pub struct TerrainCollisionDebugMesh;

/// When `true`, a green wireframe overlay is rendered on top of the physics
/// trimesh so you can visually verify collision accuracy.  Toggle with F3.
#[derive(Resource, Default)]
pub struct ShowTerrainCollision(pub bool);

/// Tracks the center of the last completed or in-flight collider build.
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
    /// Force a rebuild on the next `start_terrain_collider_build` call.
    pub fn force_rebuild(&mut self) {
        self.last_center = Vec2::splat(f32::MAX / 2.0);
    }
}

/// Holds the channel receiver for an in-flight background mesh build.
/// When `receiver` is `Some`, a build is in progress.
#[derive(Default)]
pub struct LocalColliderTask {
    receiver: Option<Receiver<ColliderBuildResult>>,
}

// SAFETY: LocalColliderTask is only accessed exclusively via ResMut (single owner at a time).
// Receiver<T> is Send but not Sync; wrapping it here is safe because no shared reference
// ever crosses thread boundaries.
unsafe impl Sync for LocalColliderTask {}
impl Resource for LocalColliderTask {}

// ---------------------------------------------------------------------------
// Internal thread-communication types
// ---------------------------------------------------------------------------

struct ColliderBuildResult {
    center: Vec2,
    n: usize,
    cell_size: f32,
    vertices: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
    fallback_tile_count: usize,
}

/// All data needed by the background thread — no Bevy types, fully `Send`.
struct BuildInput {
    center: Vec2,
    n: usize,
    cell_size: f32,
    height_scale: f32,
    tile_size: u32,
    world_scale: f32,
    /// Snapshotted level-0 tiles keyed by (tile_x, tile_z).
    tiles: std::collections::HashMap<TileKey, Vec<f32>>,
    tile_root: Option<std::path::PathBuf>,
    max_mip_level: u8,
    world_min: Vec2,
    world_max: Vec2,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COVERAGE: f32 = 256.0;
const REBUILD_DIST: f32 = COVERAGE * 0.125; // 32 m

// ---------------------------------------------------------------------------
// System: decide whether to start a new background build
// ---------------------------------------------------------------------------

/// Checks whether the terrain trimesh needs rebuilding (camera moved >32 m
/// from last build center) and, if so, snapshots the required tile heights
/// from the collision cache and spawns a background thread to compute the
/// mesh.  Returns immediately so the main thread is never blocked.
pub fn start_terrain_collider_build(
    mut state: ResMut<LocalColliderState>,
    mut task: ResMut<LocalColliderTask>,
    view: Res<TerrainViewState>,
    cache: Res<TerrainCollisionCache>,
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
) {
    if task.receiver.is_some() || cache.tile_size == 0 {
        return;
    }

    let camera_xz = view.camera_pos_ws.xz();
    let cell_size = config.world_scale;

    // Odd vertex count so half_n is a whole integer and vertices land on the
    // same grid as the render mesh (even n → half-integer offsets → ridges
    // appear shifted by 0.5 cells relative to the rendered terrain).
    let n_half = (COVERAGE * 0.5 / cell_size).ceil() as usize;
    let n = (2 * n_half + 1).min(257).max(9);

    if (camera_xz - state.last_center).length() <= REBUILD_DIST {
        return;
    }

    let cx = (camera_xz.x / cell_size).round() * cell_size;
    let cz = (camera_xz.y / cell_size).round() * cell_size;
    let center = Vec2::new(cx, cz);

    // Snapshot only the tiles that overlap this coverage area.
    let half = (n as f32 - 1.0) * 0.5 * cell_size;
    let tiles = cache.snapshot_tiles_for_region(center - Vec2::splat(half), center + Vec2::splat(half));

    let input = BuildInput {
        center,
        n,
        cell_size,
        height_scale: config.height_scale,
        tile_size: config.tile_size,
        world_scale: config.world_scale,
        tiles,
        tile_root: desc.tile_root.clone(),
        max_mip_level: desc.max_mip_level,
        world_min: desc.world_min,
        world_max: desc.world_max,
    };

    let (tx, rx) = mpsc::channel();
    task.receiver = Some(rx);
    // Mark as in-progress so we don't spawn a second build for the same center.
    state.last_center = center;

    std::thread::spawn(move || {
        let result = build_mesh_data(input);
        let _ = tx.send(result); // send fails silently if receiver was dropped
    });
}

// ---------------------------------------------------------------------------
// System: apply completed build results on the main thread
// ---------------------------------------------------------------------------

/// Polls the in-flight build channel.  When a result arrives, despawns the
/// old collider and debug mesh, spawns the new ones, and logs the build stats.
pub fn apply_terrain_collider_result(
    mut task: ResMut<LocalColliderTask>,
    show_debug: Res<ShowTerrainCollision>,
    collider_q: Query<Entity, With<LocalTerrainCollider>>,
    debug_q: Query<Entity, With<TerrainCollisionDebugMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    let Some(ref rx) = task.receiver else { return };

    let result = match rx.try_recv() {
        Ok(r) => r,
        Err(mpsc::TryRecvError::Empty) => return,
        Err(mpsc::TryRecvError::Disconnected) => {
            task.receiver = None;
            return;
        }
    };
    task.receiver = None;

    for entity in collider_q.iter() {
        commands.entity(entity).despawn();
    }
    for entity in debug_q.iter() {
        commands.entity(entity).despawn();
    }

    commands.spawn((
        LocalTerrainCollider,
        RigidBody::Static,
        Collider::trimesh(result.vertices.clone(), result.indices.clone()),
        Transform::IDENTITY,
    ));

    if show_debug.0 {
        spawn_debug_mesh(&result, &mut meshes, &mut materials, &mut commands);
    }

    let n = result.n;
    let cs = result.cell_size;
    let c = result.center;
    if result.fallback_tile_count > 0 {
        info!(
            "[Terrain] trimesh rebuilt: {}×{} @ {:.1} m/cell, centre ({:.0},{:.0}) [{} tile(s) from disk]",
            n, n, cs, c.x, c.y, result.fallback_tile_count
        );
    } else {
        info!(
            "[Terrain] trimesh rebuilt: {}×{} @ {:.1} m/cell, centre ({:.0},{:.0})",
            n, n, cs, c.x, c.y
        );
    }
}

// ---------------------------------------------------------------------------
// System: cancel an in-flight build when the terrain is hot-reloaded
// ---------------------------------------------------------------------------

/// Drops the in-flight build receiver on terrain reload so the stale result
/// is never applied after the new terrain data arrives.
pub fn cancel_collider_task_on_reload(
    mut reload_rx: MessageReader<ReloadTerrainRequest>,
    mut task: ResMut<LocalColliderTask>,
) {
    if reload_rx.read().count() > 0 {
        task.receiver = None;
    }
}

// ---------------------------------------------------------------------------
// Background thread: build vertices and indices from snapshotted tile data
// ---------------------------------------------------------------------------

fn build_mesh_data(input: BuildInput) -> ColliderBuildResult {
    let BuildInput {
        center, n, cell_size, height_scale,
        tile_size, world_scale, mut tiles,
        tile_root, max_mip_level, world_min, world_max,
    } = input;

    let half_n = (n as f32 - 1.0) * 0.5;
    let tile_world_size = tile_size as f32 * world_scale;
    let mut fallback_tile_count = 0usize;

    let mut vertices: Vec<Vec3> = Vec::with_capacity(n * n);
    for zi in 0..n {
        for xi in 0..n {
            let wx = center.x + (xi as f32 - half_n) * cell_size;
            let wz = center.y + (zi as f32 - half_n) * cell_size;

            let tx = (wx / tile_world_size).floor() as i32;
            let tz = (wz / tile_world_size).floor() as i32;
            let local_x = (wx - tx as f32 * tile_world_size) / tile_world_size;
            let local_z = (wz - tz as f32 * tile_world_size) / tile_world_size;
            let key = TileKey { level: 0, x: tx, y: tz };

            // Load from disk only if the snapshot didn't cover this tile.
            if !tiles.contains_key(&key) {
                fallback_tile_count += 1;
                let data = load_tile_data(
                    key,
                    tile_size,
                    world_scale,
                    1.0,
                    max_mip_level,
                    tile_root.as_deref(),
                    None,
                    Some((world_min, world_max)),
                    false,
                )
                .data;
                tiles.insert(key, data);
            }

            let h = tiles
                .get(&key)
                .map(|data| bilinear_sample(data, tile_size, Vec2::new(local_x, local_z)) * height_scale)
                .unwrap_or(0.0);

            vertices.push(Vec3::new(wx, h, wz));
        }
    }

    // Same winding as build_rect_mesh so physics normals match rendered geometry.
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

    ColliderBuildResult { center, n, cell_size, vertices, indices, fallback_tile_count }
}

fn bilinear_sample(data: &[f32], size: u32, uv: Vec2) -> f32 {
    let u = uv.x.clamp(0.0, 1.0) * size as f32;
    let v = uv.y.clamp(0.0, 1.0) * size as f32;
    let x0 = (u.floor() as usize).min(size as usize - 1);
    let y0 = (v.floor() as usize).min(size as usize - 1);
    let x1 = (x0 + 1).min(size as usize - 1);
    let y1 = (y0 + 1).min(size as usize - 1);
    let fx = u.fract();
    let fy = v.fract();
    let h00 = data[y0 * size as usize + x0];
    let h10 = data[y0 * size as usize + x1];
    let h01 = data[y1 * size as usize + x0];
    let h11 = data[y1 * size as usize + x1];
    let top = h00 * (1.0 - fx) + h10 * fx;
    let bot = h01 * (1.0 - fx) + h11 * fx;
    top * (1.0 - fy) + bot * fy
}

// ---------------------------------------------------------------------------
// Debug mesh (main thread only — needs Assets<Mesh>)
// ---------------------------------------------------------------------------

fn spawn_debug_mesh(
    result: &ColliderBuildResult,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    commands: &mut Commands,
) {
    let n = result.n;
    let debug_verts: Vec<[f32; 3]> = result
        .vertices
        .iter()
        .map(|v| [v.x, v.y + 0.05, v.z])
        .collect();
    let flat_indices: Vec<u32> = result
        .indices
        .iter()
        .flat_map(|tri| tri.iter().copied())
        .collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, debug_verts);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0_f32, 1.0, 0.0]; n * n]);
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
        WireframeColor { color: Color::srgb(0.0, 1.0, 0.2) },
        NoFrustumCulling,
    ));
}

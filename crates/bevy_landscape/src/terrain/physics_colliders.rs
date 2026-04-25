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
    detail_synthesis::DetailSynthesisConfig,
    resources::TerrainViewState,
    source_heightmap::SourceHeightmapState,
    synthesis_cpu::{octave_count_for_cell, synthesise_residual},
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
    /// Snapshot of the composite source heightmap, R16Unorm decoded to [0,1].
    /// Same data the GPU detail-synthesis pass samples for `base_h`, so the
    /// CPU and GPU agree to within float-rounding error.
    source_heightmap: Vec<f32>,
    source_width: u32,
    source_height: u32,
    source_origin: Vec2,
    source_extent: Vec2,
    /// Synth params snapshotted from the live config so the background thread
    /// reproduces the exact heights the GPU pass writes to the clipmap.
    synthesis_config: DetailSynthesisConfig,
    /// World-space texel size of the source heightmap (= world_scale × 2^max_mip).
    source_spacing: f32,
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
    synthesis: Res<DetailSynthesisConfig>,
    source_state: Option<Res<SourceHeightmapState>>,
    images: Res<Assets<Image>>,
) {
    if task.receiver.is_some() || cache.tile_size == 0 {
        return;
    }
    let Some(source_state) = source_state else {
        return;
    };
    let Some(source_image) = images.get(&source_state.handle) else {
        return;
    };
    let Some(source_bytes) = source_image.data.as_ref() else {
        return;
    };

    let camera_xz = view.camera_pos_ws.xz();
    // Match the finest visual mesh spacing so collision captures the same
    // detail the GPU synthesis pass writes into LOD 0.
    let cell_size = config.lod0_mesh_spacing.max(config.world_scale * 0.0625);

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

    // Drain the cached snapshot once just to keep eviction consistent — the
    // collision builder no longer reads per-tile data, but this preserves the
    // existing LRU semantics.
    let half = (n as f32 - 1.0) * 0.5 * cell_size;
    let _ = cache.snapshot_tiles_for_region(center - Vec2::splat(half), center + Vec2::splat(half));

    let source_spacing = config.world_scale * (1u32 << desc.max_mip_level as u32) as f32;

    // Decode the R16Unorm composite to [0,1] floats once, on the main thread.
    let source_size = source_image.texture_descriptor.size;
    let source_width = source_size.width;
    let source_height = source_size.height;
    let mut source_heightmap = Vec::with_capacity((source_width * source_height) as usize);
    for chunk in source_bytes.chunks_exact(2) {
        let v = u16::from_le_bytes([chunk[0], chunk[1]]);
        source_heightmap.push(v as f32 / 65535.0);
    }

    let input = BuildInput {
        center,
        n,
        cell_size,
        height_scale: config.height_scale,
        source_heightmap,
        source_width,
        source_height,
        source_origin: source_state.world_origin,
        source_extent: source_state.world_extent,
        synthesis_config: synthesis.clone(),
        source_spacing,
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
        center,
        n,
        cell_size,
        height_scale,
        source_heightmap,
        source_width,
        source_height,
        source_origin,
        source_extent,
        synthesis_config,
        source_spacing,
    } = input;

    let half_n = (n as f32 - 1.0) * 0.5;
    let octaves = octave_count_for_cell(cell_size, source_spacing, synthesis_config.lacunarity);

    let sw = source_width as i32;
    let sh = source_height as i32;
    let stride = source_width as usize;
    let tex = |x: i32, y: i32| -> f32 {
        let xi = x.clamp(0, sw - 1) as usize;
        let yi = y.clamp(0, sh - 1) as usize;
        source_heightmap[yi * stride + xi]
    };
    // Mirrors `textureSampleLevel(source_heightmap, source_samp, src_uv, 0.0)`
    // with clamp-to-edge wrap and a linear filter: texel I's centre lives at
    // uv = (I + 0.5) / size, and bilinear taps come from the four texels
    // around (uv * size - 0.5).
    let sample_norm = |wx: f32, wz: f32| -> f32 {
        let u = ((wx - source_origin.x) / source_extent.x).clamp(0.0, 1.0);
        let v = ((wz - source_origin.y) / source_extent.y).clamp(0.0, 1.0);
        let cx = u * source_width as f32 - 0.5;
        let cy = v * source_height as f32 - 0.5;
        let x0 = cx.floor() as i32;
        let y0 = cy.floor() as i32;
        let fx = cx - x0 as f32;
        let fy = cy - y0 as f32;
        let h00 = tex(x0, y0);
        let h10 = tex(x0 + 1, y0);
        let h01 = tex(x0, y0 + 1);
        let h11 = tex(x0 + 1, y0 + 1);
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        top + (bot - top) * fy
    };

    let mut vertices: Vec<Vec3> = Vec::with_capacity(n * n);
    for zi in 0..n {
        for xi in 0..n {
            let wx = center.x + (xi as f32 - half_n) * cell_size;
            let wz = center.y + (zi as f32 - half_n) * cell_size;

            let h_norm = sample_norm(wx, wz);
            let base_h = h_norm * height_scale;

            // Slope (deg) from one-source-texel finite differences — same
            // formula as the slope_mask in detail_synthesis.wgsl.
            let h_norm_x = sample_norm(wx + source_spacing, wz);
            let h_norm_z = sample_norm(wx, wz + source_spacing);
            let dh_dx = (h_norm_x - h_norm) / source_spacing * height_scale;
            let dh_dz = (h_norm_z - h_norm) / source_spacing * height_scale;
            let slope_deg = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt().atan().to_degrees();

            let residual = synthesise_residual(
                Vec2::new(wx, wz),
                octaves,
                slope_deg,
                &synthesis_config,
                source_spacing,
            );

            vertices.push(Vec3::new(wx, base_h + residual, wz));
        }
    }
    let fallback_tile_count = 0usize;

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

    ColliderBuildResult {
        center,
        n,
        cell_size,
        vertices,
        indices,
        fallback_tile_count,
    }
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

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
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
        WireframeColor {
            color: Color::srgb(0.0, 1.0, 0.2),
        },
        NoFrustumCulling,
    ));
}

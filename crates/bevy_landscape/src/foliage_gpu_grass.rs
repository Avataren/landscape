//! GPU-driven grass rendering with two LOD passes.
//!
//! # Two-pass design
//! - **Near** pass: dense spacing, covers a small radius (~80 m default).
//!   Looks good up close; individual blades are clearly visible.
//! - **Far** pass: coarse spacing, covers a large radius (~350 m default).
//!   Maintains visual coverage at distance. Inner blades (inside the near
//!   radius) are culled so the two passes don't overlap.
//!
//! Both meshes are pre-allocated at `GRASS_MAX_GRID`² vertices so the active
//! grid size can be changed at runtime without respawning.

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    mesh::{MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType,
        SpecializedMeshPipelineError, TextureDimension, TextureFormat, TextureViewDescriptor,
        TextureViewDimension,
    },
    shader::ShaderRef,
};

use crate::terrain::{
    clipmap_texture::{compute_clip_levels, TerrainClipmapState},
    config::TerrainConfig,
    math::level_scale,
    source_heightmap::SourceHeightmapState,
    components::TerrainCamera,
};

// ── Public config (main-world resource) ──────────────────────────────────────

/// Tweakable GPU grass parameters — live-edited without restarting.
#[derive(Resource, Clone, Debug)]
pub struct GpuGrassConfig {
    /// Master enable/disable for both LOD passes.
    pub enabled: bool,

    // ── Near LOD (dense, short range) ────────────────────────────────────────
    /// Blade spacing in metres for the near (dense) pass.
    pub near_spacing: f32,
    /// Coverage radius in metres for the near pass.
    pub near_range: f32,

    // ── Far LOD (sparse, long range) ─────────────────────────────────────────
    /// Blade spacing in metres for the far (sparse) pass.
    pub far_spacing: f32,
    /// Coverage radius in metres for the far pass.
    pub far_range: f32,

    // ── Shared appearance ─────────────────────────────────────────────────────
    /// Blade height in world-space metres.
    pub blade_height: f32,
    /// Blade width at the base in world-space metres.
    pub blade_width: f32,
    /// Rise/run slope ratio above which grass is suppressed (> 90 = disabled).
    pub slope_max: f32,
    /// Minimum world-Y (metres) for grass.
    pub altitude_min: f32,
    /// Maximum world-Y (metres) for grass.
    pub altitude_max: f32,
    /// Wind sway amplitude (metres at blade tip).
    pub wind_strength: f32,
    /// Wind spatial frequency (1/metres).
    pub wind_scale: f32,
    /// Base blade colour.
    pub base_color: LinearRgba,
}

impl GpuGrassConfig {
    /// Compute active grid side for the near pass (capped at GRASS_MAX_GRID).
    pub fn near_grid_size(&self) -> u32 {
        let g = ((self.near_range * 2.0) / self.near_spacing).round() as u32;
        g.clamp(4, GRASS_MAX_GRID)
    }
    /// Compute active grid side for the far pass (capped at GRASS_MAX_GRID).
    pub fn far_grid_size(&self) -> u32 {
        let g = ((self.far_range * 2.0) / self.far_spacing).round() as u32;
        g.clamp(4, GRASS_MAX_GRID)
    }
}

impl Default for GpuGrassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            near_spacing: 0.7,
            near_range: 80.0,
            far_spacing: 3.5,
            far_range: 350.0,
            blade_height: 0.7,
            blade_width: 0.12,
            slope_max: 0.7,
            altitude_min: -100_000.0,
            altitude_max: 100_000.0,
            wind_strength: 0.15,
            wind_scale: 0.035,
            base_color: LinearRgba::rgb(0.19, 0.42, 0.09),
        }
    }
}

// ── GPU uniform struct (must match GrassParams in grass_blade.wgsl) ──────────

/// 6 × vec4 = 96 bytes, all 4-byte aligned.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct GrassParamsGpu {
    /// xy = camera XZ, z = grid_size (f32), w = cell spacing (m)
    pub camera_grid: Vec4,
    /// LOD 0 clipmap level: xy = ring_center XZ, z = inv_ring_span, w = texel_world_size
    pub clip_level: Vec4,
    /// x = inner_radius_sq (cull blades closer than this), y = blade_height,
    /// z = blade_width, w = slope_max
    pub blade: Vec4,
    /// x = alt_min, y = alt_max, z = wind_time, w = wind_strength
    pub alt_wind: Vec4,
    /// x = wind_scale, yzw = base RGB colour
    pub wind_color: Vec4,
    /// xy = world_min XZ, zw = world_max XZ
    pub world_bounds: Vec4,
}

// ── Material ──────────────────────────────────────────────────────────────────

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GpuGrassMaterial {
    /// Clipmap height texture array (R32Float, world-space metres per texel).
    #[texture(0, dimension = "2d_array", visibility(vertex))]
    pub height_tex: Handle<Image>,
    #[uniform(1, visibility(vertex, fragment))]
    pub params: GrassParamsGpu,
}

impl Material for GpuGrassMaterial {
    fn vertex_shader() -> ShaderRef { "shaders/grass_blade.wgsl".into() }
    fn fragment_shader() -> ShaderRef { "shaders/grass_blade.wgsl".into() }
    fn alpha_mode(&self) -> AlphaMode { AlphaMode::Mask(0.1) }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

// ── Component markers ─────────────────────────────────────────────────────────

/// Marks the dense near-LOD grass entity.
#[derive(Component)]
pub struct GrassEntityNear;

/// Marks the sparse far-LOD grass entity.
#[derive(Component)]
pub struct GrassEntityFar;

// ── Constants ─────────────────────────────────────────────────────────────────

const VERTS_PER_BLADE: u32 = 12;

/// Maximum grid side length. Both LOD meshes are always allocated at this size
/// so range/density can be changed at runtime without respawning.
pub const GRASS_MAX_GRID: u32 = 707;

// ── Startup ───────────────────────────────────────────────────────────────────

fn spawn_grass_entities(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
    mut images: ResMut<Assets<Image>>,
    config: Res<GpuGrassConfig>,
) {
    let n_verts = (GRASS_MAX_GRID * GRASS_MAX_GRID * VERTS_PER_BLADE) as usize;
    let positions = vec![[0.0f32, 0.0, 0.0]; n_verts];

    // Both LODs share the same mesh shape — only params differ.
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    let mesh_handle = meshes.add(mesh);

    let fallback_tex = {
        let mut img = Image::new(
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            TextureDimension::D2,
            vec![0u8; 4],
            TextureFormat::R32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        img.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..default()
        });
        images.add(img)
    };

    let fallback_clip  = Vec4::new(0.0, 0.0, 1.0 / 1_000_000.0, 1.0);
    let fallback_world = Vec4::new(-500_000.0, -500_000.0, 500_000.0, 500_000.0);

    // Near entity — inner_radius_sq = 0 (no inner cull).
    let near_params = build_params(
        &config, config.near_grid_size(), config.near_spacing,
        0.0, Vec3::ZERO, 0.0, fallback_clip, fallback_world,
    );
    let near_mat = materials.add(GpuGrassMaterial {
        height_tex: fallback_tex.clone(),
        params: near_params,
    });
    commands.spawn((
        Mesh3d(mesh_handle.clone()),
        MeshMaterial3d(near_mat),
        Transform::default(),
        NoFrustumCulling,
        GrassEntityNear,
    ));

    // Far entity — inner_radius_sq culls blades inside the near radius.
    let inner_r_sq = config.near_range * config.near_range;
    let far_params = build_params(
        &config, config.far_grid_size(), config.far_spacing,
        inner_r_sq, Vec3::ZERO, 0.0, fallback_clip, fallback_world,
    );
    let far_mat = materials.add(GpuGrassMaterial {
        height_tex: fallback_tex,
        params: far_params,
    });
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(far_mat),
        Transform::default(),
        NoFrustumCulling,
        GrassEntityFar,
    ));
}

// ── Per-frame update ──────────────────────────────────────────────────────────

fn update_grass_materials(
    camera_q: Query<&Transform, With<TerrainCamera>>,
    source_state: Option<Res<SourceHeightmapState>>,
    terrain_config: Option<Res<TerrainConfig>>,
    clipmap_state: Option<Res<TerrainClipmapState>>,
    near_q: Query<&MeshMaterial3d<GpuGrassMaterial>, With<GrassEntityNear>>,
    far_q:  Query<&MeshMaterial3d<GpuGrassMaterial>, With<GrassEntityFar>>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
    config: Res<GpuGrassConfig>,
    time: Res<Time>,
) {
    let Ok(cam) = camera_q.single() else { return };

    let clip_level_0 =
        if let (Some(cs), Some(tc)) = (clipmap_state.as_deref(), terrain_config.as_deref()) {
            let scales: Vec<f32> = (0..tc.active_clipmap_levels())
                .map(|l| level_scale(tc.lod0_mesh_spacing, l))
                .collect();
            compute_clip_levels(tc, &cs.last_clip_centers, &scales)[0]
        } else {
            Vec4::new(0.0, 0.0, 1.0 / 1_000_000.0, 1.0)
        };

    let world_bounds = source_state.as_deref()
        .map(|s| {
            let max = s.world_origin + s.world_extent;
            Vec4::new(s.world_origin.x, s.world_origin.y, max.x, max.y)
        })
        .unwrap_or(Vec4::new(-500_000.0, -500_000.0, 500_000.0, 500_000.0));

    let wind_time = time.elapsed_secs();

    if let Ok(h) = near_q.single() {
        if let Some(mat) = materials.get_mut(&h.0) {
            mat.params = build_params(
                &config, config.near_grid_size(), config.near_spacing,
                0.0, cam.translation, wind_time, clip_level_0, world_bounds,
            );
            if let Some(cs) = clipmap_state.as_deref() {
                mat.height_tex = cs.height_texture_handle.clone();
            }
        }
    }

    if let Ok(h) = far_q.single() {
        if let Some(mat) = materials.get_mut(&h.0) {
            let inner_r_sq = config.near_range * config.near_range;
            mat.params = build_params(
                &config, config.far_grid_size(), config.far_spacing,
                inner_r_sq, cam.translation, wind_time, clip_level_0, world_bounds,
            );
            if let Some(cs) = clipmap_state.as_deref() {
                mat.height_tex = cs.height_texture_handle.clone();
            }
        }
    }
}

fn build_params(
    config: &GpuGrassConfig,
    grid_size: u32,
    spacing: f32,
    inner_radius_sq: f32,
    camera_pos: Vec3,
    wind_time: f32,
    clip_level_0: Vec4,
    world_bounds: Vec4,
) -> GrassParamsGpu {
    GrassParamsGpu {
        camera_grid: Vec4::new(camera_pos.x, camera_pos.z, grid_size as f32, spacing),
        clip_level: clip_level_0,
        blade: Vec4::new(inner_radius_sq, config.blade_height, config.blade_width, config.slope_max),
        alt_wind: Vec4::new(config.altitude_min, config.altitude_max, wind_time, config.wind_strength),
        wind_color: Vec4::new(
            config.wind_scale,
            config.base_color.red,
            config.base_color.green,
            config.base_color.blue,
        ),
        world_bounds,
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct GpuGrassPlugin;

impl Plugin for GpuGrassPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuGrassConfig>()
            .add_plugins(MaterialPlugin::<GpuGrassMaterial>::default())
            .add_systems(Startup, spawn_grass_entities)
            .add_systems(Update, update_grass_materials);
    }
}

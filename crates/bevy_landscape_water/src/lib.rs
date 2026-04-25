mod fft_ocean;
mod fft_ocean_compute;
mod material;

pub use fft_ocean::{OceanFftBuffers, OceanFftPlugin, OceanFftSettings};
pub use material::{StandardWaterMaterial, WaterMaterial, WaterMaterialPlugin};

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_landscape::{
    compute_clip_levels, compute_initial_clip_levels, TerrainCamera, TerrainClipmapState,
    TerrainConfig, TerrainMetadata, TerrainSourceDesc, TerrainViewState,
    MAX_SUPPORTED_CLIPMAP_LEVELS,
};

// ---------------------------------------------------------------------------
// Components (previously from bevy_water)
// ---------------------------------------------------------------------------

/// Marks the root entity that groups all water tile children.
#[derive(Component, Default)]
#[require(Transform, Visibility)]
pub struct WaterTiles;

/// Marks one water mesh tile.
#[derive(Component, Default)]
#[require(Mesh3d, MeshMaterial3d<StandardWaterMaterial>, Transform, Visibility)]
pub struct WaterTile {
    pub offset: Vec2,
}

/// Placeholder component for future wind-direction shader support.
#[derive(Component, Clone, Copy, Debug)]
pub struct WaveDirection {
    pub direction: Vec2,
}

impl Default for WaveDirection {
    fn default() -> Self {
        Self {
            direction: Vec2::new(1.0, 0.5).normalize(),
        }
    }
}

impl WaveDirection {
    pub fn new(direction: Vec2) -> Self {
        Self {
            direction: direction.normalize_or_zero(),
        }
    }
    pub fn with_duration(direction: Vec2, _blend_duration: f32) -> Self {
        Self::new(direction)
    }
}

// ---------------------------------------------------------------------------
// WaterSettings resource
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct WaterSettings {
    pub alpha_mode: AlphaMode,
    pub height: f32,
    pub amplitude: f32,
    pub base_color: Color,
    pub clarity: f32,
    pub deep_color: Color,
    pub shallow_color: Color,
    pub edge_scale: f32,
    pub edge_color: Color,
    pub update_materials: bool,
    /// Reserved for API compat — not used by LandscapeWaterPlugin.
    pub spawn_tiles: Option<UVec2>,
    pub wave_direction: Vec2,
    pub wave_direction_blend_duration: f32,
    pub wave_speed: f32,
    pub refraction_strength: f32,
    pub foam_threshold: f32,
    pub foam_color: Color,
    /// Maximum water depth (metres) that receives shoreline foam.
    pub shoreline_foam_depth: f32,
    /// Water depth range over which wave displacement fades to flat near shore.
    pub shore_wave_damp_width: f32,
    /// Multiplier on Jacobian foldover foam (0 = off, 1 = default).
    pub jacobian_foam_strength: f32,
    /// Multiplier on capillary high-frequency noise normals (0 = off).
    pub capillary_strength: f32,
    /// Macro height-noise amplitude in metres (0 disables).  Breaks the
    /// strictly periodic Gerstner sum at distance.
    pub macro_noise_amplitude: f32,
    /// Macro height-noise dominant wavelength in metres.
    pub macro_noise_scale: f32,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            alpha_mode: AlphaMode::Opaque,
            height: 1.0,
            amplitude: 1.0,
            clarity: 0.25,
            base_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
            deep_color: Color::srgba(0.2, 0.41, 0.54, 0.92),
            shallow_color: Color::srgba(0.45, 0.78, 0.81, 1.0),
            edge_scale: 0.1,
            edge_color: Color::srgba(1.0, 1.0, 1.0, 1.0),
            update_materials: true,
            spawn_tiles: None,
            wave_direction: Vec2::new(1.0, 2.0),
            wave_direction_blend_duration: 2.0,
            wave_speed: 2.35,
            refraction_strength: 15.0,
            // Normalised: fraction of max wave height above which foam appears.
            foam_threshold: 0.7,
            foam_color: Color::srgba(1.0, 1.0, 1.0, 0.9),
            shoreline_foam_depth: 2.0,
            shore_wave_damp_width: 3.0,
            jacobian_foam_strength: 1.0,
            capillary_strength: 1.0,
            macro_noise_amplitude: 2.0,
            macro_noise_scale: 110.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public resources
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Copy)]
pub struct WaterEnabled(pub bool);

// ---------------------------------------------------------------------------
// Internal resources
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Copy)]
struct InitialWaterLayout(WaterLayout);

#[derive(Resource, Clone)]
struct WaterTerrainFallback {
    height_texture: Handle<Image>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WaterLayout {
    height: f32,
    center: Vec2,
    clipmap: Option<WaterClipmapLayout>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaterClipmapLayout {
    levels: u32,
}

#[derive(Clone, Debug)]
struct PatchInstanceCpu {
    lod_level: u32,
    origin_ws: Vec2,
    level_scale_ws: f32,
}

#[derive(Resource, Default)]
struct WaterPatchEntities {
    /// Regular m×m block entities (fine and coarse rings).
    block_entities: Vec<Entity>,
    /// Interior trim strip entities (4 per ring boundary).
    trim_entities: Vec<Entity>,
    last_clip_centers: Vec<IVec2>,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct LandscapeWaterPlugin {
    pub water_height: Option<f32>,
    pub world_min: Vec2,
    pub world_max: Vec2,
}

impl Default for LandscapeWaterPlugin {
    fn default() -> Self {
        Self {
            water_height: None,
            world_min: Vec2::splat(-4096.0),
            world_max: Vec2::splat(4096.0),
        }
    }
}

impl Plugin for LandscapeWaterPlugin {
    fn build(&self, app: &mut App) {
        let initial_layout = water_layout(
            self.world_min,
            self.world_max,
            self.water_height.filter(|h| *h > 0.0),
        );

        app.insert_resource(WaterSettings {
            height: initial_layout.height,
            amplitude: 6.75,
            clarity: 0.94,
            wave_speed: 2.35,
            // Normalised [0..1]: fraction of max wave height above which foam appears.
            // 0.7 = only the top 30% of wave heights get foam (genuine crests only).
            foam_threshold: 0.7,
            shoreline_foam_depth: 1.5,
            shore_wave_damp_width: 3.0,
            spawn_tiles: None,
            ..WaterSettings::default()
        });
        app.insert_resource(WaterEnabled(initial_layout.clipmap.is_some()));
        app.insert_resource(InitialWaterLayout(initial_layout));
        app.init_resource::<WaterPatchEntities>();
        app.add_plugins(WaterMaterialPlugin);
        app.add_plugins(OceanFftPlugin);
        app.add_systems(
            Startup,
            (setup_water_terrain_fallback, spawn_initial_water_tiles).chain(),
        );
        app.add_systems(
            Update,
            (
                (sync_water_to_terrain, update_water_patch_transforms).chain(),
                sync_water_height,
                toggle_water,
                apply_water_enabled.after(toggle_water),
            ),
        );
        app.add_systems(PostUpdate, sync_water_materials);
    }
}

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const WATER_CLIPMAP_BLOCK_SIZE: u32 = 64;
// Fine-ring vertex spacing in metres.  Halved from 4.0 → 2.0 so vertex
// displacement can resolve the shorter geometry waves (down to ~21 m λ
// gets ~10 verts/wavelength instead of ~5).  Doubles fine-ring vertex count.
const WATER_CLIPMAP_BASE_SCALE: f32 = 2.0;
const WATER_CLIPMAP_MIN_LEVELS: u32 = 6;
const WATER_CLIPMAP_MAX_LEVELS: u32 = 8;
const WATER_CLIPMAP_MIN_OUTER_HALF_EXTENT: f32 = 16_384.0;

fn water_level_scale(level: u32) -> f32 {
    WATER_CLIPMAP_BASE_SCALE * (1u32 << level) as f32
}

fn water_clipmap_layout(extent: Vec2) -> WaterClipmapLayout {
    let target = (extent.max_element() * 0.75).max(WATER_CLIPMAP_MIN_OUTER_HALF_EXTENT);
    let mut levels = 1u32;
    let mut outer = 2.0 * WATER_CLIPMAP_BLOCK_SIZE as f32 * water_level_scale(0);
    while outer < target && levels < WATER_CLIPMAP_MAX_LEVELS {
        outer *= 2.0;
        levels += 1;
    }
    WaterClipmapLayout {
        levels: levels.max(WATER_CLIPMAP_MIN_LEVELS),
    }
}

fn water_layout(world_min: Vec2, world_max: Vec2, water_height: Option<f32>) -> WaterLayout {
    let extent = world_max - world_min;
    let center = (world_min + world_max) * 0.5;
    WaterLayout {
        height: water_height.unwrap_or(0.0),
        center,
        clipmap: water_height.map(|_| water_clipmap_layout(extent)),
    }
}

fn effective_water_level(level: Option<f32>) -> Option<f32> {
    level.filter(|l| *l > 0.0)
}

fn build_water_terrain_fallback_texture() -> Image {
    let layers = MAX_SUPPORTED_CLIPMAP_LEVELS as u32;
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        vec![0u8; layers as usize * 2],
        TextureFormat::R16Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn terrain_world_bounds(desc: &TerrainSourceDesc) -> Vec4 {
    Vec4::new(
        desc.world_min.x,
        desc.world_min.y,
        desc.world_max.x,
        desc.world_max.y,
    )
}

fn water_terrain_clip_levels(
    config: &TerrainConfig,
    view: Option<&TerrainViewState>,
) -> [Vec4; MAX_SUPPORTED_CLIPMAP_LEVELS] {
    match view {
        Some(view) if !view.clip_centers.is_empty() => {
            compute_clip_levels(config, &view.clip_centers, &view.level_scales)
        }
        _ => compute_initial_clip_levels(config),
    }
}

// ---------------------------------------------------------------------------
// Even-snapping: guarantees ring boundary alignment.
//
// Snap fine_center to stride = 2 × base_scale (always even result).
//
// Proof: even fine_center = 2k →
//   Fine outer max X   = (2k + 2m) × s₀
//   Coarse inner max X = (k + m)   × s₁ = (2k + 2m) × s₀  ✓
// All four ring edges align exactly.  Trim strips (below) then fill the
// T-junction cracks along these shared edges.
// ---------------------------------------------------------------------------
fn snap_to_even_grid(camera_xz: Vec2, scale: f32) -> IVec2 {
    let double_scale = scale * 2.0;
    let s = (camera_xz / double_scale).floor();
    IVec2::new(s.x as i32 * 2, s.y as i32 * 2)
}

fn build_water_clip_centers(camera_xz: Vec2, levels: u32) -> Vec<IVec2> {
    let fine = snap_to_even_grid(camera_xz, water_level_scale(0));
    (0..levels)
        .map(|l| {
            let s = l as i32;
            IVec2::new(fine.x >> s, fine.y >> s)
        })
        .collect()
}

fn build_block_origins(center: IVec2, scale: f32, m: u32, has_hole: bool) -> Vec<Vec2> {
    let m = m as i32;
    let cols = [-2 * m, -m, 0, m];
    let mut out = Vec::with_capacity(if has_hole { 12 } else { 16 });
    for &bz in &cols {
        for &bx in &cols {
            if has_hole && bx >= -m && bx < m && bz >= -m && bz < m {
                continue;
            }
            out.push(Vec2::new(
                (center.x + bx) as f32 * scale,
                (center.y + bz) as f32 * scale,
            ));
        }
    }
    out
}

fn build_patch_instances(clip_centers: &[IVec2]) -> Vec<PatchInstanceCpu> {
    let mut out = Vec::new();
    for (level, &center) in clip_centers.iter().enumerate() {
        let level = level as u32;
        let scale = water_level_scale(level);
        for origin in build_block_origins(center, scale, WATER_CLIPMAP_BLOCK_SIZE, level > 0) {
            out.push(PatchInstanceCpu {
                lod_level: level,
                origin_ws: origin,
                level_scale_ws: scale,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Coarse-scale interior trim strips  (GPU Gems 2 §2.3.3)
//
// Each ring LOD boundary has a T-junction: the fine ring has 2× more vertices
// along the shared edge than the coarse ring.  Even-snap ensures the boundary
// world positions coincide, but the coarse mesh has no vertices at the fine
// half-stride positions, leaving visible cracks when Gerstner waves displace
// vertices differently.
//
// Fix: add 4 trim strips (LEFT, RIGHT, BOTTOM, TOP) per ring level, placed
// 1 coarse unit inside the hole from each boundary edge.  Strips overlap the
// fine ring slightly (2 fine quads wide) which causes minimal double-blending
// through alpha-blended water — far less visible than the underlying crack.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct WaterTrimCpu {
    origin_ws: Vec2,
    level_scale_ws: f32,
    is_horizontal: bool,
}

fn build_trim_instances(clip_centers: &[IVec2]) -> Vec<WaterTrimCpu> {
    let m = WATER_CLIPMAP_BLOCK_SIZE as i32;
    let mut out = Vec::with_capacity(4 * clip_centers.len().saturating_sub(1));

    for level in 1..clip_centers.len() as u32 {
        let coarse_scale = water_level_scale(level);
        // Use fine (level-1) scale so trim strip vertices are at fine-ring
        // spacing — this eliminates the T-junction cracks that appear when
        // the trim has coarse vertex density but the fine ring has 2× more
        // vertices along the shared edge.
        let fine_scale = water_level_scale(level - 1);
        let c = clip_centers[level as usize];

        let min_x = (c.x - m) as f32 * coarse_scale;
        let max_x = (c.x + m) as f32 * coarse_scale;
        let min_z = (c.y - m) as f32 * coarse_scale;
        let max_z = (c.y + m) as f32 * coarse_scale;

        // LEFT  (vertical):   x ∈ [min_x, min_x+coarse], z ∈ [min_z, max_z]
        out.push(WaterTrimCpu {
            origin_ws: Vec2::new(min_x, min_z),
            level_scale_ws: fine_scale,
            is_horizontal: false,
        });
        // RIGHT (vertical):   x ∈ [max_x-coarse, max_x], z ∈ [min_z, max_z]
        // max_x - coarse_scale = max_x - 2*fine_scale (same world position)
        out.push(WaterTrimCpu {
            origin_ws: Vec2::new(max_x - coarse_scale, min_z),
            level_scale_ws: fine_scale,
            is_horizontal: false,
        });
        // BOTTOM (horizontal): x ∈ [min_x, max_x], z ∈ [min_z, min_z+coarse]
        out.push(WaterTrimCpu {
            origin_ws: Vec2::new(min_x, min_z),
            level_scale_ws: fine_scale,
            is_horizontal: true,
        });
        // TOP   (horizontal): x ∈ [min_x, max_x], z ∈ [max_z-coarse, max_z]
        out.push(WaterTrimCpu {
            origin_ws: Vec2::new(min_x, max_z - coarse_scale),
            level_scale_ws: fine_scale,
            is_horizontal: true,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

fn build_rect_mesh(nx: u32, nz: u32) -> Mesh {
    let vx = nx + 1;
    let vz = nz + 1;
    let inv_nx = 1.0 / nx as f32;
    let inv_nz = 1.0 / nz as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((vx * vz) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((vx * vz) as usize);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity((vx * vz) as usize);

    for z in 0..vz {
        for x in 0..vx {
            positions.push([x as f32, 0.0, z as f32]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 * inv_nx, z as f32 * inv_nz]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity((nx * nz * 6) as usize);
    for z in 0..nz {
        for x in 0..nx {
            let i00 = z * vx + x;
            let i10 = i00 + 1;
            let i01 = i00 + vx;
            let i11 = i01 + 1;
            indices.extend_from_slice(&[i00, i01, i10, i10, i01, i11]);
        }
    }

    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Tile rebuild
// ---------------------------------------------------------------------------

fn rebuild_water_tiles(
    commands: &mut Commands,
    settings: &WaterSettings,
    layout: WaterLayout,
    terrain_fallback: &WaterTerrainFallback,
    fft_state: Option<&OceanFftBuffers>,
    fft_settings: Option<&OceanFftSettings>,
    camera_xz: Vec2,
    patches: &mut WaterPatchEntities,
    existing_roots: &[Entity],
    root_children: &Query<&Children, With<WaterTiles>>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardWaterMaterial>,
) {
    for &root in existing_roots {
        if let Ok(children) = root_children.get(root) {
            for child in children.iter() {
                commands.entity(child).despawn();
            }
        }
        commands.entity(root).despawn();
    }
    patches.block_entities.clear();
    patches.trim_entities.clear();
    patches.last_clip_centers.clear();

    let Some(clipmap) = layout.clipmap else {
        return;
    };

    let clip_centers = build_water_clip_centers(camera_xz, clipmap.levels);
    let root_center = clip_centers[0].as_vec2() * water_level_scale(0);
    let patch_list = build_patch_instances(&clip_centers);
    let trim_list = build_trim_instances(&clip_centers);
    let water_height = layout.height;

    let trim_quads = 4 * WATER_CLIPMAP_BLOCK_SIZE;
    let block_mesh = Mesh3d(meshes.add(build_rect_mesh(
        WATER_CLIPMAP_BLOCK_SIZE,
        WATER_CLIPMAP_BLOCK_SIZE,
    )));
    // 2 quads wide × 4m quads tall (fine-scale density) so each trim strip's
    // vertices align with the fine-ring vertex grid, eliminating T-junction cracks.
    let trim_v_mesh = Mesh3d(meshes.add(build_rect_mesh(2, trim_quads)));
    let trim_h_mesh = Mesh3d(meshes.add(build_rect_mesh(trim_quads, 2)));

    let mat = MeshMaterial3d(materials.add(StandardWaterMaterial {
        base: StandardMaterial {
            base_color: settings.base_color,
            alpha_mode: AlphaMode::Opaque,
            // Low base roughness so the atmosphere IBL specular reads clearly.
            // Fragment shader overrides perceptual_roughness per-pixel.
            perceptual_roughness: 0.05,
            // F0 ≈ 0.02 for water (IOR 1.333): reflectance = sqrt(0.02/0.16) ≈ 0.354
            reflectance: 0.35,
            // Specular transmission makes water transparent and lets PBR
            // compute screen-space refraction via the depth buffer.  IOR 1.333
            // bends the refracted ray correctly.  The fragment shader drives
            // the actual per-pixel transmission value via Beer's law.
            specular_transmission: 0.94,
            thickness: 2.0,
            ior: 1.333,
            attenuation_distance: 48.0,
            attenuation_color: settings.deep_color,
            ..default()
        },
        extension: WaterMaterial {
            amplitude: settings.amplitude,
            clarity: settings.clarity,
            deep_color: settings.deep_color,
            shallow_color: settings.shallow_color,
            edge_color: settings.edge_color,
            edge_scale: settings.edge_scale,
            wave_speed: settings.wave_speed,
            quality: 4,
            refraction_strength: settings.refraction_strength,
            foam_threshold: settings.foam_threshold,
            foam_color: settings.foam_color,
            shoreline_foam_depth: settings.shoreline_foam_depth,
            wave_direction: settings.wave_direction.normalize_or_zero(),
            water_height,
            shore_wave_damp_width: settings.shore_wave_damp_width,
            jacobian_foam_strength: settings.jacobian_foam_strength,
            capillary_strength: settings.capillary_strength,
            macro_noise_amplitude: settings.macro_noise_amplitude,
            macro_noise_scale: settings.macro_noise_scale,
            terrain_height_texture: terrain_fallback.height_texture.clone(),
            terrain_world_bounds: Vec4::ZERO,
            terrain_height_scale: 0.0,
            terrain_num_levels: 0,
            terrain_clip_levels: [Vec4::ZERO; MAX_SUPPORTED_CLIPMAP_LEVELS],
            fft_displacement_texture: fft_state
                .map(|s| s.displacement.clone())
                .unwrap_or_default(),
            fft_world_size: fft_state.map(|s| s.world_size).unwrap_or(128.0),
            fft_size: fft_settings.map(|s| s.size).unwrap_or(128),
            fft_strength: match (fft_settings, fft_state) {
                (Some(s), Some(_)) if s.enabled => s.strength,
                _ => 0.0,
            },
        },
    }));

    let wave_dir = WaveDirection::with_duration(
        settings.wave_direction,
        settings.wave_direction_blend_duration,
    );

    let mut block_spawned = Vec::with_capacity(patch_list.len());
    let mut trim_spawned = Vec::with_capacity(trim_list.len());

    let root = commands
        .spawn((
            WaterTiles,
            Name::new("Water"),
            // Water height lives on the root; children use y=0 so root.y can be
            // updated live without rebuilding child transforms.
            Transform::from_xyz(root_center.x, water_height, root_center.y),
            Visibility::Inherited,
        ))
        .id();

    commands.entity(root).with_children(|parent| {
        // --- Block patches ---
        for p in &patch_list {
            let offset = p.origin_ws - root_center;
            let entity = parent
                .spawn((
                    WaterTile { offset },
                    Name::new(format!("Water Patch L{}", p.lod_level)),
                    Transform {
                        translation: Vec3::new(offset.x, 0.0, offset.y),
                        scale: Vec3::new(p.level_scale_ws, 1.0, p.level_scale_ws),
                        ..default()
                    },
                    block_mesh.clone(),
                    mat.clone(),
                    wave_dir,
                    NotShadowCaster,
                    NotShadowReceiver,
                ))
                .id();
            block_spawned.push(entity);
        }

        // --- Interior trim strips (fill T-junction cracks at ring boundaries) ---
        for trim in &trim_list {
            let offset = trim.origin_ws - root_center;
            let mesh = if trim.is_horizontal {
                trim_h_mesh.clone()
            } else {
                trim_v_mesh.clone()
            };
            let entity = parent
                .spawn((
                    WaterTile { offset },
                    Name::new("Water Trim"),
                    Transform {
                        translation: Vec3::new(offset.x, 0.0, offset.y),
                        scale: Vec3::new(trim.level_scale_ws, 1.0, trim.level_scale_ws),
                        ..default()
                    },
                    mesh,
                    mat.clone(),
                    wave_dir,
                    NotShadowCaster,
                    NotShadowReceiver,
                ))
                .id();
            trim_spawned.push(entity);
        }
    });

    patches.block_entities = block_spawned;
    patches.trim_entities = trim_spawned;
    patches.last_clip_centers = clip_centers;
}

// ---------------------------------------------------------------------------
// Camera helper
// ---------------------------------------------------------------------------

fn current_water_camera_xz(
    q: &Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
    fallback: Vec2,
) -> Vec2 {
    if let Ok(cam) = q.single() {
        cam.translation.xz()
    } else {
        fallback
    }
}

// ---------------------------------------------------------------------------
// Per-frame transform update
// ---------------------------------------------------------------------------

fn update_water_patch_transforms(
    mut patch_entities: ResMut<WaterPatchEntities>,
    camera_q: Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
    mut root_q: Query<
        &mut Transform,
        (With<WaterTiles>, Without<TerrainCamera>, Without<WaterTile>),
    >,
    mut tile_q: Query<
        (&mut Transform, &mut WaterTile),
        (Without<WaterTiles>, Without<TerrainCamera>),
    >,
) {
    if patch_entities.block_entities.is_empty() {
        return;
    }
    let Ok(mut root_t) = root_q.single_mut() else {
        return;
    };

    let clip_centers = build_water_clip_centers(
        current_water_camera_xz(&camera_q, root_t.translation.xz()),
        patch_entities.last_clip_centers.len() as u32,
    );
    if clip_centers == patch_entities.last_clip_centers {
        return;
    }

    let root_center = clip_centers[0].as_vec2() * water_level_scale(0);
    root_t.translation.x = root_center.x;
    root_t.translation.z = root_center.y;

    let patch_list = build_patch_instances(&clip_centers);
    let trim_list = build_trim_instances(&clip_centers);

    for (entity, p) in patch_entities.block_entities.iter().zip(patch_list.iter()) {
        if let Ok((mut t, mut tile)) = tile_q.get_mut(*entity) {
            let offset = p.origin_ws - root_center;
            t.translation.x = offset.x;
            t.translation.z = offset.y;
            t.scale = Vec3::new(p.level_scale_ws, 1.0, p.level_scale_ws);
            tile.offset = offset;
        }
    }

    for (entity, trim) in patch_entities.trim_entities.iter().zip(trim_list.iter()) {
        if let Ok((mut t, mut tile)) = tile_q.get_mut(*entity) {
            let offset = trim.origin_ws - root_center;
            t.translation.x = offset.x;
            t.translation.z = offset.y;
            t.scale = Vec3::new(trim.level_scale_ws, 1.0, trim.level_scale_ws);
            tile.offset = offset;
        }
    }

    patch_entities.last_clip_centers = clip_centers;
}

// ---------------------------------------------------------------------------
// apply_water_layout
// ---------------------------------------------------------------------------

fn apply_water_layout(
    commands: &mut Commands,
    settings: &mut WaterSettings,
    enabled: &mut WaterEnabled,
    layout: WaterLayout,
    terrain_fallback: &WaterTerrainFallback,
    fft_state: Option<&OceanFftBuffers>,
    fft_settings: Option<&OceanFftSettings>,
    camera_xz: Vec2,
    patches: &mut WaterPatchEntities,
    water_roots: &Query<Entity, With<WaterTiles>>,
    root_children: &Query<&Children, With<WaterTiles>>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardWaterMaterial>,
) {
    settings.height = layout.height;
    settings.spawn_tiles = None;
    enabled.0 = layout.clipmap.is_some();

    let roots: Vec<Entity> = water_roots.iter().collect();
    rebuild_water_tiles(
        commands,
        settings,
        layout,
        terrain_fallback,
        fft_state,
        fft_settings,
        camera_xz,
        patches,
        &roots,
        root_children,
        meshes,
        materials,
    );
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn setup_water_terrain_fallback(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.insert_resource(WaterTerrainFallback {
        height_texture: images.add(build_water_terrain_fallback_texture()),
    });
}

fn spawn_initial_water_tiles(
    initial: Res<InitialWaterLayout>,
    terrain_fallback: Res<WaterTerrainFallback>,
    fft_state: Option<Res<OceanFftBuffers>>,
    fft_settings: Option<Res<OceanFftSettings>>,
    mut settings: ResMut<WaterSettings>,
    mut enabled: ResMut<WaterEnabled>,
    mut patches: ResMut<WaterPatchEntities>,
    camera_q: Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
    water_roots: Query<Entity, With<WaterTiles>>,
    root_children: Query<&Children, With<WaterTiles>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardWaterMaterial>>,
    mut commands: Commands,
) {
    apply_water_layout(
        &mut commands,
        &mut settings,
        &mut enabled,
        initial.0,
        &terrain_fallback,
        fft_state.as_deref(),
        fft_settings.as_deref(),
        current_water_camera_xz(&camera_q, initial.0.center),
        &mut patches,
        &water_roots,
        &root_children,
        &mut meshes,
        &mut materials,
    );
}

fn sync_water_to_terrain(
    source_desc: Res<TerrainSourceDesc>,
    config: Res<TerrainConfig>,
    terrain_fallback: Res<WaterTerrainFallback>,
    fft_state: Option<Res<OceanFftBuffers>>,
    fft_settings: Option<Res<OceanFftSettings>>,
    mut settings: ResMut<WaterSettings>,
    mut enabled: ResMut<WaterEnabled>,
    mut patches: ResMut<WaterPatchEntities>,
    camera_q: Query<&Transform, (With<TerrainCamera>, Without<WaterTiles>)>,
    water_roots: Query<Entity, With<WaterTiles>>,
    root_children: Query<&Children, With<WaterTiles>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardWaterMaterial>>,
    mut commands: Commands,
) {
    if !source_desc.is_changed() {
        return;
    }

    let meta = source_desc
        .tile_root
        .as_deref()
        .map(|p| TerrainMetadata::load(p))
        .unwrap_or_default();

    let water_height = effective_water_level(meta.water_level).map(|wl| wl * config.height_scale);
    let layout = water_layout(source_desc.world_min, source_desc.world_max, water_height);

    apply_water_layout(
        &mut commands,
        &mut settings,
        &mut enabled,
        layout,
        &terrain_fallback,
        fft_state.as_deref(),
        fft_settings.as_deref(),
        current_water_camera_xz(&camera_q, layout.center),
        &mut patches,
        &water_roots,
        &root_children,
        &mut meshes,
        &mut materials,
    );

    info!(
        "Water level: {} (F2 to toggle)",
        if let Some(h) = water_height {
            format!("{:.1}m", h)
        } else {
            "none".into()
        }
    );
}

fn toggle_water(keys: Res<ButtonInput<KeyCode>>, mut enabled: ResMut<WaterEnabled>) {
    if keys.just_pressed(KeyCode::F2) {
        enabled.0 = !enabled.0;
        info!("Water {} (F2)", if enabled.0 { "ON" } else { "OFF" });
    }
}

fn sync_water_height(
    settings: Res<WaterSettings>,
    mut roots: Query<&mut Transform, With<WaterTiles>>,
) {
    if !settings.is_changed() {
        return;
    }
    for mut t in &mut roots {
        t.translation.y = settings.height;
    }
}

fn sync_water_materials(
    settings: Res<WaterSettings>,
    terrain_fallback: Res<WaterTerrainFallback>,
    source_desc: Res<TerrainSourceDesc>,
    config: Res<TerrainConfig>,
    terrain_view: Option<Res<TerrainViewState>>,
    terrain_clipmap: Option<Res<TerrainClipmapState>>,
    fft_settings: Option<Res<OceanFftSettings>>,
    fft_state: Option<Res<OceanFftBuffers>>,
    tiles: Query<&MeshMaterial3d<StandardWaterMaterial>, With<WaterTile>>,
    mut mats: ResMut<Assets<StandardWaterMaterial>>,
) {
    let terrain_changed = source_desc.is_changed()
        || config.is_changed()
        || terrain_view.as_ref().is_some_and(|view| view.is_changed())
        || terrain_clipmap
            .as_ref()
            .is_some_and(|clipmap| clipmap.is_changed());
    let settings_changed = settings.is_changed();
    let fft_changed = fft_settings.as_ref().is_some_and(|s| s.is_changed())
        || fft_state.as_ref().is_some_and(|s| s.is_changed());

    if !settings_changed && !terrain_changed && !fft_changed {
        return;
    }

    let Some(handle) = tiles.iter().next() else {
        return;
    };

    let terrain_height_texture = terrain_clipmap
        .as_ref()
        .map(|clipmap| clipmap.height_texture_handle.clone())
        .unwrap_or_else(|| terrain_fallback.height_texture.clone());
    let terrain_num_levels = if terrain_clipmap.is_some() {
        config.active_clipmap_levels()
    } else {
        0
    };
    let terrain_clip_levels = if terrain_num_levels > 0 {
        water_terrain_clip_levels(&config, terrain_view.as_deref())
    } else {
        [Vec4::ZERO; MAX_SUPPORTED_CLIPMAP_LEVELS]
    };
    let wave_direction = settings.wave_direction.normalize_or_zero();

    if let Some(mat) = mats.get_mut(&handle.0) {
        if settings_changed && settings.update_materials {
            mat.base.base_color = settings.base_color;
            mat.base.alpha_mode = AlphaMode::Opaque;
            mat.base.perceptual_roughness = 0.05;
            mat.base.reflectance = 0.35;
            mat.base.specular_transmission = 0.94;
            mat.base.thickness = 2.0;
            mat.base.ior = 1.333;
            mat.base.attenuation_distance = 48.0;
            mat.base.attenuation_color = settings.deep_color;

            mat.extension.amplitude = settings.amplitude;
            mat.extension.clarity = settings.clarity;
            mat.extension.deep_color = settings.deep_color;
            mat.extension.shallow_color = settings.shallow_color;
            mat.extension.edge_color = settings.edge_color;
            mat.extension.edge_scale = settings.edge_scale;
            mat.extension.wave_speed = settings.wave_speed;
            mat.extension.refraction_strength = settings.refraction_strength;
            mat.extension.foam_threshold = settings.foam_threshold;
            mat.extension.foam_color = settings.foam_color;
            mat.extension.shoreline_foam_depth = settings.shoreline_foam_depth;
            mat.extension.wave_direction = wave_direction;
        }

        if settings_changed {
            mat.extension.water_height = settings.height;
            mat.extension.shore_wave_damp_width = settings.shore_wave_damp_width;
            mat.extension.jacobian_foam_strength = settings.jacobian_foam_strength;
            mat.extension.capillary_strength = settings.capillary_strength;
            mat.extension.macro_noise_amplitude = settings.macro_noise_amplitude;
            mat.extension.macro_noise_scale = settings.macro_noise_scale;
        }

        if fft_changed {
            if let (Some(state), Some(settings)) = (fft_state.as_ref(), fft_settings.as_ref()) {
                mat.extension.fft_displacement_texture = state.displacement.clone();
                mat.extension.fft_world_size = state.world_size;
                mat.extension.fft_size = settings.size;
                mat.extension.fft_strength = if settings.enabled { settings.strength } else { 0.0 };
            } else {
                mat.extension.fft_strength = 0.0;
            }
        }

        if terrain_changed {
            mat.extension.terrain_height_texture = terrain_height_texture;
            mat.extension.terrain_world_bounds = terrain_world_bounds(&source_desc);
            mat.extension.terrain_height_scale = config.height_scale;
            mat.extension.terrain_num_levels = terrain_num_levels;
            mat.extension.terrain_clip_levels = terrain_clip_levels;
        }
    }
}

fn apply_water_enabled(
    enabled: Res<WaterEnabled>,
    mut water_tiles: Query<&mut Visibility, With<WaterTiles>>,
) {
    if !enabled.is_changed() {
        return;
    }
    let vis = if enabled.0 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut water_tiles {
        *v = vis;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        build_patch_instances, build_trim_instances, build_water_clip_centers,
        effective_water_level, snap_to_even_grid, water_clipmap_layout, water_layout,
        water_level_scale, WATER_CLIPMAP_BLOCK_SIZE,
    };
    use bevy::prelude::*;

    #[test]
    fn zero_water_level_is_treated_as_no_water() {
        assert_eq!(effective_water_level(Some(0.0)), None);
        assert_eq!(effective_water_level(Some(0.25)), Some(0.25));
    }

    #[test]
    fn layout_scales_clipmap_with_terrain_extent() {
        let layout = water_layout(
            Vec2::new(-15_360.0, -7_680.0),
            Vec2::new(15_360.0, 7_680.0),
            Some(120.0),
        );
        assert_eq!(layout.center, Vec2::ZERO);
        assert_eq!(
            layout.clipmap,
            Some(super::WaterClipmapLayout { levels: 7 })
        );
        assert_eq!(layout.height, 120.0);
    }

    #[test]
    fn clipmap_layout_keeps_minimum_levels() {
        assert_eq!(water_clipmap_layout(Vec2::splat(2_048.0)).levels, 6);
    }

    #[test]
    fn even_snap_produces_even_fine_center() {
        let scale = water_level_scale(0);
        for &cam_x in &[0.1_f32, 1.9, 3.1, 5.0, 7.5, 13.9, 100.3, -1.2, -7.8] {
            let c = snap_to_even_grid(Vec2::new(cam_x, cam_x), scale);
            assert_eq!(c.x % 2, 0, "fine_center.x must be even for cam_x={}", cam_x);
            assert_eq!(c.y % 2, 0, "fine_center.y must be even for cam_x={}", cam_x);
        }
    }

    #[test]
    fn even_snap_eliminates_ring_boundary_gap() {
        let scale0 = water_level_scale(0);
        let scale1 = water_level_scale(1);
        let m = WATER_CLIPMAP_BLOCK_SIZE as i32;

        for cam in [0.0_f32, 3.7, 13.9, 100.3, -5.5, 333.1] {
            let centers = build_water_clip_centers(Vec2::splat(cam), 4);
            let c0 = centers[0];
            let c1 = centers[1];

            let fine_outer = (c0.x + 2 * m) as f32 * scale0;
            let coarse_inner = (c1.x + m) as f32 * scale1;

            assert!(
                (fine_outer - coarse_inner).abs() < 1e-3,
                "gap at cam={}: fine_outer={} coarse_inner={}",
                cam,
                fine_outer,
                coarse_inner
            );
        }
    }

    #[test]
    fn clipmap_centers_are_nested_by_shift() {
        let centers = build_water_clip_centers(Vec2::new(13.9, 29.9), 4);
        assert_eq!(centers[0].x % 2, 0, "fine center must be even");
        assert_eq!(centers[1], centers[0] >> 1);
        assert_eq!(centers[2], centers[0] >> 2);
        assert_eq!(centers[3], centers[0] >> 3);
    }

    #[test]
    fn clipmap_patch_counts_match_gpu_gems_layout() {
        let centers = build_water_clip_centers(Vec2::ZERO, 6);
        // Blocks: 16 fine + 12 per coarse ring × 5 rings
        assert_eq!(build_patch_instances(&centers).len(), 16 + 12 * 5);
        // Trims: 4 strips per ring boundary × 5 boundaries
        assert_eq!(build_trim_instances(&centers).len(), 4 * 5);
    }
}

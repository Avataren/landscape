//! GPU-driven grass rendering with two LOD passes and PBR texture variants.
//!
//! # Two-pass design
//! - **Near** pass: dense spacing, covers a small radius (~80 m default).
//! - **Far** pass: coarse spacing, extends past the near edge (~500 m default).
//!
//! # Texture variants
//! Three grass variants are loaded from assets/grass_01/{grass_01,grass_02,grass_03}/.
//! Each variant has diffuse, normal, opacity and specular TGA maps.
//! They are combined into texture_2d_array assets (one array per map type, 3 layers)
//! and bound to the material.  The vertex shader picks a random variant per blade.

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    mesh::{MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError, TextureDimension, TextureFormat, TextureViewDescriptor,
            TextureViewDimension,
        },
    },
    image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    shader::ShaderRef,
};

use crate::terrain::{
    clipmap_texture::{compute_clip_levels, TerrainClipmapState},
    config::TerrainConfig,
    math::level_scale,
    source_heightmap::SourceHeightmapState,
    components::TerrainCamera,
};
use crate::terrain::material_slots::MaterialLibrary;

// ── Public config ─────────────────────────────────────────────────────────────

#[derive(Resource, Clone, Debug)]
pub struct GpuGrassConfig {
    pub enabled: bool,
    pub near_spacing: f32,
    pub near_range: f32,
    pub far_spacing: f32,
    /// Extension in metres *past* the near edge (total far radius = near_range + far_range).
    pub far_range: f32,
    pub blade_height: f32,
    pub blade_width: f32,
    /// Rise/run slope ratio above which grass fades (> 90 = disabled).
    pub slope_max: f32,
    pub altitude_min: f32,
    pub altitude_max: f32,
    pub wind_strength: f32,
    pub wind_scale: f32,
    pub base_color: LinearRgba,
    /// When true, altitude and slope limits are driven by material slot 0's
    /// procedural rules so grass only appears where the ground texture is active.
    pub link_to_slot0: bool,
}

impl GpuGrassConfig {
    pub fn near_grid_size(&self) -> u32 {
        let g = ((self.near_range * 2.0) / self.near_spacing).round() as u32;
        g.clamp(4, GRASS_MAX_GRID)
    }
    pub fn far_grid_size(&self) -> u32 {
        let total = self.near_range + self.far_range;
        let g = ((total * 2.0) / self.far_spacing).round() as u32;
        g.clamp(4, GRASS_MAX_GRID)
    }
}

impl Default for GpuGrassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            near_spacing: 0.7,
            near_range: 80.0,
            far_spacing: 1.5,
            far_range: 500.0,
            blade_height: 0.7,
            blade_width: 0.7,
            slope_max: 0.7,
            altitude_min: -100_000.0,
            altitude_max: 100_000.0,
            wind_strength: 0.15,
            wind_scale: 0.035,
            base_color: LinearRgba::rgb(0.19, 0.42, 0.09),
            link_to_slot0: false,
        }
    }
}

// ── GPU uniform ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default, ShaderType)]
pub struct GrassParamsGpu {
    pub camera_grid:  Vec4,  // xy=camera XZ, z=grid_size, w=spacing
    pub clip_level:   Vec4,  // xy=ring_center XZ, z=inv_ring_span, w=texel_ws
    pub blade:        Vec4,  // x=inner_radius_sq, y=blade_height, z=blade_width, w=slope_max
    pub alt_wind:     Vec4,  // x=alt_min, y=alt_max, z=wind_time, w=wind_strength
    pub wind_color:   Vec4,  // x=wind_scale, yzw=base_color (fallback when no textures)
    pub world_bounds: Vec4,  // xy=world_min XZ, zw=world_max XZ
}

// ── Material ──────────────────────────────────────────────────────────────────

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GpuGrassMaterial {
    // Terrain clipmap height (vertex-only).
    #[texture(0, dimension = "2d_array", visibility(vertex))]
    pub height_tex: Handle<Image>,
    #[uniform(1, visibility(vertex, fragment))]
    pub params: GrassParamsGpu,
    // Grass variant texture arrays (3 layers = 3 variants).
    #[texture(2, dimension = "2d_array")]
    #[sampler(3)]
    pub diffuse_arr: Handle<Image>,
    #[texture(4, dimension = "2d_array")]
    #[sampler(5)]
    pub normal_arr: Handle<Image>,
    #[texture(6, dimension = "2d_array")]
    #[sampler(7)]
    pub opacity_arr: Handle<Image>,
    #[texture(8, dimension = "2d_array")]
    #[sampler(9)]
    pub specular_arr: Handle<Image>,
}

impl Material for GpuGrassMaterial {
    fn vertex_shader()   -> ShaderRef { "shaders/grass_blade.wgsl".into() }
    fn fragment_shader() -> ShaderRef { "shaders/grass_blade.wgsl".into() }
    fn alpha_mode(&self) -> AlphaMode { AlphaMode::Mask(0.3) }

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

// ── Texture loading ───────────────────────────────────────────────────────────

/// Tracks the 12 individual TGA loads and holds the combined array handles
/// once they are ready.
#[derive(Resource)]
pub struct GrassTextureLoader {
    pub diffuse:  [Handle<Image>; 3],
    pub normal:   [Handle<Image>; 3],
    pub opacity:  [Handle<Image>; 3],
    pub specular: [Handle<Image>; 3],
    pub ready: bool,
}

fn load_grass_textures(mut commands: Commands, asset_server: Res<AssetServer>) {
    let variants = ["grass_01", "grass_02", "grass_03"];
    let load = |v: &str, map: &str| -> Handle<Image> {
        asset_server.load(format!("grass_01/{v}/{map}.png"))
    };
    commands.insert_resource(GrassTextureLoader {
        diffuse:  variants.map(|v| load(v, "diffus")),
        normal:   variants.map(|v| load(v, "normal")),
        opacity:  variants.map(|v| load(v, "opacity")),
        specular: variants.map(|v| load(v, "specular")),
        ready: false,
    });
}

/// 2×2 box-filter downsample. Works for any power-of-two bytes-per-pixel format.
fn box_downsample(src: &[u8], src_w: u32, src_h: u32, bpp: usize) -> Vec<u8> {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let mut out = vec![0u8; (dst_w * dst_h) as usize * bpp];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let x0 = (x * 2) as usize;
            let y0 = (y * 2) as usize;
            let x1 = (x * 2 + 1).min(src_w - 1) as usize;
            let y1 = (y * 2 + 1).min(src_h - 1) as usize;
            let dst_i = (y * dst_w + x) as usize * bpp;
            for c in 0..bpp {
                let sw = src_w as usize;
                let p00 = src[(y0 * sw + x0) * bpp + c] as u32;
                let p10 = src[(y0 * sw + x1) * bpp + c] as u32;
                let p01 = src[(y1 * sw + x0) * bpp + c] as u32;
                let p11 = src[(y1 * sw + x1) * bpp + c] as u32;
                out[dst_i + c] = ((p00 + p10 + p01 + p11 + 2) / 4) as u8;
            }
        }
    }
    out
}

/// Once all 12 PNGs are loaded, combines each group into a texture_2d_array
/// with a full mip chain and patches both grass materials.
fn combine_grass_textures(
    mut loader: ResMut<GrassTextureLoader>,
    mut images: ResMut<Assets<Image>>,
    near_q: Query<&MeshMaterial3d<GpuGrassMaterial>, With<GrassEntityNear>>,
    far_q:  Query<&MeshMaterial3d<GpuGrassMaterial>, With<GrassEntityFar>>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
) {
    if loader.ready { return; }

    // Check all 12 handles are loaded.
    let all_handles: Vec<&Handle<Image>> = loader.diffuse.iter()
        .chain(&loader.normal)
        .chain(&loader.opacity)
        .chain(&loader.specular)
        .collect();
    if !all_handles.iter().all(|h| images.get(*h).is_some()) { return; }

    let make_array = |handles: &[Handle<Image>; 3], images: &mut Assets<Image>| -> Handle<Image> {
        let imgs: Vec<_> = handles.iter().map(|h| images.get(h).unwrap().clone()).collect();
        let w   = imgs[0].width();
        let h   = imgs[0].height();
        let fmt = imgs[0].texture_descriptor.format;
        let bpp = fmt.block_copy_size(None).unwrap_or(4) as usize;
        let num_mips = (w.max(h) as f32).log2().floor() as u32 + 1;

        // Build mip chain for each of the 3 layers.
        let mip_chains: Vec<Vec<Vec<u8>>> = imgs.iter().map(|img| {
            let base = img.data.as_deref().unwrap_or(&[]).to_vec();
            let mut chain = vec![base];
            let mut cw = w;
            let mut ch = h;
            while cw > 1 || ch > 1 {
                let prev = chain.last().unwrap();
                chain.push(box_downsample(prev, cw, ch, bpp));
                cw = (cw / 2).max(1);
                ch = (ch / 2).max(1);
            }
            chain
        }).collect();

        // wgpu/Bevy default is LayerMajor: [layer0_mip0..N, layer1_mip0..N, layer2_mip0..N]
        let mut combined = Vec::new();
        for chain in &mip_chains {
            for mip_data in chain {
                combined.extend_from_slice(mip_data);
            }
        }

        let mut arr = Image::new(
            Extent3d { width: w, height: h, depth_or_array_layers: 3 },
            TextureDimension::D2,
            combined,
            fmt,
            RenderAssetUsages::RENDER_WORLD,
        );
        arr.texture_descriptor.mip_level_count = num_mips;
        arr.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..default()
        });
        arr.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            mipmap_filter: ImageFilterMode::Linear,
            min_filter:    ImageFilterMode::Linear,
            mag_filter:    ImageFilterMode::Linear,
            ..default()
        });
        images.add(arr)
    };

    let diffuse_h  = make_array(&loader.diffuse,  &mut images);
    let normal_h   = make_array(&loader.normal,   &mut images);
    let opacity_h  = make_array(&loader.opacity,  &mut images);
    let specular_h = make_array(&loader.specular, &mut images);

    if let Ok(mat_handle) = near_q.single() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.diffuse_arr  = diffuse_h.clone();
            mat.normal_arr   = normal_h.clone();
            mat.opacity_arr  = opacity_h.clone();
            mat.specular_arr = specular_h.clone();
        }
    }
    if let Ok(mat_handle) = far_q.single() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.diffuse_arr  = diffuse_h.clone();
            mat.normal_arr   = normal_h.clone();
            mat.opacity_arr  = opacity_h.clone();
            mat.specular_arr = specular_h.clone();
        }
    }

    loader.ready = true;
    info!("Grass textures combined into arrays and applied.");
}

// ── Entity markers ────────────────────────────────────────────────────────────

#[derive(Component)] pub struct GrassEntityNear;
#[derive(Component)] pub struct GrassEntityFar;

// ── Constants ─────────────────────────────────────────────────────────────────

const VERTS_PER_BLADE: u32 = 12;
pub const GRASS_MAX_GRID: u32 = 1000;

// ── Startup ───────────────────────────────────────────────────────────────────

fn spawn_grass_entities(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
    mut images: ResMut<Assets<Image>>,
    config: Res<GpuGrassConfig>,
) {
    let n_verts = (GRASS_MAX_GRID * GRASS_MAX_GRID * VERTS_PER_BLADE) as usize;
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; n_verts]);
    let mesh_handle = meshes.add(mesh);

    let fallback_height = make_fallback_height_tex(&mut images);
    let fallback_diffuse  = make_fallback_array(&mut images, [50, 110, 25, 255], TextureFormat::Rgba8UnormSrgb);
    let fallback_normal   = make_fallback_array(&mut images, [128, 128, 255, 255], TextureFormat::Rgba8Unorm);
    let fallback_opacity  = make_fallback_array(&mut images, [255, 0, 0, 0], TextureFormat::Rgba8Unorm);
    let fallback_specular = make_fallback_array(&mut images, [30, 0, 0, 0], TextureFormat::Rgba8Unorm);

    let fallback_clip  = Vec4::new(0.0, 0.0, 1.0 / 1_000_000.0, 1.0);
    let fallback_world = Vec4::new(-500_000.0, -500_000.0, 500_000.0, 500_000.0);

    let near_params = build_params(
        &config, config.near_grid_size(), config.near_spacing,
        0.0, Vec3::ZERO, 0.0, fallback_clip, fallback_world,
    );
    let near_mat = materials.add(GpuGrassMaterial {
        height_tex:   fallback_height.clone(),
        params:       near_params,
        diffuse_arr:  fallback_diffuse.clone(),
        normal_arr:   fallback_normal.clone(),
        opacity_arr:  fallback_opacity.clone(),
        specular_arr: fallback_specular.clone(),
    });
    commands.spawn((Mesh3d(mesh_handle.clone()), MeshMaterial3d(near_mat),
                    Transform::default(), NoFrustumCulling, GrassEntityNear));

    let inner_r_sq = config.near_range * config.near_range;
    let far_params = build_params(
        &config, config.far_grid_size(), config.far_spacing,
        inner_r_sq, Vec3::ZERO, 0.0, fallback_clip, fallback_world,
    );
    let far_mat = materials.add(GpuGrassMaterial {
        height_tex:   fallback_height,
        params:       far_params,
        diffuse_arr:  fallback_diffuse,
        normal_arr:   fallback_normal,
        opacity_arr:  fallback_opacity,
        specular_arr: fallback_specular,
    });
    commands.spawn((Mesh3d(mesh_handle), MeshMaterial3d(far_mat),
                    Transform::default(), NoFrustumCulling, GrassEntityFar));
}

fn make_fallback_height_tex(images: &mut Assets<Image>) -> Handle<Image> {
    let mut img = Image::new(
        Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        TextureDimension::D2, vec![0u8; 4], TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array), ..default()
    });
    images.add(img)
}

fn make_fallback_array(
    images: &mut Assets<Image>,
    rgba: [u8; 4],
    fmt: TextureFormat,
) -> Handle<Image> {
    let bytes_per_texel = fmt.block_copy_size(None).unwrap_or(4) as usize;
    let layer_data = rgba[..bytes_per_texel].to_vec();
    let mut combined = Vec::with_capacity(layer_data.len() * 3);
    for _ in 0..3 { combined.extend_from_slice(&layer_data); }
    let mut img = Image::new(
        Extent3d { width: 1, height: 1, depth_or_array_layers: 3 },
        TextureDimension::D2, combined, fmt, RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array), ..default()
    });
    images.add(img)
}

// ── Per-frame update ──────────────────────────────────────────────────────────

fn update_grass_materials(
    camera_q: Query<&Transform, With<TerrainCamera>>,
    source_state: Option<Res<SourceHeightmapState>>,
    terrain_config: Option<Res<TerrainConfig>>,
    clipmap_state: Option<Res<TerrainClipmapState>>,
    material_library: Option<Res<MaterialLibrary>>,
    near_q: Query<&MeshMaterial3d<GpuGrassMaterial>, With<GrassEntityNear>>,
    far_q:  Query<&MeshMaterial3d<GpuGrassMaterial>, With<GrassEntityFar>>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
    config: Res<GpuGrassConfig>,
    time: Res<Time>,
) {
    let Ok(cam) = camera_q.single() else { return };

    // Override altitude/slope from material slot 0 if requested.
    let effective = if config.link_to_slot0 {
        if let Some(lib) = material_library.as_deref() {
            if let Some(slot) = lib.slots.first() {
                let p = &slot.procedural;
                let slope_max = (p.slope_range_deg.y.to_radians()).tan();
                let mut c = config.clone();
                c.altitude_min = p.altitude_range_m.x;
                c.altitude_max = p.altitude_range_m.y;
                c.slope_max    = slope_max;
                std::borrow::Cow::Owned(c)
            } else { std::borrow::Cow::Borrowed(config.as_ref()) }
        } else { std::borrow::Cow::Borrowed(config.as_ref()) }
    } else { std::borrow::Cow::Borrowed(config.as_ref()) };
    let config = effective.as_ref();

    let fallback_clip = Vec4::new(0.0, 0.0, 1.0 / 1_000_000.0, 1.0);
    let (clip_level_near, clip_level_far) =
        if let (Some(cs), Some(tc)) = (clipmap_state.as_deref(), terrain_config.as_deref()) {
            let scales: Vec<f32> = (0..tc.active_clipmap_levels())
                .map(|l| level_scale(tc.lod0_mesh_spacing, l))
                .collect();
            let levels = compute_clip_levels(tc, &cs.last_clip_centers, &scales);

            // Near always uses LOD 0. Store the layer index in clip_level.x
            // (ring_center.x) since sample_height never reads it.
            let near = Vec4::new(0.0, levels[0].y, levels[0].z, levels[0].w);

            // Far: smallest LOD whose half-span covers near_range + far_range.
            let far_total = config.near_range + config.far_range;
            let (far_idx, far_lv) = levels.iter().copied().enumerate()
                .find(|(_, lv)| (0.5 / lv.z) >= far_total)
                .unwrap_or((levels.len() - 1, *levels.last().unwrap_or(&levels[0])));
            let far = Vec4::new(far_idx as f32, far_lv.y, far_lv.z, far_lv.w);

            (near, far)
        } else {
            (fallback_clip, fallback_clip)
        };

    let world_bounds = source_state.as_deref()
        .map(|s| {
            let max = s.world_origin + s.world_extent;
            Vec4::new(s.world_origin.x, s.world_origin.y, max.x, max.y)
        })
        .unwrap_or(Vec4::new(-500_000.0, -500_000.0, 500_000.0, 500_000.0));

    let wt = time.elapsed_secs();
    let height_handle = clipmap_state.as_deref().map(|cs| cs.height_texture_handle.clone());

    if let Ok(h) = near_q.single() {
        if let Some(mat) = materials.get_mut(&h.0) {
            mat.params = build_params(&config, config.near_grid_size(), config.near_spacing,
                                      0.0, cam.translation, wt, clip_level_near, world_bounds);
            if let Some(ref hh) = height_handle { mat.height_tex = hh.clone(); }
        }
    }
    if let Ok(h) = far_q.single() {
        if let Some(mat) = materials.get_mut(&h.0) {
            let inner_r_sq = config.near_range * config.near_range;
            mat.params = build_params(&config, config.far_grid_size(), config.far_spacing,
                                      inner_r_sq, cam.translation, wt, clip_level_far, world_bounds);
            if let Some(ref hh) = height_handle { mat.height_tex = hh.clone(); }
        }
    }
}

fn build_params(
    config: &GpuGrassConfig,
    grid_size: u32, spacing: f32, inner_radius_sq: f32,
    camera_pos: Vec3, wind_time: f32,
    clip_level_0: Vec4, world_bounds: Vec4,
) -> GrassParamsGpu {
    GrassParamsGpu {
        camera_grid: Vec4::new(camera_pos.x, camera_pos.z, grid_size as f32, spacing),
        clip_level:  clip_level_0,
        blade: Vec4::new(inner_radius_sq, config.blade_height, config.blade_width, config.slope_max),
        alt_wind: Vec4::new(config.altitude_min, config.altitude_max, wind_time, config.wind_strength),
        wind_color: Vec4::new(
            config.wind_scale,
            config.base_color.red, config.base_color.green, config.base_color.blue,
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
            .add_systems(Startup, (load_grass_textures, spawn_grass_entities))
            .add_systems(Update, (update_grass_materials, combine_grass_textures));
    }
}

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::terrain::{config::TerrainConfig, world_desc::TerrainSourceDesc};

/// GPU-side state for the composited source heightmap texture.
///
/// The texture covers `[world_origin, world_origin + world_extent]` in XZ.
/// Use these to compute UVs in shaders:
///   `uv = (world_xz - world_origin) / world_extent`
#[derive(Resource, Clone, Debug)]
pub struct SourceHeightmapState {
    pub handle: Handle<Image>,
    /// World-space XZ of texel (0, 0).
    pub world_origin: Vec2,
    /// World-space XZ size of the entire texture.
    pub world_extent: Vec2,
    /// World-space size of one texel (= world_scale × 2^max_mip_level).
    pub texel_size: f32,
}

fn read_r16_tile(path: &std::path::Path, tile_size: u32) -> Option<Vec<u16>> {
    let bytes = std::fs::read(path).ok()?;
    let expected = (tile_size * tile_size * 2) as usize;
    if bytes.len() != expected {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect(),
    )
}

fn make_image(width: u32, height: u32, data: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R16Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn fallback_image() -> Image {
    // 1×1 mid-grey — height = 0.5, safe neutral value
    make_image(1, 1, vec![0x00, 0x80])
}

pub struct SourceHeightmapLoadResult {
    pub image: Image,
    pub world_origin: Vec2,
    pub world_extent: Vec2,
    pub texel_size: f32,
}

/// Composites all tiles at `max_mip_level` into a single R16Unorm GPU texture.
///
/// The texture is aligned to tile boundaries (so it may extend slightly beyond
/// `world_min`/`world_max`). Missing tiles are left as zero (sea level).
pub fn load_source_heightmap(
    config: &TerrainConfig,
    desc: &TerrainSourceDesc,
) -> SourceHeightmapLoadResult {
    let fallback = |msg: &str| {
        warn!("[Terrain] Source heightmap: {msg} — using 1×1 fallback.");
        SourceHeightmapLoadResult {
            image: fallback_image(),
            world_origin: desc.world_min,
            world_extent: (desc.world_max - desc.world_min).max(Vec2::ONE),
            texel_size: config.world_scale,
        }
    };

    let Some(tile_root) = &desc.tile_root else {
        return fallback("no tile root configured");
    };

    let world_min = desc.world_min;
    let world_max = desc.world_max;
    if !world_max.cmpgt(world_min).all() {
        return fallback("world bounds are empty");
    }

    let source_level = desc.max_mip_level;
    let texel_size = config.world_scale * (1u32 << source_level as u32) as f32;
    let tile_size = config.tile_size;
    let tile_world_size = tile_size as f32 * texel_size;

    // Tile-aligned coverage that fully contains [world_min, world_max].
    let tx_min = (world_min.x / tile_world_size).floor() as i32;
    let tz_min = (world_min.y / tile_world_size).floor() as i32;
    let tx_max = (world_max.x / tile_world_size).ceil() as i32;
    let tz_max = (world_max.y / tile_world_size).ceil() as i32;

    let grid_w = (tx_max - tx_min) as u32;
    let grid_h = (tz_max - tz_min) as u32;

    if grid_w == 0 || grid_h == 0 {
        return fallback("computed tile grid is empty");
    }

    let tex_w = grid_w * tile_size;
    let tex_h = grid_h * tile_size;
    let mut data = vec![0u16; (tex_w * tex_h) as usize];
    let mut tiles_loaded = 0u32;

    for tz in tz_min..tz_max {
        for tx in tx_min..tx_max {
            let path = tile_root.join(format!(
                "height/L{source_level}/{tx}_{tz}.bin"
            ));
            let Some(tile_data) = read_r16_tile(&path, tile_size) else {
                continue;
            };
            tiles_loaded += 1;

            let dst_x = (tx - tx_min) as u32;
            let dst_z = (tz - tz_min) as u32;

            for row in 0..tile_size {
                let src_row_base = (row * tile_size) as usize;
                let dst_row_base = ((dst_z * tile_size + row) * tex_w + dst_x * tile_size) as usize;
                let len = tile_size as usize;
                if dst_row_base + len <= data.len() {
                    data[dst_row_base..dst_row_base + len]
                        .copy_from_slice(&tile_data[src_row_base..src_row_base + len]);
                }
            }
        }
    }

    let world_origin = Vec2::new(tx_min as f32 * tile_world_size, tz_min as f32 * tile_world_size);
    let world_extent = Vec2::new(grid_w as f32 * tile_world_size, grid_h as f32 * tile_world_size);

    info!(
        "[Terrain] Source heightmap: {}×{} texels, {:.1}m/texel, \
         {:.1}×{:.1}km coverage ({}/{} tiles loaded from L{source_level}).",
        tex_w,
        tex_h,
        texel_size,
        world_extent.x / 1000.0,
        world_extent.y / 1000.0,
        tiles_loaded,
        grid_w * grid_h,
    );

    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    SourceHeightmapLoadResult {
        image: make_image(tex_w, tex_h, bytes),
        world_origin,
        world_extent,
        texel_size,
    }
}

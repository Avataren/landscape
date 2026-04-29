use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum MipFilter {
    Box,
    Rgba8Triangle,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureArraySampler {
    pub address_mode: ImageAddressMode,
    pub mag_filter: ImageFilterMode,
    pub min_filter: ImageFilterMode,
    pub mipmap_filter: ImageFilterMode,
    pub anisotropy: Option<u16>,
}

impl TextureArraySampler {
    pub fn clamp_linear_mip() -> Self {
        Self {
            address_mode: ImageAddressMode::ClampToEdge,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            anisotropy: None,
        }
    }

    pub fn repeat_linear_mip(anisotropy: Option<u16>) -> Self {
        Self {
            address_mode: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            anisotropy,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureArrayConfig {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: TextureFormat,
    pub usage: RenderAssetUsages,
    pub sampler: TextureArraySampler,
    pub view_dimension: Option<TextureViewDimension>,
}

pub(crate) fn mip_level_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height).max(1);
    u32::BITS - max_dim.leading_zeros()
}

pub(crate) fn solid_layer(width: u32, height: u32, texel: &[u8]) -> Vec<u8> {
    let texel_count = width as usize * height as usize;
    let mut out = Vec::with_capacity(texel_count * texel.len());
    for _ in 0..texel_count {
        out.extend_from_slice(texel);
    }
    out
}

pub(crate) fn build_texture_2d_array(
    layers: &[Vec<u8>],
    config: TextureArrayConfig,
    mip_filter: MipFilter,
) -> Image {
    assert_eq!(
        layers.len(),
        config.layers as usize,
        "texture array layer count mismatch"
    );

    let bytes_per_texel = config.format.block_copy_size(None).unwrap_or(4) as usize;
    let expected_base_len = config.width as usize * config.height as usize * bytes_per_texel;
    for layer in layers {
        assert_eq!(
            layer.len(),
            expected_base_len,
            "texture array layer byte size mismatch"
        );
    }

    let mut data = Vec::new();
    for layer in layers {
        for mip in generate_mip_chain(
            layer,
            config.width,
            config.height,
            bytes_per_texel,
            mip_filter,
        ) {
            data.extend_from_slice(&mip);
        }
    }

    let mut image = Image::new_uninit(
        Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: config.layers,
        },
        TextureDimension::D2,
        config.format,
        config.usage,
    );
    image.texture_descriptor.mip_level_count = mip_level_count(config.width, config.height);
    image.data = Some(data);
    if let Some(dimension) = config.view_dimension {
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(dimension),
            ..default()
        });
    }
    image.sampler = ImageSampler::Descriptor(sampler_descriptor(config.sampler));
    image
}

fn sampler_descriptor(sampler: TextureArraySampler) -> ImageSamplerDescriptor {
    let mut desc = ImageSamplerDescriptor {
        address_mode_u: sampler.address_mode,
        address_mode_v: sampler.address_mode,
        address_mode_w: sampler.address_mode,
        mag_filter: sampler.mag_filter,
        min_filter: sampler.min_filter,
        mipmap_filter: sampler.mipmap_filter,
        ..default()
    };
    if let Some(anisotropy) = sampler.anisotropy {
        desc.set_anisotropic_filter(anisotropy);
    }
    desc
}

fn generate_mip_chain(
    base: &[u8],
    width: u32,
    height: u32,
    bytes_per_texel: usize,
    filter: MipFilter,
) -> Vec<Vec<u8>> {
    let mut chain = vec![base.to_vec()];
    let mut cur_w = width;
    let mut cur_h = height;

    while cur_w > 1 || cur_h > 1 {
        let prev = chain.last().expect("base mip exists");
        let next = match filter {
            MipFilter::Box => box_downsample(prev, cur_w, cur_h, bytes_per_texel),
            MipFilter::Rgba8Triangle => rgba8_triangle_downsample(prev, cur_w, cur_h),
        };
        chain.push(next);
        cur_w = (cur_w / 2).max(1);
        cur_h = (cur_h / 2).max(1);
    }

    chain
}

fn box_downsample(src: &[u8], src_w: u32, src_h: u32, bytes_per_texel: usize) -> Vec<u8> {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let mut out = vec![0u8; dst_w as usize * dst_h as usize * bytes_per_texel];

    for y in 0..dst_h {
        for x in 0..dst_w {
            let x0 = (x * 2) as usize;
            let y0 = (y * 2) as usize;
            let x1 = (x * 2 + 1).min(src_w - 1) as usize;
            let y1 = (y * 2 + 1).min(src_h - 1) as usize;
            let dst_i = (y as usize * dst_w as usize + x as usize) * bytes_per_texel;

            for c in 0..bytes_per_texel {
                let sw = src_w as usize;
                let p00 = src[(y0 * sw + x0) * bytes_per_texel + c] as u32;
                let p10 = src[(y0 * sw + x1) * bytes_per_texel + c] as u32;
                let p01 = src[(y1 * sw + x0) * bytes_per_texel + c] as u32;
                let p11 = src[(y1 * sw + x1) * bytes_per_texel + c] as u32;
                out[dst_i + c] = ((p00 + p10 + p01 + p11 + 2) / 4) as u8;
            }
        }
    }

    out
}

fn rgba8_triangle_downsample(src: &[u8], src_w: u32, src_h: u32) -> Vec<u8> {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let img = image::RgbaImage::from_raw(src_w, src_h, src.to_vec())
        .expect("RGBA8 mip generation: invalid dimensions");
    image::imageops::resize(&img, dst_w, dst_h, image::imageops::FilterType::Triangle).into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_count_handles_non_power_of_two() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(4, 4), 3);
        assert_eq!(mip_level_count(5, 3), 3);
    }

    #[test]
    fn solid_layer_repeats_texel() {
        assert_eq!(
            solid_layer(2, 2, &[1, 2, 3, 4]),
            vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        );
    }

    #[test]
    fn box_mips_are_layer_major_in_built_image() {
        let layer0 = vec![0, 0, 0, 255, 4, 4, 4, 255, 8, 8, 8, 255, 12, 12, 12, 255];
        let layer1 = vec![
            16, 16, 16, 255, 20, 20, 20, 255, 24, 24, 24, 255, 28, 28, 28, 255,
        ];
        let image = build_texture_2d_array(
            &[layer0, layer1],
            TextureArrayConfig {
                width: 2,
                height: 2,
                layers: 2,
                format: TextureFormat::Rgba8Unorm,
                usage: RenderAssetUsages::RENDER_WORLD,
                sampler: TextureArraySampler::clamp_linear_mip(),
                view_dimension: Some(TextureViewDimension::D2Array),
            },
            MipFilter::Box,
        );

        assert_eq!(image.texture_descriptor.mip_level_count, 2);
        let data = image.data.as_deref().unwrap();
        assert_eq!(&data[16..20], &[6, 6, 6, 255]);
        assert_eq!(&data[36..40], &[22, 22, 22, 255]);
    }
}

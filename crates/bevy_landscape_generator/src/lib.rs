mod compute;
mod erosion_compute;
pub mod erosion_images;
pub mod erosion_params;
pub mod export;
pub mod images;
pub mod params;
mod uniforms;

use bevy::prelude::*;

use compute::GeneratorComputePlugin;
use erosion_compute::ErosionComputePlugin;
use erosion_images::ErosionBuffers;
use erosion_params::{ErosionControlState, ErosionParams, ErosionUniform};
use export::GeneratorExportPlugin;
use images::{
    build_generator_image, build_normalization_image, GeneratorImage, NormalizationImage,
};
use uniforms::{GeneratorDisplayGeneration, GeneratorParamGeneration, GeneratorUniform};

pub use erosion_images::ErosionBuffers as GeneratorErosionBuffers;
pub use erosion_params::ErosionControlState as GeneratorErosionControlState;
pub use erosion_params::ErosionParams as GeneratorErosionParams;
pub use images::GeneratorImage as HeightfieldImage;
pub use params::GeneratorParams;

pub struct LandscapeGeneratorPlugin;

impl Plugin for LandscapeGeneratorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeneratorParams>()
            .init_resource::<GeneratorUniform>()
            .init_resource::<GeneratorParamGeneration>()
            .init_resource::<GeneratorDisplayGeneration>()
            .init_resource::<ErosionParams>()
            .init_resource::<ErosionUniform>()
            .insert_resource(ErosionControlState::new_dirty())
            .add_plugins((
                GeneratorComputePlugin,
                ErosionComputePlugin,
                GeneratorExportPlugin,
            ))
            .add_systems(Startup, setup_generator)
            .add_systems(
                PostUpdate,
                (
                    sync_generator_image,
                    sync_normalization_image,
                    sync_erosion_buffers,
                    sync_uniform,
                )
                    .chain(),
            );
    }
}

fn setup_generator(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    params: Res<GeneratorParams>,
) {
    let handle = build_generator_image(&mut images, params.resolution);
    commands.insert_resource(GeneratorImage {
        heightfield: handle,
    });
    let raw_handle = build_normalization_image(&mut images, params.resolution);
    commands.insert_resource(NormalizationImage {
        raw_heights: raw_handle,
    });
    commands.insert_resource(ErosionBuffers::new(&mut images, params.resolution));
}

fn sync_generator_image(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    params: Res<GeneratorParams>,
    current: Option<Res<GeneratorImage>>,
) {
    if !params.is_changed() {
        return;
    }

    let needs_rebuild = current
        .as_ref()
        .and_then(|image| images.get(&image.heightfield))
        .map(|image| {
            image.texture_descriptor.size.width != params.resolution
                || image.texture_descriptor.size.height != params.resolution
        })
        .unwrap_or(true);

    if needs_rebuild {
        let handle = build_generator_image(&mut images, params.resolution);
        commands.insert_resource(GeneratorImage {
            heightfield: handle,
        });
    }
}

fn sync_normalization_image(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    params: Res<GeneratorParams>,
    current: Option<Res<NormalizationImage>>,
) {
    if !params.is_changed() {
        return;
    }

    let needs_rebuild = current
        .as_ref()
        .and_then(|norm| images.get(&norm.raw_heights))
        .map(|image| {
            image.texture_descriptor.size.width != params.resolution
                || image.texture_descriptor.size.height != params.resolution
        })
        .unwrap_or(true);

    if needs_rebuild {
        let raw_handle = build_normalization_image(&mut images, params.resolution);
        commands.insert_resource(NormalizationImage {
            raw_heights: raw_handle,
        });
    }
}

fn sync_erosion_buffers(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    params: Res<GeneratorParams>,
    current: Option<Res<ErosionBuffers>>,
    erosion_ctrl: Option<ResMut<ErosionControlState>>,
    mut erosion_params: ResMut<ErosionParams>,
) {
    if !params.is_changed() {
        return;
    }

    let needs_rebuild = current
        .as_ref()
        .map(|eb| eb.resolution != params.resolution)
        .unwrap_or(true);

    if needs_rebuild {
        commands.insert_resource(ErosionBuffers::new(&mut images, params.resolution));
        // Invalidate any previous erosion result — it was at a different resolution.
        // Prevent copy_out from blasting the fresh raw_heights with zeros from the
        // newly-created (empty) height_a buffers.
        if let Some(ctrl) = erosion_ctrl {
            ctrl.mark_dirty();
        }
        erosion_params.enabled = false;
    }
}

fn sync_uniform(
    params: Res<GeneratorParams>,
    erosion_params: Res<ErosionParams>,
    mut uniform: ResMut<GeneratorUniform>,
    mut generation: ResMut<GeneratorParamGeneration>,
    mut display_generation: ResMut<GeneratorDisplayGeneration>,
    mut was_erosion_enabled: Local<bool>,
) {
    // Detect erosion being cleared (enabled → disabled) so we can restore the
    // un-eroded noise preview. params_changed alone isn't enough because no
    // noise/shape field changes on Clear — only erosion.enabled flips.
    let erosion_just_cleared = *was_erosion_enabled && !erosion_params.enabled;
    *was_erosion_enabled = erosion_params.enabled;

    if params.is_changed() {
        let new = GeneratorUniform::from_params(&params);
        // Check whether any *noise/shape* field changed, ignoring the display-only
        // `grayscale` flag.  If only grayscale flipped, we must NOT increment the raw
        // generation — that would re-dispatch `preview_generate_raw` and overwrite
        // the eroded `raw_heights` with fresh un-eroded noise.
        let raw_changed = new.resolution != uniform.resolution
            || new.octaves != uniform.octaves
            || new.seed != uniform.seed
            || new.offset != uniform.offset
            || new.frequency != uniform.frequency
            || new.lacunarity != uniform.lacunarity
            || new.gain != uniform.gain
            || new.height_scale != uniform.height_scale
            || new.continent_frequency != uniform.continent_frequency
            || new.continent_strength != uniform.continent_strength
            || new.ridge_strength != uniform.ridge_strength
            || new.warp_frequency != uniform.warp_frequency
            || new.warp_strength != uniform.warp_strength
            || new.erosion_strength != uniform.erosion_strength;
        *uniform = new;
        // Always bump display_generation so GeneratorNormNode re-runs the
        // normalize/colorize pass (needed for grayscale toggle, erosion, etc.).
        display_generation.0 += 1;
        // Only bump the raw generation when the noise/shape params actually changed.
        if raw_changed {
            generation.0 += 1;
        }
    }

    if erosion_just_cleared {
        // Force preview_generate_raw to re-run so raw_heights is overwritten with
        // fresh noise, replacing the eroded heights that copy_out left behind.
        display_generation.0 += 1;
        generation.0 += 1;
    }
}

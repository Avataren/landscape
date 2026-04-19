mod compute;
pub mod export;
pub mod images;
pub mod params;
mod terrain_fn;
mod uniforms;

use bevy::prelude::*;

use compute::GeneratorComputePlugin;
use images::{build_generator_image, GeneratorImage};
use uniforms::GeneratorUniform;

pub use images::GeneratorImage as HeightfieldImage;
pub use params::GeneratorParams;

pub struct LandscapeGeneratorPlugin;

impl Plugin for LandscapeGeneratorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeneratorParams>()
            .init_resource::<GeneratorUniform>()
            .add_plugins(GeneratorComputePlugin)
            .add_systems(Startup, setup_generator)
            .add_systems(PostUpdate, (sync_generator_image, sync_uniform).chain());
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

fn sync_uniform(params: Res<GeneratorParams>, mut uniform: ResMut<GeneratorUniform>) {
    if params.is_changed() {
        *uniform = GeneratorUniform::from_params(&params);
    }
}

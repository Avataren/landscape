mod compute;
pub mod export;
pub mod images;
pub mod params;
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
            .add_systems(PostUpdate, sync_uniform);
    }
}

fn setup_generator(mut commands: Commands, mut images: ResMut<Assets<Image>>, params: Res<GeneratorParams>) {
    let handle = build_generator_image(&mut images, params.resolution);
    commands.insert_resource(GeneratorImage { heightfield: handle });
}

fn sync_uniform(params: Res<GeneratorParams>, mut uniform: ResMut<GeneratorUniform>) {
    if params.is_changed() {
        *uniform = GeneratorUniform::from_params(&params);
    }
}

mod cloud_panel;
mod diffusion_panel;
mod fog_panel;
mod foliage_panel;
mod generator_panel;
mod import;
mod level_io;
mod material_panel;
mod preferences;
mod sky_panel;
mod synthesis_panel;
mod texture_browser;
mod toolbar;
mod water_panel;

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use cloud_panel::CloudPanelPlugin;
use fog_panel::FogPanelPlugin;
use foliage_panel::FoliagePanelPlugin;
use generator_panel::GeneratorPanelPlugin;
use import::ImportPlugin;
use level_io::LevelIoPlugin;
use material_panel::MaterialPanelPlugin;
use preferences::PreferencesPlugin;
use sky_panel::SkyPanelPlugin;
use synthesis_panel::SynthesisPanelPlugin;
use texture_browser::TextureBrowserPlugin;
use water_panel::WaterPanelPlugin;

pub use level_io::LevelIoState;
pub use preferences::AppPreferences;

pub struct LandscapeEditorPlugin;

impl Plugin for LandscapeEditorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.init_resource::<toolbar::ToolbarState>();
        app.add_plugins(CloudPanelPlugin)
            .add_plugins(FogPanelPlugin)
            .add_plugins(FoliagePanelPlugin)
            .add_plugins(GeneratorPanelPlugin)
            .add_plugins(MaterialPanelPlugin)
            .add_plugins(TextureBrowserPlugin)
            .add_plugins(ImportPlugin)
            .add_plugins(LevelIoPlugin)
            .add_plugins(PreferencesPlugin)
            .add_plugins(SkyPanelPlugin)
            .add_plugins(SynthesisPanelPlugin)
            .add_plugins(WaterPanelPlugin)
            .add_systems(EguiPrimaryContextPass, toolbar::toolbar_system);
    }
}

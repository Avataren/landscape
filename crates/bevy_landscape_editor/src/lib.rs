mod import;
mod level_io;
mod material_panel;
mod preferences;
mod sky_panel;
mod toolbar;

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use import::ImportPlugin;
use level_io::LevelIoPlugin;
use material_panel::MaterialPanelPlugin;
use preferences::PreferencesPlugin;
use sky_panel::SkyPanelPlugin;

pub use level_io::LevelIoState;
pub use preferences::AppPreferences;

pub struct LandscapeEditorPlugin;

impl Plugin for LandscapeEditorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.add_plugins(MaterialPanelPlugin)
            .add_plugins(ImportPlugin)
            .add_plugins(LevelIoPlugin)
            .add_plugins(PreferencesPlugin)
            .add_plugins(SkyPanelPlugin)
            .add_systems(EguiPrimaryContextPass, toolbar::toolbar_system);
    }
}

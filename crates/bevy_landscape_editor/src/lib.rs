mod material_panel;
mod toolbar;

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use material_panel::MaterialPanelPlugin;

pub struct LandscapeEditorPlugin;

impl Plugin for LandscapeEditorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.add_plugins(MaterialPanelPlugin)
            .add_systems(EguiPrimaryContextPass, toolbar::toolbar_system);
    }
}

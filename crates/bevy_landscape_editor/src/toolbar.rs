use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::import::ImportWizard;
use crate::level_io::LevelIoState;
use crate::material_panel::MaterialPanelState;
use crate::preferences::{AppPreferences, PreferencesDialog};
use crate::sky_panel::SkyPanelState;

pub(crate) fn toolbar_system(
    mut contexts: EguiContexts,
    mut app_exit: MessageWriter<AppExit>,
    mut material_panel: ResMut<MaterialPanelState>,
    mut import: ResMut<ImportWizard>,
    mut level_io: ResMut<LevelIoState>,
    prefs: Res<AppPreferences>,
    mut prefs_dialog: ResMut<PreferencesDialog>,
    mut sky_panel: ResMut<SkyPanelState>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Import Heightmap…").clicked() {
                    import.as_mut().open();
                    ui.close();
                }
                ui.separator();
                if ui.button("Save Landscape…").clicked() {
                    level_io.as_mut().start_save();
                    ui.close();
                }
                if ui.button("Load Landscape…").clicked() {
                    level_io.as_mut().start_load();
                    ui.close();
                }
                ui.separator();
                if ui.button("Preferences…").clicked() {
                    prefs_dialog.as_mut().open(&prefs);
                    ui.close();
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    app_exit.write(AppExit::Success);
                    ui.close();
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui.checkbox(&mut material_panel.open, "Materials").clicked() {
                    ui.close();
                }
                if ui.checkbox(&mut sky_panel.open, "Sky / Time of Day").clicked() {
                    ui.close();
                }
            });
        });
    });
    Ok(())
}

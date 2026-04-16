use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::material_panel::MaterialPanelState;

pub(crate) fn toolbar_system(
    mut contexts: EguiContexts,
    mut app_exit: MessageWriter<AppExit>,
    mut material_panel: ResMut<MaterialPanelState>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Exit").clicked() {
                    app_exit.write(AppExit::Success);
                    ui.close();
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui
                    .checkbox(&mut material_panel.open, "Materials")
                    .clicked()
                {
                    ui.close();
                }
            });
        });
    });
    Ok(())
}

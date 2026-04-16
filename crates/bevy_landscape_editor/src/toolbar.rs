use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub(crate) fn toolbar_system(
    mut contexts: EguiContexts,
    mut app_exit: MessageWriter<AppExit>,
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
        });
    });
    Ok(())
}

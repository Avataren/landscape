use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use bevy_landscape::{
    MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainSourceDesc,
    MAX_SUPPORTED_CLIPMAP_LEVELS,
};

use crate::fog_panel::FogPanelState;
use crate::import::ImportWizard;
use crate::level_io::LevelIoState;
use crate::material_panel::MaterialPanelState;
use crate::preferences::{AppPreferences, PreferencesDialog};
use crate::sky_panel::SkyPanelState;

#[derive(Resource)]
pub(crate) struct ToolbarState {
    view_distance_rings: u32,
    dragging_view_distance: bool,
    pending_view_distance: Option<u32>,
}

impl Default for ToolbarState {
    fn default() -> Self {
        Self {
            view_distance_rings: TerrainConfig::default().clipmap_levels,
            dragging_view_distance: false,
            pending_view_distance: None,
        }
    }
}

pub(crate) fn toolbar_system(
    mut contexts: EguiContexts,
    mut app_exit: MessageWriter<AppExit>,
    mut toolbar: ResMut<ToolbarState>,
    mut material_panel: ResMut<MaterialPanelState>,
    mut import: ResMut<ImportWizard>,
    mut level_io: ResMut<LevelIoState>,
    prefs: Res<AppPreferences>,
    mut prefs_dialog: ResMut<PreferencesDialog>,
    mut sky_panel: ResMut<SkyPanelState>,
    mut fog_panel: ResMut<FogPanelState>,
    config: Res<TerrainConfig>,
    desc: Res<TerrainSourceDesc>,
    library: Res<MaterialLibrary>,
    mut reload_tx: MessageWriter<ReloadTerrainRequest>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    if toolbar.pending_view_distance == Some(config.clipmap_levels) {
        toolbar.pending_view_distance = None;
    }
    if toolbar.pending_view_distance.is_none()
        && !toolbar.dragging_view_distance
        && toolbar.view_distance_rings != config.clipmap_levels
    {
        toolbar.view_distance_rings = config.clipmap_levels;
    }

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
                if ui
                    .checkbox(&mut sky_panel.open, "Sky / Time of Day")
                    .clicked()
                {
                    ui.close();
                }
                if ui.checkbox(&mut fog_panel.open, "Fog").clicked() {
                    ui.close();
                }
            });

            ui.separator();
            ui.label("View Distance");
            let response = ui
                .add_sized(
                    [220.0, 0.0],
                    egui::Slider::new(
                        &mut toolbar.view_distance_rings,
                        4..=MAX_SUPPORTED_CLIPMAP_LEVELS as u32,
                    )
                    .show_value(true),
                )
                .on_hover_text(
                    "Number of clipmap rings. Higher values extend terrain view distance at the cost of memory and rebuild time.",
                );

            if response.dragged() {
                toolbar.dragging_view_distance = true;
            }

            let should_apply = (response.drag_stopped()
                || (response.changed() && !toolbar.dragging_view_distance))
                && toolbar.view_distance_rings != config.clipmap_levels;

            if response.drag_stopped() {
                toolbar.dragging_view_distance = false;
            }

            if should_apply {
                let mut new_config = config.clone();
                new_config.clipmap_levels = toolbar.view_distance_rings;
                toolbar.pending_view_distance = Some(toolbar.view_distance_rings);
                toolbar.dragging_view_distance = false;
                reload_tx.write(ReloadTerrainRequest {
                    config: new_config,
                    source: desc.clone(),
                    material_library: library.clone(),
                });
            }
        });
    });
    Ok(())
}

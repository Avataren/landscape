use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use bevy_landscape::{
    MaterialLibrary, ReloadTerrainRequest, TerrainConfig, TerrainSourceDesc,
    MAX_SUPPORTED_CLIPMAP_LEVELS,
};
use bevy_landscape_water::{WaterEnabled, WaterSettings};

use crate::cloud_panel::CloudPanelState;
use crate::fog_panel::FogPanelState;
use crate::generator_panel::GeneratorPanelState;
use crate::import::ImportWizard;
use crate::level_io::LevelIoState;
use crate::material_panel::MaterialPanelState;
use crate::preferences::{AppPreferences, PreferencesDialog};
use crate::sky_panel::SkyPanelState;
use crate::synthesis_panel::SynthesisPanelState;

/// Bundled water params — counts as a single system parameter.
#[derive(SystemParam)]
pub(crate) struct WaterParams<'w> {
    settings: Option<ResMut<'w, WaterSettings>>,
    enabled: Option<Res<'w, WaterEnabled>>,
}

/// Bundled read-only terrain resources.
#[derive(SystemParam)]
pub(crate) struct TerrainParams<'w> {
    config: Res<'w, TerrainConfig>,
    desc: Res<'w, TerrainSourceDesc>,
    library: Res<'w, MaterialLibrary>,
}

/// Bundled preferences params — counts as a single system parameter.
#[derive(SystemParam)]
pub(crate) struct PrefsUi<'w> {
    prefs: Res<'w, AppPreferences>,
    dialog: ResMut<'w, PreferencesDialog>,
}

#[derive(Resource)]
pub(crate) struct ToolbarState {
    view_distance_rings: u32,
    dragging_view_distance: bool,
    pending_view_distance: Option<u32>,
    /// Water height as set by the level / generator (world units).
    water_base_height: f32,
    /// User offset applied on top of the base height, range [-100, 100] m.
    water_height_offset: f32,
    /// True once we've read the initial height from WaterSettings at least once.
    water_height_initialized: bool,
}

impl Default for ToolbarState {
    fn default() -> Self {
        Self {
            view_distance_rings: TerrainConfig::default().clipmap_levels,
            dragging_view_distance: false,
            pending_view_distance: None,
            water_base_height: 0.0,
            water_height_offset: 0.0,
            water_height_initialized: false,
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
    mut prefs_ui: PrefsUi<'_>,
    mut sky_panel: ResMut<SkyPanelState>,
    mut cloud_panel: ResMut<CloudPanelState>,
    mut fog_panel: ResMut<FogPanelState>,
    mut generator_panel: ResMut<GeneratorPanelState>,
    mut synthesis_panel: ResMut<SynthesisPanelState>,
    mut reload_tx: MessageWriter<ReloadTerrainRequest>,
    mut water: WaterParams<'_>,
    terrain: TerrainParams<'_>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    if toolbar.pending_view_distance == Some(terrain.config.clipmap_levels) {
        toolbar.pending_view_distance = None;
    }
    if toolbar.pending_view_distance.is_none()
        && !toolbar.dragging_view_distance
        && toolbar.view_distance_rings != terrain.config.clipmap_levels
    {
        toolbar.view_distance_rings = terrain.config.clipmap_levels;
    }

    // Sync water height slider from resource.
    // On first access, capture the current runtime height as the base.
    // If the level reloads and changes the base height externally, preserve
    // the user's offset and reapply it on top of the new base.
    if let Some(ref mut ws) = water.settings {
        if !toolbar.water_height_initialized {
            toolbar.water_base_height = ws.height;
            toolbar.water_height_offset = 0.0;
            toolbar.water_height_initialized = true;
        } else if !ws.is_changed() {
            let expected = toolbar.water_base_height + toolbar.water_height_offset;
            if (ws.height - expected).abs() > 0.01 {
                // External change (e.g. clipmap/view-distance reload) —
                // re-anchor the base while preserving the user offset.
                toolbar.water_base_height = ws.height;
                if toolbar.water_height_offset.abs() > 0.01 {
                    ws.height = toolbar.water_base_height + toolbar.water_height_offset;
                }
            }
        }
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
                    prefs_ui.dialog.open(&prefs_ui.prefs);
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
                if ui.checkbox(&mut cloud_panel.open, "Clouds").clicked() {
                    ui.close();
                }
                if ui.checkbox(&mut fog_panel.open, "Fog").clicked() {
                    ui.close();
                }
                if ui.checkbox(&mut generator_panel.open, "Terrain Generator").clicked() {
                    ui.close();
                }
                if ui
                    .checkbox(&mut synthesis_panel.open, "Landscape Synthesis")
                    .clicked()
                {
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
                && toolbar.view_distance_rings != terrain.config.clipmap_levels;

            if response.drag_stopped() {
                toolbar.dragging_view_distance = false;
            }

            if should_apply {
                let mut new_config = terrain.config.clone();
                new_config.clipmap_levels = toolbar.view_distance_rings;
                toolbar.pending_view_distance = Some(toolbar.view_distance_rings);
                toolbar.dragging_view_distance = false;
                reload_tx.write(ReloadTerrainRequest {
                    config: new_config,
                    source: terrain.desc.clone(),
                    material_library: terrain.library.clone(),
                });
            }

            // --- Water height slider ---
            let water_active = water.enabled.as_deref().map_or(false, |e| e.0);
            if water_active {
                ui.separator();
                ui.label("Water Offset");
                let water_resp = ui
                    .add_sized(
                        [180.0, 0.0],
                        egui::Slider::new(&mut toolbar.water_height_offset, -100.0_f32..=100.0)
                            .suffix(" m")
                            .show_value(true),
                    )
                    .on_hover_text("Offset the water plane relative to its level height (F2 to toggle water).");
                if water_resp.changed() {
                    if let Some(ref mut ws) = water.settings {
                        ws.height = toolbar.water_base_height + toolbar.water_height_offset;
                    }
                }
            }

        });
    });
    Ok(())
}

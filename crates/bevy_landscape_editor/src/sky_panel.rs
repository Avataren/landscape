use bevy::{light::light_consts::lux, prelude::*};
use bevy_egui::{egui, EguiContexts};
use serde::{Deserialize, Serialize};

#[derive(Resource, Serialize, Deserialize)]
pub struct TimeOfDay {
    pub hours: f32,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self { hours: 10.0 }
    }
}

#[derive(Resource, Default)]
pub struct SkyPanelState {
    pub open: bool,
}

pub struct SkyPanelPlugin;

impl Plugin for SkyPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimeOfDay>()
            .init_resource::<SkyPanelState>()
            .add_systems(bevy_egui::EguiPrimaryContextPass, sky_panel_system)
            .add_systems(Update, update_sun_from_time);
    }
}

pub(crate) fn sky_panel_system(
    mut contexts: EguiContexts,
    mut state: ResMut<SkyPanelState>,
    mut tod: ResMut<TimeOfDay>,
) -> Result {
    if !state.open {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("Sky / Time of Day")
        .open(&mut state.open)
        .resizable(false)
        .default_width(640.0)
        .show(ctx, |ui| {
            let h = tod.hours;
            let hh = h as u32;
            let mm = ((h - hh as f32) * 60.0) as u32;
            ui.label(format!("Time: {:02}:{:02}", hh, mm));
            ui.spacing_mut().slider_width = 580.0;
            ui.add(
                egui::Slider::new(&mut tod.hours, 0.0_f32..=24.0)
                    .show_value(false)
                    .step_by(1.0 / 60.0),
            );
        });
    Ok(())
}

fn update_sun_from_time(
    tod: Res<TimeOfDay>,
    mut lights: Query<(&mut Transform, &mut DirectionalLight)>,
) {
    if !tod.is_changed() {
        return;
    }

    // solar_angle sweeps a full circle over 24h, offset so that:
    //   6h  → solar_angle = 0        → elevation = sin(0)    = 0  (sunrise)
    //   12h → solar_angle = π/2      → elevation = sin(π/2)  = 1  (noon peak)
    //   18h → solar_angle = π        → elevation = sin(π)    = 0  (sunset)
    //   0h  → solar_angle = -π/2     → elevation = sin(-π/2) = -1 (midnight)
    let solar_angle = (tod.hours - 6.0) / 24.0 * std::f32::consts::TAU;

    // 30° max elevation keeps the sun at a low-to-mid angle throughout the
    // day: directional shadows remain visible, and the Atmosphere always
    // scatters some warm colour.  60° produced harsh overhead noon lighting
    // that washed out terrain colours even at the same RAW_SUNLIGHT value.
    let max_elevation = 30.0_f32.to_radians();
    let elevation_rad = solar_angle.sin() * max_elevation;

    // Negative X pitch tilts light downward; 0 = on horizon, -max_elev = noon.
    let pitch = -elevation_rad;

    // Azimuth: east (yaw = -π/2) at 6h → south (0) at 12h → west (+π/2) at 18h.
    let yaw = (tod.hours - 12.0) / 6.0 * std::f32::consts::FRAC_PI_2;

    for (mut transform, mut light) in &mut lights {
        transform.rotation = Quat::from_euler(EulerRot::XYZ, pitch, yaw, 0.0);

        light.illuminance = if elevation_rad <= 0.0 {
            0.0
        } else {
            let daylight = (elevation_rad / max_elevation).clamp(0.0, 1.0);
            // Avoid starving the atmosphere/environment-map pass at sunrise and
            // sunset. The low sun angle and atmospheric transmittance still keep
            // the direct light subdued, but sky fill remains visible.
            lux::RAW_SUNLIGHT * daylight.max(0.25)
        };

        light.shadows_enabled = elevation_rad > 0.05;
    }
}

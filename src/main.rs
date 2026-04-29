mod app_config;
mod player;

use bevy::{
    camera::{Exposure, ScreenSpaceTransmissionQuality},
    core_pipeline::{prepass::DepthPrepass, tonemapping::Tonemapping},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::AccumulatedMouseMotion,
    light::{
        light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, FogVolume,
        VolumetricLight,
    },
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium},
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        render_resource::WgpuFeatures,
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
    window::PrimaryWindow,
};
use bevy_landscape::{
    level::load_level, DetailSynthesisConfig, FoliageSourceDesc, GpuGrassPlugin, TerrainCamera,
    TerrainConfig, TerrainDebugPlugin, TerrainPlugin, TerrainSourceDesc, TerrainSystemSet,
};
use bevy_landscape_clouds::{CloudsConfig, VolumetricCloudsPlugin};
use bevy_landscape_editor::{AppPreferences, LandscapeEditorPlugin};
use bevy_landscape_generator::LandscapeGeneratorPlugin;
use bevy_landscape_water::{LandscapeWaterPlugin, OceanFftSettings, WaterSettings};
use player::{CameraMode, PlayerPlugin, PlayerSystemSet};
use serde::Deserialize;

const WINDOW_TITLE: &str = "Landscape Renderer";

#[derive(Resource)]
struct ShadowsEnabled(bool);

#[derive(Deserialize)]
struct WaterSettingsDto {
    amplitude: f32,
    wave_speed: f32,
    wave_direction: [f32; 2],
    clarity: f32,
    deep_color: [f32; 3],
    shallow_color: [f32; 3],
    edge_scale: f32,
    edge_color: [f32; 3],
    refraction_strength: f32,
    foam_threshold: f32,
    foam_color: [f32; 3],
    shoreline_foam_depth: f32,
    shore_wave_damp_width: f32,
    jacobian_foam_strength: f32,
    capillary_strength: f32,
    macro_noise_amplitude: f32,
    macro_noise_scale: f32,
}

impl WaterSettingsDto {
    fn apply_to(&self, s: &mut WaterSettings) {
        fn color(rgb: [f32; 3]) -> Color {
            Color::linear_rgb(rgb[0], rgb[1], rgb[2])
        }

        s.amplitude = self.amplitude;
        s.wave_speed = self.wave_speed;
        s.wave_direction = Vec2::new(self.wave_direction[0], self.wave_direction[1]);
        s.clarity = self.clarity;
        s.deep_color = color(self.deep_color);
        s.shallow_color = color(self.shallow_color);
        s.edge_scale = self.edge_scale;
        s.edge_color = color(self.edge_color);
        s.refraction_strength = self.refraction_strength;
        s.foam_threshold = self.foam_threshold;
        s.foam_color = color(self.foam_color);
        s.shoreline_foam_depth = self.shoreline_foam_depth;
        s.shore_wave_damp_width = self.shore_wave_damp_width;
        s.jacobian_foam_strength = self.jacobian_foam_strength;
        s.capillary_strength = self.capillary_strength;
        s.macro_noise_amplitude = self.macro_noise_amplitude;
        s.macro_noise_scale = self.macro_noise_scale;
    }
}

#[derive(Deserialize)]
struct OceanFftDto {
    enabled: bool,
    size: u32,
    world_size: f32,
    wind_speed: f32,
    wind_direction: [f32; 2],
    amplitude: f32,
    choppy: f32,
    seed: u32,
    strength: f32,
}

impl OceanFftDto {
    fn into_settings(self) -> OceanFftSettings {
        OceanFftSettings {
            enabled: self.enabled,
            size: self.size,
            world_size: self.world_size,
            wind_speed: self.wind_speed,
            wind_direction: Vec2::new(self.wind_direction[0], self.wind_direction[1]),
            amplitude: self.amplitude,
            choppy: self.choppy,
            seed: self.seed,
            strength: self.strength,
        }
    }
}

#[derive(Deserialize)]
struct WaterDesc {
    settings: WaterSettingsDto,
    fft: OceanFftDto,
}

#[derive(Deserialize)]
struct SynthesisDto {
    enabled: bool,
    max_amplitude: f32,
    lacunarity: f32,
    gain: f32,
    erosion_strength: f32,
    seed: [f32; 2],
    slope_mask_threshold_deg: f32,
    slope_mask_falloff_deg: f32,
    normal_detail_strength: f32,
}

impl SynthesisDto {
    fn into_config(self) -> DetailSynthesisConfig {
        DetailSynthesisConfig {
            enabled: self.enabled,
            max_amplitude: self.max_amplitude,
            lacunarity: self.lacunarity,
            gain: self.gain,
            erosion_strength: self.erosion_strength,
            seed: Vec2::new(self.seed[0], self.seed[1]),
            slope_mask_threshold_deg: self.slope_mask_threshold_deg,
            slope_mask_falloff_deg: self.slope_mask_falloff_deg,
            normal_detail_strength: self.normal_detail_strength,
        }
    }
}

fn initial_water_settings(water_height: Option<f32>) -> WaterSettings {
    WaterSettings {
        height: water_height.unwrap_or(0.0),
        amplitude: 6.75,
        clarity: 0.94,
        wave_speed: 2.35,
        foam_threshold: 0.7,
        shoreline_foam_depth: 1.5,
        shore_wave_damp_width: 3.0,
        spawn_tiles: None,
        ..WaterSettings::default()
    }
}

fn main() {
    // Priority: --level arg > preferences default > empty editor (no terrain)
    let level_arg = {
        let args: Vec<String> = std::env::args().collect();
        let pos = args.iter().position(|a| a == "--level");
        pos.and_then(|i| args.get(i + 1)).cloned()
    };
    let level_arg = level_arg.or_else(|| AppPreferences::load().default_level);

    let (
        terrain_config,
        terrain_source,
        loaded_library,
        clouds_config,
        water_plugin,
        water_desc,
        synthesis_config,
    ) = if let Some(ref path) = level_arg {
        match load_level(path) {
            Ok(desc) => {
                let loaded_clouds: Option<CloudsConfig> = desc
                    .clouds
                    .as_ref()
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let loaded_water: Option<WaterDesc> = desc
                    .water
                    .as_ref()
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let loaded_synthesis: Option<SynthesisDto> = desc
                    .synthesis
                    .as_ref()
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                let (mut config, source, library, wmin, wmax, meta) = desc.into_runtime();
                config.height_scale *= 1.0; // into_runtime already multiplies
                let water_height = meta
                    .water_level
                    .filter(|wl| *wl > 0.0)
                    .map(|wl| wl * config.height_scale);
                let water = LandscapeWaterPlugin {
                    water_height,
                    world_min: wmin,
                    world_max: wmax,
                };
                (
                    config,
                    source,
                    Some(library),
                    loaded_clouds,
                    water,
                    loaded_water,
                    loaded_synthesis,
                )
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load level '{path}': {e}. Starting with empty editor."
                );
                (
                    TerrainConfig::default(),
                    TerrainSourceDesc::default(),
                    None,
                    None,
                    LandscapeWaterPlugin::default(),
                    None,
                    None,
                )
            }
        }
    } else {
        // No level configured: use landscape.toml as the editable workspace terrain.
        // Use File → Import Heightmap to load data, then Save Landscape and
        // set it as the default level in the preferences to auto-load on startup.
        let (config, source) = app_config::load().into_runtime();
        let water = LandscapeWaterPlugin {
            world_min: source.world_min,
            world_max: source.world_max,
            ..default()
        };
        (config, source, None, None, water, None, None)
    };

    let mut app = App::new();
    if let Some(cc) = clouds_config {
        app.insert_resource(cc);
    }
    if let Some(lib) = loaded_library {
        app.insert_resource(lib);
    }
    // Runtime grass is generated on the GPU. The foliage root is still kept as
    // an authoring location for future brush-painted density/type masks.
    let foliage_root_str = terrain_source.foliage_root.clone();
    if let Some(root) = &foliage_root_str {
        app.insert_resource(FoliageSourceDesc {
            foliage_root: Some(std::path::PathBuf::from(root)),
        });
    }
    let water_height_for_settings = water_plugin.water_height;

    app.insert_resource(ClearColor(Color::BLACK))
        // Terrain uses the atmosphere cubemap for ambient IBL; non-terrain PBR
        // objects also receive IBL from AtmosphereEnvironmentMapLight on the camera.
        .insert_resource(GlobalAmbientLight {
            brightness: 0.0,
            ..default()
        })
        .insert_resource(ShadowsEnabled(true))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: WgpuFeatures::POLYGON_MODE_LINE,
                        ..default()
                    }),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: WINDOW_TITLE.into(),
                        present_mode: bevy::window::PresentMode::Immediate,
                        resolution: (1920u32, 1080u32).into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(TerrainPlugin {
            config: terrain_config,
            source: terrain_source,
        })
        .add_plugins(water_plugin);

    if let Some(water) = water_desc {
        let mut settings = initial_water_settings(water_height_for_settings);
        water.settings.apply_to(&mut settings);
        app.insert_resource(settings);
        app.insert_resource(water.fft.into_settings());
    }
    if let Some(synthesis) = synthesis_config {
        app.insert_resource(synthesis.into_config());
    }

    app.add_plugins(VolumetricCloudsPlugin)
        .add_plugins(LandscapeGeneratorPlugin)
        .add_plugins(TerrainDebugPlugin)
        .add_plugins(GpuGrassPlugin)
        .add_plugins(LandscapeEditorPlugin)
        .add_plugins(PlayerPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (camera_move, camera_look)
                .chain()
                .after(PlayerSystemSet::Movement)
                .before(PlayerSystemSet::CameraSync)
                .before(TerrainSystemSet::View),
        )
        .add_systems(Update, toggle_shadows)
        .add_systems(Update, update_window_title)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    desc: Res<TerrainSourceDesc>,
) {
    // Spawn camera above terrain centre, angled downward so terrain is visible
    // immediately in freecam mode.  preload_terrain_startup uses the XZ position
    // to decide which tiles to load first.
    commands.spawn((
        Camera3d {
            screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
            ..default()
        },
        DepthPrepass,
        // 4× MSAA — switched away from TAA because the temporal accumulation
        // smeared the high-frequency synthesis detail and ghosted around fast
        // camera movement.  4× is the typical sweet spot for forward-rendered
        // terrain on desktop GPUs.
        Msaa::Sample4,
        Projection::Perspective(PerspectiveProjection {
            near: 0.1,
            // 2 000 km covers a 700 km terrain fully regardless of camera
            // position.  Bevy uses reverse-Z by default which keeps depth
            // precision acceptable even at this range.
            far: 2_000_000.0,
            ..default()
        }),
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        AtmosphereEnvironmentMapLight {
            // The terrain relies on the atmosphere environment map for diffuse
            // sky fill. A modest lift keeps low-angle sun scenes readable
            // without reintroducing flat global ambient by default.
            intensity: 2.0,
            ..default()
        },
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        TerrainCamera::default(),
        Transform::from_xyz(0.0, 800.0, 1200.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            // Use raw sunlight so the atmosphere does the scattering itself.
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        // Low-angle sun (~14° elevation) casts long shadows across terrain features.
        // X rotates the light down from horizontal; Y sets the azimuth.
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.25, 0.8, 0.0)),
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            minimum_distance: 1.0,
            first_cascade_far_bound: 400.0,
            maximum_distance: 5_000.0,
            overlap_proportion: 0.3,
        }
        .build(),
        VolumetricLight,
    ));

    // FogVolume renders only when the camera is OUTSIDE the volume: Bevy uses
    // back-face culling (cull_mode: Back), so when the camera is inside the
    // AABB all visible faces are back-faces and nothing is drawn.
    // Solution: keep the volume as a ground-level slab the camera looks down
    // at from above. The slab covers the full terrain footprint + generous XZ
    // padding, but only reaches halfway up the terrain height so the camera
    // (starting at 800 m) is above it and can see the fog below.
    // When the camera descends into the slab the effect disappears naturally
    // (like flying into real low-lying cloud).
    let fog_center_xz = (desc.world_min + desc.world_max) * 0.5;
    // Fog slab: sea level (-500 m) to 600 m altitude.
    // Center at 50 m, half-height 550 m → top = 600 m.
    // Camera starts at 800 m → 200 m above the slab top.
    commands.spawn((
        FogVolume {
            density_factor: 0.0,
            ..default()
        },
        Transform::from_xyz(fog_center_xz.x, 50.0, fog_center_xz.y)
            .with_scale(Vec3::new(200_000.0, 1100.0, 200_000.0)),
    ));
}

// ---------------------------------------------------------------------------
// Keyboard movement
// WASD move in the camera's horizontal plane (projected onto XZ).
// Q/E move straight up/down.
// Hold Shift for 5× speed.
// ---------------------------------------------------------------------------
fn camera_move(
    mode: Res<CameraMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    if *mode != CameraMode::Freecam {
        return;
    }
    let Ok(mut t) = query.single_mut() else {
        return;
    };

    let speed = if keys.pressed(KeyCode::ShiftLeft) {
        5000.0
    } else {
        500.0
    };
    let dt = time.delta_secs();

    let forward: Vec3 = t.forward().into();
    let right: Vec3 = t.right().into();

    if keys.pressed(KeyCode::KeyW) {
        t.translation += forward * speed * dt;
    }
    if keys.pressed(KeyCode::KeyS) {
        t.translation -= forward * speed * dt;
    }
    if keys.pressed(KeyCode::KeyA) {
        t.translation -= right * speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        t.translation += right * speed * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        t.translation.y += speed * dt;
    }
    if keys.pressed(KeyCode::KeyQ) {
        t.translation.y -= speed * dt;
    }
}

// ---------------------------------------------------------------------------
// Mouse look — hold right mouse button and drag to look around.
// Yaw rotates around the world Y axis; pitch rotates around the local X axis.
// ---------------------------------------------------------------------------
fn camera_look(
    mode: Res<CameraMode>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut query: Query<&mut Transform, With<TerrainCamera>>,
) {
    if *mode != CameraMode::Freecam {
        return;
    }
    if !mouse_buttons.pressed(MouseButton::Right) {
        return;
    }
    let delta = mouse_motion.delta;
    if delta == Vec2::ZERO {
        return;
    }
    let Ok(mut t) = query.single_mut() else {
        return;
    };

    let yaw = -delta.x * 0.002;
    let pitch = -delta.y * 0.002;

    // Rotate yaw around world Y (prevents roll accumulation).
    let yaw_rot = Quat::from_rotation_y(yaw);
    // Rotate pitch around the camera's local X axis.
    let right: Vec3 = t.right().into();
    let pitch_rot = Quat::from_axis_angle(right, pitch);

    t.rotation = (yaw_rot * pitch_rot * t.rotation).normalize();
}

fn toggle_shadows(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut enabled: ResMut<ShadowsEnabled>,
    mut lights: Query<(Entity, &mut DirectionalLight)>,
) {
    if !keys.just_pressed(KeyCode::F7) {
        return;
    }
    enabled.0 = !enabled.0;
    for (entity, mut light) in &mut lights {
        light.shadows_enabled = enabled.0;
        // VolumetricLight requires the shadow map to be active; remove it when
        // shadows are disabled so Bevy doesn't keep the shadow pass alive.
        if enabled.0 {
            commands.entity(entity).insert(VolumetricLight);
        } else {
            commands.entity(entity).remove::<VolumetricLight>();
        }
    }
    info!("Shadows {}", if enabled.0 { "ON (F7)" } else { "OFF (F7)" });
}

fn update_window_title(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut update_timer: Local<Option<Timer>>,
) {
    let timer = update_timer.get_or_insert_with(|| Timer::from_seconds(0.25, TimerMode::Repeating));
    if !timer.tick(time.delta()).just_finished() {
        return;
    }

    let Ok(mut window) = windows.single_mut() else {
        return;
    };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed());

    window.title = match fps {
        Some(fps) => format!("{WINDOW_TITLE} - {:.0} FPS", fps),
        None => format!("{WINDOW_TITLE} - -- FPS"),
    };
}

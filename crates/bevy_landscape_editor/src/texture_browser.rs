use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Mutex};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};
use bevy_landscape::{MaterialLibrary, PbrTexturesDirty};

// ─── Data ────────────────────────────────────────────────────────────────────

pub struct TextureSet {
    pub name: String,
    pub albedo: Option<PathBuf>,
    pub normal: Option<PathBuf>,
    pub orm: Option<PathBuf>,
    pub height: Option<PathBuf>,
}

enum ThumbnailState {
    Pending,
    Loading(Mutex<mpsc::Receiver<Option<(u32, u32, Vec<u8>)>>>),
    Ready(egui::TextureHandle),
    Failed,
}

pub struct BrowserEntry {
    set: TextureSet,
    thumb: ThumbnailState,
}

// ─── Resource ────────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct TextureBrowser {
    pub open: bool,
    pub target_slot: Option<usize>,
    entries: Vec<BrowserEntry>,
    previews: HashMap<PathBuf, ThumbnailState>,
    scanned: bool,
}

impl TextureBrowser {
    pub fn open_for(&mut self, slot: usize) {
        self.target_slot = Some(slot);
        self.open = true;
        if !self.scanned {
            self.scan();
        }
    }

    pub fn rescan(&mut self) {
        self.entries.clear();
        self.scanned = false;
        self.scan();
    }

    fn scan(&mut self) {
        let root = PathBuf::from("assets/textures");
        self.entries = scan_texture_sets(&root)
            .into_iter()
            .map(|set| BrowserEntry {
                set,
                thumb: ThumbnailState::Pending,
            })
            .collect();
        self.scanned = true;
    }

    pub(crate) fn preview_for(
        &mut self,
        ctx: &egui::Context,
        path: Option<&Path>,
    ) -> Option<egui::TextureHandle> {
        let path = path.map(resolve_texture_path)?;
        let thumb = self
            .previews
            .entry(path.clone())
            .or_insert(ThumbnailState::Pending);
        request_thumbnail(thumb, path.clone());
        poll_thumbnail(thumb, ctx, &format!("material-preview:{}", path.display()));
        match thumb {
            ThumbnailState::Ready(handle) => Some(handle.clone()),
            _ => None,
        }
    }
}

// ─── Plugin ──────────────────────────────────────────────────────────────────

pub struct TextureBrowserPlugin;

impl Plugin for TextureBrowserPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TextureBrowser>()
            .add_systems(EguiPrimaryContextPass, texture_browser_system);
    }
}

// ─── System ──────────────────────────────────────────────────────────────────

fn texture_browser_system(
    mut contexts: EguiContexts,
    mut browser: ResMut<TextureBrowser>,
    mut library: ResMut<MaterialLibrary>,
    mut pbr_dirty: ResMut<PbrTexturesDirty>,
) -> Result {
    if !browser.open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;

    // Kick off pending thumbnail loads on background threads.
    for entry in &mut browser.entries {
        if let Some(ref path) = entry.set.albedo {
            request_thumbnail(&mut entry.thumb, path.clone());
        }
    }

    // Poll loading thumbnails, upgrade to Ready via egui's texture cache.
    for entry in &mut browser.entries {
        if let Some(ref path) = entry.set.albedo {
            poll_thumbnail(
                &mut entry.thumb,
                ctx,
                &format!("texture-browser:{}", path.display()),
            );
        }
    }

    // Draw window.
    let mut open = browser.open;
    let mut selected: Option<usize> = None;
    let mut do_rescan = false;

    let target_name = browser
        .target_slot
        .and_then(|i| library.slots.get(i))
        .map(|s| s.name.clone())
        .unwrap_or_default();

    egui::Window::new(format!("Texture Atlas — {target_name}"))
        .id(egui::Id::new("texture_browser"))
        .open(&mut open)
        .default_width(660.0)
        .default_height(540.0)
        .min_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} texture sets", browser.entries.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("↺ Rescan").clicked() {
                        do_rescan = true;
                    }
                });
            });

            // Legend
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                dot_legend(ui, egui::Color32::from_rgb(100, 150, 255), "Normal");
                dot_legend(ui, egui::Color32::from_rgb(100, 220, 100), "ORM");
                dot_legend(ui, egui::Color32::from_rgb(220, 180, 100), "Height");
            });

            ui.separator();

            if browser.entries.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        "No texture sets found.\n\
                         Place Poly Haven-style PBR sets in assets/textures/\n\
                         (files named *_diff_4k.jpg / .png).",
                    );
                });
                return;
            }

            egui::ScrollArea::vertical()
                .id_salt("tex_browser_scroll")
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                        for (i, entry) in browser.entries.iter().enumerate() {
                            if texture_card(ui, entry).clicked() {
                                selected = Some(i);
                            }
                        }
                    });
                });
        });

    browser.open = open;

    if do_rescan {
        browser.rescan();
        return Ok(());
    }

    if let Some(entry_idx) = selected {
        if let (Some(slot_idx), Some(entry)) = (browser.target_slot, browser.entries.get(entry_idx))
        {
            if let Some(slot) = library.slots.get_mut(slot_idx) {
                // Slot paths are relative to assets/ (build functions prepend "assets/").
                // Browser scans from "assets/textures", so strip the "assets/" prefix.
                slot.albedo_path = entry.set.albedo.as_ref().map(strip_assets_prefix);
                slot.normal_path = entry.set.normal.as_ref().map(strip_assets_prefix);
                slot.orm_path = entry.set.orm.as_ref().map(strip_assets_prefix);
                slot.height_path = entry.set.height.as_ref().map(strip_assets_prefix);
                // Auto-name the slot from the texture set if it still has the default name.
                if slot.name.starts_with("Slot ") {
                    slot.name = entry.set.name.clone();
                }
            }
        }
        // Signal the terrain system to rebuild PBR texture arrays.
        pbr_dirty.0 = true;
        browser.open = false;
    }

    Ok(())
}

// ─── Card widget ─────────────────────────────────────────────────────────────

const THUMB: f32 = 120.0;
const CARD_W: f32 = THUMB + 8.0;
const CARD_H: f32 = THUMB + 36.0;

fn texture_card(ui: &mut egui::Ui, entry: &BrowserEntry) -> egui::Response {
    let (card_rect, response) =
        ui.allocate_exact_size(egui::vec2(CARD_W, CARD_H), egui::Sense::click());

    if ui.is_rect_visible(card_rect) {
        let bg = if response.hovered() {
            egui::Color32::from_rgb(60, 80, 110)
        } else {
            egui::Color32::from_gray(45)
        };
        ui.painter().rect_filled(card_rect, 6.0, bg);

        let thumb_rect = egui::Rect::from_min_size(
            card_rect.min + egui::vec2(4.0, 4.0),
            egui::vec2(THUMB, THUMB),
        );

        match &entry.thumb {
            ThumbnailState::Ready(handle) => {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                ui.painter()
                    .image(handle.id(), thumb_rect, uv, egui::Color32::WHITE);
            }
            ThumbnailState::Failed => {
                ui.painter()
                    .rect_filled(thumb_rect, 2.0, egui::Color32::from_rgb(50, 20, 20));
                ui.painter().text(
                    thumb_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "?",
                    egui::FontId::proportional(28.0),
                    egui::Color32::DARK_RED,
                );
            }
            _ => {
                // Still loading — grey placeholder
                ui.painter()
                    .rect_filled(thumb_rect, 2.0, egui::Color32::from_gray(30));
            }
        }

        // Name label below thumbnail
        let name = clamp_str(&entry.set.name, 16);
        ui.painter().text(
            egui::pos2(card_rect.center().x, thumb_rect.max.y + 4.0),
            egui::Align2::CENTER_TOP,
            name,
            egui::FontId::proportional(11.0),
            egui::Color32::LIGHT_GRAY,
        );

        // PBR map availability dots
        let dot_y = card_rect.max.y - 7.0;
        let colors = [
            (
                entry.set.normal.is_some(),
                egui::Color32::from_rgb(100, 150, 255),
            ),
            (
                entry.set.orm.is_some(),
                egui::Color32::from_rgb(100, 220, 100),
            ),
            (
                entry.set.height.is_some(),
                egui::Color32::from_rgb(220, 180, 100),
            ),
        ];
        let total_w = colors.len() as f32 * 12.0 - 2.0;
        let dot_x0 = card_rect.center().x - total_w / 2.0 + 5.0;
        for (i, (present, color)) in colors.iter().enumerate() {
            let cx = dot_x0 + i as f32 * 12.0;
            let center = egui::pos2(cx, dot_y);
            if *present {
                ui.painter().circle_filled(center, 4.0, *color);
            } else {
                ui.painter().circle_stroke(
                    center,
                    4.0,
                    egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
                );
            }
        }
    }

    response.on_hover_text(format!(
        "{}\nNormal: {}\nORM: {}\nHeight: {}",
        entry.set.name,
        avail(entry.set.normal.as_ref()),
        avail(entry.set.orm.as_ref()),
        avail(entry.set.height.as_ref()),
    ))
}

fn dot_legend(ui: &mut egui::Ui, color: egui::Color32, label: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
    ui.label(
        egui::RichText::new(label)
            .small()
            .color(egui::Color32::GRAY),
    );
    ui.add_space(6.0);
}

fn avail(p: Option<&PathBuf>) -> &'static str {
    if p.is_some() {
        "available"
    } else {
        "not found"
    }
}

fn clamp_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

// ─── Directory scan ───────────────────────────────────────────────────────────

fn scan_texture_sets(root: &Path) -> Vec<TextureSet> {
    let mut albedo_files = Vec::new();
    collect_albedo_files(root, &mut albedo_files);

    let mut sets: Vec<TextureSet> = albedo_files
        .into_iter()
        .filter_map(derive_texture_set)
        .collect();

    sets.sort_by(|a, b| a.name.cmp(&b.name));
    sets
}

fn collect_albedo_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_albedo_files(&path, out);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let lo = name.to_ascii_lowercase();
            if lo.contains("_diff_") && (lo.ends_with(".jpg") || lo.ends_with(".png")) {
                out.push(path);
            }
        }
    }
}

fn derive_texture_set(albedo: PathBuf) -> Option<TextureSet> {
    let dir = albedo.parent()?;
    let file_name = albedo.file_name()?.to_str()?;
    let lo = file_name.to_ascii_lowercase();
    let prefix_end = lo.find("_diff_")?;
    let prefix = &file_name[..prefix_end];

    let name = prefix
        .split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    let normal = find_pbr_file(dir, prefix, "_nor_gl_");
    let orm = find_pbr_file(dir, prefix, "_rough_");
    let height = find_pbr_file(dir, prefix, "_disp_");

    Some(TextureSet {
        name,
        albedo: Some(albedo),
        normal,
        orm,
        height,
    })
}

/// Prefer PNG > JPG > EXR for each PBR map (best quality available without EXR overhead).
fn find_pbr_file(dir: &Path, prefix: &str, pattern: &str) -> Option<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return None;
    };
    let prefix_lo = prefix.to_ascii_lowercase();
    let mut best: Option<(u8, PathBuf)> = None;

    for entry in rd.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lo = name.to_ascii_lowercase();
        if !lo.starts_with(&prefix_lo) || !lo.contains(pattern) {
            continue;
        }
        let priority = if lo.ends_with(".png") {
            3
        } else if lo.ends_with(".jpg") || lo.ends_with(".jpeg") {
            2
        } else if lo.ends_with(".exr") {
            1
        } else {
            continue;
        };
        if best.as_ref().map_or(true, |(p, _)| priority > *p) {
            best = Some((priority, path));
        }
    }

    best.map(|(_, p)| p)
}

// ─── Path helpers ─────────────────────────────────────────────────────────────

/// Slot paths are stored relative to `assets/` (build functions prepend it).
/// The browser scans from `assets/textures`, so strip the leading `assets/`.
fn strip_assets_prefix(path: &PathBuf) -> PathBuf {
    path.strip_prefix("assets")
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| path.clone())
}

// ─── Background thumbnail loading ─────────────────────────────────────────────

fn resolve_texture_path(path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with("assets") {
        path.to_path_buf()
    } else {
        PathBuf::from("assets").join(path)
    }
}

fn request_thumbnail(state: &mut ThumbnailState, path: PathBuf) {
    if !matches!(state, ThumbnailState::Pending) {
        return;
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(load_thumbnail(&path));
    });
    *state = ThumbnailState::Loading(Mutex::new(rx));
}

fn poll_thumbnail(state: &mut ThumbnailState, ctx: &egui::Context, cache_key: &str) {
    let old = std::mem::replace(state, ThumbnailState::Pending);
    *state = match old {
        ThumbnailState::Loading(rx) => {
            let received = match rx.lock() {
                Ok(guard) => guard.try_recv().ok(),
                Err(_) => Some(None),
            };
            match received {
                Some(Some((w, h, bytes))) => {
                    let ci =
                        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &bytes);
                    ThumbnailState::Ready(ctx.load_texture(
                        cache_key,
                        ci,
                        egui::TextureOptions::LINEAR,
                    ))
                }
                Some(None) => ThumbnailState::Failed,
                None => ThumbnailState::Loading(rx),
            }
        }
        other => other,
    };
}

fn load_thumbnail(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let img = image::open(path).ok()?;
    let thumb = img.resize_to_fill(128, 128, image::imageops::FilterType::Triangle);
    let rgba = thumb.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some((w, h, rgba.into_raw()))
}

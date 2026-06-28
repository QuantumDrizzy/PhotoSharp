//! PhotoSharp GUI — a native (egui) front-end for the lucky-imaging stacker.
//!
//! Open a telescope video, tune the controls, stack, compare before/after, export. The
//! stacking runs on a worker thread (with progress) so the window stays responsive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // no console window in release

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use eframe::egui;
use photosharp_core::{decode, image_io, pipeline, roi, Gray};

const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x25, 0x63, 0xeb); // a calm blue
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(0xf0, 0xf2, 0xf6); // soft off-white
const SAVE_GREEN: egui::Color32 = egui::Color32::from_rgb(0x16, 0x7a, 0x3b); // quick-save button
const STAGE_BG: egui::Color32 = egui::Color32::from_rgb(0x0d, 0x11, 0x17); // dark image stage

/// The repo-local `captures/` folder (created on demand, already gitignored) — every stacked
/// result lands here so they pile up in one known place instead of being hunted in dialogs.
fn captures_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("captures");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Next free `captures/<stem>-NN.png`, so repeated saves in a session never overwrite.
fn next_capture_path(stem: &str) -> std::path::PathBuf {
    let dir = captures_dir();
    (1..1000)
        .map(|n| dir.join(format!("{stem}-{n:02}.png")))
        .find(|p| !p.exists())
        .unwrap_or_else(|| dir.join(format!("{stem}.png")))
}

enum Msg {
    Progress { stage: &'static str, done: usize, total: usize },
    Done { stacked: Box<Gray>, single: Box<Gray>, report: String },
    Cancelled,
    Error(String),
}

#[derive(PartialEq, Clone, Copy)]
enum Status {
    Idle,
    Running,
    Done,
    Failed,
}

#[derive(PartialEq, Clone, Copy)]
enum View {
    Stacked,
    Single,
}

struct App {
    video: Option<PathBuf>,
    info: Option<decode::Summary>,
    roi: usize,
    max_frames: usize,
    keep: f32,
    centroid_k: f32,
    sigma: f32,
    amount: f32,
    status: Status,
    rx: Option<Receiver<Msg>>,
    cancel: Arc<AtomicBool>,
    progress: f32,
    phase: String,
    stacked: Option<Gray>,
    single: Option<Gray>,
    tex_stacked: Option<egui::TextureHandle>,
    tex_single: Option<egui::TextureHandle>,
    view: View,
    report: String,
    saved: Option<String>,
    error: Option<String>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            video: None,
            info: None,
            roi: 512,
            max_frames: 1000,
            keep: 0.3,
            centroid_k: 1.0,
            sigma: 1.5,
            amount: 1.0,
            status: Status::Idle,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: 0.0,
            phase: String::new(),
            stacked: None,
            single: None,
            tex_stacked: None,
            tex_single: None,
            view: View::Stacked,
            report: String::new(),
            saved: None,
            error: None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_stack(
    video: PathBuf,
    roi_size: usize,
    max_frames: usize,
    total_hint: usize,
    keep: f32,
    centroid_k: f32,
    sigma: f32,
    amount: f32,
    cancel: &AtomicBool,
    tx: &Sender<Msg>,
) -> anyhow::Result<bool> {
    let p = video.to_str().ok_or_else(|| anyhow::anyhow!("video path is not valid UTF-8"))?;
    let mut frames: Vec<Gray> = Vec::new();
    decode::decode_gray(p, max_frames, Some(cancel), |i, frame| {
        let (cx, cy) = roi::bright_centroid(&frame, centroid_k);
        frames.push(roi::crop_centered(&frame, cx, cy, roi_size));
        if i % 4 == 0 {
            // total_hint = the video's real frame count when known (capped by max_frames),
            // so the bar reflects reality instead of always dividing by the cap.
            let _ = tx.send(Msg::Progress { stage: "decoding", done: i + 1, total: total_hint.max(i + 1) });
        }
    })?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(false);
    }
    let n = frames.len();
    let _ = tx.send(Msg::Progress { stage: "decoding", done: n, total: n }); // decode complete
    if frames.is_empty() {
        anyhow::bail!("decoded 0 frames — is this a video, and is ffmpeg on PATH?");
    }

    let params = pipeline::Params { keep_fraction: keep, unsharp_sigma: sigma, unsharp_amount: amount };
    let (img, rep) = pipeline::process_progress(&frames, &params, |stage, done, total| {
        let _ = tx.send(Msg::Progress { stage, done, total });
    });
    if cancel.load(Ordering::Relaxed) {
        return Ok(false);
    }
    let gain = if rep.stacked_noise > 0.0 { rep.ref_noise / rep.stacked_noise } else { 0.0 };
    let report = format!("Stacked {} of {} frames  ·  {gain:.1}× less noise", rep.kept, rep.total);
    let single = frames[rep.ref_index].stretched();
    let _ = tx.send(Msg::Done { stacked: Box::new(img.stretched()), single: Box::new(single), report });
    Ok(true)
}

fn gray_to_color_image(g: &Gray) -> egui::ColorImage {
    let pixels = g
        .data
        .iter()
        .map(|v| egui::Color32::from_gray((v.clamp(0.0, 1.0) * 255.0) as u8))
        .collect();
    egui::ColorImage { size: [g.w, g.h], pixels }
}

fn human_duration(s: f64) -> String {
    let s = s.max(0.0) as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// A labelled slider: caption above, full-width control below, with a hover hint.
fn slider_row(ui: &mut egui::Ui, label: &str, slider: egui::Slider<'_>, hint: &str) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(label).small().weak());
    ui.add(slider).on_hover_text(hint);
}

/// A rounded white card with a soft border and shadow.
fn card(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::default()
        .fill(egui::Color32::WHITE)
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(228)))
        .corner_radius(egui::CornerRadius::same(10))
        .inner_margin(egui::Margin::same(13))
        .shadow(egui::Shadow {
            offset: [0, 1],
            blur: 6,
            spread: 0,
            color: egui::Color32::from_black_alpha(14),
        })
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add(ui);
        });
}

impl App {
    /// Name stem for output files — the loaded video's name, or "photosharp".
    fn stem(&self) -> String {
        self.video
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "photosharp".to_owned())
    }

    /// The image currently shown (stacked result or the single reference frame).
    fn current_image(&self) -> Option<&Gray> {
        match self.view {
            View::Stacked => self.stacked.as_ref(),
            View::Single => self.single.as_ref(),
        }
    }

    fn open_video(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("video", &["mp4", "mov", "avi", "mkv", "m4v", "ser"])
            .pick_file()
        {
            self.info = path.to_str().and_then(|p| decode::summary(p).ok());
            self.video = Some(path);
            self.status = Status::Idle;
            self.error = None;
            self.saved = None;
            self.report.clear();
            self.stacked = None;
            self.single = None;
            self.tex_stacked = None;
            self.tex_single = None;
        }
    }

    fn start_stack(&mut self) {
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.cancel = Arc::new(AtomicBool::new(false));
        self.status = Status::Running;
        self.error = None;
        self.saved = None;
        self.report.clear();
        self.progress = 0.0;
        self.phase = "Decoding…".to_owned();
        self.stacked = None;
        self.single = None;
        self.tex_stacked = None;
        self.tex_single = None;

        // The video's real frame count (capped by max_frames) makes the decode bar honest;
        // fall back to the cap if the container does not store an exact count.
        let total_hint = self
            .info
            .as_ref()
            .map(|i| i.frames)
            .filter(|&f| f > 0)
            .map(|f| f.min(self.max_frames))
            .unwrap_or(self.max_frames);

        let v = self.video.clone().unwrap();
        let cancel = self.cancel.clone();
        let (roi_size, mf, keep, ck, sigma, amount) =
            (self.roi, self.max_frames, self.keep, self.centroid_k, self.sigma, self.amount);
        thread::spawn(move || {
            match run_stack(v, roi_size, mf, total_hint, keep, ck, sigma, amount, &cancel, &tx) {
                Ok(true) => {}                                 // Done already sent
                Ok(false) => { let _ = tx.send(Msg::Cancelled); }
                Err(e) => { let _ = tx.send(Msg::Error(e.to_string())); }
            }
        });
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let mut done: Option<(Gray, Gray, String)> = None;
        let mut clear_rx = false;
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Progress { stage, done: d, total: t } => {
                        // Grading/aligning/decoding fill the bar 0→1; stacking/sharpening are the
                        // quick finishing steps, shown full with a label.
                        self.progress = match stage {
                            "stacking" | "sharpening" => 1.0,
                            _ if t > 0 => (d as f32 / t as f32).min(1.0),
                            _ => 0.0,
                        };
                        self.phase = match stage {
                            "decoding" => format!("Decoding frames…  {d} / {t}"),
                            "grading" => format!("Grading {t} frames…"),
                            "aligning" => format!("Aligning {t} frames…"),
                            "stacking" => "Stacking…".to_owned(),
                            "sharpening" => "Sharpening…".to_owned(),
                            other => other.to_owned(),
                        };
                    }
                    Msg::Done { stacked, single, report } => done = Some((*stacked, *single, report)),
                    Msg::Cancelled => {
                        self.status = Status::Idle;
                        self.phase.clear();
                        self.progress = 0.0;
                        clear_rx = true;
                    }
                    Msg::Error(e) => {
                        self.error = Some(e);
                        self.status = Status::Failed;
                        clear_rx = true;
                    }
                }
            }
        }
        if clear_rx {
            self.rx = None;
        }
        if let Some((stacked, single, report)) = done {
            self.tex_stacked =
                Some(ctx.load_texture("stacked", gray_to_color_image(&stacked), egui::TextureOptions::LINEAR));
            self.tex_single =
                Some(ctx.load_texture("single", gray_to_color_image(&single), egui::TextureOptions::LINEAR));
            self.stacked = Some(stacked);
            self.single = Some(single);
            self.report = report;
            self.view = View::Stacked;
            self.status = Status::Done;
            self.rx = None;
        }
        if self.status == Status::Running {
            ctx.request_repaint();
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        card(ui, |ui| {
            ui.label(egui::RichText::new("1 · Source").strong().color(ACCENT));
            ui.add_space(6.0);
            if ui.add_sized([ui.available_width(), 30.0], egui::Button::new("Open video…")).clicked() {
                self.open_video();
            }
            ui.add_space(4.0);
            match (&self.video, &self.info) {
                (Some(p), Some(info)) => {
                    ui.label(
                        egui::RichText::new(format!(
                            "✓ {}",
                            p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
                        ))
                        .color(egui::Color32::from_rgb(0x16, 0x7a, 0x3b)),
                    );
                    let frames = if info.frames > 0 { info.frames.to_string() } else { "?".into() };
                    ui.label(
                        egui::RichText::new(format!(
                            "{}×{}  ·  {} frames  ·  {}",
                            info.w, info.h, frames, human_duration(info.duration)
                        ))
                        .weak()
                        .small(),
                    );
                }
                (Some(p), None) => {
                    // Video picked but ffprobe couldn't read it — still loaded, length unknown.
                    ui.label(
                        egui::RichText::new(format!(
                            "✓ {}",
                            p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
                        ))
                        .color(egui::Color32::from_rgb(0x16, 0x7a, 0x3b)),
                    );
                    ui.label(
                        egui::RichText::new("length unknown — frames counted while decoding")
                            .weak()
                            .small(),
                    );
                }
                _ => {
                    ui.label(egui::RichText::new("no video selected").weak());
                }
            }
        });

        ui.add_space(12.0);
        card(ui, |ui| {
            ui.label(egui::RichText::new("2 · Target").strong().color(ACCENT));
            slider_row(ui, "crop (px)", egui::Slider::new(&mut self.roi, 128..=2400),
                "Square crop centred on the target.\n~512 for a small planet, ~1900 for the whole Moon.");
            slider_row(ui, "detect", egui::Slider::new(&mut self.centroid_k, 0.0..=4.0),
                "Brightness threshold (mean + k·σ).\nLow (~0.5) for a big bright Moon, high (~3) for a small planet.");
            slider_row(ui, "max frames", egui::Slider::new(&mut self.max_frames, 50..=4000),
                "Cap on how many frames to read (bounds memory).");
        });

        ui.add_space(12.0);
        card(ui, |ui| {
            ui.label(egui::RichText::new("3 · Stack & sharpen").strong().color(ACCENT));
            slider_row(ui, "keep", egui::Slider::new(&mut self.keep, 0.05..=1.0),
                "Fraction of the sharpest frames to stack.\nLower keeps only the steadiest moments.");
            slider_row(ui, "sharpen radius", egui::Slider::new(&mut self.sigma, 0.3..=4.0),
                "Unsharp-mask radius — larger softens broader structure.");
            slider_row(ui, "sharpen amount", egui::Slider::new(&mut self.amount, 0.0..=3.0),
                "Sharpening strength. 0 = none.");
        });

        ui.add_space(14.0);
        card(ui, |ui| {
            let running = self.status == Status::Running;
            if running {
                // While a run is in flight, the primary action is to stop it cleanly.
                let btn = egui::Button::new(egui::RichText::new("Cancel").size(15.0).color(egui::Color32::WHITE))
                    .fill(egui::Color32::from_rgb(0xb0, 0x3a, 0x2e));
                if ui.add_sized([ui.available_width(), 42.0], btn).clicked() {
                    self.cancel.store(true, Ordering::Relaxed);
                    self.phase = "Cancelling…".to_owned();
                }
            } else {
                ui.add_enabled_ui(self.video.is_some(), |ui| {
                    let btn = egui::Button::new(egui::RichText::new("Stack").size(16.0).color(egui::Color32::WHITE))
                        .fill(ACCENT);
                    if ui.add_sized([ui.available_width(), 42.0], btn).clicked() {
                        self.start_stack();
                    }
                });
            }
            ui.add_space(6.0);
            ui.add_enabled_ui(self.stacked.is_some() && !running, |ui| {
                // Quick save: one click, auto-named, straight into the repo's captures/ folder.
                let save_btn = egui::Button::new(
                    egui::RichText::new("⬇  Save to captures/").size(15.0).color(egui::Color32::WHITE),
                )
                .fill(SAVE_GREEN);
                if ui.add_sized([ui.available_width(), 34.0], save_btn).clicked() {
                    let path = next_capture_path(&self.stem());
                    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                    let res = self.current_image().map(|img| image_io::save_gray(img, &path));
                    match res {
                        Some(Ok(())) => {
                            self.saved = Some(format!("captures/{name}"));
                            self.error = None;
                        }
                        Some(Err(e)) => self.error = Some(format!("couldn't save: {e}")),
                        None => {}
                    }
                }
                ui.add_space(4.0);
                // Export elsewhere: a dialog that still defaults into captures/ with a fresh name.
                if ui.add_sized([ui.available_width(), 28.0], egui::Button::new("Export to…")).clicked() {
                    let start_name = next_capture_path(&self.stem())
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let chosen = rfd::FileDialog::new()
                        .add_filter("png", &["png"])
                        .set_directory(captures_dir())
                        .set_file_name(start_name)
                        .save_file();
                    if let Some(path) = chosen {
                        let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                        let res = self.current_image().map(|img| image_io::save_gray(img, &path));
                        match res {
                            Some(Ok(())) => {
                                self.saved = Some(name);
                                self.error = None;
                            }
                            Some(Err(e)) => self.error = Some(format!("couldn't save: {e}")),
                            None => {}
                        }
                    }
                }
            });
            if running {
                ui.add_space(8.0);
                ui.add(egui::ProgressBar::new(self.progress).text(self.phase.clone()).animate(true));
            }
            if !self.report.is_empty() {
                ui.add_space(6.0);
                ui.label(egui::RichText::new(&self.report).strong());
            }
            if let Some(name) = &self.saved {
                ui.add_space(4.0);
                ui.colored_label(egui::Color32::from_rgb(0x16, 0x7a, 0x3b), format!("✓ Saved  {name}"));
            }
            if let Some(err) = &self.error {
                ui.add_space(6.0);
                ui.colored_label(egui::Color32::from_rgb(0xc0, 0x30, 0x20), format!("⚠ {err}"));
            }
        });
        ui.add_space(8.0);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("PhotoSharp").size(22.0).strong().color(ACCENT));
                ui.label(
                    egui::RichText::new("· turn a shaky telescope video into one sharp frame")
                        .italics()
                        .weak(),
                );
            });
            ui.add_space(8.0);
        });

        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(346.0)
            .frame(egui::Frame::default().fill(PANEL_BG).inner_margin(egui::Margin::same(14)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    self.controls(ui);
                });
            });

        // The image stage is a dark canvas — the Moon and planets read far better on black,
        // and it makes the tool feel like a real imaging app rather than a form.
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(STAGE_BG).inner_margin(egui::Margin::same(12)))
            .show(ctx, |ui| {
                ui.visuals_mut().override_text_color = Some(egui::Color32::from_gray(210));
                if self.tex_stacked.is_some() {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.view, View::Single, "Single frame");
                        ui.selectable_value(&mut self.view, View::Stacked, "Stacked result");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new("compare before / after").weak().small());
                        });
                    });
                    ui.add_space(6.0);
                }
                let tex = match self.view {
                    View::Stacked => self.tex_stacked.as_ref(),
                    View::Single => self.tex_single.as_ref(),
                };
                if let Some(tex) = tex {
                    ui.centered_and_justified(|ui| {
                        ui.add(
                            egui::Image::new(egui::load::SizedTexture::new(tex.id(), tex.size_vec2()))
                                .max_size(ui.available_size()),
                        );
                    });
                } else if self.status == Status::Running {
                    ui.centered_and_justified(|ui| {
                        ui.label(egui::RichText::new(self.phase.clone()).size(18.0));
                    });
                } else {
                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.available_height() * 0.26);
                        ui.label(egui::RichText::new("🌙").size(54.0));
                        ui.add_space(12.0);
                        ui.label(
                            egui::RichText::new("Turn a shaky telescope video into one sharp frame")
                                .size(18.0)
                                .color(egui::Color32::from_gray(225)),
                        );
                        ui.add_space(16.0);
                        ui.label(
                            egui::RichText::new("1 · Open video     2 · Crop & detect     3 · Stack     4 · Save to captures/")
                                .size(14.0)
                                .color(egui::Color32::from_gray(140)),
                        );
                    });
                }
            });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1120.0, 740.0])
            .with_min_inner_size([860.0, 560.0])
            .with_title("PhotoSharp"),
        ..Default::default()
    };
    eframe::run_native(
        "PhotoSharp",
        options,
        Box::new(|cc| {
            let mut visuals = egui::Visuals::light();
            let r = egui::CornerRadius::same(8);
            visuals.widgets.noninteractive.corner_radius = r;
            visuals.widgets.inactive.corner_radius = r;
            visuals.widgets.hovered.corner_radius = r;
            visuals.widgets.active.corner_radius = r;
            visuals.widgets.open.corner_radius = r;
            visuals.window_fill = PANEL_BG;
            visuals.panel_fill = egui::Color32::WHITE;
            cc.egui_ctx.set_visuals(visuals);

            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = egui::vec2(8.0, 8.0);
            style.spacing.button_padding = egui::vec2(12.0, 7.0);
            style.spacing.slider_width = 200.0;
            cc.egui_ctx.set_style(style);

            Ok(Box::new(App::default()))
        }),
    )
}

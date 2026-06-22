//! PhotoSharp GUI — a native (egui) front-end for the lucky-imaging stacker.
//!
//! Open a telescope video, tune the controls, stack, compare before/after, export. The
//! stacking runs on a worker thread (with progress) so the window stays responsive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // no console window in release

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use eframe::egui;
use photosharp_core::{decode, image_io, pipeline, roi, Gray};

const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x25, 0x63, 0xeb); // a calm blue

enum Msg {
    Progress(usize, usize),
    Phase(&'static str),
    Done { stacked: Box<Gray>, single: Box<Gray>, report: String },
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
    // parameters
    roi: usize,
    max_frames: usize,
    keep: f32,
    centroid_k: f32,
    sigma: f32,
    amount: f32,
    // run state
    status: Status,
    rx: Option<Receiver<Msg>>,
    progress: f32,
    phase: String,
    stacked: Option<Gray>,
    single: Option<Gray>,
    tex_stacked: Option<egui::TextureHandle>,
    tex_single: Option<egui::TextureHandle>,
    view: View,
    report: String,
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
            progress: 0.0,
            phase: String::new(),
            stacked: None,
            single: None,
            tex_stacked: None,
            tex_single: None,
            view: View::Stacked,
            report: String::new(),
            error: None,
        }
    }
}

/// The work, off the UI thread: decode -> crop -> grade -> align -> stack -> sharpen.
fn run_stack(
    video: PathBuf,
    roi_size: usize,
    max_frames: usize,
    keep: f32,
    centroid_k: f32,
    sigma: f32,
    amount: f32,
    tx: &Sender<Msg>,
) -> anyhow::Result<()> {
    let p = video.to_str().ok_or_else(|| anyhow::anyhow!("video path is not valid UTF-8"))?;
    let mut frames: Vec<Gray> = Vec::new();
    decode::decode_gray(p, max_frames, |i, frame| {
        let (cx, cy) = roi::bright_centroid(&frame, centroid_k);
        frames.push(roi::crop_centered(&frame, cx, cy, roi_size));
        if i % 8 == 0 {
            let _ = tx.send(Msg::Progress(i + 1, max_frames));
        }
    })?;
    if frames.is_empty() {
        anyhow::bail!("decoded 0 frames — is this a video file?");
    }
    let _ = tx.send(Msg::Phase("aligning & stacking"));
    let params = pipeline::Params { keep_fraction: keep, unsharp_sigma: sigma, unsharp_amount: amount };
    let (img, rep) = pipeline::process(&frames, &params);
    let gain = if rep.stacked_noise > 0.0 { rep.ref_noise / rep.stacked_noise } else { 0.0 };
    let report = format!(
        "Stacked {} of {} frames  ·  {gain:.1}x less background noise",
        rep.kept, rep.total
    );
    let single = frames[rep.ref_index].stretched();
    let _ = tx.send(Msg::Done {
        stacked: Box::new(img.stretched()),
        single: Box::new(single),
        report,
    });
    Ok(())
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

impl App {
    fn open_video(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("video", &["mp4", "mov", "avi", "mkv", "m4v", "ser"])
            .pick_file()
        {
            self.info = path.to_str().and_then(|p| decode::summary(p).ok());
            self.video = Some(path);
            self.status = Status::Idle;
            self.error = None;
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
        self.status = Status::Running;
        self.error = None;
        self.report.clear();
        self.progress = 0.0;
        self.phase = "decoding".to_owned();
        self.stacked = None;
        self.single = None;
        self.tex_stacked = None;
        self.tex_single = None;

        let v = self.video.clone().unwrap();
        let (roi_size, mf, keep, ck, sigma, amount) =
            (self.roi, self.max_frames, self.keep, self.centroid_k, self.sigma, self.amount);
        thread::spawn(move || {
            if let Err(e) = run_stack(v, roi_size, mf, keep, ck, sigma, amount, &tx) {
                let _ = tx.send(Msg::Error(e.to_string()));
            }
        });
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let mut done: Option<(Gray, Gray, String)> = None;
        let mut clear_rx = false;
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Progress(d, t) => {
                        self.progress = if t > 0 { (d as f32 / t as f32).min(1.0) } else { 0.0 };
                        self.phase = "decoding".to_owned();
                    }
                    Msg::Phase(p) => self.phase = p.to_owned(),
                    Msg::Done { stacked, single, report } => {
                        done = Some((*stacked, *single, report));
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
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("PhotoSharp").color(ACCENT));
                ui.label(
                    egui::RichText::new("· turn a shaky telescope video into one sharp frame")
                        .italics()
                        .weak(),
                );
            });
            ui.add_space(6.0);
        });

        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(300.0)
            .show(ctx, |ui| {
                ui.add_space(10.0);

                // 1. Source
                ui.label(egui::RichText::new("1 · Source").strong().color(ACCENT));
                if ui.button("Open video…").clicked() {
                    self.open_video();
                }
                match (&self.video, &self.info) {
                    (Some(p), Some(info)) => {
                        ui.label(p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
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
                    _ => {
                        ui.label(egui::RichText::new("no video selected").weak());
                    }
                }

                ui.add_space(10.0);
                ui.separator();

                // 2. Target
                ui.label(egui::RichText::new("2 · Target").strong().color(ACCENT));
                ui.add(egui::Slider::new(&mut self.roi, 128..=2400).text("crop px"))
                    .on_hover_text("Size of the square crop centred on the target.\nA small planet needs ~512; the whole Moon needs ~1900.");
                ui.add(egui::Slider::new(&mut self.centroid_k, 0.0..=4.0).text("detect"))
                    .on_hover_text("How bright the target must be to be found (mean + k·σ).\nLow (~0.5) for a big bright Moon, high (~3) for a small planet on black.");
                ui.add(egui::Slider::new(&mut self.max_frames, 50..=4000).text("max frames"))
                    .on_hover_text("Cap on how many frames to read (bounds memory).");

                ui.add_space(10.0);
                ui.separator();

                // 3. Stack & sharpen
                ui.label(egui::RichText::new("3 · Stack & sharpen").strong().color(ACCENT));
                ui.add(egui::Slider::new(&mut self.keep, 0.05..=1.0).text("keep"))
                    .on_hover_text("Fraction of the sharpest frames to stack.\nLower keeps only the very best moments of steady seeing.");
                ui.add(egui::Slider::new(&mut self.sigma, 0.3..=4.0).text("sharpen radius"))
                    .on_hover_text("Unsharp-mask radius — larger softens broader structure.");
                ui.add(egui::Slider::new(&mut self.amount, 0.0..=3.0).text("sharpen amount"))
                    .on_hover_text("How hard to sharpen. 0 = none.");

                ui.add_space(14.0);

                // Actions
                let can_run = self.video.is_some() && self.status != Status::Running;
                ui.add_enabled_ui(can_run, |ui| {
                    let btn = egui::Button::new(egui::RichText::new("Stack").size(16.0).color(egui::Color32::WHITE))
                        .fill(ACCENT)
                        .min_size(egui::vec2(ui.available_width(), 34.0));
                    if ui.add(btn).clicked() {
                        self.start_stack();
                    }
                });
                ui.add_enabled_ui(self.stacked.is_some(), |ui| {
                    if ui.button("Export PNG…").clicked() {
                        let img = match self.view {
                            View::Stacked => self.stacked.as_ref(),
                            View::Single => self.single.as_ref(),
                        };
                        if let (Some(img), Some(path)) = (
                            img,
                            rfd::FileDialog::new().add_filter("png", &["png"]).set_file_name("photosharp.png").save_file(),
                        ) {
                            let _ = image_io::save_gray(img, path);
                        }
                    }
                });

                ui.add_space(10.0);
                if self.status == Status::Running {
                    ui.add(egui::ProgressBar::new(self.progress).text(self.phase.clone()).animate(true));
                }
                if !self.report.is_empty() {
                    ui.label(egui::RichText::new(&self.report).strong());
                }
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(0xc0, 0x30, 0x20), format!("⚠ {err}"));
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.tex_stacked.is_some() {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.view, View::Single, "Single frame");
                    ui.selectable_value(&mut self.view, View::Stacked, "Stacked result");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new("← compare before / after").weak().small());
                    });
                });
                ui.separator();
            }
            let tex = match self.view {
                View::Stacked => self.tex_stacked.as_ref(),
                View::Single => self.tex_single.as_ref(),
            };
            ui.centered_and_justified(|ui| {
                if let Some(tex) = tex {
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(tex.id(), tex.size_vec2()))
                            .max_size(ui.available_size()),
                    );
                } else if self.status == Status::Running {
                    ui.label(egui::RichText::new(format!("{}…", self.phase)).size(18.0).weak());
                } else {
                    ui.label(
                        egui::RichText::new(
                            "Open a telescope video, then press Stack.\n\n\
                             1 · Open video      2 · Set crop & detect for your target\n\
                             3 · Stack           4 · Compare and Export PNG",
                        )
                        .size(15.0)
                        .weak(),
                    );
                }
            });
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1120.0, 740.0])
            .with_min_inner_size([840.0, 560.0])
            .with_title("PhotoSharp"),
        ..Default::default()
    };
    eframe::run_native(
        "PhotoSharp",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = egui::vec2(8.0, 8.0);
            style.spacing.button_padding = egui::vec2(10.0, 6.0);
            style.spacing.slider_width = 150.0;
            cc.egui_ctx.set_style(style);
            Ok(Box::new(App::default()))
        }),
    )
}

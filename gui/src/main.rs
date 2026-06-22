//! PhotoSharp GUI — a native (egui) front-end for the lucky-imaging stacker.
//!
//! Pick a video, tune the crop / keep-fraction / sharpening, hit Stack, see the result,
//! export a PNG. The stacking runs on a worker thread so the window stays responsive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // no console window in release

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;

use eframe::egui;
use photosharp_core::{decode, image_io, pipeline, roi, Gray};

enum Msg {
    Done(Box<Gray>, String),
    Error(String),
}

#[derive(PartialEq)]
enum Status {
    Idle,
    Running,
    Done,
    Failed,
}

struct App {
    video: Option<PathBuf>,
    roi: usize,
    max_frames: usize,
    keep: f32,
    centroid_k: f32,
    sigma: f32,
    amount: f32,
    status: Status,
    rx: Option<Receiver<Msg>>,
    result: Option<Gray>,
    texture: Option<egui::TextureHandle>,
    report: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            video: None,
            roi: 512,
            max_frames: 1000,
            keep: 0.3,
            centroid_k: 1.0,
            sigma: 1.5,
            amount: 1.0,
            status: Status::Idle,
            rx: None,
            result: None,
            texture: None,
            report: String::new(),
        }
    }
}

/// The work: decode the video, crop the target per frame, stack. Runs off the UI thread.
fn run_stack(
    video: PathBuf,
    roi_size: usize,
    max_frames: usize,
    keep: f32,
    centroid_k: f32,
    sigma: f32,
    amount: f32,
) -> anyhow::Result<(Gray, String)> {
    let p = video.to_str().ok_or_else(|| anyhow::anyhow!("video path is not valid UTF-8"))?;
    let mut frames: Vec<Gray> = Vec::new();
    decode::decode_gray(p, max_frames, |_i, frame| {
        let (cx, cy) = roi::bright_centroid(&frame, centroid_k);
        frames.push(roi::crop_centered(&frame, cx, cy, roi_size));
    })?;
    if frames.is_empty() {
        anyhow::bail!("decoded 0 frames");
    }
    let params = pipeline::Params { keep_fraction: keep, unsharp_sigma: sigma, unsharp_amount: amount };
    let (img, rep) = pipeline::process(&frames, &params);
    let gain = if rep.stacked_noise > 0.0 { rep.ref_noise / rep.stacked_noise } else { 0.0 };
    let report = format!(
        "kept {}/{} sharpest  ·  noise {:.4} -> {:.4}  ({gain:.1}x cleaner)",
        rep.kept, rep.total, rep.ref_noise, rep.stacked_noise
    );
    Ok((img.stretched(), report))
}

fn gray_to_color_image(g: &Gray) -> egui::ColorImage {
    let pixels = g
        .data
        .iter()
        .map(|v| egui::Color32::from_gray((v.clamp(0.0, 1.0) * 255.0) as u8))
        .collect();
    egui::ColorImage { size: [g.w, g.h], pixels }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain worker messages.
        if let Some(rx) = &self.rx {
            match rx.try_recv() {
                Ok(Msg::Done(img, report)) => {
                    self.texture = Some(ctx.load_texture(
                        "result",
                        gray_to_color_image(&img),
                        egui::TextureOptions::LINEAR,
                    ));
                    self.result = Some(*img);
                    self.report = report;
                    self.status = Status::Done;
                    self.rx = None;
                }
                Ok(Msg::Error(e)) => {
                    self.report = format!("error: {e}");
                    self.status = Status::Failed;
                    self.rx = None;
                }
                Err(_) => ctx.request_repaint(), // still running — keep polling
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("PhotoSharp");
                ui.label("lucky-imaging stacker");
            });
            ui.add_space(4.0);
        });

        egui::SidePanel::left("controls").resizable(false).exact_width(290.0).show(ctx, |ui| {
            ui.add_space(8.0);
            if ui.button("Open video...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("video", &["mp4", "mov", "avi", "mkv", "ser"])
                    .pick_file()
                {
                    self.video = Some(path);
                    self.status = Status::Idle;
                }
            }
            ui.label(match &self.video {
                Some(p) => p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                None => "no video selected".to_owned(),
            });

            ui.separator();
            ui.add(egui::Slider::new(&mut self.roi, 128..=2400).text("crop (px)"));
            ui.add(egui::Slider::new(&mut self.max_frames, 50..=4000).text("max frames"));
            ui.add(egui::Slider::new(&mut self.keep, 0.05..=1.0).text("keep fraction"));
            ui.add(egui::Slider::new(&mut self.centroid_k, 0.0..=4.0).text("detect k"));
            ui.add(egui::Slider::new(&mut self.sigma, 0.3..=4.0).text("sharpen radius"));
            ui.add(egui::Slider::new(&mut self.amount, 0.0..=3.0).text("sharpen amount"));

            ui.separator();
            let running = self.status == Status::Running;
            ui.add_enabled_ui(self.video.is_some() && !running, |ui| {
                if ui.button("Stack").clicked() {
                    let (tx, rx) = channel();
                    self.rx = Some(rx);
                    self.status = Status::Running;
                    self.result = None;
                    self.texture = None;
                    self.report = "stacking...".to_owned();
                    let v = self.video.clone().unwrap();
                    let (roi_size, mf, keep, ck, sigma, amount) =
                        (self.roi, self.max_frames, self.keep, self.centroid_k, self.sigma, self.amount);
                    thread::spawn(move || {
                        let msg = match run_stack(v, roi_size, mf, keep, ck, sigma, amount) {
                            Ok((img, rep)) => Msg::Done(Box::new(img), rep),
                            Err(e) => Msg::Error(e.to_string()),
                        };
                        let _ = tx.send(msg);
                    });
                }
            });
            ui.add_enabled_ui(self.result.is_some(), |ui| {
                if ui.button("Export PNG...").clicked() {
                    if let (Some(img), Some(path)) = (
                        self.result.as_ref(),
                        rfd::FileDialog::new()
                            .add_filter("png", &["png"])
                            .set_file_name("photosharp.png")
                            .save_file(),
                    ) {
                        let _ = image_io::save_gray(img, path);
                    }
                }
            });

            ui.add_space(10.0);
            if running {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("stacking...");
                });
            }
            if !self.report.is_empty() {
                ui.label(&self.report);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| match &self.texture {
                Some(tex) => {
                    let avail = ui.available_size();
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(tex.id(), tex.size_vec2()))
                            .max_size(avail),
                    );
                }
                None => {
                    ui.label("open a video and hit Stack");
                }
            });
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native("PhotoSharp", options, Box::new(|_cc| Ok(Box::new(App::default()))))
}

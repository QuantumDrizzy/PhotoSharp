//! PhotoSharp CLI — stack a burst of planetary/lunar frames into one sharp image.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use photosharp_core::{decode, image_io, pipeline, roi, synthetic, Gray};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "photosharp",
    version,
    about = "Lucky-imaging stacking for planetary & lunar frames"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Stack into one sharp image. Source is a video (--video) or a folder of frames (--input).
    Stack {
        /// A video file (mp4/mov/…), decoded via ffmpeg; the planet is auto-cropped per frame.
        #[arg(long)]
        video: Option<PathBuf>,
        /// A directory of frames (PNG/JPEG/TIFF), instead of a video.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Crop window (pixels) around the planet, used with --video.
        #[arg(long, default_value_t = 512)]
        roi: usize,
        /// Cap on the number of frames decoded from --video (bounds memory).
        #[arg(long, default_value_t = 1500)]
        max_frames: usize,
        /// Detection threshold = mean + k·std. Lower k (~0.5) for a big bright Moon,
        /// higher (~3) for a small planet on black.
        #[arg(long, default_value_t = 3.0)]
        centroid_k: f32,
        /// Fraction of the sharpest frames to keep (0..1).
        #[arg(long, default_value_t = 0.3)]
        keep: f32,
        /// Unsharp-mask radius (Gaussian sigma).
        #[arg(long, default_value_t = 1.5)]
        sigma: f32,
        /// Unsharp-mask strength.
        #[arg(long, default_value_t = 1.0)]
        amount: f32,
        /// Combiner: mean | median | sigma (sigma-clipped mean — rejects per-frame outliers).
        #[arg(long, default_value = "sigma")]
        stack: String,
        /// Sigma-clip threshold (only used with --stack sigma).
        #[arg(long, default_value_t = 2.5)]
        kappa: f32,
        /// Output PNG path.
        #[arg(long, default_value = "photosharp-out.png")]
        out: PathBuf,
        /// Min-max stretch the result for visibility.
        #[arg(long)]
        stretch: bool,
    },
    /// Run on synthetic frames (no real data) to verify the pipeline end to end.
    Demo {
        #[arg(long, default_value_t = 200)]
        frames: usize,
        #[arg(long, default_value = "photosharp-demo")]
        out_prefix: String,
    },
    /// Write synthetic capture frames as numbered PNGs (to test the folder/video path).
    GenFrames {
        #[arg(long, default_value_t = 200)]
        frames: usize,
        #[arg(long, default_value_t = 256)]
        size: usize,
        #[arg(long, default_value = "frames")]
        out_dir: PathBuf,
    },
}

fn load_dir(dir: &PathBuf) -> Result<Vec<Gray>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()).as_deref(),
                Some("png" | "jpg" | "jpeg" | "tif" | "tiff")
            )
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no image frames found in {}", dir.display());
    }
    let mut frames = Vec::with_capacity(paths.len());
    for p in &paths {
        frames.push(image_io::load_gray(p)?);
    }
    Ok(frames)
}

/// Decode a video and crop the planet ROI from each frame (streaming — full frames never
/// all live in memory at once).
fn load_video(path: &PathBuf, roi_size: usize, max_frames: usize, centroid_k: f32) -> Result<Vec<Gray>> {
    let p = path.to_str().context("video path is not valid UTF-8")?;
    let mut frames: Vec<Gray> = Vec::new();
    let n = decode::decode_gray(p, max_frames, None, |_i, frame| {
        let (cx, cy) = roi::bright_centroid(&frame, centroid_k);
        frames.push(roi::crop_centered(&frame, cx, cy, roi_size));
    })?;
    if frames.is_empty() {
        bail!("decoded 0 frames from {}", path.display());
    }
    println!(
        "[photosharp] decoded {n} frames from {}, cropped {}x{} around the planet",
        path.display(), frames[0].w, frames[0].h
    );
    Ok(frames)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Stack { video, input, roi, max_frames, centroid_k, keep, sigma, amount, stack, kappa, out, stretch } => {
            let frames = match (video, input) {
                (Some(v), _) => load_video(&v, roi, max_frames, centroid_k)?,
                (None, Some(d)) => {
                    let f = load_dir(&d)?;
                    println!("[photosharp] loaded {} frames from {}", f.len(), d.display());
                    f
                }
                (None, None) => bail!("give --video <file> or --input <folder>"),
            };
            let stack_method = match stack.to_lowercase().as_str() {
                "mean" => pipeline::StackMethod::Mean,
                "median" => pipeline::StackMethod::Median,
                "sigma" | "sigma-clip" | "sigmaclip" => pipeline::StackMethod::SigmaClip { kappa, iters: 2 },
                other => bail!("unknown --stack '{other}' (use mean | median | sigma)"),
            };
            let params = pipeline::Params {
                keep_fraction: keep,
                stack_method,
                unsharp_sigma: sigma,
                unsharp_amount: amount,
            };
            let (img, rep) = pipeline::process(&frames, &params);
            let img = if stretch { img.stretched() } else { img };
            image_io::save_gray(&img, &out)?;
            let gain = if rep.stacked_noise > 0.0 { rep.ref_noise / rep.stacked_noise } else { 0.0 };
            println!(
                "[photosharp] kept {}/{} sharpest | background noise {:.4} -> {:.4} ({:.1}x cleaner) | saved {}",
                rep.kept, rep.total, rep.ref_noise, rep.stacked_noise, gain, out.display()
            );
        }
        Cmd::Demo { frames, out_prefix } => {
            let truth = synthetic::planet(256);
            let caps = synthetic::capture(&truth, frames, 42);
            let raw: Vec<Gray> = caps.into_iter().map(|c| c.img).collect();
            let (img, rep) = pipeline::process(&raw, &pipeline::Params::default());

            image_io::save_gray(&raw[rep.ref_index].stretched(), format!("{out_prefix}-single.png"))?;
            image_io::save_gray(&img.stretched(), format!("{out_prefix}-stacked.png"))?;
            image_io::save_gray(&truth.stretched(), format!("{out_prefix}-truth.png"))?;
            let gain = if rep.stacked_noise > 0.0 { rep.ref_noise / rep.stacked_noise } else { 0.0 };
            println!(
                "[photosharp] demo: {} frames, kept {} sharpest | background noise {:.4} -> {:.4} ({:.1}x cleaner)",
                rep.total, rep.kept, rep.ref_noise, rep.stacked_noise, gain
            );
            println!(
                "[photosharp] wrote {out_prefix}-single.png, {out_prefix}-stacked.png, {out_prefix}-truth.png"
            );
        }
        Cmd::GenFrames { frames, size, out_dir } => {
            std::fs::create_dir_all(&out_dir)?;
            let truth = synthetic::planet(size);
            let caps = synthetic::capture(&truth, frames, 42);
            for (i, c) in caps.iter().enumerate() {
                image_io::save_gray(&c.img, out_dir.join(format!("frame_{i:05}.png")))?;
            }
            println!("[photosharp] wrote {frames} frames to {}", out_dir.display());
        }
    }
    Ok(())
}

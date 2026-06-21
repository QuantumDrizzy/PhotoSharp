//! PhotoSharp CLI — stack a burst of planetary/lunar frames into one sharp image.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use photosharp_core::{image_io, pipeline, synthetic};
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
    /// Stack a folder of frames (PNG/JPEG/TIFF) into one sharp image.
    Stack {
        /// Directory containing the input frames.
        #[arg(long)]
        input: PathBuf,
        /// Fraction of the sharpest frames to keep (0..1).
        #[arg(long, default_value_t = 0.3)]
        keep: f32,
        /// Unsharp-mask radius (Gaussian sigma).
        #[arg(long, default_value_t = 1.5)]
        sigma: f32,
        /// Unsharp-mask strength.
        #[arg(long, default_value_t = 1.0)]
        amount: f32,
        /// Output PNG path.
        #[arg(long, default_value = "photosharp-out.png")]
        out: PathBuf,
        /// Min-max stretch the result for visibility.
        #[arg(long)]
        stretch: bool,
    },
    /// Run on synthetic frames (no real data) to verify the pipeline end to end.
    Demo {
        /// Number of synthetic frames to generate.
        #[arg(long, default_value_t = 200)]
        frames: usize,
        /// Output filename prefix.
        #[arg(long, default_value = "photosharp-demo")]
        out_prefix: String,
    },
}

fn load_dir(dir: &PathBuf) -> Result<Vec<photosharp_core::Gray>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_lowercase())
                    .as_deref(),
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

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Stack { input, keep, sigma, amount, out, stretch } => {
            let frames = load_dir(&input)?;
            println!("[photosharp] loaded {} frames from {}", frames.len(), input.display());
            let params = pipeline::Params {
                keep_fraction: keep,
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
            let raw: Vec<photosharp_core::Gray> = caps.into_iter().map(|c| c.img).collect();
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
    }
    Ok(())
}

//! The lucky-imaging pipeline: grade -> select -> align -> stack -> sharpen.

use rayon::prelude::*;

use crate::gray::Gray;
use crate::{align, quality, sharpen, stack};

/// How the kept, aligned frames are combined into one image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StackMethod {
    /// Plain average — lowest noise, but smears per-frame outliers.
    Mean,
    /// Per-pixel median — robust to outliers, no threshold.
    Median,
    /// Sigma-clipped mean — rejects values beyond `kappa`·σ, then averages (the lucky-imaging
    /// default). `iters` re-estimates the bounds.
    SigmaClip { kappa: f32, iters: usize },
}

/// Tunable parameters for a stacking run.
pub struct Params {
    /// Fraction of the sharpest frames to keep, in `(0, 1]`.
    pub keep_fraction: f32,
    /// How to combine the kept frames.
    pub stack_method: StackMethod,
    /// Gaussian sigma for the unsharp mask.
    pub unsharp_sigma: f32,
    /// Unsharp strength.
    pub unsharp_amount: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            keep_fraction: 0.3,
            stack_method: StackMethod::Mean,
            unsharp_sigma: 1.5,
            unsharp_amount: 1.0,
        }
    }
}

/// Result of the grade -> select -> align -> stack stage (before sharpening).
pub struct StackResult {
    pub stacked: Gray,
    pub kept: usize,
    pub ref_index: usize,
    /// Measured background noise of the sharpest single frame.
    pub ref_noise: f32,
    /// Measured background noise of the stacked result.
    pub stacked_noise: f32,
}

/// What the pipeline did, for honest reporting.
pub struct Report {
    pub total: usize,
    pub kept: usize,
    pub ref_index: usize,
    pub ref_noise: f32,
    pub stacked_noise: f32,
}

/// Grade every frame, keep the sharpest fraction, align them onto the sharpest frame,
/// and mean-stack. This is the noise-reducing heart of lucky imaging.
pub fn stack_frames(frames: &[Gray], keep_fraction: f32) -> StackResult {
    stack_frames_method(frames, keep_fraction, StackMethod::Mean, |_, _, _| {})
}

/// Like [`stack_frames`], but reports progress through each stage via `on(stage, done, total)`
/// — `stage` is one of `"grading"`, `"aligning"`, `"stacking"`. Mean-combines; use
/// [`stack_frames_method`] to choose the combiner.
pub fn stack_frames_progress<F: FnMut(&'static str, usize, usize)>(
    frames: &[Gray],
    keep_fraction: f32,
    on: F,
) -> StackResult {
    stack_frames_method(frames, keep_fraction, StackMethod::Mean, on)
}

/// Grade → keep the sharpest fraction → align → combine with `method`, reporting progress via
/// `on(stage, done, total)`. The align loop is the slow part; it runs in parallel (rayon).
pub fn stack_frames_method<F: FnMut(&'static str, usize, usize)>(
    frames: &[Gray],
    keep_fraction: f32,
    method: StackMethod,
    mut on: F,
) -> StackResult {
    assert!(!frames.is_empty(), "no frames to stack");
    let n = frames.len();

    // 1. Grade by sharpness (parallel — each frame is independent), sharpest first.
    on("grading", 0, n);
    let mut scored: Vec<(usize, f32)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| (i, quality::laplacian_variance(f)))
        .collect();
    on("grading", n, n);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 2. Keep the sharpest fraction (at least one frame).
    let keep = ((n as f32 * keep_fraction).ceil() as usize).clamp(1, n);
    let best: Vec<usize> = scored.iter().take(keep).map(|s| s.0).collect();

    // 3. Reference = the single sharpest frame; align every kept frame onto it. This was the
    //    slow, sequential part — each alignment is independent, so run them in parallel (rayon).
    //    `phase_correlate` builds its own FFT planner per call, so this is thread-safe.
    let ref_index = best[0];
    let reference = &frames[ref_index];
    on("aligning", 0, keep);
    let aligned: Vec<Gray> = best
        .par_iter()
        .map(|&i| {
            let (dx, dy) = align::phase_correlate(reference, &frames[i]);
            align::shift_image(&frames[i], dx, dy)
        })
        .collect();
    on("aligning", keep, keep);

    // 4. Combine the aligned frames with the chosen method.
    on("stacking", keep, keep);
    let refs: Vec<&Gray> = aligned.iter().collect();
    let stacked = match method {
        StackMethod::Mean => stack::mean_stack(&refs),
        StackMethod::Median => stack::median_stack(&refs),
        StackMethod::SigmaClip { kappa, iters } => stack::sigma_clip_stack(&refs, kappa, iters),
    };
    let ref_noise = quality::background_noise(reference);
    let stacked_noise = quality::background_noise(&stacked);
    StackResult { stacked, kept: keep, ref_index, ref_noise, stacked_noise }
}

/// Run the full pipeline on a set of frames (all the same size): stack, then sharpen.
pub fn process(frames: &[Gray], p: &Params) -> (Gray, Report) {
    process_progress(frames, p, |_, _, _| {})
}

/// Like [`process`], but reports progress through each stage via `on(stage, done, total)`
/// (`"grading"`, `"aligning"`, `"stacking"`, `"sharpening"`).
pub fn process_progress<F: FnMut(&'static str, usize, usize)>(
    frames: &[Gray],
    p: &Params,
    mut on: F,
) -> (Gray, Report) {
    let sr = stack_frames_method(frames, p.keep_fraction, p.stack_method, &mut on);
    on("sharpening", 0, 1);
    let out = sharpen::unsharp(&sr.stacked, p.unsharp_sigma, p.unsharp_amount);
    on("sharpening", 1, 1);
    let report = Report {
        total: frames.len(),
        kept: sr.kept,
        ref_index: sr.ref_index,
        ref_noise: sr.ref_noise,
        stacked_noise: sr.stacked_noise,
    };
    (out, report)
}

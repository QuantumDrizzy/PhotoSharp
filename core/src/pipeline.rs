//! The lucky-imaging pipeline: grade -> select -> align -> stack -> sharpen.

use crate::gray::Gray;
use crate::{align, quality, sharpen, stack};

/// Tunable parameters for a stacking run.
pub struct Params {
    /// Fraction of the sharpest frames to keep, in `(0, 1]`.
    pub keep_fraction: f32,
    /// Gaussian sigma for the unsharp mask.
    pub unsharp_sigma: f32,
    /// Unsharp strength.
    pub unsharp_amount: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self { keep_fraction: 0.3, unsharp_sigma: 1.5, unsharp_amount: 1.0 }
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
    stack_frames_progress(frames, keep_fraction, |_, _, _| {})
}

/// Like [`stack_frames`], but reports progress through each stage via `on(stage, done, total)`
/// — `stage` is one of `"grading"`, `"aligning"`, `"stacking"`. Lets a GUI show what the
/// (otherwise silent) align loop is actually doing instead of looking frozen.
pub fn stack_frames_progress<F: FnMut(&'static str, usize, usize)>(
    frames: &[Gray],
    keep_fraction: f32,
    mut on: F,
) -> StackResult {
    assert!(!frames.is_empty(), "no frames to stack");
    let n = frames.len();

    // 1. Grade by sharpness, sharpest first.
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
    for (i, f) in frames.iter().enumerate() {
        scored.push((i, quality::laplacian_variance(f)));
        if i % 8 == 0 {
            on("grading", i + 1, n);
        }
    }
    on("grading", n, n);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 2. Keep the sharpest fraction (at least one frame).
    let keep = ((n as f32 * keep_fraction).ceil() as usize).clamp(1, n);
    let best: Vec<usize> = scored.iter().take(keep).map(|s| s.0).collect();

    // 3. Reference = the single sharpest frame; align every kept frame onto it (the slow part).
    let ref_index = best[0];
    let reference = &frames[ref_index];
    let mut aligned: Vec<Gray> = Vec::with_capacity(keep);
    for (k, &i) in best.iter().enumerate() {
        let (dx, dy) = align::phase_correlate(reference, &frames[i]);
        aligned.push(align::shift_image(&frames[i], dx, dy));
        on("aligning", k + 1, keep);
    }

    // 4. Stack.
    on("stacking", keep, keep);
    let refs: Vec<&Gray> = aligned.iter().collect();
    let stacked = stack::mean_stack(&refs);
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
    let sr = stack_frames_progress(frames, p.keep_fraction, &mut on);
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

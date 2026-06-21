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
    assert!(!frames.is_empty(), "no frames to stack");

    // 1. Grade by sharpness, sharpest first.
    let mut scored: Vec<(usize, f32)> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| (i, quality::laplacian_variance(f)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 2. Keep the sharpest fraction (at least one frame).
    let keep = ((frames.len() as f32 * keep_fraction).ceil() as usize).clamp(1, frames.len());
    let best: Vec<usize> = scored.iter().take(keep).map(|s| s.0).collect();

    // 3. Reference = the single sharpest frame; align every kept frame onto it.
    let ref_index = best[0];
    let reference = &frames[ref_index];
    let aligned: Vec<Gray> = best
        .iter()
        .map(|&i| {
            let (dx, dy) = align::phase_correlate(reference, &frames[i]);
            align::shift_image(&frames[i], dx, dy)
        })
        .collect();
    let refs: Vec<&Gray> = aligned.iter().collect();

    let stacked = stack::mean_stack(&refs);
    let ref_noise = quality::background_noise(reference);
    let stacked_noise = quality::background_noise(&stacked);
    StackResult { stacked, kept: keep, ref_index, ref_noise, stacked_noise }
}

/// Run the full pipeline on a set of frames (all the same size): stack, then sharpen.
pub fn process(frames: &[Gray], p: &Params) -> (Gray, Report) {
    let sr = stack_frames(frames, p.keep_fraction);
    let out = sharpen::unsharp(&sr.stacked, p.unsharp_sigma, p.unsharp_amount);
    let report = Report {
        total: frames.len(),
        kept: sr.kept,
        ref_index: sr.ref_index,
        ref_noise: sr.ref_noise,
        stacked_noise: sr.stacked_noise,
    };
    (out, report)
}

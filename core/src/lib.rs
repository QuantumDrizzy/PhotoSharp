//! PhotoSharp core — lucky-imaging stacking for planetary & lunar frames.
//!
//! The pipeline turns a burst of poorly-seen frames (atmospheric blur, wind, a
//! shaky phone on an eyepiece adapter) into one sharp image, by *grading* every
//! frame, *selecting* the sharpest, *aligning* them sub-pixel, *stacking* them to
//! beat down noise, and *sharpening* the result. This is the classic "lucky
//! imaging" technique — with the luck replaced by a deterministic sharpness metric.

pub mod align;
pub mod decode;
pub mod gray;
pub mod image_io;
pub mod pipeline;
pub mod quality;
pub mod roi;
pub mod sharpen;
pub mod stack;
pub mod synthetic;

pub use gray::Gray;

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard deviation of a 16x16 sky patch in the corner — a noise estimate.
    fn background_std(g: &Gray) -> f32 {
        let mut v = Vec::with_capacity(256);
        for y in 0..16 {
            for x in 0..16 {
                v.push(g.at(x, y));
            }
        }
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|p| (p - m) * (p - m)).sum::<f32>() / v.len() as f32).sqrt()
    }

    #[test]
    fn phase_correlation_recovers_the_alignment_shift() {
        let truth = synthetic::planet(128);
        // Move the content by (5, -3); the shift that *undoes* it is (-5, 3).
        let shifted = align::shift_image(&truth, 5, -3);
        let (dx, dy) = align::phase_correlate(&truth, &shifted);
        assert!(
            (dx - (-5)).abs() <= 1 && (dy - 3).abs() <= 1,
            "expected alignment shift near (-5, 3), got ({dx}, {dy})"
        );
        // And applying it must reconstruct the original.
        let realigned = align::shift_image(&shifted, dx, dy);
        assert!((realigned.at(64, 64) - truth.at(64, 64)).abs() < 1e-3);
    }

    #[test]
    fn stacking_reduces_noise() {
        let truth = synthetic::planet(128);
        let caps = synthetic::capture(&truth, 200, 42);
        let frames: Vec<Gray> = caps.iter().map(|c| c.img.clone()).collect();

        let sr = pipeline::stack_frames(&frames, 0.3);

        // The sharpest single frame is the noisiest one worth keeping; stacking ~60
        // aligned frames should cut its background noise by well over 2x (~sqrt(N)).
        let ref_noise = background_std(&frames[sr.ref_index]);
        let stack_noise = background_std(&sr.stacked);
        assert!(
            stack_noise < ref_noise * 0.5,
            "stacking should cut background noise: ref {ref_noise:.5} -> stack {stack_noise:.5}"
        );
        assert_eq!(sr.stacked.w, truth.w);
        assert_eq!(sr.stacked.h, truth.h);
    }
}

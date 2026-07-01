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
pub mod rgb;
pub mod roi;
pub mod sharpen;
pub mod stack;
pub mod synthetic;

pub use gray::Gray;
pub use rgb::Rgb;

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
        let shifted = align::shift_image(&truth, 5.0, -3.0);
        let (dx, dy) = align::phase_correlate(&truth, &shifted);
        assert!(
            (dx + 5.0).abs() <= 0.5 && (dy - 3.0).abs() <= 0.5,
            "expected alignment shift near (-5, 3), got ({dx}, {dy})"
        );
        // And applying it must reconstruct the original.
        let realigned = align::shift_image(&shifted, dx, dy);
        assert!((realigned.at(64, 64) - truth.at(64, 64)).abs() < 5e-3);
    }

    #[test]
    fn phase_correlation_is_sub_pixel() {
        // A fractional shift must be recovered better than the ½-pixel error that
        // nearest-integer registration would leave (rounding 5.4 → 5 is a 0.4 px error).
        let truth = synthetic::planet(128);
        let shifted = align::shift_image(&truth, 5.4, -3.0);
        let (dx, dy) = align::phase_correlate(&truth, &shifted);
        assert!((dx + 5.4).abs() < 0.3, "sub-pixel dx off: got {dx}, want ~-5.4");
        assert!((dy - 3.0).abs() < 0.3, "dy off: got {dy}, want ~3.0");
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

    #[test]
    fn color_pipeline_reduces_noise_and_keeps_colour() {
        let truth = synthetic::planet_color(128);
        let frames = synthetic::capture_color(&truth, 200, 7);

        // Sharpening re-adds high-frequency noise; test the *stacking* by disabling it (amount 0),
        // parallel to the grayscale stack test.
        let params = pipeline::Params { unsharp_amount: 0.0, ..Default::default() };
        let (out, rep) = pipeline::process_color(&frames, &params);

        // Noise (measured on luminance) drops like the grayscale path.
        let ref_noise = background_std(&frames[rep.ref_index].luminance());
        let out_noise = background_std(&out.luminance());
        assert!(
            out_noise < ref_noise * 0.5,
            "colour stack should cut background noise: {ref_noise:.5} -> {out_noise:.5}"
        );

        // Colour survives: the warm gold disc stays warm (mean R well above mean B).
        let mean = |g: &Gray| g.data.iter().sum::<f32>() / g.data.len() as f32;
        assert!(
            mean(&out.r) > mean(&out.b) * 1.3,
            "warm colour lost after stacking: R {:.3} vs B {:.3}",
            mean(&out.r),
            mean(&out.b)
        );
        assert_eq!(out.w(), truth.w());
    }
}

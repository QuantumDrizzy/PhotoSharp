//! Frame stacking — combine aligned frames to beat down random noise.
//!
//! Three combiners, from lowest-noise to most robust:
//! - [`mean_stack`] — lowest noise (~1/√N) but smears any per-frame outlier into the result;
//! - [`median_stack`] — rejects outliers with no threshold, ~1.25× the mean's noise;
//! - [`sigma_clip_stack`] — the lucky-imaging workhorse: a mean that first throws out the
//!   per-frame outliers (planes, satellites, cosmic rays, hot pixels), keeping most of the
//!   mean's noise reduction *and* a clean result.

use rayon::prelude::*;

use crate::gray::Gray;

/// Mean-stack a set of already-aligned frames. Random noise falls as ~1/sqrt(N),
/// so stacking N frames lifts the signal-to-noise ratio by ~sqrt(N).
pub fn mean_stack(frames: &[&Gray]) -> Gray {
    assert!(!frames.is_empty(), "cannot stack zero frames");
    let (w, h) = (frames[0].w, frames[0].h);
    let mut acc = Gray::new(w, h);
    for f in frames {
        acc.add_inplace(f);
    }
    acc.scale(1.0 / frames.len() as f32);
    acc
}

/// Median-stack: each output pixel is the median of that pixel across all frames. More robust
/// than the mean to outliers (a satellite trail, a hot pixel, one gust-blurred frame) and needs
/// no threshold, at a small cost in noise (~1.25× the mean's).
pub fn median_stack(frames: &[&Gray]) -> Gray {
    assert!(!frames.is_empty(), "cannot stack zero frames");
    let (w, h) = (frames[0].w, frames[0].h);
    let mut out = Gray::new(w, h);
    out.data.par_iter_mut().enumerate().for_each(|(i, o)| {
        let mut vals: Vec<f32> = frames.iter().map(|f| f.data[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = vals.len() / 2;
        *o = if vals.len() % 2 == 1 { vals[m] } else { 0.5 * (vals[m - 1] + vals[m]) };
    });
    out
}

/// Sigma-clipped mean: for each pixel, iteratively reject values more than `kappa` standard
/// deviations from the mean across the frame stack, then average the survivors. This keeps the
/// noise-beating power of the mean while discarding the per-frame outliers a plain mean would
/// smear in — the standard "professional" stacker for lucky imaging. `iters` re-estimates the
/// clip bounds (2 is plenty). `kappa ≈ 2.5` is a good default.
pub fn sigma_clip_stack(frames: &[&Gray], kappa: f32, iters: usize) -> Gray {
    assert!(!frames.is_empty(), "cannot stack zero frames");
    let (w, h) = (frames[0].w, frames[0].h);
    let mut out = Gray::new(w, h);
    out.data.par_iter_mut().enumerate().for_each(|(i, o)| {
        let mut lo = f32::NEG_INFINITY;
        let mut hi = f32::INFINITY;
        let mut mean = 0.0f32;
        for _ in 0..iters.max(1) {
            let (mut sum, mut cnt) = (0.0f32, 0usize);
            for f in frames {
                let v = f.data[i];
                if v >= lo && v <= hi {
                    sum += v;
                    cnt += 1;
                }
            }
            if cnt == 0 {
                break;
            }
            mean = sum / cnt as f32;
            let mut vsum = 0.0f32;
            for f in frames {
                let v = f.data[i];
                if v >= lo && v <= hi {
                    vsum += (v - mean) * (v - mean);
                }
            }
            let sd = (vsum / cnt as f32).sqrt();
            if sd == 0.0 {
                break; // all surviving values identical — nothing left to clip
            }
            lo = mean - kappa * sd;
            hi = mean + kappa * sd;
        }
        *o = mean;
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, v: f32) -> Gray {
        Gray { w, h, data: vec![v; w * h] }
    }

    #[test]
    fn sigma_clip_rejects_a_hot_pixel_that_the_mean_smears() {
        // 8 frames are a flat 0.2 field; one frame has a hot pixel (0.9) at index 0.
        let base = flat(4, 4, 0.2);
        let mut hot = base.clone();
        hot.data[0] = 0.9;
        let frames: Vec<Gray> =
            (0..8).map(|k| if k == 0 { hot.clone() } else { base.clone() }).collect();
        let refs: Vec<&Gray> = frames.iter().collect();

        // The mean is pulled up by the outlier; sigma-clip rejects it and recovers ~0.2.
        let mean = mean_stack(&refs);
        let clipped = sigma_clip_stack(&refs, 2.0, 2);
        assert!(mean.data[0] > 0.27, "mean should be dragged up: {}", mean.data[0]);
        assert!((clipped.data[0] - 0.2).abs() < 0.02, "sigma-clip should reject it: {}", clipped.data[0]);
        // away from the outlier the two agree
        assert!((clipped.data[5] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn median_is_robust_to_an_outlier() {
        let base = flat(2, 2, 0.2);
        let mut spike = base.clone();
        spike.data[0] = 5.0;
        let frames: Vec<Gray> =
            (0..7).map(|k| if k == 0 { spike.clone() } else { base.clone() }).collect();
        let refs: Vec<&Gray> = frames.iter().collect();
        let med = median_stack(&refs);
        assert!((med.data[0] - 0.2).abs() < 1e-6, "median ignores the spike: {}", med.data[0]);
    }

    #[test]
    fn all_methods_agree_with_no_outliers() {
        // On clean, identical frames the three combiners give the same answer.
        let frames: Vec<Gray> = (0..6).map(|_| flat(3, 3, 0.4)).collect();
        let refs: Vec<&Gray> = frames.iter().collect();
        for g in [mean_stack(&refs), median_stack(&refs), sigma_clip_stack(&refs, 2.5, 2)] {
            assert!((g.data[4] - 0.4).abs() < 1e-6);
        }
    }
}

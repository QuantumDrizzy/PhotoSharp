//! Synthetic capture generator — for tests and a no-data demo.
//!
//! There is no real telescope data here. This models the *conditions* PhotoSharp must
//! beat: a planetary disk seen through jitter, variable atmospheric blur ("seeing"),
//! and sensor noise — most frames soft, a few sharp ("the lucky ones").

use crate::gray::Gray;
use crate::{align, sharpen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A synthetic planetary disk: limb darkening plus Jupiter-like horizontal banding.
/// Ground truth — what a perfect frame would look like.
pub fn planet(size: usize) -> Gray {
    let mut g = Gray::new(size, size);
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let radius = size as f32 * 0.32;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt();
            if r <= radius {
                // Limb darkening (brighter centre, darker edge).
                let mu = (1.0 - (r / radius).powi(2)).max(0.0).sqrt();
                let mut v = 0.35 + 0.5 * mu;
                // Horizontal cloud bands.
                v += 0.12 * (dy / radius * 9.0).sin();
                g.set(x, y, v.clamp(0.0, 1.0));
            } else {
                g.set(x, y, 0.02); // sky background
            }
        }
    }
    g
}

/// One simulated capture: ground truth jittered, blurred and noised.
pub struct Frame {
    pub img: Gray,
    pub true_shift: (i32, i32),
}

/// Generate `n` frames simulating a poor capture run.
pub fn capture(truth: &Gray, n: usize, seed: u64) -> Vec<Frame> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let sx = rng.gen_range(-6..=6);
        let sy = rng.gen_range(-6..=6);
        let shifted = align::shift_image(truth, sx as f32, sy as f32);

        // Variable seeing: most frames soft, a few sharp.
        let sigma = rng.gen_range(0.4f32..3.0);
        let mut frame = sharpen::gaussian_blur(&shifted, sigma);

        // Additive noise.
        let noise = 0.05f32;
        for v in frame.data.iter_mut() {
            let n: f32 = (rng.gen::<f32>() - 0.5) * 2.0 * noise;
            *v = (*v + n).clamp(0.0, 1.0);
        }
        out.push(Frame { img: frame, true_shift: (sx, sy) });
    }
    out
}

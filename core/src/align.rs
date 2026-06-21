//! Frame registration via phase correlation (FFT cross-power spectrum).
//!
//! Planetary frames jitter (wind, a hand-held phone, mount drift). To stack them we
//! must align each to a common reference. Phase correlation finds the translation
//! between two images from the location of the peak in the inverse FFT of their
//! normalized cross-power spectrum — robust to the brightness changes that defeat a
//! naive cross-correlation. A Hann window suppresses edge leakage from the non-periodic
//! borders so the peak is clean.

use crate::gray::Gray;
use rustfft::{num_complex::Complex, FftPlanner};

fn hann(n: usize) -> Vec<f32> {
    if n < 2 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| {
            let v = (std::f32::consts::PI * i as f32 / (n - 1) as f32).sin();
            v * v
        })
        .collect()
}

/// In-place 2D FFT (rows then columns), unnormalized.
fn fft2(buf: &mut [Complex<f32>], w: usize, h: usize, planner: &mut FftPlanner<f32>, inverse: bool) {
    let fft_row = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    for r in 0..h {
        fft_row.process(&mut buf[r * w..(r + 1) * w]);
    }
    let fft_col = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col = vec![Complex::new(0.0, 0.0); h];
    for c in 0..w {
        for r in 0..h {
            col[r] = buf[r * w + c];
        }
        fft_col.process(&mut col);
        for r in 0..h {
            buf[r * w + c] = col[r];
        }
    }
}

/// Estimate the integer shift `(dx, dy)` to apply to `mov` via [`shift_image`] so that it
/// aligns onto `reference` (equivalently: `reference ≈ shift_image(mov, dx, dy)`).
pub fn phase_correlate(reference: &Gray, mov: &Gray) -> (i32, i32) {
    let (w, h) = (reference.w, reference.h);
    assert_eq!(mov.w, w);
    assert_eq!(mov.h, h);

    let wx = hann(w);
    let wy = hann(h);
    let mean_a = reference.mean();
    let mean_b = mov.mean();

    let windowed = |src: &Gray, mean: f32| -> Vec<Complex<f32>> {
        (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                Complex::new((src.data[i] - mean) * wx[x] * wy[y], 0.0)
            })
            .collect()
    };

    let mut planner = FftPlanner::<f32>::new();
    let mut a = windowed(reference, mean_a);
    let mut b = windowed(mov, mean_b);

    fft2(&mut a, w, h, &mut planner, false);
    fft2(&mut b, w, h, &mut planner, false);

    // Normalized cross-power spectrum R = (A · conj(B)) / |A · conj(B)|.
    let mut r: Vec<Complex<f32>> = a
        .iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let cp = av * bv.conj();
            let mag = cp.norm();
            if mag > 1e-12 {
                cp / mag
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();

    fft2(&mut r, w, h, &mut planner, true);

    // Peak of the correlation surface gives the shift.
    let mut best_i = 0usize;
    let mut best_v = f32::MIN;
    for (i, c) in r.iter().enumerate() {
        if c.re > best_v {
            best_v = c.re;
            best_i = i;
        }
    }
    let py = best_i / w;
    let px = best_i % w;
    let dx = if px > w / 2 { px as i32 - w as i32 } else { px as i32 };
    let dy = if py > h / 2 { py as i32 - h as i32 } else { py as i32 };
    (dx, dy)
}

/// Shift an image by integer `(dx, dy)`: positive `dx` moves content right, positive
/// `dy` down. Vacated edges are filled with 0.
pub fn shift_image(g: &Gray, dx: i32, dy: i32) -> Gray {
    let mut out = Gray::new(g.w, g.h);
    for y in 0..g.h as i32 {
        let sy = y - dy;
        if sy < 0 || sy >= g.h as i32 {
            continue;
        }
        for x in 0..g.w as i32 {
            let sx = x - dx;
            if sx < 0 || sx >= g.w as i32 {
                continue;
            }
            out.set(x as usize, y as usize, g.at(sx as usize, sy as usize));
        }
    }
    out
}

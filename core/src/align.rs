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

/// Estimate the **sub-pixel** shift `(dx, dy)` to apply to `mov` via [`shift_image`] so that it
/// aligns onto `reference` (equivalently: `reference ≈ shift_image(mov, dx, dy)`). The integer
/// peak of the correlation surface is refined by a parabolic fit against its neighbours, so the
/// registration is not rounded to the nearest whole pixel — which visibly sharpens a stack.
pub fn phase_correlate(reference: &Gray, mov: &Gray) -> (f32, f32) {
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
    let int_dx = if px > w / 2 { px as i32 - w as i32 } else { px as i32 };
    let int_dy = if py > h / 2 { py as i32 - h as i32 } else { py as i32 };

    // Parabolic sub-pixel refinement: fit a quadratic to the peak and its two neighbours along
    // each axis (the correlation surface is periodic, so neighbours wrap around).
    let val = |x: usize, y: usize| r[y * w + x].re;
    let cx = val(px, py);
    let (xm, xp) = ((px + w - 1) % w, (px + 1) % w);
    let (ym, yp) = ((py + h - 1) % h, (py + 1) % h);
    let sub = |vm: f32, vc: f32, vp: f32| -> f32 {
        let denom = vm - 2.0 * vc + vp;
        if denom.abs() > 1e-12 {
            (0.5 * (vm - vp) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    };
    let ddx = sub(val(xm, py), cx, val(xp, py));
    let ddy = sub(val(px, ym), cx, val(px, yp));
    (int_dx as f32 + ddx, int_dy as f32 + ddy)
}

/// Shift an image by a (possibly fractional) `(dx, dy)`: positive `dx` moves content right,
/// positive `dy` down. Fractional shifts are resolved by bilinear interpolation (so sub-pixel
/// registration is honoured); vacated edges are filled with 0. An integer shift reproduces the
/// exact nearest-neighbour copy.
pub fn shift_image(g: &Gray, dx: f32, dy: f32) -> Gray {
    let (w, h) = (g.w as i32, g.h as i32);
    let sample = |xx: i32, yy: i32| -> f32 {
        if xx < 0 || yy < 0 || xx >= w || yy >= h {
            0.0
        } else {
            g.at(xx as usize, yy as usize)
        }
    };
    let mut out = Gray::new(g.w, g.h);
    for y in 0..g.h {
        for x in 0..g.w {
            let sx = x as f32 - dx;
            let sy = y as f32 - dy;
            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let v = sample(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + sample(x0 + 1, y0) * fx * (1.0 - fy)
                + sample(x0, y0 + 1) * (1.0 - fx) * fy
                + sample(x0 + 1, y0 + 1) * fx * fy;
            out.set(x, y, v);
        }
    }
    out
}

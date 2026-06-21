//! Sharpening — recover detail the atmosphere and stacking softened.

use crate::gray::Gray;

/// Separable Gaussian blur with reflected edges.
pub fn gaussian_blur(g: &Gray, sigma: f32) -> Gray {
    if sigma <= 0.0 {
        return g.clone();
    }
    let radius = (3.0 * sigma).ceil() as i32;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0f32;
    for i in -radius..=radius {
        let v = (-(i as f32 * i as f32) / (2.0 * sigma * sigma)).exp();
        kernel.push(v);
        sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    let reflect = |i: i32, n: i32| -> usize {
        let mut j = i;
        if j < 0 {
            j = -j - 1;
        }
        if j >= n {
            j = 2 * n - j - 1;
        }
        j.clamp(0, n - 1) as usize
    };

    // Horizontal pass.
    let mut tmp = Gray::new(g.w, g.h);
    for y in 0..g.h {
        for x in 0..g.w {
            let mut acc = 0.0;
            for (ki, k) in kernel.iter().enumerate() {
                let xx = reflect(x as i32 + ki as i32 - radius, g.w as i32);
                acc += *k * g.at(xx, y);
            }
            tmp.set(x, y, acc);
        }
    }
    // Vertical pass.
    let mut out = Gray::new(g.w, g.h);
    for y in 0..g.h {
        for x in 0..g.w {
            let mut acc = 0.0;
            for (ki, k) in kernel.iter().enumerate() {
                let yy = reflect(y as i32 + ki as i32 - radius, g.h as i32);
                acc += *k * tmp.at(x, yy);
            }
            out.set(x, y, acc);
        }
    }
    out
}

/// Unsharp mask: `out = img + amount * (img - blur(img))`, clamped to `[0, 1]`.
pub fn unsharp(g: &Gray, sigma: f32, amount: f32) -> Gray {
    let blur = gaussian_blur(g, sigma);
    let mut out = Gray::new(g.w, g.h);
    for i in 0..g.data.len() {
        let high_pass = g.data[i] - blur.data[i];
        out.data[i] = (g.data[i] + amount * high_pass).clamp(0.0, 1.0);
    }
    out
}

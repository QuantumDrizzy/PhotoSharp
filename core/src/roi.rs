//! Region-of-interest: find the planet and crop a window around it.
//!
//! A planetary capture is a small bright disk drifting on a dark sky (wind, an imperfect
//! mount, a hand-held phone). Cropping a fixed-size window centred on the disk does two
//! jobs at once: it bounds memory (we stack small crops, not 4K frames) and it coarsely
//! centres every frame, leaving only sub-pixel drift for the FFT alignment.

use crate::gray::Gray;

/// Intensity-weighted centroid of the bright region (threshold = mean + k·std).
/// Falls back to the image centre when there is no clear bright blob (e.g. the Moon
/// filling the frame).
pub fn bright_centroid(g: &Gray, k: f32) -> (f32, f32) {
    let mean = g.mean();
    let var = g.data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / g.data.len() as f32;
    let thr = mean + k * var.sqrt();

    let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.0f32);
    for y in 0..g.h {
        for x in 0..g.w {
            let v = g.at(x, y);
            if v > thr {
                sx += v * x as f32;
                sy += v * y as f32;
                sw += v;
            }
        }
    }
    if sw <= 0.0 {
        (g.w as f32 / 2.0, g.h as f32 / 2.0)
    } else {
        (sx / sw, sy / sw)
    }
}

/// Crop a `size`×`size` window centred on `(cx, cy)`, clamped to the image. The returned
/// crop is always `min(size, w)` × `min(size, h)` so every frame's crop is identical in
/// shape (required for stacking).
pub fn crop_centered(g: &Gray, cx: f32, cy: f32, size: usize) -> Gray {
    let sx = size.min(g.w);
    let sy = size.min(g.h);
    let x0 = ((cx.round() as i32 - sx as i32 / 2).clamp(0, (g.w - sx) as i32)) as usize;
    let y0 = ((cy.round() as i32 - sy as i32 / 2).clamp(0, (g.h - sy) as i32)) as usize;

    let mut out = Gray::new(sx, sy);
    for y in 0..sy {
        for x in 0..sx {
            out.set(x, y, g.at(x0 + x, y0 + y));
        }
    }
    out
}

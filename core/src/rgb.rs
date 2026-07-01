//! Three-channel RGB image built on [`Gray`] planes.
//!
//! The whole grayscale pipeline (grade, align, stack, sharpen) is reused per channel, while
//! grading and alignment run on a single **luminance** plane — so planetary colour (Saturn's
//! gold, Jupiter's belts, Mars's rust) survives stacking instead of being thrown away at decode.

use crate::gray::Gray;

#[derive(Clone, Debug)]
pub struct Rgb {
    pub r: Gray,
    pub g: Gray,
    pub b: Gray,
}

impl Rgb {
    pub fn new(w: usize, h: usize) -> Self {
        Self { r: Gray::new(w, h), g: Gray::new(w, h), b: Gray::new(w, h) }
    }

    #[inline]
    pub fn w(&self) -> usize { self.r.w }
    #[inline]
    pub fn h(&self) -> usize { self.r.h }

    /// Rec. 601 luma — the plane grading and alignment run on. Colour is carried along the shifts.
    pub fn luminance(&self) -> Gray {
        let n = self.r.data.len();
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            data[i] = 0.299 * self.r.data[i] + 0.587 * self.g.data[i] + 0.114 * self.b.data[i];
        }
        Gray { w: self.r.w, h: self.r.h, data }
    }

    /// Linear contrast stretch driven by luminance, applied identically to all three channels so
    /// colour balance is preserved (a per-channel stretch would tint the result).
    pub fn stretched(&self) -> Rgb {
        let lum = self.luminance();
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &v in &lum.data {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        let range = (hi - lo).max(1e-6);
        let map = |g: &Gray| Gray {
            w: g.w,
            h: g.h,
            data: g.data.iter().map(|v| ((v - lo) / range).clamp(0.0, 1.0)).collect(),
        };
        Rgb { r: map(&self.r), g: map(&self.g), b: map(&self.b) }
    }
}

//! Single-channel f32 image — the working representation for the whole pipeline.

/// A grayscale image stored row-major as `f32`, values in `[0, 1]` by convention.
#[derive(Clone, Debug)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.w + x] = v;
    }

    /// Accumulate another same-sized image into this one.
    pub fn add_inplace(&mut self, other: &Gray) {
        debug_assert_eq!(self.w, other.w);
        debug_assert_eq!(self.h, other.h);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// Multiply every pixel by a scalar.
    pub fn scale(&mut self, s: f32) {
        for a in self.data.iter_mut() {
            *a *= s;
        }
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Min-max stretch to `[0, 1]` — used to make faint detail visible on export.
    pub fn stretched(&self) -> Gray {
        let mut mn = f32::MAX;
        let mut mx = f32::MIN;
        for &v in &self.data {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let range = (mx - mn).max(1e-6);
        let data = self.data.iter().map(|v| (v - mn) / range).collect();
        Gray { w: self.w, h: self.h, data }
    }
}

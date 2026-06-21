//! Image loading and saving (PNG/JPEG/TIFF) to and from [`Gray`].

use crate::gray::Gray;
use anyhow::{Context, Result};
use std::path::Path;

/// Load any supported image as single-channel f32 luma in `[0, 1]`.
pub fn load_gray<P: AsRef<Path>>(path: P) -> Result<Gray> {
    let p = path.as_ref();
    let img = image::open(p)
        .with_context(|| format!("opening {}", p.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data = img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
    Ok(Gray { w, h, data })
}

/// Save an f32 image to PNG. Values are clamped to `[0, 1]` and scaled to 8-bit.
pub fn save_gray<P: AsRef<Path>>(g: &Gray, path: P) -> Result<()> {
    let p = path.as_ref();
    let mut buf = image::GrayImage::new(g.w as u32, g.h as u32);
    for (i, px) in buf.pixels_mut().enumerate() {
        let v = (g.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        *px = image::Luma([v]);
    }
    buf.save(p).with_context(|| format!("saving {}", p.display()))?;
    Ok(())
}

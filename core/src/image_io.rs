//! Image loading and saving (PNG/JPEG/TIFF) to and from [`Gray`].

use crate::gray::Gray;
use crate::rgb::Rgb;
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

/// Load any supported image as three f32 channels in `[0, 1]`.
pub fn load_rgb<P: AsRef<Path>>(path: P) -> Result<Rgb> {
    let p = path.as_ref();
    let img = image::open(p)
        .with_context(|| format!("opening {}", p.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let n = w * h;
    let raw = img.into_raw();
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        r[i] = raw[i * 3] as f32 / 255.0;
        g[i] = raw[i * 3 + 1] as f32 / 255.0;
        b[i] = raw[i * 3 + 2] as f32 / 255.0;
    }
    Ok(Rgb { r: Gray { w, h, data: r }, g: Gray { w, h, data: g }, b: Gray { w, h, data: b } })
}

/// Save an [`Rgb`] to PNG. Each channel is clamped to `[0, 1]` and scaled to 8-bit.
pub fn save_rgb<P: AsRef<Path>>(img: &Rgb, path: P) -> Result<()> {
    let p = path.as_ref();
    let mut buf = image::RgbImage::new(img.w() as u32, img.h() as u32);
    for (i, px) in buf.pixels_mut().enumerate() {
        let r = (img.r.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (img.g.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (img.b.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        *px = image::Rgb([r, g, b]);
    }
    buf.save(p).with_context(|| format!("saving {}", p.display()))?;
    Ok(())
}

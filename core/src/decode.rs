//! Video decoding by streaming raw gray frames from ffmpeg.
//!
//! No H.264/H.265 decoder is reimplemented — ffmpeg is the standard native video layer
//! (open source, ubiquitous). We spawn it, ask for `rawvideo`/`gray`, and read one frame
//! at a time from its stdout, so a multi-gigabyte 4K capture never lives in RAM at once.
//! Requires `ffmpeg` and `ffprobe` on PATH.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};

use crate::gray::Gray;

pub struct VideoInfo {
    pub w: usize,
    pub h: usize,
}

/// Read the video's frame dimensions with ffprobe.
pub fn probe(path: &str) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", path,
        ])
        .output()
        .context("running ffprobe (is ffmpeg installed and on PATH?)")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let nums: Vec<usize> = s.trim().split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if nums.len() < 2 || nums[0] == 0 || nums[1] == 0 {
        bail!("could not parse video dimensions from ffprobe output: {s:?}");
    }
    Ok(VideoInfo { w: nums[0], h: nums[1] })
}

/// A human-readable summary of a video, for the UI.
pub struct Summary {
    pub w: usize,
    pub h: usize,
    pub frames: usize,
    pub duration: f64,
}

/// Probe dimensions, frame count and duration (best-effort; `frames` may be 0 if the
/// container does not store an exact count).
pub fn summary(path: &str) -> Result<Summary> {
    let VideoInfo { w, h } = probe(path)?;
    let field = |args: &[&str]| -> String {
        Command::new("ffprobe")
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let frames = field(&[
        "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path,
    ])
    .parse()
    .unwrap_or(0);
    let duration = field(&[
        "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path,
    ])
    .parse()
    .unwrap_or(0.0);
    Ok(Summary { w, h, frames, duration })
}

/// Stream gray frames from `path`, calling `on_frame(index, frame)` for up to `max_frames`.
/// Returns the number of frames decoded. If `cancel` is set partway through, decoding stops
/// early (cooperative cancellation for a responsive GUI) and returns the frames read so far.
pub fn decode_gray<F: FnMut(usize, Gray)>(
    path: &str,
    max_frames: usize,
    cancel: Option<&AtomicBool>,
    mut on_frame: F,
) -> Result<usize> {
    let info = probe(path)?;
    let (w, h) = (info.w, info.h);

    let mut child = Command::new("ffmpeg")
        // -noautorotate: keep the coded WxH that ffprobe reported; otherwise a phone's
        // rotation metadata makes ffmpeg emit transposed frames that mismatch (w, h).
        .args(["-v", "error", "-noautorotate", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning ffmpeg")?;
    let mut stdout = child.stdout.take().expect("piped stdout");

    let mut buf = vec![0u8; w * h];
    let mut idx = 0usize;
    while idx < max_frames {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            break; // cooperative cancel: stop reading, return what we have
        }
        match stdout.read_exact(&mut buf) {
            Ok(()) => {
                let data = buf.iter().map(|&b| b as f32 / 255.0).collect();
                on_frame(idx, Gray { w, h, data });
                idx += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break, // last partial / EOF
            Err(e) => {
                let _ = child.kill();
                return Err(e).context("reading a frame from ffmpeg");
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    Ok(idx)
}

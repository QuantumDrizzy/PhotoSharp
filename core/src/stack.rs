//! Frame stacking — combine aligned frames to beat down random noise.

use crate::gray::Gray;

/// Mean-stack a set of already-aligned frames. Random noise falls as ~1/sqrt(N),
/// so stacking N frames lifts the signal-to-noise ratio by ~sqrt(N).
pub fn mean_stack(frames: &[&Gray]) -> Gray {
    assert!(!frames.is_empty(), "cannot stack zero frames");
    let (w, h) = (frames[0].w, frames[0].h);
    let mut acc = Gray::new(w, h);
    for f in frames {
        acc.add_inplace(f);
    }
    acc.scale(1.0 / frames.len() as f32);
    acc
}

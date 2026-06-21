//! Per-frame sharpness grading.

use crate::gray::Gray;

/// Variance of the Laplacian — a classic focus/sharpness metric.
///
/// A sharp frame has strong high-frequency content, so its Laplacian has high
/// variance; a frame blurred by bad seeing has little. This is the deterministic
/// score that replaces "luck" in lucky imaging: we rank frames by it and keep the
/// best. Takes `&Gray` (works directly with `Iterator::map`).
pub fn laplacian_variance(g: &Gray) -> f32 {
    if g.w < 3 || g.h < 3 {
        return 0.0;
    }
    let mut vals = Vec::with_capacity((g.w - 2) * (g.h - 2));
    for y in 1..g.h - 1 {
        for x in 1..g.w - 1 {
            // 4-neighbour Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]
            let lap = g.at(x, y - 1) + g.at(x - 1, y) + g.at(x + 1, y) + g.at(x, y + 1)
                - 4.0 * g.at(x, y);
            vals.push(lap);
        }
    }
    let n = vals.len() as f32;
    let mean = vals.iter().sum::<f32>() / n;
    vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n
}

/// Measured background-noise estimate: the standard deviation of the high-frequency
/// residual (`pixel - local 3x3 mean`) over the darkest quartile of the image — i.e. the
/// sky around the target, where any fluctuation is noise rather than detail. This is the
/// honest, scene-independent way to show what stacking buys: noise falls, detail stays.
pub fn background_noise(g: &Gray) -> f32 {
    if g.w < 3 || g.h < 3 {
        return 0.0;
    }
    let mut sorted = g.data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = sorted[sorted.len() / 4]; // 25th percentile == sky
    let mut residuals = Vec::new();
    for y in 1..g.h - 1 {
        for x in 1..g.w - 1 {
            if g.at(x, y) > threshold {
                continue;
            }
            let mut sum = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    sum += g.at((x as i32 + dx) as usize, (y as i32 + dy) as usize);
                }
            }
            residuals.push(g.at(x, y) - sum / 9.0);
        }
    }
    if residuals.is_empty() {
        return 0.0;
    }
    let m = residuals.iter().sum::<f32>() / residuals.len() as f32;
    (residuals.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / residuals.len() as f32).sqrt()
}

# PhotoSharp — Roadmap

The pipeline is validated end-to-end (grade → align → sigma-clip stack → sharpen, video +
folder paths, colour). This roadmap tracks improvements found by running it on **real captures**
from the current rig (Celestron 114EQ, afocal phone video), calibrating the tool now so it is
polished by the time the GoTo upgrade lands (~2027, no rush).

## Verified on real data (2026-07-03 Moon test — PXL 4K/60fps afocal)

The pipeline stacked a handheld-ish afocal Moon clip cleanly (auto-tracked the disk across mount
drift, kept the sharpest 15 % of 300 frames, sigma-clipped). Two real issues surfaced:

- [ ] **Lunar / full-disk noise metric is wrong.** The `background noise a -> b (Nx cleaner)`
  readout assumes a *small planet on a black field* and measures the dark background. When the
  target fills the ROI (the Moon), there is no background to measure, so the number is meaningless
  and even reads "worse" after a clean stack. Fix: detect a full-disk/large target (ROI mostly
  above threshold) and either suppress the metric or switch to a target-independent quality proxy
  (e.g. high-frequency energy / gradient variance of the result).
- [ ] **Memory ceiling at 4K.** Decoding every frame's crop into RAM OOMs around
  ~500 frames × 1600² in colour (crops + f32 working copies). This caps how much of a 73 s clip we
  can use for lucky imaging. Fix: **stream/batch** — accumulate the stack incrementally (online
  sigma-clip, or a two-pass mean/variance) or spill decoded crops to disk, so the usable frame
  count is not bound by RAM. Unlocks using the whole clip, not just the first few seconds.

## Colour pipeline

- [x] RGB stacking that carries planetary colour (grading/alignment on luminance) — `--color`.
- [ ] **Real chroma validation on a planet.** The Moon is near-neutral, so it only proves the path
  runs. The Saturn clip (gold body/rings) is the actual colour test — process it and confirm the
  hue survives grade → align → stack.

## Nice-to-haves seen in the afocal workflow

- [ ] Optional **eyepiece field-stop / vignette crop** for afocal projection (the dark circular
  border), for a clean final frame.
- [ ] Auto-trim the alignment "staircase" border (the union of shifted/rotated crops).
- [ ] Per-target parameter presets (Moon vs small planet) so `--centroid-k` / `--roi` don't have to
  be hand-tuned each time.

## Not now (deliberate)

- GPU decode/stack — the RTX path is worth it once memory batching lands and clips get longer; the
  CPU pipeline is fast enough for current clip sizes.

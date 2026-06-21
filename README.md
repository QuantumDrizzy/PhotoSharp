# PhotoSharp

> Lucky-imaging stacking for planetary & lunar astrophotography — one clean frame from a shaky burst.

You point a telescope at Saturn, record a minute of 4K through a phone adapter, and almost
every frame is smeared by the atmosphere. **PhotoSharp** grades the whole burst, keeps the
sharpest frames, aligns them sub-pixel, stacks them to beat down noise, and sharpens the
result — turning a wobbly video into the still, detailed image you actually saw at the
eyepiece.

It is the open-source, native, *maintained* alternative to the classic but abandoned
PIPP → AutoStakkert! → RegiStax toolchain: one tool, no antivirus headaches, yours to
extend.

## The result

The bundled `demo` (no telescope required) simulates 200 jittered, blurred, noisy frames
of a planet and recovers a clean image — a **measured 4.9× drop in background noise** from
stacking the 61 sharpest frames:

| One raw frame | PhotoSharp (stacked) | Ground truth |
|:---:|:---:|:---:|
| ![one raw frame](docs/showcase/demo-single.png) | ![stacked result](docs/showcase/demo-stacked.png) | ![ground truth](docs/showcase/demo-truth.png) |

The grain on the left is what a single frame gives you; the middle is what grading,
aligning, stacking and sharpening recover. On a real Saturn or Jupiter capture the same
pipeline pulls the rings and cloud bands out of a shaky 4K video — Phase 2 wires in the
video decode so you can point it straight at your own footage.

## How it works

```
decode → grade (sharpness) → keep sharpest % → align (FFT) → stack → sharpen → export
```

The technique is **lucky imaging**, with the luck removed: instead of hoping for a good
frame, PhotoSharp scores every frame by the **variance of its Laplacian** (a deterministic
sharpness metric), keeps only the best, and registers them with **phase correlation** (the
FFT cross-power spectrum) before stacking. Stacking N frames lifts the signal-to-noise
ratio by ~√N; a final unsharp mask restores the detail the atmosphere softened.

## Status

**Phase 1 — the core pipeline — works and is tested.** It runs on a sequence of frames and
ships with a synthetic `demo` that verifies the whole chain end to end with no real data.

| Phase | Scope | State |
|-------|-------|-------|
| 1 | grade · align · stack · sharpen · CLI · synthetic demo | ✅ done |
| 2 | MP4/MOV/SER decode (real 4K captures end to end) | planned |
| 3 | native `egui` GUI (load, tune, preview, export) | planned |
| 4 | CUDA path · 16-bit TIFF · drizzle · sub-pixel · batch | planned |

See [`docs/ADR-0001-architecture.md`](docs/ADR-0001-architecture.md) for the design and its
trade-offs.

## Try it (no telescope needed)

```bash
cargo run --release -p photosharp-cli -- demo --frames 200 --out-prefix demo
```

This generates a synthetic planet, simulates 200 jittered/blurred/noisy frames, and writes
`demo-single.png` (one raw frame) next to `demo-stacked.png` (the recovered result) so you
can see what stacking buys you.

On real frames (a folder of PNG/JPEG/TIFF exported from a capture):

```bash
cargo run --release -p photosharp-cli -- stack --input ./frames --keep 0.3 --stretch --out saturn.png
```

## Build

```bash
cargo build --release
cargo test
```

Pure Rust, no system dependencies (video decode in Phase 2 will add `ffmpeg`).

## Licence

MIT — see [LICENSE](LICENSE). Built to be made yours.

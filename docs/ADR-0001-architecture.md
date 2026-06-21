# ADR-0001: PhotoSharp architecture

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** QuantumDrizzy

## Context

Planetary and lunar astrophotography through a small telescope is dominated by
*atmospheric seeing*: turbulence smears almost every frame, and only in brief, random
instants does the air settle enough for a sharp one. The established answer is **lucky
imaging** — record a video of hundreds to thousands of frames, keep the sharpest, align
and stack them to beat down noise, then sharpen. The classic toolchain (PIPP →
AutoStakkert! → RegiStax) does this across three or four separate Windows programs that
are years out of date, frequently trip antivirus heuristics, and cannot be modified.

I have a telescope with a phone adapter, lunar/planetary/nebula filters, and real 4K
captures of Saturn, Jupiter, Venus and the Moon. I want **one** tool, mine, that takes a
capture and returns the clean frame — open source, native, maintained, and trustworthy.

This is deliberately **separate** from the data-cycle work in SUBSTRATE (ingest → compress
→ store → analyse public datasets like NASA/TESS). That recovers discoveries from data
that already exists. PhotoSharp produces a *new* image from *my own* capture. Different
problem, different repo.

## Decision

Build PhotoSharp as a **Rust workspace** implementing the lucky-imaging pipeline from
first principles, with a native GUI and a headless CLI. No web stack. No heavyweight
frameworks. The processing runs on the PC (where the compute lives); the phone is only the
camera.

**Pipeline:** decode → grade (sharpness) → select sharpest % → register (sub-pixel) →
stack → sharpen → export.

## Options Considered

### Option A: Keep using PIPP / AutoStakkert! / RegiStax
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low (already exist) |
| Cost | Zero upfront |
| Control | None — closed, abandoned, antivirus-flagged |

**Pros:** mature, feature-rich, free.
**Cons:** unmaintained for years, trigger antivirus, three tools for one job, not modifiable, Windows-only binaries I cannot trust or extend.

### Option B: Rust workspace, native GUI + CLI (chosen)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — classic image processing (FFT registration, stacking, wavelets) |
| Cost | My time, phased |
| Control | Full — mine, open source, maintained |

**Pros:** one clean tool; bare-metal performance; a trustworthy signed binary; modifiable; CUDA-ready for 4K bursts; serves the community.
**Cons:** I have to build and maintain it; matching RegiStax's sharpening polish is iterative.

### Option C: Python (OpenCV/astropy) script
**Pros:** fastest to a first result.
**Cons:** packaging and distribution pain, slower on large bursts, not the native bare-metal tool I want to stand behind.

## Trade-off Analysis

The algorithms are not research-hard — they are well-understood image processing, the
same level as the CFAR/Kalman tracking in NIGHTWATCH. The real costs are (1) decoding the
phone's video codecs and (2) iterating on sharpening quality. Option B pays those costs
once in exchange for a tool I own and can keep alive — which is the whole point, since the
incumbents died from *not* being maintained.

## Stack

- **Core:** Rust. FFT via `rustfft`; image I/O via `image`; no GPU dependency by default.
- **GUI:** `egui`/`eframe`, native (planned Phase 3). Never web.
- **CLI:** `clap`. Headless `stack` and a `demo` that needs no real data.
- **Video decode:** `ffmpeg` (Phase 2) — the one external dependency, isolated behind the decode module.
- **GPU:** optional, feature-gated CUDA for large 4K bursts (Phase 4), off by default so it builds anywhere — the TESSERA/GeoPulse pattern.

## Build Sequence

- **Phase 0** — workspace, this ADR, README, MIT licence.
- **Phase 1** — the core science: grade + FFT registration + stack + sharpen, on a frame
  sequence, plus a synthetic `demo` that verifies the pipeline with no real data. **(this commit)**
- **Phase 2** — MP4/MOV/SER decode, so a real 4K capture works end to end.
- **Phase 3** — `egui` GUI: load, tune keep-% and sharpening, live preview, export.
- **Phase 4** — CUDA path, 16-bit TIFF output, drizzle upscaling, batch mode, sub-pixel
  resampling.

## Consequences

- **Easier:** a single trustworthy tool; reproducible results; a clean base to extend.
- **Harder:** I own maintenance; sub-pixel resampling and wavelet sharpening are deferred
  to later phases, so Phase 1 alignment is integer-pixel.
- **Revisit:** registration model (translation-only today; rotation/affine if mounts
  warrant it), and whether a phone-side capture companion is worth building later.

## Known limits (honest)

- Phase 1 registration is **integer-pixel**; sub-pixel resampling lands in Phase 4.
- No real video decode yet (Phase 2) — Phase 1 consumes a frame sequence or the synthetic demo.
- Sharpening is a single unsharp mask; multi-scale wavelets come later.

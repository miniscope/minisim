# 01 - Anatomy of a 1-photon miniscope recording

**Build a recording forward from its physics - the inverse of the minian analysis pipeline.**

This is the first of two training notebooks. It **constructs** a synthetic
miniscope recording one physically-meaningful stage at a time, using
[`minisim`](https://github.com/miniscope/minisim). Each stage follows the same rhythm:
*understand* the physics, *explore* it with sliders, then *commit* the values you
want and move on - so the recording grows in front of you. Because every stage is
built by hand, the *exact* ground truth is known at each step, which is precisely
what the analysis pipeline has to recover. Notebook 2 (`02_demixing`) then reads the
biology back out by demixing a recording like this, and Notebook 3 (`03_metrics`)
scores how well a pipeline recovers that truth.

> **The full forward pipeline is built.** All stages are in place, in physical
> order: place neurons, calcium activity, photobleaching, optics (depth blur +
> dimming), composite (the first movie), neuropil background, brain motion,
> illumination profile, vignette, leakage, and the image sensor (noisy integer
> counts). The notebook ends by streaming the complete recording to a grayscale AVI.

## Run it

No data download is needed - the recording is generated on the fly.

```bash
pip install "minisim[notebook]"   # engine + ipywidgets, matplotlib, mediapy
jupyter notebook 01_anatomy.ipynb
```

Run top to bottom; each stage uses the values committed above it. The **explore**
cells are interactive and need a live kernel (e.g. the Stage-1 scope sliders for
NA / magnification / pixel pitch); in a statically-rendered copy they show their
default state - run the notebook locally to drag them.

## Dependencies beyond core Minisim

`ipywidgets` (sliders), `matplotlib` (figures), and `mediapy` (inline movie
playback, used by the movie stages; it relies on `ffmpeg`). All are lightweight
and need no GPU.

## What you'll learn

- A recording is a physical chain, and the `Spec` (`acquisition` + an ordered
  list of `steps`) *is* that chain written down.
- How numerical aperture (light collection ∝ NA²), magnification, pixel pitch,
  depth, and tissue scattering shape what the sensor actually sees - and set an
  **irreducible limit** on what any analysis can recover.
- What each minian stage is the inverse of: motion correction ↔ `shifts`,
  background/glow removal ↔ neuropil/leakage, denoising ↔ the sensor model.

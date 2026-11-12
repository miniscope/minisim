# Concepts

## The forward model

An analysis pipeline runs *backward*: it takes a movie and tries to recover what
produced it (where the cells are, when they fired). Minisim runs the same chain
*forward*. It starts from biology and optics and produces the movie, so it
already knows every answer the analysis pipeline is trying to recover.

```
place neurons -> cell activity -> bleaching -> optics -> composite -> neuropil
             -> brain motion -> illumination profile -> vignette -> leakage
             -> (sensor defects) -> image sensor
```

Each arrow is a small, inspectable physical model. A neuron is placed in tissue;
its calcium activity becomes a fluorescence trace; the objective blurs and dims
it by depth; the result is composited onto a canvas; the brain moves under the lens;
the illumination falls off toward the edges; and finally a sensor turns photons
into integer counts (optionally with hot pixels and fixed-pattern noise). Run the chain and you get a movie that *looks* like
miniscope data because it was made the way miniscope data is made.

## Ground truth is a byproduct, not an annotation

Because the recording is built forward, every quantity an analysis pipeline
would estimate is something Minisim *chose* on the way in. They are attached to
the result as {py:class}`~minisim.GroundTruth`:

- cell locations, depths, and spatial footprints,
- per-cell calcium traces and spike times,
- the brain-motion trajectory (rigid `shifts`, plus the `nonrigid_field` when
  non-rigid motion is on; {py:meth}`~minisim.GroundTruth.displacement` gives the
  full motion at any pixel),
- the per-pixel optical fields (illumination, vignette),
- the sensor's static defect maps (hot/stuck pixels, PRNU, row/column offsets)
  when {py:class}`~minisim.SensorDefects` is in the chain,
- per-cell halo parameters (`observed_halo_*`) when the tissue has a scatter halo.

This is what makes the data useful for benchmarking: you are not comparing
against a human-labeled annotation that might be wrong, you are comparing
against the exact signal that generated the pixels.

## Detectability: which cells a pipeline could recover

Not every planted cell can be found: a deep, defocused, dim or vessel-covered cell
may leave too little signal above the noise. Each cell's `detectable` flag records
whether it could be, and by default {py:func}`~minisim.testing.score` computes
recall over the detectable cells only (see the
{doc}`stability contract <reference/stability>`).

The test is the **matched-filter SNR** ({py:func}`~minisim.matched_filter_snr`):
the SNR of the ideal detector that knows the cell's footprint and its activity.
In detected electrons, cell `k` adds `g_i·a_i·Δ_t` to pixel `i` at frame `t`.
Here `a` is its realized footprint, after defocus, scatter and halo; `g` is the
per-pixel gain from vessels, illumination, vignette, exposure and QE; and
`Δ_t = (C_t − min C)·B_t` is its emitted transient above rest, with bleaching `B`.
Against independent per-pixel noise of variance `σ_i² = λ_i + r²`, the matched
filter's SNR is

```
SNR² = [ Σ_i g_i² a_i² / σ_i² ] · [ Σ_t Δ_t² ]
```

Here `λ_i` is the shot noise: the mean electrons from all the light at that pixel
(other cells' out-of-focus light, neuropil haze, leakage). `r` is the read noise.
The first factor pools the cell's pixels and the second pools all its transients,
the way factorization pipelines such as CNMF fit a footprint and a trace jointly.
Two consequences:

- **Defocus costs less than the peak suggests.** Blur conserves light. A defocused
  cell's peak falls as `σ₀²/σ²`, but its matched-filter SNR falls only as `σ₀/σ`,
  because the light lands on more pixels. Up to minisim 1.0 `detectable` was a
  single-pixel peak test ({py:func}`~minisim.peak_snr`, kept as a diagnostic).
  With sharp PSFs it undercounted the cells pipelines recover, and under field
  curvature the detectable cells formed a ring.
- **Activity counts.** More transients mean more signal energy, so a busier cell or
  a longer recording is more detectable, as it is for a pipeline.

The fluctuations of the neuropil and of neighbouring cells are not counted as
noise, because pipelines model them as components of their own; their light
still adds shot noise.

**Calibration.** The SNR above is for an ideal observer. A real pipeline must also
estimate the footprint and the trace, so it needs more. The threshold
{py:data}`~minisim.DETECT_SNR_THRESHOLD` = 110 is the SNR at which a strong
pipeline (cala, on simulated CA1 recordings at the real recording's cell size and
blur) recovers half the cells. It was fitted on 6 px cells and checked on 10 px
cells, where recovery also crosses 50 % near 100. On cells much larger than that
pipeline's cell-size setting (14-16 px against its 10 px) recovery needs 3-4 times
the SNR. That is the pipeline's size prior, not noise, so the threshold is not
raised for it.

## The three objects you work with

Minisim's surface is small. Almost everything you do is:

```
    Spec  ──simulate──▶  Recording  ──your pipeline──▶  estimates
  (typed input)      (movie + GroundTruth)                  │
                            │                               │
                            └──────────── metrics ──────────┘
                                    (recovery scores)
```

- {py:class}`~minisim.Spec` - a fully typed, validated description of a
  recording: an {py:class}`~minisim.Acquisition` (optics, sensor, tissue,
  sampling), a `seed`, and an ordered list of `steps`. It is reproducible: the
  same `Spec` always yields the same recording, and `spec.cache_key()` is a
  stable hash of its JSON form.
- {py:class}`~minisim.Recording` - the result: `observed` (the movie),
  `ground_truth`, and optional per-stage `snapshots`.
- The {doc}`metrics <reference/metrics>` - functions that score recovered
  cells, traces, spikes, and motion against the ground truth.

## Steps and the pipeline

`Spec.steps` is the forward chain, as a list of typed step specs
({py:class}`~minisim.PlaceNeurons`, {py:class}`~minisim.CellActivity`,
{py:class}`~minisim.Composite`, {py:class}`~minisim.Sensor`, ...). Each `kind`
appears at most once, and the `Spec` validates the chain on construction:
genuinely invalid orderings raise, while unusual-but-legal ones emit a
{py:class}`~minisim.SpecWarning`. You can stop the chain early with
`simulate(spec, until="<stage name>")` to inspect an intermediate stage, or set
`Output.save_intermediates=True` to keep every stage in `Recording.snapshots`.

The minimal chain to get a movie is place → activity → optics → composite → sensor;
the remaining steps (neuropil, vasculature, bleaching, brain motion,
illumination, vignette, leakage) layer on the physical effects that make the
data realistic. Three further effects are opt-in and absent from a default spec:
the heavy-tailed {doc}`scatter halo <howto/scatter_psf>` (`Tissue` fields, not a
step), {doc}`realistic and non-rigid brain motion <howto/realistic_motion>`
(`BrainMotion` options), and the {doc}`sensor defects <howto/sensor_defects>` step
(placed just before `sensor`). See the {doc}`reference/spec` for every step and its parameters.

## Getting a recording out

`simulate(spec)` returns a {py:class}`~minisim.Recording` in memory;
`Recording.save()` writes it as a zarr directory (movie, ground truth and spec).
To hand simulated data to a pipeline that reads real Miniscope output, use
{py:func}`~minisim.write_miniscope_daq` or the `minisim generate` command line
(see the {doc}`DAQ folder guide <howto/daq_folder>` and the
{doc}`CLI reference <reference/cli>`), which stream a Miniscope-DAQ folder of
`N.avi` files with the ground truth beside it.

## Design principle: simulate the biology, not the algorithm

Minisim deliberately models the real physical and biological process, *not* the
assumptions of any particular analysis pipeline. The point of benchmarking
against it is to find where a pipeline's model diverges from the biology, so the
data is never shaped to flatter a given algorithm.

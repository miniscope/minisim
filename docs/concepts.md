# Concepts

## The forward model

An analysis pipeline runs *backward*: it takes a movie and tries to recover what
produced it (where the cells are, when they fired). Minisim runs the same chain
*forward*. It starts from biology and optics and produces the movie, so it
already knows every answer the analysis pipeline is trying to recover.

```
place neurons -> cell activity -> bleaching -> optics -> composite -> neuropil
             -> brain motion -> illumination profile -> vignette -> leakage -> image sensor
```

Each arrow is a small, inspectable physical model. A neuron is placed in tissue;
its calcium activity becomes a fluorescence trace; the objective blurs and dims
it by depth; the result is composited onto a canvas; the brain moves under the lens;
the illumination falls off toward the edges; and finally a sensor turns photons
into integer counts. Run the chain and you get a movie that *looks* like
miniscope data because it was made the way miniscope data is made.

## Ground truth is a byproduct, not an annotation

Because the recording is built forward, every quantity an analysis pipeline
would estimate is something Minisim *chose* on the way in. They are attached to
the result as {py:class}`~minisim.GroundTruth`:

- cell locations, depths, and spatial footprints,
- per-cell calcium traces and spike times,
- the brain-motion trajectory,
- the per-pixel optical fields (illumination, vignette).

This is what makes the data useful for benchmarking: you are not comparing
against a human-labeled annotation that might be wrong, you are comparing
against the exact signal that generated the pixels.

## The three objects you work with

Minisim's surface is small. Almost everything you do is:

```
    Spec  ──simulate──▶  Recording  ──your pipeline──▶  estimates
  (typed input)      (movie + GroundTruth)                  │
                            │                               │
                            └──────────── metrics ──────────┘
                                    (recovery scores)
```

- {py:class}`~minisim.Spec` — a fully typed, validated description of a
  recording: an {py:class}`~minisim.Acquisition` (optics, sensor, tissue,
  sampling), a `seed`, and an ordered list of `steps`. It is reproducible: the
  same `Spec` always yields the same recording, and `spec.cache_key()` is a
  stable hash of its JSON form.
- {py:class}`~minisim.Recording` — the result: `observed` (the movie),
  `ground_truth`, and optional per-stage `snapshots`.
- The {doc}`metrics <reference/metrics>` — functions that score recovered
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
data realistic. See the {doc}`reference/spec` for every step and its parameters.

## Design principle: simulate the biology, not the algorithm

Minisim deliberately models the real physical and biological process, *not* the
assumptions of any particular analysis pipeline. The point of benchmarking
against it is to find where a pipeline's model diverges from the biology, so the
data is never shaped to flatter a given algorithm.

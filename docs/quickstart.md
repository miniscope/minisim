# Quickstart

## Install

```bash
pip install minisim                # engine only
pip install "minisim[notebook]"    # + the interactive teaching notebooks
```

## Simulate a recording

Build a {py:class}`~minisim.Spec`, then {py:func}`~minisim.simulate` it. The
minimal forward chain is place neurons → cell activity → optics → composite →
sensor; defaults are filled in for everything you do not set.

```python
from minisim import (
    Acquisition, Optics, ImageSensor,
    PlaceNeurons, CellActivity, CellOptics, Composite, Sensor,
    Spec, simulate,
)

spec = Spec(
    acquisition=Acquisition(
        fps=20.0,
        duration_s=10.0,
        optics=Optics(magnification=8.0, na=0.45),
        image_sensor=ImageSensor(n_px_height=256, n_px_width=256, pixel_pitch_um=8.0),
    ),
    seed=0,
    steps=[
        PlaceNeurons(),
        CellActivity(),
        CellOptics(),
        Composite(),
        Sensor(),
    ],
)

rec = simulate(spec)
movie = rec.observed          # (frame, height, width) array of sensor counts
truth = rec.ground_truth      # cells, traces, spikes, optical fields
```

## Inspect the ground truth

Everything the recording was built from is on
{py:class}`~minisim.GroundTruth`:

```python
gt = rec.ground_truth
gt.n_units          # how many cells were planted
gt.A_planted        # spatial footprints, as planted
gt.A_observed       # footprints after optics (blurred, dimmed by depth)
gt.depth_um         # each cell's depth below the tissue surface

detectable = gt.detectable_subset()   # the cells actually recoverable in this movie
```

## Reproducibility and caching

A `Spec` fully determines its recording: same `Spec`, same movie, every time.
`spec.cache_key()` is a stable hash of the spec, and
{py:func}`~minisim.simulate_cached` memoizes results to disk by that key, so a
repeated simulation is a fast load instead of a recompute.

```python
from minisim import simulate_cached

rec = simulate_cached(spec)   # computes once, then loads on subsequent calls
```

## Stop early to inspect a stage

Pass `until` to stop the chain at a named stage, or keep every stage by setting
`Output.save_intermediates=True` and reading them back with `Recording.stage()`.
A stage name is usually the step's `kind`; the one exception is the `composite`
step, whose stage is named `"cells_only"` (the cells-on-black movie it produces).

```python
from minisim import Output

partial = simulate(spec, until="cells_only")  # stop after the composite step

spec_full = spec.model_copy(update={"output": Output(save_intermediates=True)})
rec = simulate(spec_full)
cells_only = rec.stage("cells_only")          # the movie as of the composite stage
```

## Where to next

- {doc}`How-to: benchmark a pipeline <howto/benchmark>`
- {doc}`How-to: sweep parameters <howto/sweep>`
- {doc}`How-to: export to video <howto/video>`
- {doc}`API reference <reference/index>`

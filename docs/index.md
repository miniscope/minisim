---
sd_hide_title: true
---

# Minisim

```{image} _static/logo/minisim_wordmark_sim.png
:alt: Minisim
:width: 360px
:align: center
```

```{div} sd-text-center sd-fs-3 sd-font-weight-light
Physically-driven synthetic 1-photon miniscope data
```

```{div} sd-text-center sd-fs-5 sd-text-muted
A forward-model generator and teaching tool, the inverse of an analysis
pipeline like [minian](https://github.com/miniscope/minian).
```

---

Minisim builds a miniscope recording *forward* from its physical components, the
inverse of an analysis pipeline. Instead of recovering signals from a movie, it
starts from biology and optics and produces the movie, together with the exact
ground truth that generated it: cell locations, footprints, calcium traces,
spike times, motion trajectory, and per-pixel optical fields.

```{code-block} python
from minisim import (
    Acquisition, Optics, ImageSensor,
    PlaceNeurons, CellActivity, CellOptics, Composite, Sensor,
    Spec, simulate,
)

spec = Spec(
    acquisition=Acquisition(
        fps=20.0, duration_s=10.0,
        optics=Optics(magnification=8.0, na=0.45),
        image_sensor=ImageSensor(n_px_height=256, n_px_width=256, pixel_pitch_um=8.0),
    ),
    seed=0,
    steps=[PlaceNeurons(), CellActivity(), CellOptics(), Composite(), Sensor()],
)

rec = simulate(spec)         # -> Recording, with rec.ground_truth attached
movie = rec.observed         # the simulated movie: (frame, height, width) array
```

Because every recording ships with its ground truth, Minisim is built for:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`graph` Benchmarking
Score calcium-imaging pipelines (minian, CaImAn, suite2p) against known truth
with the recovery [metrics](reference/metrics).
:::

:::{grid-item-card} {octicon}`mortar-board` Teaching
Walk the anatomy of miniscope data: what each physical effect does to the
image. See the [tutorials](tutorials/index).
:::

:::{grid-item-card} {octicon}`beaker` Testing
Reproducible, parameterized fixtures for analysis code, with a typed
[`Spec`](reference/spec) and disk caching.
:::

::::

## Install

```bash
pip install minisim                # engine only
pip install "minisim[notebook]"    # + the interactive teaching notebooks
```

Requires Python >= 3.10. Core dependencies are numpy, scipy, xarray, zarr,
pydantic, and numpydantic.

The teaching notebooks ship inside the package; list them with
`minisim-notebooks list` and copy one out with `minisim-notebooks copy 01_anatomy`
(see the {doc}`tutorials <tutorials/index>`).

## Where to go next

- New here? Start with the {doc}`concepts` page for the mental model, then the
  {doc}`quickstart`.
- Want to run something specific? The {doc}`how-to guides <howto/index>` cover
  benchmarking, parameter sweeps, and video export.
- Looking for a class or function? The {doc}`API reference <reference/index>`
  is generated from the package.

```{toctree}
:hidden:
:caption: Getting started

concepts
quickstart
```

```{toctree}
:hidden:
:caption: Guides

howto/index
tutorials/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

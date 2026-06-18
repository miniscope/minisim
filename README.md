<p align="center">
  <img src="https://raw.githubusercontent.com/miniscope/minisim/main/docs/_static/logo/minisim_wordmark_sim.png" alt="Minisim" width="380">
</p>

[![pytest](https://github.com/miniscope/minisim/actions/workflows/testandcov.yml/badge.svg)](https://github.com/miniscope/minisim/actions/workflows/testandcov.yml)
[![codecov](https://codecov.io/gh/miniscope/minisim/branch/main/graph/badge.svg)](https://codecov.io/gh/miniscope/minisim)
[![Documentation Status](https://readthedocs.org/projects/minisim/badge/?version=latest)](https://minisim.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/minisim.svg)](https://pypi.org/project/minisim/)
[![Python versions](https://img.shields.io/pypi/pyversions/minisim.svg)](https://pypi.org/project/minisim/)
[![License](https://img.shields.io/pypi/l/minisim.svg)](https://github.com/miniscope/minisim/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Physically-driven synthetic 1-photon miniscope data: a forward-model
generator and teaching tool.**

Minisim builds a miniscope recording *forward* from its physical components, the
inverse of an analysis pipeline like [minian](https://github.com/miniscope/minian).
Instead of recovering signals from a movie, it starts from biology and optics and
produces the movie, together with the ground truth that generated it:

```
place neurons -> cell activity -> bleaching -> optics -> composite -> neuropil
             -> brain motion -> illumination profile -> vignette -> leakage -> image sensor
```

Each stage is a small, inspectable physical model. Because the recording is built
forward, every recording ships with exact ground truth (cell locations,
footprints, calcium traces, spike times, motion trajectory, per-pixel optical
fields), which makes Minisim useful for:

- **Benchmarking** calcium-imaging pipelines (minian, CaImAn, suite2p, ...)
  against known ground truth.
- **Teaching** the anatomy of miniscope data: what each physical effect does to
  the image, via interactive notebooks.
- **Testing** analysis code with reproducible, parameterized fixtures.

📖 **Full documentation: [minisim.readthedocs.io](https://minisim.readthedocs.io/)** - concepts,
quickstart, how-to guides, and the API reference.

## Install

```bash
pip install minisim                # engine only
pip install "minisim[notebook]"    # + the interactive teaching notebooks
```

Requires Python >= 3.10. Core dependencies are just numpy, scipy, xarray, zarr,
pydantic, and numpydantic.

The teaching notebooks ship inside the package; list them and copy the ones you
want out to a writable directory with the bundled command:

```bash
minisim-notebooks list                  # see what's available
minisim-notebooks copy 01_anatomy       # -> ./minisim-notebooks/01_anatomy
# then: cd minisim-notebooks/01_anatomy && jupyter lab
```

## Quick start

```python
from minisim import (
    Acquisition, Optics, ImageSensor, PlaceNeurons, CellActivity,
    CellOptics, Composite, Sensor, Spec, simulate,
)

spec = Spec(
    acquisition=Acquisition(
        fps=20.0, duration_s=10.0,
        optics=Optics(magnification=8.0, na=0.45),
        image_sensor=ImageSensor(n_px_height=256, n_px_width=256, pixel_pitch_um=8.0),
    ),
    seed=0,
    steps=[
        PlaceNeurons(density_per_mm3=400000.0, soma_radius_um=4.0),
        CellActivity(),
        CellOptics(),
        Composite(),
        Sensor(),
    ],
)

rec = simulate(spec)
movie = rec.observed          # xarray DataArray (frame, height, width)
truth = rec.ground_truth      # cells, traces, spikes, optical fields
```

## Relationship to minian

Minisim is the forward (generative) counterpart to minian's inverse (analysis)
pipeline. The dependency is designed to be strictly one-directional: Minisim
never imports minian. The intended integration is for minian to use Minisim as a
test dependency, supplying ground-truth fixtures for its recovery tests; that
wiring is planned, not yet in place.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

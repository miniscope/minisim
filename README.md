# minisim

[![pytest](https://github.com/miniscope/minisim/actions/workflows/testandcov.yml/badge.svg)](https://github.com/miniscope/minisim/actions/workflows/testandcov.yml)
[![codecov](https://codecov.io/gh/miniscope/minisim/branch/main/graph/badge.svg)](https://codecov.io/gh/miniscope/minisim)
[![PyPI](https://img.shields.io/pypi/v/minisim.svg)](https://pypi.org/project/minisim/)
[![Python versions](https://img.shields.io/pypi/pyversions/minisim.svg)](https://pypi.org/project/minisim/)
[![License](https://img.shields.io/pypi/l/minisim.svg)](https://github.com/miniscope/minisim/blob/main/LICENSE)

**Physically-driven synthetic 1-photon miniscope data: a forward-model
generator and teaching tool.**

minisim builds a miniscope recording *forward* from its physical components, the
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
fields), which makes minisim useful for:

- **Benchmarking** calcium-imaging pipelines (minian, CaImAn, suite2p, ...)
  against known ground truth.
- **Teaching** the anatomy of miniscope data: what each physical effect does to
  the image, via interactive notebooks.
- **Testing** analysis code with reproducible, parameterized fixtures.

## Install

```bash
pip install minisim                # engine only
pip install "minisim[notebook]"    # + the interactive teaching notebooks
```

Requires Python >= 3.10. Core dependencies are just numpy, scipy, xarray, zarr,
pydantic, and numpydantic.

The teaching notebooks ship inside the package; copy them somewhere writable
with the bundled command:

```bash
minisim-notebooks ./minisim-notebooks   # then: cd minisim-notebooks && jupyter lab
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

minisim is the forward (generative) counterpart to minian's inverse (analysis)
pipeline. The dependency is strictly one-directional: minisim never imports
minian. minian uses minisim as a test dependency to supply ground-truth fixtures
for its recovery tests.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

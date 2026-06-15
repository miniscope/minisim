# Sweep parameters

{py:func}`~minisim.sweep` takes a base {py:class}`~minisim.Spec` and a dict of
*dotted-path → list of values* and yields one validated spec per point in their
Cartesian product. Because the axes are physically meaningful (depth in µm, NA,
density in cells/mm³), a sweep traces a scientifically interpretable surface,
for example "recall vs depth at NA 0.45".

## Define the axes

Paths address any field in the spec:

- nested models: `"acquisition.optics.na"`,
- a step by its (unique) `kind`: `"steps.place_neurons.density_per_mm3"`,
- a top-level field: `"seed"`.

```python
from minisim import sweep

for variant in sweep(base_spec, {
    "acquisition.optics.na": [0.3, 0.45, 0.6],
    "acquisition.focal_depth_in_tissue_um": [50.0, 100.0, 150.0],
}):
    print(variant.axes)   # e.g. {'acquisition.optics.na': 0.45, ...}
```

Each yielded {py:class}`~minisim.SweptSpec` is a real `Spec` (it drops straight
into `simulate`) tagged with an `axes` dict recording the chosen value per path.
Every cross-field validator re-runs per combination, so an axis value that
produces an invalid spec raises immediately rather than at simulate time.

## Collect a benchmark surface

Combine a sweep with the {doc}`benchmarking recipe <benchmark>` to build a tidy
table keyed on the physical axes:

```python
import pandas as pd
from minisim import simulate, hungarian_match

rows = []
for variant in sweep(base_spec, {
    "acquisition.optics.na": [0.3, 0.45, 0.6],
    "acquisition.focal_depth_in_tissue_um": [50.0, 100.0, 150.0],
}):
    rec = simulate(variant)
    det = rec.ground_truth.detectable_subset()
    A_est, _, _ = run_my_pipeline(rec.observed)
    match = hungarian_match(A_est, det.A_observed)
    rows.append({**variant.axes, "recall": match.recall()})

df = pd.DataFrame(rows)   # one row per (na, depth) point, ready to pivot/plot
```

## Worked example: a focus × separation grid

A concrete two-axis sweep. Place **two independent cells** at a known depth, then
sweep the **focal plane** (a scalar axis) against their **lateral separation**
(swept by overriding the whole `place_neurons.populations` field with a fresh
pair per value). At each grid point we do the most naive trace extraction there
is - a small ROI dropped on each cell's true position, pixels averaged per frame -
and correlate the two ROI traces. The cells fire independently, so their true
correlation is ~0; anything above that is optical blur bleeding one cell into the
other's ROI.

```python
import numpy as np
import pandas as pd

from minisim import (
    Acquisition, CellActivity, CellOptics, Composite, NeuronPopulation,
    PlaceNeurons, Sensor, Spec, Tissue, simulate, sweep,
)
from minisim.presets import miniscope_v4

DEPTHS_UM, ROI_DIAM_UM = (100.0, 110.0), 20.0
FOCAL_PLANES_UM = [85.0, 95.0, 105.0, 115.0, 125.0]
SEPARATIONS_UM = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


def two_cells(sep_um):                        # a pair straddling the axis at y = 0
    return NeuronPopulation(
        positions_um=[(DEPTHS_UM[0], 0.0, -sep_um / 2), (DEPTHS_UM[1], 0.0, sep_um / 2)],
        soma_radius_um=5.0, morphology="cytosolic",
    )


def roi_trace(observed, acq, y_um, x_um):     # brute-force extraction: mean of an ROI
    h, w = observed.shape[1:]
    cr, cc = acq.um_to_index(y_um, x_um, (h, w))      # true position -> pixel
    radius_px = (ROI_DIAM_UM / 2) / acq.pixel_size_um
    rr, cols = np.ogrid[:h, :w]
    mask = (rr - cr) ** 2 + (cols - cc) ** 2 <= radius_px**2
    return observed[:, mask].mean(axis=1)


# V4 optics with the sensor cropped to 128 px (so the grid runs in seconds), a
# default Tissue scatter model, and the standard cell chain. focal_depth and the
# cell pair here are placeholders that the sweep overrides per grid point.
v4 = miniscope_v4()
small = v4.image_sensor.model_copy(update={"n_px_height": 128, "n_px_width": 128})
base = Spec(
    acquisition=Acquisition(
        optics=v4.optics, image_sensor=small, tissue=Tissue(),
        fps=10.0, duration_s=60.0,           # 60 s so the true correlation settles to ~0
    ),
    seed=0,
    steps=[
        PlaceNeurons(populations=[two_cells(0.0)]),
        CellActivity(), CellOptics(), Composite(),
        Sensor(photons_per_unit=250.0),
    ],
)

rows = []
for variant in sweep(base, {
    "acquisition.focal_depth_in_tissue_um": FOCAL_PLANES_UM,            # scalar axis
    "steps.place_neurons.populations": [[two_cells(s)] for s in SEPARATIONS_UM],
}):
    rec = simulate(variant)
    obs, gt, acq = np.asarray(rec.observed), rec.ground_truth, rec.spec.acquisition
    (_, y0, x0), (_, y1, x1) = gt.centers_um
    t0, t1 = roi_trace(obs, acq, y0, x0), roi_trace(obs, acq, y1, x1)
    sep = x1 - x0
    rows.append({
        "focal_depth_um": variant.axes["acquisition.focal_depth_in_tissue_um"],
        "separation_um": sep,
        "roi_corr": 1.0 if sep == 0.0 else float(np.corrcoef(t0, t1)[0, 1]),
    })

grid = pd.DataFrame(rows).pivot(
    index="focal_depth_um", columns="separation_um", values="roi_corr"
)
```

Sweeping a list-valued field (`populations`) works because each combination is
re-validated from its canonical dump - the same round-trip `simulate_cached` and
JSON reload use - so a hand-placed pair survives the sweep intact.

:::{figure} /_static/examples/overlap_grid_images.png
:alt: a 5x6 grid of max-projection thumbnails over focal plane and separation, each with the ROI drawn to scale

Max projection at each grid point (focal plane down, separation across; red circle
= the 20 µm ROI, to scale). The pair is sharpest and best-separated on the in-focus
105 µm row and blurs back together off-focus.
:::

:::{figure} /_static/examples/overlap_grid_corr.png
:alt: a heatmap of ROI-trace correlation over focal plane and separation

The `grid` table as a heatmap. Correlation is 1.0 when the ROIs overlap (Δ = 0),
stays high while the cells share blur (Δ ≤ 10 µm), and relaxes to the ~0.1
independent-cell floor once they separate (Δ ≥ 30 µm). At the marginal Δ = 20 µm
the focal plane matters most - the in-focus 105 µm row roughly halves the crosstalk.
:::

The full figure-generating script is `scripts/gen_overlap_focus_grid.py`.

## Notes

- `axes` is excluded from serialization, so `SweptSpec.cache_key()` equals the
  equivalent plain spec's. Sweeping does not perturb {doc}`cache
  <../reference/caching>` dedup, and the tag vanishes when a recording is saved.
- An empty `axes` dict yields the base spec once, with `axes={}`.
- For expensive sweeps, swap `simulate` for {py:func}`~minisim.simulate_cached`
  so repeated points load instead of recompute.

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

## Notes

- `axes` is excluded from serialization, so `SweptSpec.cache_key()` equals the
  equivalent plain spec's. Sweeping does not perturb {doc}`cache
  <../reference/caching>` dedup, and the tag vanishes when a recording is saved.
- An empty `axes` dict yields the base spec once, with `axes={}`.
- For expensive sweeps, swap `simulate` for {py:func}`~minisim.simulate_cached`
  so repeated points load instead of recompute.

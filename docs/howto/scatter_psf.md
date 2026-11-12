# Add a heavy-tailed scatter halo

By default a cell's image is its planted shape blurred by **one Gaussian**
(diffraction, defocus and scatter added in quadrature). Real 1-photon footprints have
a long tail that a single Gaussian cannot make: multiply-scattered light is
re-collected from a broad neighbourhood, so a Miniscope V4 CA1 cell sits on a roughly
1 % plateau out to ~50 px. The opt-in **scatter halo** adds that tail:

```
PSF = (1 − f) · Gaussian(σ_core) + f · Gaussian(σ_halo)
```

```python
from minisim import Tissue, presets

region = presets.ca1().model_copy(
    update={"tissue": Tissue(scatter_halo_fraction=0.45, scatter_halo_blur_per_um=0.37)}
)
spec = presets.build_spec(presets.miniscope_v4(), region)
```

Both fields live on {py:class}`~minisim.Tissue`:

- `scatter_halo_fraction` (`f`, default `0.0` = off). With `0` every output, spec hash
  and cache key is bit-identical to a build without the feature.
- `scatter_halo_blur_per_um` (default `0.37`, µm of σ per µm of depth). The wing is the
  core image further broadened by diffuse scatter, so
  `σ_halo = hypot(σ_core, scatter_halo_blur_per_um · z)`. It scales linearly with depth
  like the core's `scatter_blur_per_um`, but several times faster: diffuse photons
  random-walk on the *transport* length, not the ballistic MFP. It is ignored while
  `f = 0`.

## Energy accounting

The halo is **redistributed from the core**, not added on top. Both Gaussians are
sum-normalized, so a cell's footprint integral is `gain · Σ planted` for every `f`;
`attenuation(z)` already describes the light the objective *collects*, which includes
the recollected diffuse light. The price is a lower core: the peak-brightness scalar
(`optical_brightness`, a diagnostic) becomes
`attenuation · collection · [(1−f)·σ₀²/σ_core² + f·σ₀²/σ_halo²]`, i.e. it falls by
about `1 − f`. `detectable` uses the whole realized footprint, halo included
({py:func}`~minisim.matched_filter_snr`), so the wing lowers it less than it lowers
the peak. `in_focus` stays the geometric depth-of-field test. Light past the
canvas edge is lost for the halo exactly as for the core.

## Ground truth

`GroundTruth.observed_halo_sigma_px` and `observed_halo_fraction` (per cell, `None`
when `f = 0`) join `observed_sigma_px` / `observed_gain`, so
`GroundTruth.A_observed` reproduces the full footprint. Observed footprints are still
regenerated on demand; the halo kernel is truncated at 4σ and applied as two small
matrix products, so rendering costs about 1.1× the single-Gaussian path
(256 px, 300 frames) and memory is unchanged. Streaming
({py:func}`~minisim.simulate_video`, {py:func}`~minisim.write_miniscope_daq`) matches
{py:func}`~minisim.simulate` exactly.

## Choosing `f`

Part of a measured tail may be co-activation rather than optics, so `f` is uncertain
(between 0 and ~0.45 for the V4 CA1 data we calibrated against). For a V4 + CA1 cell at
the preset depth (150 µm, 1.655 µm/px) the radial profile relative to peak is:

| radius (px) | 5 | 10 | 15 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| real V4 CA1 | 0.54 | 0.14 | 0.02 | 0.003 | 0.012 | 0.010 |
| `f=0.45`, rate 0.37 (σ_halo ≈ 34 px) | 0.571 | 0.122 | 0.022 | 0.014 | 0.011 | 0.008 |
| `f=0.30`, rate 0.29 | 0.570 | 0.120 | 0.019 | 0.010 | 0.007 | 0.004 |
| `f=0.15`, rate 0.27 | 0.567 | 0.114 | 0.013 | 0.005 | 0.003 | 0.002 |
| no halo (default) | 0.564 | 0.109 | 0.007 | 0.0001 | 0 | 0 |

A Gaussian wing is monotonic, so it cannot reproduce the dip at 20 px followed by a
rise to the plateau; that feature is more likely co-activation or segmentation.

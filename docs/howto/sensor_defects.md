# Add hot pixels and fixed-pattern noise

Real sensors carry a static signature that does not average away and does not move
with the brain: a sparse set of **hot pixels**, a small pixel-to-pixel **gain**
variation (PRNU), **column / row offsets** from the readout electronics, and
sometimes pixels **stuck** at one value. The opt-in {py:class}`~minisim.SensorDefects`
step adds them to the `sensor` step's digitization. It is **off by default**: a spec
without the step produces exactly the same counts (and the same `cache_key`) as
before.

```python
from minisim import SensorDefects
from minisim.testing import make_recording

rec = make_recording(extra_steps=[SensorDefects()])
gt = rec.ground_truth
gt.hot_pixel_mask       # (H, W) bool
gt.hot_pixel_excess     # (H, W) dark-current excess in counts (0 elsewhere)
gt.prnu_gain            # (H, W) multiplicative gain, 1 = nominal
gt.fpn_column           # (W,)  additive offset per column, counts
gt.fpn_row              # (H,)  additive offset per row, counts
gt.stuck_pixel_mask     # (H, W) bool
```

With `build_spec`, pass it through `extra_steps=[SensorDefects(...)]`. It needs a
`sensor` step and works with {py:func}`~minisim.simulate`,
{py:func}`~minisim.simulate_video` and {py:func}`~minisim.write_miniscope_daq`
(the maps land in `ground_truth/arrays.npz` too).

## How it enters the sensor model

In physical order, per frame: PRNU multiplies the expected photons; hot-pixel dark
current adds an equivalent photon offset (so it carries shot noise); the
column/row offsets are added in ADU after the noise and before quantization; stuck
pixels are overwritten last. The defect maps are drawn **once**, from a generator
derived from the spec seed (or `SensorDefects(seed=...)`) and **independent of the
simulation RNG**: adding the step never changes the cells, motion, or noise of an
otherwise identical spec, and changing one effect's parameters does not reshuffle
another's map.

## Parameters and defaults

| field | default | meaning |
| --- | --- | --- |
| `hot_pixel_fraction` / `n_hot_pixels` | `5.8e-4` / `None` | fraction of pixels, or an exact count (overrides) |
| `hot_pixel_min_counts`, `hot_pixel_tail_shape`, `hot_pixel_max_counts` | `10`, `2.67`, `150` | excess is Pareto: `P(excess > x) ~ x^-shape` above the min, clipped at the max |
| `stuck_pixel_fraction`, `stuck_pixel_value` | `0`, `None` (full scale) | stuck pixels (off by default) |
| `prnu_sd` | `0.01` | std of the multiplicative per-pixel gain |
| `column_fpn_sd`, `row_fpn_sd` | `0.5`, `0.0` | std of the per-column / per-row offset, counts |

The hot-pixel defaults are calibrated to a real V4 recording (600x600): about 209
pixels more than 10 grey levels and 18 more than 25 above a 5x5 local median of the
mean image, maximum about +126. On a simulated 608x608 V4 frame the defaults give
about 190-210 and 20 (maximum 90-130, depending on the draw). The PRNU and
column/row magnitudes are modest placeholders, not measured values; set them to
match your sensor.

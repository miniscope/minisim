# Add realistic brain motion

The default `"physical"` motion model is a tidy oscillator. Real Miniscope recordings
move differently: a broadband, 1/f-like spectrum, a dominant axis, short bursts of
larger movement, a slow drift, and a **non-rigid** component (patches of the FOV move
slightly differently from the whole). The `"miniscope"` model and the opt-in
`nonrigid` deformation reproduce those statistics. Both are **off by default** for
existing specs: a spec that does not use them serializes, hashes (`cache_key`) and
renders exactly as before.

```python
from minisim import presets, simulate

motion = presets.realistic_motion()          # model="miniscope", nonrigid=True
spec = presets.build_spec(
    presets.miniscope_v4(), presets.ca1(), duration_s=120, extra_steps=[motion]
)
rec = simulate(spec)
gt = rec.ground_truth
gt.shifts             # (frame, 2) rigid (dy, dx), px
gt.nonrigid_field     # (frame, gy, gx, 2) non-rigid control-node displacement, px
gt.nonrigid_nodes_y   # (gy,) node rows in FOV pixels (x: nonrigid_nodes_x)
```

`BrainMotion(model="miniscope")` gives rigid motion only;
`BrainMotion(model="walk" | "physical", nonrigid=True)` adds the deformation to any
rigid model. Override any field, e.g. `presets.realistic_motion(motion_rms_um=1.0,
burst_rate_hz=0.2)`.

## The rigid model

Three additive parts in a principal frame whose major axis points `motion_axis_deg`
from +y toward +x (the other axis has `motion_axis_ratio` times the RMS):

- **Bursts**: Poisson onsets (`burst_rate_hz`), 1-3 s long
  (`burst_min/max_duration_s`), peaking at 2-4 px (`burst_min/max_peak_um`), fast
  rise and slower relaxation, random sign, mostly along the major axis.
- **Broadband noise** whose per-band power is the target spectrum *minus what the
  bursts already contribute*, so the total follows the measured band shares
  (below 0.1 / 0.5 / 1 / 2 / 5 Hz: 39 / 63 / 76 / 86 / 96 %) with bursts included.
- **Drift**: a linear ramp, `drift_um_per_min` (about 1 px per 20 min).

Frame 0 is the reference (zero shift). The `*_um` defaults are the measured pixel
values converted at the V4 pixel size (4.8 µm / 2.9 = 1.655 µm); for another scope
the physical motion stays the same in µm. The measured V4 CA1 numbers it targets: std
0.9 px (y) and 1.1 px (x), frame-to-frame step std about 0.5 px.

## The non-rigid model

A displacement field on a coarse `nonrigid_grid` (default 4x4) of control nodes,
natural-cubic-spline interpolated to full resolution and applied with `cv2.remap`
(bilinear, zero border, like the rigid path). Per frame its RMS is
`sqrt(quiet² + (coupling·m)²)`, where `m` is the distance of the rigid trajectory
from its mean: about 0.4 px in quiet frames and about 1 px during large movements.
It is temporally correlated (`nonrigid_tau_s`), larger toward the edges, with zero
spatial mean (so `shifts` stays the global motion), and clipped to
`nonrigid_max_um` (the tissue margin grows by that much automatically).

The field is the displacement **at the observed pixel**: content at observed pixel
`p` came from `p - d(p)`, the same sign convention as `shifts` (positive moves content
down/right). A motion-correction estimate is its negation.

## Ground truth and scoring

```python
d = gt.displacement(ys, xs)                 # (frame, point, 2): rigid + non-rigid
d = gt.displacement(ys, xs, frames=slice(0, 100))
```

`displacement` evaluates the full motion at any pixels, e.g. patch centers. Score a
non-rigid correction's per-patch shifts with
{py:func}`~minisim.patch_shift_rmse`:

```python
from minisim import patch_shift_rmse
patch_shift_rmse(est_shifts, gt, centers_yx, correction=True, align=True)
# est_shifts: (frame, patch, 2); centers_yx: (patch, 2)
```

`shifts` and the field are also in `ground_truth/arrays.npz` and the zarr ground truth
written by {py:func}`~minisim.write_miniscope_daq` (the streaming writer produces the
same field as `simulate`), and the nodes are `linspace(0, size - 1, g)` per axis.

## Command line

```bash
minisim generate --preset miniscope_v4 --region ca1 --duration 120 --seed 0 \
    --motion miniscope --nonrigid --out ./sim_v4_realmotion
```

`--nonrigid` works with any `--motion` model except `none`; for `miniscope`,
`--motion-um` sets the RMS along the dominant axis.

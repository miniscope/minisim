# Benchmark a pipeline against ground truth

The point of a forward simulation is that you know the answer. This recipe runs
your analysis pipeline on a simulated movie and scores what it recovered against
the ground truth, using the {doc}`recovery metrics <../reference/metrics>`.

## 1. Simulate, then run your pipeline

```python
from minisim import simulate

rec = simulate(spec)
movie = rec.observed          # (frame, height, width) sensor counts
gt = rec.ground_truth

# Your analysis pipeline (minian, CaImAn, suite2p, ...) returns:
#   A_est: (n_units, height, width) spatial footprints
#   C_est: (n_units, frame)         calcium traces
#   S_est: (n_units, frame)         deconvolved spikes
A_est, C_est, S_est = run_my_pipeline(movie)
```

## 2. Match estimated cells to true cells

Recovery is only meaningful once you know which estimated cell corresponds to
which true cell. {py:func}`~minisim.hungarian_match` solves the optimal
one-to-one assignment by spatial overlap (IoU). Match against `A_observed` (the
optically degraded footprint, the recoverable target through the optics), not
`A_planted` (the optics-free ideal).

```python
from minisim import hungarian_match

match = hungarian_match(A_est, gt.A_observed)

match.recall()       # fraction of true cells recovered (IoU >= 0.5)
match.precision()    # fraction of estimated cells that are real
match.mean_iou       # mean spatial overlap over matched pairs
```

Recall should be scored against the *detectable* cells, not every planted cell:
a cell too deep or too dim to appear in the movie is not a fair miss. Use
{py:meth}`~minisim.GroundTruth.detectable_subset` for the honest denominator:

```python
det = gt.detectable_subset()
match = hungarian_match(A_est, det.A_observed)
print(f"recall over detectable cells: {match.recall():.2f}")
```

```{note}
Detectability is decided by a peak-SNR cut ({py:data}`~minisim.DETECT_SNR_THRESHOLD`,
currently 3.0). That threshold is provisional: it has not yet been calibrated
against the recovery behavior of a real pipeline, so it sets the recall
denominator but should be read as a sensible default rather than a settled value.
```

## 3. Score the recovered traces and spikes

`match.pairing` is the list of `(est_idx, true_idx)` pairs; feed it straight to
the temporal metrics.

```python
import numpy as np
from minisim import trace_pearson, spike_precision_recall

r = trace_pearson(C_est, gt.C, match.pairing)   # one Pearson r per matched pair
print(f"median trace correlation: {float(np.nanmedian(r)):.2f}")

spikes = spike_precision_recall(S_est, gt.S, match.pairing, tol_frames=2)
print(f"spike precision={spikes.precision:.2f}, recall={spikes.recall:.2f}")
```

## 4. Score motion recovery (optional)

If your pipeline estimates a rigid-motion trajectory and the spec has a
{py:class}`~minisim.BrainMotion` step, compare against `gt.shifts` with
{py:func}`~minisim.shift_rmse`:

```python
from minisim import shift_rmse

rmse_px = shift_rmse(shifts_est, gt.shifts)   # per-frame (dy, dx), in pixels
```

## Scaling up

To trace a metric across a physical axis (recall vs depth, vs NA, vs density),
drive this same scoring from a {doc}`parameter sweep <sweep>` and collect the
results into a DataFrame.

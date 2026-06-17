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
#   S_est: (n_units, frame)         deconvolved activity (not a spike train)
A_est, C_est, S_est = run_my_pipeline(movie)
```

## 2. Match estimated cells to true cells

Recovery is only meaningful once you know which estimated cell corresponds to
which true cell. {py:func}`~minisim.hungarian_match` solves the optimal
one-to-one assignment by spatial overlap. Match against `A_observed` (the
optically degraded footprint, the recoverable target through the optics), not
`A_planted` (the optics-free ideal).

```python
from minisim import hungarian_match

match = hungarian_match(A_est, gt.A_observed)

match.recall()        # fraction of true cells recovered (similarity >= 0.5)
match.precision()     # fraction of estimated cells that are real
match.mean_similarity # mean footprint overlap over matched pairs
```

By default the overlap is binary IoU (energy-mask Jaccard). Pass
`metric="cosine"` or `metric="weighted_jaccard"` to compare the *intensity
profile* instead, so a footprint's pixel weights, not just which pixels are lit,
drive the match. If the estimate has been motion-corrected, its footprints can
sit a few pixels off minisim's reference frame; pass `shift="auto"` to find the
global translation that maximizes overlap (or a known `(dy, dx)`) so a uniform
offset is not scored as a miss:

```python
match = hungarian_match(A_est, gt.A_observed, metric="cosine", shift="auto")
match.shift          # the (dy, dx) applied to A_est, in pixels
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

## 3. Score the recovered traces and activity

`match.pairing` is the list of `(est_idx, true_idx)` pairs; feed it straight to
the temporal metrics.

```python
import numpy as np
from minisim import trace_pearson, activity_similarity

r = trace_pearson(C_est, gt.C, match.pairing)   # one Pearson r per matched pair
print(f"median trace correlation: {float(np.nanmedian(r)):.2f}")

# The deconvolved S is not a spike train: it is a non-negative activity rate,
# scaled by an unknown factor. Score it without binarizing and up to that scale.
act = activity_similarity(S_est, gt.S, match.pairing)
print(f"median activity correlation: {float(np.nanmedian(act.correlation)):.2f}")
print(f"median variance explained:   {float(np.nanmedian(act.variance_explained)):.2f}")
```

## 4. Score motion recovery (optional)

If your pipeline estimates a rigid-motion trajectory and the spec has a
{py:class}`~minisim.BrainMotion` step, compare against `gt.shifts` with
{py:func}`~minisim.shift_rmse`:

```python
from minisim import shift_rmse

# correction=True negates the estimate (a correction undoes the applied motion);
# align=True removes a constant origin offset, since each pipeline registers to its
# own template and the absolute zero frame is arbitrary.
rmse_px = shift_rmse(shifts_est, gt.shifts, correction=True, align=True)
```

That same constant offset between the pipeline's template and minisim's reference
is what shifts the recovered footprints, so it can be read straight off the two
trajectories with {py:func}`~minisim.global_shift_from_trajectories` and handed to
`hungarian_match(..., shift=...)` to align the footprints exactly.

## Scaling up

To trace a metric across a physical axis (recall vs depth, vs NA, vs density),
drive this same scoring from a {doc}`parameter sweep <sweep>` and collect the
results into a DataFrame.

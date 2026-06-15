# Use minisim in your test suite

If you maintain a calcium-imaging analysis pipeline (minian, CaImAn, suite2p,
...), minisim can supply small, reproducible recordings whose answer you already
know, so a test can assert that your pipeline recovers them. The
{py:mod}`minisim.testing` module gives you a one-call fixture and a one-call
scorecard built for exactly this.

## A recording in one call

{py:func}`~minisim.testing.make_recording` returns a small, fast, deterministic
{py:class}`~minisim.Recording`: the same arguments (and `seed`) always produce
the same movie and the same ground truth.

```python
from minisim.testing import make_recording

rec = make_recording(n_cells=6, n_px=128, duration_s=2.0, seed=0)
movie = rec.observed          # (frame, height, width) sensor counts
truth = rec.ground_truth      # exact cells, traces, spikes
```

The defaults are tuned for CI (a 128 px field at 1 µm/px, six cells at 50 µm
depth, two seconds at 20 fps). Shrink `n_px` / `duration_s` for an even faster
fixture, raise `n_cells` for a denser one, pass `motion=True` to exercise motion
correction, or hand in your own `activity=` / `sensor=` / `extra_steps=`.

## Scoring your pipeline in one call

Wrap your pipeline's output in an {py:class}`~minisim.testing.Estimate` and pass
it to {py:func}`~minisim.testing.score`. It applies the conventions the
{doc}`benchmarking guide <benchmark>` spells out (match against `A_observed`,
score recall over the detectable cells, nan-safe trace median, treat the motion
estimate as a correction) and returns a {py:class}`~minisim.testing.Report`.

```python
from minisim.testing import Estimate, score

A_est, C_est, S_est = run_my_pipeline(rec.observed)
report = score(Estimate(A=A_est, C=C_est, S=S_est), rec.ground_truth)

assert report.recall > 0.8, report.summary()
```

`A` is the only required field of `Estimate`; leave `C` / `S` / `shifts` out and
the matching scores come back as `nan` / `None`. The `Report` carries `recall`,
`precision`, `mean_iou`, `trace_corr` (median Pearson r), `spike_precision`,
`spike_recall`, and `shift_rmse`. Arrays may be `numpy` or `xarray` (minian's CNMF
returns `xr.DataArray`); both are accepted.

When you need more than the common case, the underlying primitives
({py:func}`~minisim.hungarian_match`, {py:func}`~minisim.trace_pearson`,
{py:func}`~minisim.spike_precision_recall`, {py:func}`~minisim.shift_rmse`) stay
fully available; `score` is just the 90%-path on top of them.

## A pytest fixture

Drop a fixture in your `conftest.py`. Build the recording once per test (or per
session, since a `Recording` is immutable and safe to share):

```python
# conftest.py
import pytest
from minisim.testing import make_recording

@pytest.fixture(scope="session")
def sim_recording():
    return make_recording(n_cells=8, duration_s=3.0, seed=0)
```

```python
# test_recovery.py
from minisim.testing import Estimate, score

def test_pipeline_recovers_cells(sim_recording):
    A, C, S = run_my_pipeline(sim_recording.observed)
    report = score(Estimate(A=A, C=C, S=S), sim_recording.ground_truth)
    assert report.recall > 0.8, report.summary()
```

Use plain `make_recording` / {py:func}`~minisim.simulate` in tests. The on-disk
{py:func}`~minisim.simulate_cached` is meant for parameter sweeps, not CI: in a
fresh CI environment its cache is cold (no speedup, just disk writes).

## Add minisim as a test-only dependency

minisim never imports an analysis pipeline, so the dependency is strictly
one-directional and adding minisim to *your* test extra cannot create an import
cycle. In your `pyproject.toml`:

```toml
[project.optional-dependencies]
test = ["minisim", "pytest"]
```

minisim is then present when your tests run, but is not a runtime dependency of
your package.

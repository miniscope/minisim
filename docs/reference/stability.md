# Reproducibility & stability contract

minisim's second job is to be a *test fixture* for analysis pipelines (minian,
CaImAn, suite2p). That job has an unusual failure mode: the thing a consumer pins
is not only the names they import, but **the number on the right of**
`assert report.recall > 0.8`. A change that leaves every symbol intact but quietly
moves that number - a new default, a livelier activity model, a recalibrated
detection threshold, a different RNG order - would flip downstream test suites red
for a reason invisible in their own diff.

This page states what minisim treats as **frozen** (changing it is a breaking
change, announced in the changelog) versus **incidental** (free to change), so you
know what you are allowed to lean on.

## The core guarantee

> For a fixed minisim version, the same inputs and the same `seed` produce a
> **byte-identical** `observed` movie and an identical `ground_truth`.

This holds for {py:func}`~minisim.simulate`, {py:func}`~minisim.testing.make_recording`,
and the streamed {py:func}`~minisim.simulate_video` (which is bit-for-bit equal to
`simulate`). It is enforced in minisim's own CI by a golden-master test
(`test_reproducibility.py`) that pins the hash of `make_recording(seed=0)`; that
test failing is the signal that the fixture contract moved.

## Frozen across patch releases (changing these is breaking)

These are the inputs to a consumer's assertion, so they are versioned. A change is
a minor-version bump with a changelog entry.

- **`make_recording` defaults** - cell count, layout, depth, the default activity
  and exposure, the resulting cell positions and pixel values.
- **`score` semantics** - footprints are matched against `A_observed` (the
  recoverable target), not the optics-free `A_planted`; trace correlation is
  reduced with a nan-safe median; `restrict_to_detectable` defaults to `True`
  (recall is over the detectable subset); a motion estimate is treated as a
  *correction* (negated) before comparison.
- **The `Report` / `Estimate` shape** - existing fields and their meaning. The
  types are frozen dataclasses precisely so a field can be *added* without breaking
  callers; removing or renaming one is breaking.
- **`DETECT_SNR_THRESHOLD`** - the SNR a cell must clear to count as `detectable`.
  This sets the `restrict_to_detectable` recall denominator and the "auto"
  focus/exposure objective. **It is provisional and not yet calibrated against a
  real pipeline, so it may change before 1.0.** When it changes, recall
  denominators move under downstream tests, so it is treated as a breaking change.
  If you need a denominator that cannot drift, pass `restrict_to_detectable=False`
  and read `n_requested` off the `Report`.

## Reading the recall denominator

Because `recall` is over the detectable cells by default, the `Report` always
reports what the denominator was drawn from:

- `n_requested` - cells planted (the full population), invariant to the filter.
- `n_detectable` - cells that clear the detection floor.
- `n_true` - the denominator `recall` actually used (= `n_detectable` under the
  default filter, `n_requested` without it).

So `recall = 1.0` with `n_detectable = 4 < n_requested = 6` reads as "recovered
every detectable cell, but two planted cells were too dim", not "recovered
everything". `report.summary()` prints all three.

## On-disk format

A saved {py:class}`~minisim.Recording` (`save()` / `load()`) is a zarr directory
stamped with a `format_version`. The compatibility boundary is that version: a
recording is loadable by any minisim whose reader understands its `format_version`.
The sibling `spec.json` is additionally checked against the spec hash to catch a
stale or hand-edited cache. Persisted fixtures are a supported pattern only within
a stable `format_version`; for CI, prefer regenerating in-memory with
`make_recording` (a cold cache gives no speedup anyway).

## Not frozen (free to change)

- Exact wall-clock performance and memory use.
- Internal module layout, private helpers (leading `_`), and anything not exported
  from the top level or `minisim.testing`.
- The *incidental* numeric value of any quantity a consumer does not, and should
  not, assert on (e.g. the exact bytes of an intermediate snapshot).

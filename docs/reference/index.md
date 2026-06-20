# API reference

Generated from the package. Most things documented here are importable directly
from the top level (`from minisim import ...`); the testing helpers live in the
`minisim.testing` submodule.

```{toctree}
:maxdepth: 2

spec
presets
simulate
recording
scene
metrics
testing
caching
stability
```

## At a glance

```{list-table}
:header-rows: 1
:widths: 30 70

* - Symbol
  - Role
* - {py:class}`~minisim.Spec`
  - The typed, validated recording specification (the input you build).
* - {py:func}`~minisim.build_spec`
  - Compose a {py:class}`~minisim.Scope` × {py:class}`~minisim.Region` preset into a `Spec`.
* - {py:func}`~minisim.simulate`
  - Run the forward pipeline; returns a {py:class}`~minisim.Recording`.
* - {py:func}`~minisim.simulate_cached`
  - As `simulate`, memoized to disk by `spec.cache_key()`.
* - {py:func}`~minisim.simulate_video`
  - Simulate straight to a grayscale video, streamed to disk.
* - {py:func}`~minisim.sweep`
  - Cartesian-product generator over spec overrides.
* - {py:class}`~minisim.Recording`
  - The result: `observed` movie, `ground_truth`, per-stage `snapshots`.
* - {py:class}`~minisim.GroundTruth`
  - The exact generators (footprints, traces, spikes, optical fields).
* - {mod}`metrics <minisim.metrics>`
  - Score recovered cells, traces, spikes, and motion against truth.
* - {py:func}`~minisim.testing.make_recording` / {py:func}`~minisim.testing.score`
  - One-call CI fixture and recovery scorecard (in `minisim.testing`).
```

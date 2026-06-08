# API reference

Generated from the package. Everything documented here is importable directly
from the top level (`from minisim import ...`).

```{toctree}
:maxdepth: 2

spec
simulate
recording
scene
metrics
caching
```

## At a glance

```{list-table}
:header-rows: 1
:widths: 30 70

* - Symbol
  - Role
* - {py:class}`~minisim.Spec`
  - The typed, validated recording specification (the input you build).
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
```

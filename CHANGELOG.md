<!-- This changelog is managed by commitizen (`cz bump`); entries are generated
from conventional-commit messages. The first release is v0.1.0. -->

## v1.1.0 (2026-11-12)

### BREAKING CHANGE (minor bump under the stability contract)

- `detectable` is now a **matched-filter SNR over space and time**
  (`matched_filter_snr`): the ideal detector's SNR for each cell's realized footprint
  (defocus, scatter, halo, vessels, illumination, vignette) and all its transients,
  `SNR² = Σ_i g_i² a_i² / σ_i² · Σ_t Δ_t²`, against per-pixel shot noise on all the
  light at that pixel (other cells, neuropil, leakage) plus read noise. It replaces
  the single-pixel, single-transient peak test, which undercounted what pipelines
  recover once PSFs are sharp (defocus drops the peak as `σ₀²/σ²` but the pooled SNR
  only as `σ₀/σ`) and, under field curvature, left the detectable cells in a ring.
  `DETECT_SNR_THRESHOLD` changes from 3.0 (peak SNR, uncalibrated) to **110**
  (matched-filter SNR), calibrated as the SNR at which cala recovers half the cells
  on CaLab's simulated CA1 recordings with 6 px cells at the real recording's edge
  blur (fit 111, 95 % CI 104-117, 1527 cells, two seeds), and checked without
  retuning on 10 px cells (recovery crosses 50 % near 103; cala's matches to cells
  not flagged detectable fall from 115 to 0 on one 480 s recording, and the
  curved-field ring disappears). The "auto" focus yield objective uses the same
  criterion (with a closed-form Gaussian footprint), so an `"auto"` focal plane can
  move slightly.
- **Migration: recall denominators move.** `n_detectable`, `detectable_subset()` and
  `score(..., restrict_to_detectable=True)` recall change for most recordings, mostly
  upward for sharp or defocused cells and for active cells, downward for cells that
  barely fire in a short recording (the SNR grows with the number of transients).
  Re-pin any `assert report.recall > X` against 1.1.0, or pass
  `restrict_to_detectable=False` for a denominator that cannot drift. Saved recordings
  keep the flag they were written with; recompute it with
  `matched_filter_snr(gt, spec) >= DETECT_SNR_THRESHOLD`.

### Feat

- `matched_filter_snr(ground_truth, spec)` and its parts
  (`recording.detection_noise_fields`, `recording.spatial_snr2`,
  `recording.transient_energy`), computable from any saved ground truth;
  `GroundTruth.observed_footprints()` (sparse, FOV-cropped observed footprints).
- `peak_snr`, the old single-pixel peak SNR, kept as a diagnostic; `detection_snr`
  remains as a deprecated alias of it.
- Teaching notebooks: the illumination and sensor panels plot the matched-filter SNR
  (projected to the full recording length); notebook 3 re-executed; notebook 2's text
  follows the new counts.
- heavy-tailed scatter PSF: opt-in `Tissue.scatter_halo_fraction` (default 0) and
  `scatter_halo_blur_per_um` add a broad diffuse wing, `PSF = (1-f)·G(σ_core) +
  f·G(σ_halo)`, redistributed from the core (footprint integral unchanged; peak
  brightness and `detectable` follow). New `Acquisition.halo_optics`,
  `degrade_footprint(..., halo_sigma_px, halo_fraction)` and per-cell
  `GroundTruth.observed_halo_sigma_px` / `observed_halo_fraction`, so `A_observed`
  reproduces the full footprint; streaming matches `simulate()`. Defaults are
  bit-identical and spec hashes unchanged; rendering costs ~1.1x. See the
  "Add a heavy-tailed scatter halo" how-to.

- write recordings in Miniscope DAQ folder layout: `minisim.write_miniscope_daq`
  (streamed `N.avi` + `metaData.json` + `timeStamps.csv`, ground truth in a sibling
  `ground_truth/`), a dependency-free streaming `GREY` AVI writer
  (`minisim.avi.AviWriter`), `save_ground_truth` / `load_ground_truth`, and a
  `minisim generate` command line.

- realistic brain motion: `BrainMotion(model="miniscope")` (broadband 1/f-like,
  anisotropic rigid motion with Poisson bursts and slow drift, calibrated to a real
  V4 CA1 recording) and opt-in `nonrigid=True` (a smooth, rigid-coupled deformation
  field on a coarse control grid, applied with `cv2.remap`) for any model;
  `presets.realistic_motion()`, `GroundTruth.nonrigid_field` / `nonrigid_nodes_*` /
  `displacement()`, `patch_shift_rmse`, and `minisim generate --motion miniscope
  --nonrigid`. Existing specs serialize, hash and render unchanged.
- hot pixels and fixed-pattern noise: a new opt-in `SensorDefects` step (hot pixels with
  a heavy-tailed dark-current excess, stuck pixels, PRNU, column/row offsets), seeded
  independently of the simulation RNG, applied in the sensor digitization (also when
  streaming), with the maps saved in `GroundTruth`. Absent by default: existing
  output and spec hashes are unchanged.

### Fix

- `simulate_video(progress=False)` no longer prints progress when `tqdm` is absent.

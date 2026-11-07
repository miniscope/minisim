<!-- This changelog is managed by commitizen (`cz bump`); entries are generated
from conventional-commit messages. The first release is v0.1.0. -->

## Unreleased

### Feat

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

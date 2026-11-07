<!-- This changelog is managed by commitizen (`cz bump`); entries are generated
from conventional-commit messages. The first release is v0.1.0. -->

## Unreleased

### Feat

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

### Fix

- `simulate_video(progress=False)` no longer prints progress when `tqdm` is absent.

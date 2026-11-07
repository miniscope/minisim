<!-- This changelog is managed by commitizen (`cz bump`); entries are generated
from conventional-commit messages. The first release is v0.1.0. -->

## Unreleased

### Feat

- write recordings in Miniscope DAQ folder layout: `minisim.write_miniscope_daq`
  (streamed `N.avi` + `metaData.json` + `timeStamps.csv`, ground truth in a sibling
  `ground_truth/`), a dependency-free streaming `GREY` AVI writer
  (`minisim.avi.AviWriter`), `save_ground_truth` / `load_ground_truth`, and a
  `minisim generate` command line.

### Fix

- `simulate_video(progress=False)` no longer prints progress when `tqdm` is absent.

# Command line

Installing minisim provides a `minisim` command (the `minisim.cli:main` entry point;
it uses only argparse and the standard library). It currently has one subcommand,
`generate`, which simulates a recording and writes it to disk. A second entry point,
`minisim-notebooks`, copies the bundled teaching notebooks (see the
{doc}`tutorials <../tutorials/index>`).

```bash
minisim generate --preset miniscope_v4 --region ca1 --duration 60 --seed 0 \
    --motion miniscope --nonrigid --out ./sim_v4_ca1
minisim generate --spec spec.json --out ./sim_from_spec
```

## `minisim generate`

What to simulate (or `--spec` to load a full {py:class}`~minisim.Spec` JSON, which
overrides everything in this group):

```{list-table}
:header-rows: 1
:widths: 28 72

* - Option
  - Meaning
* - `--spec PATH`
  - Spec JSON file; overrides the options below.
* - `--preset {miniscope_v4,generic_1p}`
  - Scope preset (default `miniscope_v4`); see {doc}`presets`.
* - `--region {ca1,cortex_l23}`
  - Tissue region preset (default `ca1`).
* - `--duration SECONDS`, `--fps HZ`, `--seed N`
  - Recording length (default 60), frame rate (default 20) and master RNG seed
    (default 0).
* - `--motion {none,physical,walk,miniscope}`
  - Brain-motion model (default `none`). `miniscope` is the realistic broadband model.
* - `--motion-um UM`
  - Motion amplitude: extreme excursion for `physical`, RMS along the dominant axis
    for `miniscope`, max shift for `walk`.
* - `--nonrigid`
  - Add the smooth non-rigid deformation field on top of the rigid motion (needs
    `--motion` other than `none`). See the {doc}`realistic motion guide
    <../howto/realistic_motion>`.
```

Output:

```{list-table}
:header-rows: 1
:widths: 28 72

* - Option
  - Meaning
* - `--out DIR`
  - Output directory (required).
* - `--layout {daq,zarr}`
  - `daq` (default) streams a Miniscope-DAQ folder with the ground truth in a sibling
    `ground_truth/`; `zarr` writes the single-directory {py:meth}`Recording.save
    <minisim.Recording.save>` format (simulated in memory).
* - `--frames-per-file N`, `--device-name NAME`, `--jitter-ms MS`, `--no-ground-truth`
  - `daq` only: frames per `N.avi` (default 1000), device folder name (default
    `Miniscope`), timestamp jitter, and skipping `ground_truth/`.
* - `--overwrite`, `--quiet`
  - Replace existing output; suppress progress output.
```

The `daq` layout is {py:func}`~minisim.write_miniscope_daq`; see the
{doc}`DAQ folder guide <../howto/daq_folder>` for what is written. Run
`minisim generate --help` for the full option list.

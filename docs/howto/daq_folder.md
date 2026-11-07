# Write a Miniscope DAQ folder

{py:func}`~minisim.write_miniscope_daq` writes a simulated recording in the same
folder layout the [Miniscope DAQ software](https://github.com/Aharoni-Lab/Miniscope-DAQ-QT-Software)
produces, so anything that reads real DAQ output (a streaming extraction pipeline,
minian's loaders, your own reader) can be pointed at it unchanged, while the exact
ground truth stays available next to it. It uses only numpy and the standard
library for the AVI encoding, so no OpenCV codec or ffmpeg is involved in the
output.

```python
from minisim import build_spec, write_miniscope_daq
from minisim.presets import ca1, miniscope_v4

spec = build_spec(miniscope_v4(), ca1(), duration_s=60.0, seed=0)
folder = write_miniscope_daq(spec, "sim_v4_ca1", frames_per_file=1000)
print(folder.device_dir, folder.n_frames)
```

## What is written

```
sim_v4_ca1/
    metaData.json            experiment level (start time, device list, ...)
    Miniscope/               the device folder (device_name=...)
        metaData.json        ROI, compression "GREY", frameRate "20FPS", framesPerFile, gain, ...
        timeStamps.csv       Frame Number,Time Stamp (ms),Buffer Index
        0.avi  1.avi  ...    framesPerFile frames each (the last holds the remainder)
    ground_truth/            NOT part of the DAQ layout: the simulator's truth
        spec.json            the full Spec that produced the recording
        ground_truth.zarr/   the GroundTruth; reload with minisim.load_ground_truth
        arrays.npz           dense arrays for readers without minisim
```

The AVIs are plain RIFF (no OpenDML), fourcc `GREY` in both the stream header and
the `BITMAPINFOHEADER`, 8 bits per pixel, one `00dc` chunk per frame, with an
`idx1` index at the end (offsets relative to the `movi` fourcc). Rows are stored top
to bottom, so a decoded frame equals the array that was written. Unlike the real
DAQ files, the header carries the **true** frame rate (the DAQ stamps 60 fps there
regardless of the sensor rate). The frame counter in `timeStamps.csv` runs
continuously across files.

Counts are mapped to 8-bit exactly as in {doc}`video`: for the 8-bit V4 sensor the
counts are written verbatim, so the files hold the same numbers as
`simulate(spec).observed`.

## Ground truth

`arrays.npz` holds, for the `n` ground-truth cells (order matches `C` and `S`):

| key | shape | meaning |
| --- | --- | --- |
| `A_observed` | `(n, H, W)` float32 | optically degraded footprints: the recoverable target |
| `A_planted` | `(n, H, W)` float32 | the sharp, pre-optics footprints |
| `C`, `S` | `(n, frame)` | calcium traces and the spike/activity train that drove them |
| `shifts` | `(frame, 2)` | `(dy, dx)` brain motion in pixels (positive moves tissue down/right); only with a `BrainMotion` step |
| `detectable`, `in_focus`, `centers_um`, `amplitude_per_cell` | `(n, ...)` | per-cell attributes |
| `illumination`, `vignette`, `leakage`, `vasculature_mask`, `neuropil_*`, `bleaching` | | per-effect fields, present when the step is |

`fps` and `pixel_size_um` are stored as scalars. Load the same truth as typed
objects with:

```python
from minisim import load_ground_truth
gt, spec = load_ground_truth("sim_v4_ca1/ground_truth/ground_truth.zarr")
```

## Timestamps and metadata

`timeStamps.csv` holds nominal frame times plus optional Gaussian jitter:

```python
write_miniscope_daq(spec, out, jitter_ms=2.0, start_offset_ms=-21)
```

The jitter is drawn from its own seeded generator (`timestamp_seed`, default the
spec seed), so it never changes the simulated movie. `Buffer Index` is always 0
(the software never fell behind) and there are no dropped frames. Pass
`device_metadata={...}` / `experiment_metadata={...}` to override or add
`metaData.json` keys (e.g. `gain`, `led0`, `deviceType`).

## Streaming and memory

Given a `Spec`, the movie is rendered in frame chunks and each frame is written
straight into the current AVI, so peak memory does not grow with duration (tune
`chunk_frames`). A `Recording` you already hold is written from memory.

The **ground truth is not streamed**: while the movie renders, the traces `C` and
`S` (`(unit, frame)` float64) and the neuropil components are held in memory, and
at the end `arrays.npz` materializes the dense footprints. As a rule of thumb that
is `~16 B * units * frames` for the traces (a thousand cells over an hour at 20 fps
is about 1 GB) plus `8 B * units * H * W` for the footprints (a hundred cells on a
608 px sensor is about 150 MB). For very large populations pass `write_npz=False`
and read the zarr, whose footprints are sparse and regenerate on demand. A single
AVI is limited to 4 GiB (no OpenDML); `AviWriter` raises rather than write a corrupt
file, so keep `frames_per_file * H * W` below that (the DAQ's default of 1000
frames at 600x600 is 360 MB).

## From the command line

```bash
minisim generate --preset miniscope_v4 --region ca1 --duration 60 --seed 0 \
    --motion physical --motion-um 3 --out sim_v4_ca1
minisim generate --spec my_spec.json --out sim_from_spec --frames-per-file 500
minisim generate --preset generic_1p --duration 10 --layout zarr --out rec.zarr
```

`--layout daq` (default) is the folder described above, streamed; `--layout zarr`
simulates in memory and writes {py:meth}`Recording.save <minisim.Recording.save>`.
`--motion-um` is the extreme excursion for the `physical` model (one V4 pixel is
about 1.6 um). Use `--jitter-ms`, `--device-name`, `--no-ground-truth`,
`--overwrite` and `--quiet` as needed; `minisim generate --help` lists everything.

# Export a recording to video

{py:func}`~minisim.simulate_video` simulates a spec straight to a grayscale
video on disk, rendering and digitizing in frame chunks so the whole movie is
never held in memory (unlike `simulate(spec).observed`). The counts it writes
match `simulate(spec).observed` exactly.

Requires the `notebook` extra (for `mediapy`):

```bash
pip install "minisim[notebook]"
```

## Basic export

```python
from minisim import simulate_video

path = simulate_video(spec, "recording.avi")   # returns the written path
```

By default it writes uncompressed 8-bit grayscale (`codec="rawvideo"`, fourcc
`Y800`), so the file carries the exact sensor counts with no compression
artifacts and opens directly in ImageJ/Fiji. That file is large
(~`n_frames * H * W` bytes).

## Controlling the brightness mapping

`vmax` sets the count mapped to white (`vmin` maps to black). For a spec with a
{py:class}`~minisim.Sensor` step it defaults to the sensor's full ADC range
(`2**bit_depth - 1`), so the file honestly shows the true ADC utilization (a
dim, faithful frame). A sensorless, continuous-intensity spec has no natural
scale, so you **must** pass `vmax`:

```python
simulate_video(sensorless_spec, "out.avi", vmax=4.0)
```

## Smaller files

For a compact file pass a lossy codec. `"mjpeg"` is Fiji-readable; `"png"` /
`"ffv1"` are smaller and lossless but ffmpeg tags them in a way Fiji's built-in
AVI reader rejects.

```python
simulate_video(spec, "small.avi", codec="mjpeg", fps=30)
```

## Useful arguments

- `chunk_frames` — frames rendered per write; tune for the memory/throughput
  trade-off.
- `fps` — playback frame rate of the file (defaults to the spec's acquisition fps).
- `progress` — set `False` to silence the progress bar.

See {py:func}`~minisim.simulate_video` in the reference for the full signature.

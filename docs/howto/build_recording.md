# Build a recording interactively

The **`build_recording` studio** is an interactive notebook for *making usable
synthetic recordings*. Where the {doc}`anatomy tutorial <../tutorials/index>`
isolates one physical effect at a time to *explain* it, the studio exposes every
knob at once so you can tune a recording to taste and write it to disk, ground
truth included.

:::{note}
This is an **interactive** notebook (`ipywidgets` + a live simulation), so it is
meant to be *run*, not read statically. It needs the `notebook` extra.
:::

## Get the notebook

The studio ships inside the package, alongside the teaching notebooks. Copy it out
with the bundled `minisim-notebooks` command:

```bash
pip install "minisim[notebook]"
minisim-notebooks list                       # shows 01_anatomy and build_recording
minisim-notebooks copy build_recording       # -> ./minisim-notebooks/build_recording
cd minisim-notebooks/build_recording
jupyter lab build_recording.ipynb
```

Working from a clone instead? Open it directly at
[`minisim/notebooks/studio/build_recording/build_recording.ipynb`](https://github.com/miniscope/minisim/blob/main/minisim/notebooks/studio/build_recording/build_recording.ipynb).

## The three panels

The notebook is three panels that all read and write **one shared configuration**,
so the file you generate is exactly what you previewed. Run the cells top to bottom.

1. **Anatomy & scope** — every non-activity knob (optics, image sensor, tissue
   scatter, cell placement, and the optical confounds: vasculature, illumination
   falloff, vignette, stray-light glow, brain motion). The live preview is a
   max-projection of a short full-pipeline render. To stay responsive it renders a
   centred window **auto-sized to a fixed cell budget** (~350 cells) at the true
   pixel scale rather than the whole FOV, so as you raise the density the window
   shrinks and the image's apparent crowding stays roughly constant; density instead
   shows in the readout's full-FOV count (`~N cells over full FOV`) and in the
   side-view dot count. A readout reports the field of view, pixel size, expected
   neurons over the full FOV, the resolved focus, and the preview-window detectable
   count. A side-view schematic below shows where each layer sits in depth across
   the FOV (cell band — its density visible in the dots — focal surface, and
   vasculature), with the focal surface bowing shallower at the edges when field
   curvature is on. **Preset** buttons seed the whole panel from a real
   configuration. This panel takes a moment to redraw after each slider change (it
   re-runs the optics, recomputing every cell footprint's depth-dependent blur and
   dimming, which is also why it renders a cell-budgeted window); panels 2 and 3 are
   fast.
2. **Neural activity** — the calcium model (firing gate, rates, kinetics, and the
   cell-to-cell brightness spread), previewed as the clean ground-truth traces `C`
   with spike ticks `S`. `Quiet` / `Moderate` / `Active` presets set typical levels.
3. **Generate** — set the duration, frame rate, and seed, then write the recording
   at the full field of view.

## Presets

The anatomy panel ships three starting points, applied with one click:

- **Generic 1p scope** — the library defaults.
- **Miniscope V4 — CA1** — a Miniscope V4 imaging a dense, thin hippocampal CA1
  pyramidal band, with milder vasculature.
- **Miniscope V4 — cortex L2/3** — a V4 imaging sparser neocortical layer 2/3
  cells, with thick vessels sitting on top of the band.

The V4 optics (NA, sensor format, pixel pitch, working distance) are the real
values; the activity panel adds its own `Quiet`/`Moderate`/`Active` firing presets.

## Output formats

```{list-table}
:header-rows: 1
:widths: 12 40 30

* - Format
  - Contents
  - Reload
* - **zarr** (default)
  - The complete {py:class}`~minisim.Recording`: the movie *and* the ground truth
    (footprints `A`, traces `C`, spikes `S`, positions, detectable flags, vessel
    mask) plus the {py:class}`~minisim.Spec`. This is what makes the data usable
    for scoring an analysis pipeline.
  - `minisim.Recording.load(path)`
* - **avi**
  - A viewable grayscale movie only (no ground truth), streamed via
    {py:func}`~minisim.simulate_video`. The tuned spec is written beside it as
    `<name>.spec.json`.
  - re-simulate from the saved spec
```

zarr goes through the in-memory {py:func}`~minisim.simulate`, so the whole movie
is held in RAM while it is written; the panel's size estimate flags how large the
file (and the in-RAM movie) will be. For a long recording, dial the duration down
or use avi, which streams to disk with flat memory.

## Reproducibility

A studio configuration is just a {py:class}`~minisim.Spec`. The zarr output embeds
its `spec.json`, and the avi output drops one beside the movie, so any generated
recording is reproducible and scriptable headlessly without the notebook:

```python
from minisim import Spec, simulate

spec = Spec.model_validate_json(open("recording.spec.json").read())
rec = simulate(spec)        # bit-for-bit the same recording
```

See {doc}`video` for the streaming video writer on its own, and
{doc}`../tutorials/index` for the physics behind each knob.

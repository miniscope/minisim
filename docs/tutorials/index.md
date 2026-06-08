# Tutorials

minisim is also a teaching tool. The tutorials walk the *anatomy* of miniscope
data: starting from a clean simulated signal and adding one physical effect at a
time (optics, brain motion, illumination falloff, sensor noise), so you can see
exactly what each does to the image.

## Anatomy of a recording (interactive notebook)

The flagship tutorial is an interactive Jupyter notebook that builds the forward
pipeline stage by stage, with sliders to vary the physics and see the movie
respond in real time.

:::{note}
This notebook is **interactive** (it uses `ipywidgets` and runs a live
simulation), so it is meant to be *run*, not read statically.
:::

### Get the notebook

The notebooks ship inside the package. After installing, copy them out to a
directory you own with the bundled `minisim-notebooks` command:

```bash
pip install "minisim[notebook]"
minisim-notebooks ./minisim-notebooks      # copies the bundle(s) here
cd minisim-notebooks/01_anatomy
jupyter lab 01_anatomy.ipynb
```

`minisim-notebooks` takes an optional destination (default `./minisim-notebooks`)
and `--force` to overwrite an existing copy. No data download is needed: minisim
*generates* the recording from code as the notebook runs.

Working from a clone of the repository instead? Open it directly at
[`minisim/notebooks/training/01_anatomy/01_anatomy.ipynb`](https://github.com/miniscope/minisim/blob/main/minisim/notebooks/training/01_anatomy/01_anatomy.ipynb).

The stages mirror the forward chain described in {doc}`../concepts`:

1. Place neurons and generate calcium activity (the clean signal).
2. Optics: depth-dependent blur and dimming.
3. Render to the sensor canvas.
4. Neuropil and vasculature background.
5. Photobleaching over the recording.
6. Brain motion under the lens.
7. Illumination profile and emission vignette.
8. Stray-light leakage.
9. Sensor digitization to raw counts, where the auto-focus yield is realized.

:::{admonition} Coming soon
:class: seealso

A second notebook, the **demixing capstone**, shows why naive per-ROI traces are
contaminated by neighbor bleed and neuropil, and how demixing recovers the true
signals, quantified against the ground-truth `A`/`C`.
:::

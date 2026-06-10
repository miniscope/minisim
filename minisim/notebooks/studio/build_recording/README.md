# Build a recording: tune every knob, then generate usable simulated data

A hands-on **data-generation studio** (not a lesson). Where the `01_anatomy`
teaching notebook isolates one physical effect at a time to *explain* it, this
notebook exposes every knob at once so you can *make a usable synthetic
recording*: tune the optics, tissue, and confounds against a live example frame;
tune the calcium activity against live traces; set the recording length; and
generate a complete recording on disk, ground truth included.

## What you get

Three interactive panels, each feeding one shared, carried-forward `Spec`:

1. **Anatomy & scope** - optics (NA, magnification, emission, focal depth), sensor
   (pixel pitch/count, QE, read noise, gain, bit depth, exposure), tissue scatter,
   vasculature, illumination falloff, vignette, leakage, and motion.
   Live preview: a max-projection of a short full-pipeline render.
   **Preset buttons** seed the whole panel from a real configuration
   (*Generic 1p scope*, *Miniscope V4 - CA1*, *Miniscope V4 - cortex L2/3*).
2. **Neural activity** - the calcium model (firing rates, kinetics, brightness
   spread). Live preview: stacked ground-truth traces.
3. **Generate** - duration, frame rate, seed, output path and format, then a
   single button that streams the recording to disk.

## Output formats

- **zarr** (default) keeps the *ground truth* alongside the movie - footprints
  `A`, traces `C`, spikes `S`, positions, detectable flags, vessel masks, and the
  full `Spec`. This is what makes the data usable for testing an analysis
  pipeline. Reload with `minisim.Recording.load(path)`.
- **AVI** writes a viewable grayscale movie only (no ground truth); the tuned
  `Spec` is dropped beside it as `<name>.spec.json`.

The tuned `Spec` is also exported as JSON, so any configuration is reproducible
and scriptable headlessly: `minisim.simulate(Spec.model_validate_json(...))`.

## Running

```
pip install 'minisim[notebook]'
minisim-notebooks copy build_recording
cd minisim-notebooks/build_recording && jupyter lab
```

The sliders need a **live kernel** - run the notebook, don't just read it.

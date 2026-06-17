# 03 - Score a pipeline against the truth

**Measure how well an analysis pipeline recovered the ground truth - and avoid the
traps that make naive scoring lie.**

The third training notebook turns minisim around. Notebook 1 built a recording
*forward* from its physics; this one uses the exact ground truth that came with it to
**grade a recovery**. Because the recording was generated, the answer key is known -
true footprints, calcium traces, spike counts, and motion - so recovery can be
measured rather than guessed.

It is a **static, step-by-step walkthrough** (no live widgets): each demo perturbs the
truth in one controlled way, scores it, and plots the whole effect, so it reads
correctly even in a statically-rendered copy. Two halves:

1. **Concepts and pitfalls** - each recovery metric on a deliberate perturbation:
   footprint matching, why **pixel weights** matter (not just a binary mask), why a
   **global shift** after motion correction is not a miss, trace correlation, why the
   deconvolved `S` is **not a spike train** (score it without binarizing and up to an
   unknown scale), and scoring **motion** independent of an arbitrary origin.
2. **End-to-end walkthrough** - mock an imperfect pipeline output, score it with
   `minisim.testing.score`, and read every field of the `Report`.

## Run it

No data download is needed - the recording is generated on the fly.

```bash
pip install "minisim[notebook]"   # engine + matplotlib (mediapy/ipywidgets not needed here)
jupyter notebook 03_metrics.ipynb
```

Run top to bottom. Every demo cell starts with a `# try:` knob - edit it and re-run to
explore (e.g. a different perturbation range or cell).

## Dependencies beyond core Minisim

`matplotlib` only (figures). This notebook uses no widgets or video, so it is the
lightest of the three to run.

## What you'll learn

- How `hungarian_match` pairs estimated cells to true ones, and where `recall`,
  `precision`, `f1`, and mean overlap come from.
- When to use **weighted** footprint overlap (`cosine`, `weighted_jaccard`) instead of
  binary IoU, and how `shift="auto"` / the motion trajectory absorb a global offset.
- Why the deconvolved `S` is a scaled activity rate, not spikes, and how
  `activity_similarity` scores it (correlation, recovered scale, variance explained).
- How `shift_rmse` scores motion tracking independent of the registration origin.
- How `minisim.testing.score` bundles all of this into one honest `Report` - with the
  recall denominator (`n_requested` / `n_detectable` / `n_true`) always in view.

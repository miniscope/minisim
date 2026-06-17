# 02 - Cross-talk and demixing

**Why averaging an ROI does not recover a cell's calcium - and what does.**

The second training notebook inverts the first. Notebook 1 built a recording forward
as `movie = A·C + b·f` (each cell's footprint `A_i` times its calcium `C_i`, on a
diffuse background). This one asks the question that motivates the whole analysis
pipeline: **given the movie, how do you read one cell's calcium back out?**

The intuitive answer - draw a region of interest and average its pixels - is wrong, and
seeing *why* is the lesson. Optical blur and 1-photon tissue scatter spread every
footprint, so a cell's ROI also collects its **neighbours'** light (cross-talk) and the
**neuropil** pedestal. Because minisim generated the recording, the true `A`, `C`, and
background are known, so the contamination can be shown exactly, then removed.

It is a **static, step-by-step walkthrough** (no live widgets), so it reads correctly
even in a statically-rendered copy. Three parts:

1. **The problem** - the naive footprint-ROI trace, decomposed into its sources (own
   `C` + neighbour bleed + neuropil pedestal) against the ground truth.
2. **The fix** - demixing: solve `movie = A·C + b·f` for the traces instead of masking.
   Recovered `C` tracks the truth and the cross-talk collapses. This is the `A·C`
   factorization **CNMF** performs, run as the inverse of Notebook 1.
3. **The limit** - sweep density: even *oracle* demixing (perfect knowledge of `A`)
   breaks down once footprints overlap too much, the irreducible limit physics sets on
   any pipeline.

## Run it

No data download is needed - the recording is generated on the fly.

```bash
pip install "minisim[notebook]"   # engine + matplotlib (no widgets or video here)
jupyter notebook 02_demixing.ipynb
```

Run top to bottom. Every demo cell starts with a `# try:` knob - edit it and re-run to
explore (a different cell, more neighbours, a different density sweep).

## Dependencies beyond core Minisim

`matplotlib` only (figures). Like Notebook 3, it uses no widgets or video.

## What you'll learn

- Why a hand-drawn ROI trace is a **mixture**, not a cell: blur and scatter make
  footprints overlap, so the mask collects neighbour bleed and the neuropil pedestal.
- How **demixing** solves the linear system `movie = A·C + b·f` for the traces, and why
  that is exactly the factorization a CNMF pipeline (minian) performs.
- That demixing has an **irreducible limit** - when footprints overlap too much the
  system is ill-conditioned and no method, however perfect its spatial model, can
  separate the traces.

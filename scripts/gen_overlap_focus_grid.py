"""Two cells vs (lateral separation x focal plane), driven by sweep(), with
brute-force ROI crosstalk.

A worked sweep(): from one base Spec (a Miniscope V4 imaging two cells), sweep
the focal plane 85 -> 125 um (the cells sit at 100 / 110 um) against the lateral
separation 0 -> 50 um. Focal plane is a plain scalar axis; separation is swept by
overriding the whole `place_neurons.populations` field with a fresh two-cell pair
per value. sweep() yields the full Cartesian product as validated specs, each
tagged with an `axes` dict, which we collect into a tidy DataFrame.

For every grid point we do the most naive trace extraction possible - drop a
~20 um ROI right on each cell's known position, average its pixels per frame -
and correlate the two ROI traces. The two cells fire *independently*, so the true
trace correlation is ~0; anything above that is pure ROI contamination from
optical blur bleeding one cell into the other's ROI. Close + out-of-focus -> high
correlation; far + in-focus -> clean.

Writes two PNGs to docs/_static/examples/:
  overlap_grid_images.png   - the max-projection at each (focal, separation)
  overlap_grid_corr.png     - the ROI-trace correlation heatmap

Run from the repo root: .venv/Scripts/python.exe scripts/gen_overlap_focus_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minisim import (
    Acquisition,
    CellActivity,
    CellOptics,
    Composite,
    NeuronPopulation,
    PlaceNeurons,
    Sensor,
    Spec,
    Tissue,
    simulate,
    sweep,
)
from minisim.presets import miniscope_v4

# ---- the experiment knobs ---------------------------------------------------
FOCAL_PLANES_UM = [85.0, 95.0, 105.0, 115.0, 125.0]
SEPARATIONS_UM = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
DEPTHS_UM = (100.0, 110.0)  # shallow enough that scatter/defocus stay modest
MID_DEPTH_UM = sum(DEPTHS_UM) / 2.0  # the best-focus plane (105 µm)
ROI_DIAM_UM = 20.0
# The two cells fire independently, but over a short window two bursty traces (0.5 s
# calcium decay) correlate by chance: at 10 s a single seed can hit +/-0.4. A 60 s
# recording has enough events that the true correlation settles to ~0, so any ROI
# correlation that remains is pure optical-blur crosstalk.
DURATION_S = 60.0
FPS = 10.0
# This test only needs the patch of FOV around the optical axis, so we keep every
# V4 optic (NA, magnification, pixel pitch -> pixel_size_um, field curvature) but
# crop the sensor to a small window. Field curvature / vignette are radial from the
# axis, so for these near-axis cells the numbers are identical to the full 608 px
# V4 - this is purely a speed crop (~22x fewer pixels).
SENSOR_N_PX = 128
# Shallower cells are brighter (less tissue attenuation), so drop the exposure
# from the deep-tissue 600 to keep the 8-bit ADC off its 255 ceiling.
PHOTONS_PER_UNIT = 250.0
_V4 = miniscope_v4()
_SMALL_SENSOR = _V4.image_sensor.model_copy(
    update={"n_px_height": SENSOR_N_PX, "n_px_width": SENSOR_N_PX}
)


def two_cells(separation_um: float) -> NeuronPopulation:
    """A pair on the optical-axis row (y = 0), straddling center by +/- sep/2."""
    return NeuronPopulation(
        positions_um=[
            (DEPTHS_UM[0], 0.0, -separation_um / 2.0),
            (DEPTHS_UM[1], 0.0, +separation_um / 2.0),
        ],
        soma_radius_um=5.0,
        morphology="cytosolic",
    )


def roi_trace(observed: np.ndarray, acq, y_um: float, x_um: float) -> np.ndarray:
    """Mean of the pixels inside a circular ROI of ROI_DIAM_UM at (y, x) um, per frame."""
    h, w = observed.shape[1:]
    cr, cc = acq.um_to_index(y_um, x_um, (h, w))
    radius_px = (ROI_DIAM_UM / 2.0) / acq.pixel_size_um
    rows, cols = np.ogrid[:h, :w]
    mask = (rows - cr) ** 2 + (cols - cc) ** 2 <= radius_px**2
    return observed[:, mask].mean(axis=1)


# ---- SETUP: one base spec, swept over (focal plane x separation) ------------
# V4 optics, the cropped sensor, a default Tissue scatter model, and the standard
# cell chain. focal_depth and the cell pair here are placeholders that the sweep
# overrides per grid point.
base = Spec(
    acquisition=Acquisition(
        optics=_V4.optics,
        image_sensor=_SMALL_SENSOR,
        tissue=Tissue(),
        fps=FPS,
        duration_s=DURATION_S,
    ),
    seed=0,
    steps=[
        PlaceNeurons(populations=[two_cells(0.0)]),
        CellActivity(),
        CellOptics(),
        Composite(),
        Sensor(photons_per_unit=PHOTONS_PER_UNIT),
    ],
)
AXES = {
    "acquisition.focal_depth_in_tissue_um": FOCAL_PLANES_UM,  # scalar axis
    "steps.place_neurons.populations": [[two_cells(s)] for s in SEPARATIONS_UM],
}

WIN_UM = 42.0
rows: list[dict] = []
thumbs: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
max_count = 0.0
sat_frac = 0.0

for variant in sweep(base, AXES):
    rec = simulate(variant)
    acq = rec.spec.acquisition
    obs = np.asarray(rec.observed)
    gt = rec.ground_truth
    full_scale = 2 ** acq.image_sensor.bit_depth - 1
    max_count = max(max_count, float(obs.max()))
    sat_frac = max(sat_frac, float((obs >= full_scale).mean()))

    # brute-force ROI traces at each cell's true position, then correlate.
    (_, y0, x0), (_, y1, x1) = gt.centers_um
    t0, t1 = roi_trace(obs, acq, y0, x0), roi_trace(obs, acq, y1, x1)
    sep = x1 - x0  # the two cells straddle the axis, so their x-gap is the separation
    rows.append({
        "focal_depth_um": variant.axes["acquisition.focal_depth_in_tissue_um"],
        "separation_um": sep,
        # sep = 0 puts both ROIs on the same spot -> identical traces -> corr = 1
        "roi_corr": 1.0 if sep == 0.0 else float(np.corrcoef(t0, t1)[0, 1]),
        "true_corr": float(np.corrcoef(gt.C[0], gt.C[1])[0, 1]),
    })

    # keep a cropped max projection for the image montage
    img = obs.max(axis=0)
    h, w = img.shape
    cr, cc = int(round((h - 1) / 2.0)), int(round((w - 1) / 2.0))
    win = int(round(WIN_UM / acq.pixel_size_um))
    crop = img[cr - win : cr + win, cc - win : cc + win]
    focal = variant.axes["acquisition.focal_depth_in_tissue_um"]
    thumbs[(focal, sep)] = (crop, gt.centers_um[:, 2], np.zeros(gt.n_units))

df = pd.DataFrame(rows)
roi_grid = df.pivot(index="focal_depth_um", columns="separation_um", values="roi_corr")
print(f"ADC peak count = {max_count:.0f} of 255  (saturated pixels: {sat_frac * 100:.3f}%)")
print(f"mean underlying C correlation = {df['true_corr'].mean():+.3f} "
      f"(near 0 -> the cells really are independent; ROI correlation is crosstalk)")

OUT = Path("docs/_static/examples")
OUT.mkdir(parents=True, exist_ok=True)

# ---- figure 1: the image grid (focal x separation), ROIs drawn to scale -----
vmax = max(float(c.max()) for c, _, _ in thumbs.values())  # true peak, no display clip
nf, ns = len(FOCAL_PLANES_UM), len(SEPARATIONS_UM)
fig, axes = plt.subplots(nf, ns, figsize=(1.7 * ns, 1.7 * nf))
for fi, focal in enumerate(FOCAL_PLANES_UM):
    for si, sep in enumerate(SEPARATIONS_UM):
        ax = axes[fi, si]
        crop, xc, yc = thumbs[(focal, sep)]
        ax.imshow(crop, cmap="gray", vmin=0.0, vmax=vmax,
                  extent=[-WIN_UM, WIN_UM, WIN_UM, -WIN_UM])
        # draw each ROI to scale (radius in µm, same units as the extent) plus a
        # dot at the true center it is placed on
        for cx, cy in zip(xc, yc, strict=True):
            ax.add_patch(plt.Circle((cx, cy), ROI_DIAM_UM / 2.0, fill=False,
                                    edgecolor="tab:red", linewidth=1.1))
        ax.scatter(xc, yc, s=4, color="tab:red")
        ax.set_xticks([])
        ax.set_yticks([])
        if fi == 0:
            ax.set_title(f"Δ = {sep:.0f} µm", fontsize=10)
        if si == 0:
            focus_tag = "  (in focus)" if focal == MID_DEPTH_UM else ""
            ax.set_ylabel(f"focal\n{focal:.0f} µm{focus_tag}", fontsize=9)
fig.suptitle(
    f"Max projection vs focal plane × separation "
    f"(cells at {DEPTHS_UM[0]:.0f} / {DEPTHS_UM[1]:.0f} µm, V4 optics, {SENSOR_N_PX} px crop)\n"
    f"red circle = the {ROI_DIAM_UM:.0f} µm extraction ROI, to scale",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "overlap_grid_images.png", dpi=115)
plt.close(fig)

# ---- figure 2: the ROI-trace correlation heatmap ---------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
im = ax.imshow(roi_grid.values, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto", origin="upper")
ax.set_xticks(range(ns), [f"{s:.0f}" for s in roi_grid.columns])
ax.set_yticks(range(nf), [f"{f:.0f}" for f in roi_grid.index])
ax.set_xlabel("lateral separation Δ (µm)")
ax.set_ylabel("focal plane depth (µm)")
ax.set_title(
    f"Brute-force ROI trace correlation, two independent cells ({DURATION_S:.0f} s each)\n"
    f"{ROI_DIAM_UM:.0f} µm ROI on each true position; the cells' true correlation is ~0, "
    "so any signal here\nis optical-blur crosstalk - worst when close, and away from the "
    f"in-focus {MID_DEPTH_UM:.0f} µm plane",
    fontsize=10,
)
for fi in range(nf):
    for si in range(ns):
        v = roi_grid.values[fi, si]
        ax.text(si, fi, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.6 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r of ROI traces")
fig.tight_layout()
fig.savefig(OUT / "overlap_grid_corr.png", dpi=120)
plt.close(fig)

print("wrote:", OUT / "overlap_grid_images.png", "and", OUT / "overlap_grid_corr.png")

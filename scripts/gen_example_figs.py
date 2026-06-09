"""Generate the figures for docs/howto/examples.md (minimal -> full sim ladder).

The per-rung bodies below are the SAME code shown in the doc, so the figures
always reflect the example. Writes PNGs to docs/_static/examples/.

Run from the repo root: .venv/Scripts/python.exe scripts/gen_example_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from minisim import (
    Acquisition,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    Composite,
    IlluminationProfile,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    PlaceNeurons,
    Sensor,
    Spec,
    Vasculature,
    VesselLayer,
    Vignette,
    simulate,
)

OUT = Path("docs/_static/examples")
OUT.mkdir(parents=True, exist_ok=True)


# ---- shared setup + tiny plotting helpers (shown once in the doc) ----------
acq = Acquisition(
    fps=20.0,
    duration_s=20.0,
    optics=Optics(magnification=8.0),
    image_sensor=ImageSensor(n_px_height=200, n_px_width=200, pixel_pitch_um=8.0),
)


def show(ax, img, vmax, title):
    """Grayscale panel, fixed black point, no ticks."""
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def lively(movie):
    """Index of the frame carrying the most total signal (cells firing together)."""
    return int(np.asarray(movie).sum(axis=(1, 2)).argmax())


# ===========================================================================
# 1. Minimal: place_neurons + cell_activity + composite  (sharp cells on black)
# ===========================================================================
rec = simulate(Spec(acquisition=acq, seed=1, steps=[
    PlaceNeurons(), CellActivity(), Composite(),
]))
gt = rec.ground_truth
movie = np.asarray(rec.observed)
f = lively(movie)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8))
show(a, movie[f], np.percentile(movie[f], 99.8), f"frame {f}: cells on black")
z, y, x = gt.centers_um.T                          # (z, y, x) µm per cell
sc = b.scatter(x, y, c=z, s=18, cmap="viridis")
b.invert_yaxis(); b.set_aspect("equal")
b.set(title="ground truth: cell positions", xlabel="x (µm)", ylabel="y (µm)")
fig.colorbar(sc, ax=b, fraction=0.046, pad=0.04, label="depth z (µm)")
t = np.arange(gt.C.shape[1]) / acq.fps
top = np.argsort(gt.C.max(axis=1))[::-1][:4]       # 4 most active cells
for k, i in enumerate(top):
    c.plot(t, gt.C[i] + k * 0.6 * gt.C[top].max())
c.set(title="ground truth: calcium traces C", xlabel="time (s)"); c.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "01_minimal.png", dpi=110); plt.close(fig)


# ===========================================================================
# 2. + optics: depth-dependent blur and dimming  (ONE sim; before/after is the
#    same recording's planted vs observed footprint, so no hidden baseline run)
# ===========================================================================
rec_o = simulate(Spec(acquisition=acq, seed=1, steps=[
    PlaceNeurons(), CellActivity(), CellOptics(), Composite(),
]))
gt_o = rec_o.ground_truth

# Planted (sharp) vs observed (degraded) footprint of one well-centered cell -
# both live in this single recording's ground truth.
px = acq.pixel_size_um
cyx = gt_o.centers_um[:, 1:] / px
sig = gt_o.observed_sigma_px
ok = np.where(
    (sig > 6) & (sig < 13)
    & (cyx[:, 0] > 50) & (cyx[:, 0] < 150)
    & (cyx[:, 1] > 50) & (cyx[:, 1] < 150)
)[0]
i = int(ok[np.argmax(gt_o.A_planted[ok].reshape(len(ok), -1).sum(axis=1))])
cyi, cxi, R = int(round(cyx[i, 0])), int(round(cyx[i, 1])), 45
crop = (slice(cyi - R, cyi + R), slice(cxi - R, cxi + R))
ob = gt_o.A_observed[i][crop]

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8))
show(a, gt_o.A_planted[i][crop], 1.0, f"one cell: planted A (z = {gt_o.depth_um[i]:.0f} µm)")
show(b, ob, float(ob.max()), "observed A (blurred; own scale)")
# Decompose the per-cell blur (dots) into its depth terms, plus the brightness falloff.
zg = np.linspace(0.0, 200.0, 200)
focal = gt_o.focal_depth_um
defocus = np.array([acq.optics.defocus_sigma_um(zi, focal) for zi in zg]) / acq.pixel_size_um
scatter = np.array([acq.tissue.scatter_sigma_um(zi) for zi in zg]) / acq.pixel_size_um
gain = np.array([acq.tissue.attenuation(zi) for zi in zg]) * acq.optics.collection_efficiency
c.scatter(gt_o.depth_um, gt_o.observed_sigma_px, s=10, c="tab:blue", alpha=0.45, label="total σ (per cell)")
c.plot(zg, defocus, "tab:blue", lw=1.3, label="defocus |z − focal|")
c.plot(zg, scatter, "tab:blue", lw=1.3, ls="--", label="scatter (depth)")
c.axvline(focal, color="0.6", lw=0.8, ls=":")
c.set(title="ground truth: optics vs depth", xlabel="depth z (µm)", ylabel="blur σ (px)")
c.legend(fontsize=7, loc="upper center")
c2 = c.twinx(); c2.plot(zg, gain, "tab:red", lw=1.3)
c2.set_ylabel("brightness gain", color="tab:red"); c2.tick_params(axis="y", labelcolor="tab:red")
fig.tight_layout(); fig.savefig(OUT / "02_optics.png", dpi=110); plt.close(fig)


# ===========================================================================
# 3. + brain_motion: rigid motion (the ground-truth per-frame shift)
# ===========================================================================
rec_m = simulate(Spec(acquisition=acq, seed=1, steps=[
    PlaceNeurons(), CellActivity(), CellOptics(), Composite(), BrainMotion(),
]))
shifts = rec_m.ground_truth.shifts                 # (frame, 2) = (dy, dx) px
ts = np.arange(shifts.shape[0]) / acq.fps

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
a.plot(ts, shifts[:, 1], lw=0.9, label="dx")
a.plot(ts, shifts[:, 0], lw=0.9, label="dy")
a.set(title="ground truth: per-frame shift", xlabel="time (s)", ylabel="shift (px)"); a.legend(fontsize=8)
b.plot(shifts[:, 1], shifts[:, 0], lw=0.7); b.scatter([0], [0], c="k", s=12, zorder=3)
b.set_aspect("equal"); b.set(title="motion path", xlabel="dx (px)", ylabel="dy (px)")
fig.tight_layout(); fig.savefig(OUT / "03_motion.png", dpi=110); plt.close(fig)


# ===========================================================================
# 4. + neuropil background (cells-only vs + diffuse haze)
# ===========================================================================
base = [PlaceNeurons(), CellActivity(), CellOptics(), Composite()]
rec_n = simulate(Spec(acquisition=acq, seed=1, steps=[*base, Neuropil()]))
gt_n = rec_n.ground_truth
m_n = np.asarray(rec_n.observed)
f_n = lively(m_n)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8))
show(a, m_n[f_n], np.percentile(m_n[f_n], 99.8), f"frame {f_n}: cells + neuropil haze")
# neuropil structure straight from this recording's ground truth.
im = b.imshow(gt_n.neuropil_spatial.sum(axis=0), cmap="magma")
b.set(title="ground truth: neuropil spatial"); b.set_xticks([]); b.set_yticks([])
fig.colorbar(im, ax=b, fraction=0.046, pad=0.04)
tt = np.arange(gt_n.neuropil_temporal.shape[1]) / acq.fps
for comp in gt_n.neuropil_temporal:
    c.plot(tt, comp, lw=0.7, alpha=0.6)
c.plot(tt, gt_n.neuropil_population, "k", lw=1.1, label="population driver")
c.set(title="ground truth: neuropil temporal", xlabel="time (s)", ylabel="envelope"); c.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "04_neuropil.png", dpi=110); plt.close(fig)


# ===========================================================================
# 5. + vasculature: a dark, static absorbing vessel mask (landmark + confound)
# ===========================================================================
# Near the (auto-resolved ~100 µm) focal plane, so the vessel tree stays sharp;
# a shallower layer would soften into a broad shadow (the same depth blur as cells).
vasc_layer = VesselLayer(depth_um=100.0, n_roots=4, root_radius_um=10.0, opacity=0.8)
rec_v = simulate(Spec(acquisition=acq, seed=1, steps=[
    *base, Neuropil(), Vasculature(enabled=True, layers=[vasc_layer]),
]))
gt_v = rec_v.ground_truth
m_v = np.asarray(rec_v.observed)
f_v = lively(m_v)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8))
show(a, m_v[f_v], np.percentile(m_v[f_v], 99.8), f"frame {f_v}: cells under a vessel shadow")
mask = gt_v.vasculature_mask                       # (H, W) transmission in (0, 1]
im = b.imshow(mask, cmap="gray", vmin=float(mask.min()), vmax=1.0)
b.set(title="ground truth: vessel transmission"); b.set_xticks([]); b.set_yticks([])
fig.colorbar(im, ax=b, fraction=0.046, pad=0.04)
# the scoreable confound: per-cell footprint-weighted vessel occlusion.
zc, yc, xc = gt_v.centers_um.T
ov = gt_v.vessel_overlap_fraction
sc = c.scatter(xc, yc, c=ov, s=18, cmap="inferno", vmin=0.0, vmax=max(0.2, float(ov.max())))
c.invert_yaxis(); c.set_aspect("equal")
c.set(title="ground truth: per-cell vessel overlap", xlabel="x (µm)", ylabel="y (µm)")
fig.colorbar(sc, ax=c, fraction=0.046, pad=0.04, label="overlap fraction")
fig.tight_layout(); fig.savefig(OUT / "05_vasculature.png", dpi=110); plt.close(fig)


# ===========================================================================
# 6. + static fields: illumination x vignette, and additive leakage glow
# ===========================================================================
rec_f = simulate(Spec(acquisition=acq, seed=1, steps=[
    *base, Neuropil(), IlluminationProfile(), Vignette(), Leakage(),
]))
gt_f = rec_f.ground_truth
m_f = np.asarray(rec_f.observed)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8))
im0 = a.imshow(gt_f.illumination * gt_f.vignette, cmap="magma")
a.set(title="illumination × vignette"); a.set_xticks([]); a.set_yticks([])
fig.colorbar(im0, ax=a, fraction=0.046, pad=0.04)
im1 = b.imshow(gt_f.leakage, cmap="magma")
b.set(title="leakage (additive glow)"); b.set_xticks([]); b.set_yticks([])
fig.colorbar(im1, ax=b, fraction=0.046, pad=0.04)
show(c, m_f[lively(m_f)], np.percentile(m_f, 99.8), "frame with fields applied")
fig.tight_layout(); fig.savefig(OUT / "06_fields.png", dpi=110); plt.close(fig)


# ===========================================================================
# 7. Full recording: + sensor (clean intensity -> noisy integer counts)
# ===========================================================================
spec_full = Spec(acquisition=acq, seed=1, steps=[
    PlaceNeurons(), CellActivity(), Bleaching(), CellOptics(), Composite(), Neuropil(),
    BrainMotion(), IlluminationProfile(), Vignette(), Leakage(), Sensor(),
])
# Both panels from the SAME run: with a sensor present the optics "auto" focus
# switches to the yield-maximizing plane, so a separate sensorless run would focus
# elsewhere. until="leakage" gives the exact intensity the sensor digitizes.
pre = np.asarray(simulate(spec_full, until="leakage").observed)
post = np.asarray(simulate(spec_full).observed)
f_s = lively(post)
hw = acq.image_sensor
expected = np.clip(np.floor(pre[f_s] * 100.0 * hw.quantum_efficiency * hw.gain_adu_per_e),
                   0.0, 2 ** hw.bit_depth - 1)      # noise-free expected counts

fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8))
show(a, expected, 255.0, "expected counts (noise-free)")
show(b, post[f_s], 255.0, "+ sensor: realized counts (same frame)")
c.hist(post[f_s].ravel(), bins=60, color="0.3")
c.set(title="ground truth: count histogram", xlabel="ADC counts", ylabel="pixels"); c.set_yscale("log")
fig.tight_layout(); fig.savefig(OUT / "07_full.png", dpi=110); plt.close(fig)

print("wrote:", *sorted(p.name for p in OUT.glob("*.png")))

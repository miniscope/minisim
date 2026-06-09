"""Generate the minisim logo - which is itself a minisim recording.

Two concepts, both kept:
  * Concept A - an icon: an 'M' of neurons imaged through a round miniscope FOV
    (neuropil glow + illumination/vignette falloff), plus the 'minisim' wordmark.
  * Concept B - a wordmark whose 'sim' is literally simulated: its letters are
    filled with neurons and imaged, while 'mini' stays as plain text.

Neurons are placed with `positions_um` and pushed through the real forward chain,
so the mark dogfoods the library. Run from the repo root:

    .venv/Scripts/python.exe scripts/make_logo.py

Writes the finished assets (transparent background) to docs/_static/logo/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from minisim import (
    Acquisition,
    CellActivity,
    CellOptics,
    Composite,
    IlluminationProfile,
    ImageSensor,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Spec,
    Vignette,
    simulate,
)
from minisim.notebooks._support import GCAMP

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static" / "logo"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 3
MORPH = "cytosolic"  # soma + proximal dendrites, for both concepts

matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"

GRAY = "#4f5d54"  # solid dark slate-green; the old light gray read as washed-out
# Green colormap that stays green at the top (no white blowout) so cells read on white.
SIMGREEN = LinearSegmentedColormap.from_list(
    "simgreen", [(0, "#04140a"), (0.35, "#1f9a45"), (0.7, "#3fe06a"), (1.0, "#9ff7bb")]
)


def _cell(morph, dendrite_width_um=3.0, **kw):
    """A PlaceNeurons with longer cytosolic dendrites, shared by both concepts."""
    return PlaceNeurons(
        morphology=morph, soma_radius_um=7.0, n_dendrites=5,
        dendrite_length_um=38.0, dendrite_width_um=dendrite_width_um, **kw,
    )


# --------------------------------------------------------------------------- #
# Concept A: the 'M' icon, imaged in a round FOV of structured neuropil.
# --------------------------------------------------------------------------- #
PX = 768
acq_M = Acquisition(
    fps=20.0, duration_s=6.0,
    optics=Optics(magnification=5.0, na=0.5),  # lower mag -> wider field of view
    image_sensor=ImageSensor(n_px_height=PX, n_px_width=PX, pixel_pitch_um=8.0),
)
FOV = PX * acq_M.pixel_size_um
MBOX = (0.26, 0.74, 0.23, 0.77)  # x0, x1, y0, y1 (fraction of FOV)
STROKE_W_UM = 17.0  # thickness of the M strokes
DISC_R0, DISC_R1 = 0.44, 0.495  # circular cutoff: solid inside R0, faded by R1
CROP = (int(0.005 * PX), int(0.995 * PX))


def m_positions(spacing_um=6.0, jitter=3.0, depth=(35.0, 110.0)):
    """Soma (z, y, x) centers filling the four strokes of a thick capital M."""
    BL, TL, MID, TR, BR = (0, 1), (0, 0), (0.5, 0.52), (1, 0), (1, 1)
    strokes = [(BL, TL), (TL, MID), (MID, TR), (TR, BR)]
    x0, x1 = MBOX[0] * FOV, MBOX[1] * FOV
    y0, y1 = MBOX[2] * FOV, MBOX[3] * FOV
    rng = np.random.default_rng(SEED)
    half = STROKE_W_UM / 2
    n_cross = max(1, int(STROKE_W_UM / spacing_um))
    pts = []
    for (ax, ay), (bx, by) in strokes:
        a = np.array([x0 + ax * (x1 - x0), y0 + ay * (y1 - y0)])
        b = np.array([x0 + bx * (x1 - x0), y0 + by * (y1 - y0)])
        d = b - a
        L = float(np.hypot(*d))
        u = d / (L + 1e-9)
        perp = np.array([-u[1], u[0]])  # across the stroke
        for t in np.linspace(0, 1, max(2, int(L / spacing_um))):
            base = a + t * d
            for off in np.linspace(-half, half, n_cross):
                p = base + (off + rng.normal(0, jitter)) * perp + rng.normal(0, jitter, 2)
                pts.append((float(rng.uniform(*depth)), float(p[1]), float(p[0])))
    return pts


def disc_mask():
    yy, xx = np.mgrid[0:PX, 0:PX]
    r = np.hypot(yy - PX / 2, xx - PX / 2) / PX
    m = np.clip((DISC_R1 - r) / (DISC_R1 - DISC_R0), 0, 1)
    return m * m * (3 - 2 * m)  # smoothstep edge


def make_mark():
    """Render the circular 'M' mark; return the (H, W) green-normalized array."""
    steps = [
        _cell(MORPH, dendrite_width_um=2.0, positions_um=m_positions()),  # thinner dendrites
        CellActivity(), CellOptics(), Composite(),
        # structured neuropil: finer features (smaller sigma) + more components
        Neuropil(spatial_sigma_um=26.0, n_components=5, amplitude=1.8),
        IlluminationProfile(falloff=0.45, exponent=2.0),
        Vignette(falloff=0.0, exponent=3.0),
    ]
    rec = simulate(Spec(acquisition=acq_M, seed=SEED, output=Output(save_intermediates=True), steps=steps))
    gt = rec.ground_truth
    cells = np.asarray(rec.stage("cells_only").values).max(0)
    haze = np.clip((np.asarray(rec.stage("neuropil").values) - np.asarray(rec.stage("cells_only").values)).mean(0), 0, None)
    vfield = np.asarray(gt.illumination) * np.asarray(gt.vignette)  # the round falloff
    disc = disc_mask()

    cn = np.clip(cells / (np.percentile(cells, 99.6) + 1e-9), 0, 1)
    hn = haze / (np.percentile(haze[disc > 0.5], 90) + 1e-9)
    combined = (cn + 0.28 * np.clip(hn, 0, 1)) * vfield * disc  # neuropil structure, then round it
    combined = combined / (np.percentile(combined, 99.7) + 1e-9)
    arr = np.clip(combined, 0, 1)[CROP[0] : CROP[1], CROP[0] : CROP[1]]
    # save the bare circular mark on black
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(arr, cmap=GCAMP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    fig.savefig(OUT / "minisim_mark.png", bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)
    print("wrote minisim_mark.png")
    return arr


def _rounded_icon(ax, gray_arr):
    rgb = GCAMP(np.clip(gray_arr, 0, 1))
    h, w = rgb.shape[:2]
    im = ax.imshow(rgb, zorder=2)
    box = FancyBboxPatch(
        (0, 0), w - 1, h - 1, boxstyle=f"round,pad=0,rounding_size={0.16 * w}",
        transform=ax.transData, fc="black", ec="none", zorder=1,
    )
    ax.add_patch(box)
    im.set_clip_path(box)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()


def make_icon(mark):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    fig.patch.set_alpha(0)
    _rounded_icon(fig.add_axes([0, 0, 1, 1]), mark)
    fig.savefig(OUT / "minisim_icon.png", transparent=True)
    plt.close(fig)
    print("wrote minisim_icon.png")


# --------------------------------------------------------------------------- #
# Concept B: the wordmark whose 'sim' is simulated.
# --------------------------------------------------------------------------- #
def text_mask(text):
    fig = plt.figure(figsize=(8, 2), dpi=200)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=120, fontweight="bold")
    fig.canvas.draw()
    W, H = int(fig.bbox.width), int(fig.bbox.height)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(H, W, 4)
    plt.close(fig)
    alpha = buf[:, :, 3] > 40
    ys, xs = np.where(alpha)
    return alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _text_mapping(mask, fov_w, fov_h):
    mh, mw = mask.shape
    mx, my = 0.05 * fov_w, 0.16 * fov_h
    scale = min((fov_w - 2 * mx) / mw, (fov_h - 2 * my) / mh)
    return scale, (fov_w - mw * scale) / 2, (fov_h - mh * scale) / 2


def sample_text(mask, scale, ox, oy, spacing_px=7, jitter=2.0, depth=(30.0, 72.0)):
    mh, mw = mask.shape
    rng = np.random.default_rng(SEED)
    pts = []
    for r in range(0, mh, spacing_px):
        for c in range(0, mw, spacing_px):
            if mask[r, c]:
                x = ox + (c + rng.normal(0, jitter)) * scale
                y = oy + (r + rng.normal(0, jitter)) * scale
                pts.append((float(rng.uniform(*depth)), float(y), float(x)))
    return pts


def simulate_word(text):
    from scipy.ndimage import gaussian_filter, zoom

    PXW, PXH = 1000, 360
    acq = Acquisition(
        fps=20.0, duration_s=6.0,
        optics=Optics(magnification=5.0, na=0.5),
        image_sensor=ImageSensor(n_px_height=PXH, n_px_width=PXW, pixel_pitch_um=8.0),
    )
    ps = acq.pixel_size_um
    tmask = text_mask(text)
    scale, ox, oy = _text_mapping(tmask, PXW * ps, PXH * ps)
    pos = sample_text(tmask, scale, ox, oy)
    steps = [
        _cell(MORPH, positions_um=pos), CellActivity(), CellOptics(), Composite(),
        Neuropil(spatial_sigma_um=18.0, n_components=6, amplitude=0.25),
    ]
    rec = simulate(Spec(acquisition=acq, seed=SEED, output=Output(save_intermediates=True), steps=steps))
    cells = np.asarray(rec.stage("cells_only").values).max(0)
    haze = np.clip((np.asarray(rec.stage("neuropil").values) - np.asarray(rec.stage("cells_only").values)).mean(0), 0, None)

    # a soft letter-shaped mask at sim resolution, so neuropil fills only the glyphs
    lm = zoom(tmask.astype(float), scale / ps, order=1)
    canvas = np.zeros((PXH, PXW))
    r0, c0 = int(round(oy / ps)), int(round(ox / ps))
    lh, lw = min(lm.shape[0], PXH - r0), min(lm.shape[1], PXW - c0)
    canvas[r0 : r0 + lh, c0 : c0 + lw] = lm[:lh, :lw]
    letter = gaussian_filter(np.clip(canvas, 0, 1), 5)

    cn = np.clip(cells / (np.percentile(cells, 99.6) + 1e-9), 0, 1)
    sel = letter > 0.3
    hn = haze / (np.percentile(haze[sel], 88) + 1e-9) if sel.any() else haze
    fill = np.clip(hn, 0, 1) * letter
    a = np.clip(cn + 0.18 * fill, 0, 1)  # bright cells over a faint glyph-shaped glow
    rgba = SIMGREEN(a)
    # opaque across the letters; only the empty background is transparent (soft edge
    # via smoothstep so there is no dark halo, but the letter bodies are solid).
    t = np.clip((a - 0.05) / (0.26 - 0.05), 0, 1)
    rgba[..., 3] = t * t * (3 - 2 * t)
    # the exact glyph box (the text bbox region) inside the full image, so the
    # caller can place it at the same size + baseline as plain text.
    gh, gw = int(round(tmask.shape[0] * scale / ps)), int(round(tmask.shape[1] * scale / ps))
    return rgba, (int(round(oy / ps)), int(round(ox / ps)), gh, gw)


def make_wordmark_sim():
    FS = 120
    rgba, (gr, gc, gh, gw) = simulate_word("sim")
    # crop to the glyph + a margin for glow, so the placed image is not mostly empty FOV
    mv, mh = int(0.22 * gh), int(0.28 * gw)
    R0, R1 = max(0, gr - mv), min(rgba.shape[0], gr + gh + mv)
    C0, C1 = max(0, gc - mh), min(rgba.shape[1], gc + gw + mh)
    rgba = rgba[R0:R1, C0:C1]
    gr, gc = gr - R0, gc - C0
    PXH, PXW = rgba.shape[:2]

    figw, figh = 13, 4
    fig = plt.figure(figsize=(figw, figh), dpi=150)
    fig.patch.set_alpha(0)
    base, x0 = 0.42, 0.04  # `base` is the shared text baseline (fig fraction)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    t1 = fig.text(x0, base, "Mini", ha="left", va="baseline", fontsize=FS, fontweight="bold", color=GRAY)
    e1 = t1.get_window_extent(rend)
    x1 = x0 + e1.width / fig.bbox.width + 0.012  # small kerning gap before 'sim'
    # measure how big 'sim' renders as text at the same font size, then size/place the
    # simulated glyph to match exactly (same height, same baseline, flush after 'mini').
    tref = fig.text(0.5, 0.5, "sim", fontsize=FS, fontweight="bold", alpha=0)
    e2 = tref.get_window_extent(rend)
    # shrink a touch so the glow-haloed cells match 'mini' height (glow reads taller)
    sim_scale = 0.93
    Wsim, Hsim = sim_scale * e2.width / fig.bbox.width, sim_scale * e2.height / fig.bbox.height
    tref.remove()  # measurement only - must not affect the tight bbox
    sx, sy = Wsim / gw, Hsim / gh
    axw, axh = PXW * sx, PXH * sy
    axl = x1 - sx * gc
    # glyph bottom row sits on the baseline; nudge down so the bright cell mass (inset
    # from the glyph edge by ~one cell radius) lines up with 'mini', not the faint glow.
    axb = base - axh * (1 - (gr + gh) / PXH) - 0.11 * Hsim
    ax = fig.add_axes([axl, axb, axw, axh])
    ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(OUT / "minisim_wordmark_sim.png", bbox_inches="tight", pad_inches=0.06, transparent=True)
    plt.close(fig)
    print("wrote minisim_wordmark_sim.png")


if __name__ == "__main__":
    make_icon(make_mark())
    make_wordmark_sim()
    print("done ->", OUT)

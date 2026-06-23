"""Presentation plumbing for the training notebooks - not part of the public API.

The teaching notebooks (``minisim/notebooks/training/``) are about *physics*, so
the matplotlib / ipywidgets / mediapy machinery that turns a recording into an
inline video or an interactive panel is factored out here, where it stops
crowding the lesson. Nothing in this module is part of minisim's forward-model
contract: it is imported only by the notebooks (which require the ``[notebook]``
extra), never by the engine, and the heavy plotting dependencies are imported at
module load - so importing it without the extra installed fails loudly, by
design. If these helpers ever earn external use, promote them to a public
``minisim.viz``; until then they live here, contained.

The footprint mask / ROI threshold is *not* re-derived here - the dashboards call
:func:`minisim.metrics.footprint_mask` and :func:`minisim.metrics.footprint_roi_trace`,
so the visuals read the data exactly as the engine does. The per-cell detectability
SNR is the same story one module over: the panels in ``_anatomy_panels.py`` compute it
with the recording's helpers (:func:`minisim.detection_snr`, :func:`minisim.sample_field_at`)
and hand the result to :func:`plot_snr_vs_radius` here purely to draw.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import mediapy
import numpy as np
from IPython.display import Image, display
from ipywidgets import HBox, VBox
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.ndimage import binary_dilation
from scipy.ndimage import zoom as ndzoom

from minisim.metrics import footprint_mask, footprint_roi_trace

# GCaMP-like LUT (black -> green) shared by every movie panel; the 256-entry table
# is for fast numpy colourization (no per-frame matplotlib).
GCAMP = LinearSegmentedColormap.from_list("gcamp", ["#000000", "#00b140", "#b6ffb6"])
_LUT = (GCAMP(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

# R/G/B channel primaries for the neuropil components video: bright for the dark
# field thumbnails, slightly darker for the trace lines so they read on white.
_RGB_THUMB = ["#ff5050", "#50ff50", "#6d7dff"]
_RGB_TRACE = ["#e02020", "#149014", "#2a44d6"]
_RGB_CMAPS = [
    LinearSegmentedColormap.from_list(f"c{k}", ["#000000", c])
    for k, c in enumerate(_RGB_THUMB)
]


def _agg_figure(figsize, dpi=100):
    """A standalone Agg-backed figure for offline frame compositing.

    The notebooks run under ``%matplotlib widget`` (ipympl) for the interactive
    sandboxes, but the dashboards rasterize a figure to an RGB array via
    :meth:`~matplotlib.backend_bases.FigureCanvasBase.buffer_rgba` (see
    :func:`_cursor_panel`). Once an ipympl canvas has been displayed, a retina
    frontend stamps ``device_pixel_ratio = 2`` onto subsequently-created figures,
    so that buffer comes back at ``2 x (figsize x dpi)`` with the figure drawn into
    one corner - which dropped all but the first dashboard row in browser
    JupyterLab. An explicit Agg canvas always rasterizes exactly ``figsize x dpi``,
    on every frontend, so the composite is backend-independent.
    """
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def play(movie, fps=20, height=280, title=None, vmax=None):
    """Normalize a ``(frame, h, w)`` movie to ``[0, 1]`` and show a looping clip.

    Maps ``[min, vmax]`` to ``[0, 1]``; ``vmax`` defaults to the movie's max. Pass
    a percentile (e.g. ``np.percentile(movie, 99.9)``) to keep a few bright pixels
    from compressing everything else into the dark - the same display scaling the
    dashboard colourizers use, so a raw ``play`` of a render reads as bright as its
    dashboard counterpart instead of washing the neuropil haze out to near-black.

    Given only a ``height``, mediapy derives the width and can land on an *odd*
    value, which cannot encode as ``yuv420p`` - so browsers (JupyterLab) refuse to
    play it and show a blank "no supported format" box. We pass an explicit, even
    ``(height, width)`` instead: the requested ``height`` (mediapy scales the clip
    up or down to it, so the small dashboards still read), the width aspect-matched
    to the source, both rounded down to even. Even dimensions always encode as
    yuv420p and play in every browser.
    """
    arr = np.asarray(movie, dtype=float)
    lo = float(arr.min())
    hi = float(vmax) if vmax is not None else float(arr.max())
    src_h, src_w = arr.shape[1], arr.shape[2]
    out_h = height - height % 2
    out_w = round(out_h * src_w / src_h)
    out_w -= out_w % 2  # round down to even -> yuv420p-encodable
    mediapy.show_video(
        np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0),
        fps=fps,
        height=out_h,
        width=out_w,
        title=title,
        codec="h264",
    )


def plot_snr_vs_radius(ax, radius_um, snr, threshold, *, title=None):
    """Scatter per-cell transient SNR against distance from the FOV centre.

    Each point is one cell: green if its realized SNR clears ``threshold`` (the
    photon budget can recover it), red if it sinks below the shot+read floor (the
    dashed line). The log y-axis spans the order-of-magnitude spread. ``radius_um``
    and ``snr`` are matched per-cell arrays - compute ``snr`` with
    :func:`minisim.detection_snr` and use :data:`minisim.DETECT_SNR_THRESHOLD` for
    the usual floor. This is the picture of *which cells are recoverable*, the same
    question ``finalize()`` answers with its ``detectable`` flag, so it reads the
    same whether the knob being explored is the illumination falloff or the sensor
    exposure (and it is the natural recovered-vs-true view for a later pipeline
    notebook).
    """
    radius_um, snr = np.asarray(radius_um), np.asarray(snr)
    ok = snr >= threshold
    ax.clear()
    ax.scatter(
        radius_um[ok],
        snr[ok],
        s=12,
        color="#2ca02c",
        label=f"detectable ({int(ok.sum())})",
    )
    ax.scatter(
        radius_um[~ok],
        snr[~ok],
        s=12,
        color="#d62728",
        label=f"below floor ({int((~ok).sum())})",
    )
    ax.axhline(threshold, color="k", ls="--", lw=1.0)
    ax.set(
        yscale="log", xlabel="distance from center (um)", ylabel="cell SNR", title=title
    )
    ax.legend(fontsize=7, loc="lower left", frameon=False)


def plot_population(
    ax_top,
    ax_side,
    centers_um,
    fov_um,
    *,
    depth_max,
    soma_radius_um,
    depth_range,
    morph_label="",
):
    """Top-down + side scatter of placed cell bodies, coloured by depth.

    ``ax_top`` looks straight down the optical axis - each cell a *true-radius* disk
    (so crowding reads honestly) coloured by its depth ``z``; ``ax_side`` plots the
    same lateral ``x`` against ``z`` with the placement depth band shaded, the view
    the top-down picture hides. ``centers_um`` is the ``(n, 3)`` ``(z, y, x)`` array
    :func:`minisim.steps.sample_neurons` (or ``GroundTruth.centers_um``) returns;
    ``fov_um`` is ``(height, width)`` and ``depth_range`` the ``(lo, hi)`` band. The
    natural "where are the cells" view for either a placement preview or a later
    recovered-vs-true positions comparison.
    """
    centers = np.asarray(centers_um, dtype=float).reshape(-1, 3)
    z, y, x = (
        (centers[:, 0], centers[:, 1], centers[:, 2]) if len(centers) else ([], [], [])
    )
    fov_h, fov_w = fov_um
    lo, hi = depth_range
    ax_top.clear()
    if len(centers):
        pc = PatchCollection(
            [Circle((xi, yi), soma_radius_um) for xi, yi in zip(x, y, strict=True)],
            cmap="viridis",
            norm=Normalize(0, depth_max),
            alpha=0.8,
        )
        pc.set_array(z)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.2)
        ax_top.add_collection(pc)
    # optical-center frame: the axis (0, 0) is the FOV center, so axes span ±fov/2.
    ax_top.set_xlim(-fov_w / 2, fov_w / 2)
    ax_top.set_ylim(-fov_h / 2, fov_h / 2)
    ax_top.invert_yaxis()
    ax_top.set_aspect("equal")
    suffix = f"  |  GCaMP: {morph_label}" if morph_label else ""
    ax_top.set(
        title=f"top view: {len(centers)} neurons over the {fov_w:.0f} x {fov_h:.0f} um FOV "
        f"(color = depth){suffix}",
        ylabel="y (um)",
    )
    ax_top.tick_params(labelbottom=False)
    ax_side.clear()
    ax_side.axhspan(lo, max(hi, lo), color="0.88", zorder=0)
    if len(centers):
        ax_side.scatter(x, z, c=z, cmap="viridis", vmin=0, vmax=depth_max, s=9)
    ax_side.set_xlim(-fov_w / 2, fov_w / 2)
    ax_side.set_ylim(0, depth_max)
    ax_side.invert_yaxis()
    ax_side.set(
        title="side view: depth distribution", xlabel="x (um)", ylabel="depth z (um)"
    )


def plot_traces(
    ax,
    t,
    C,
    spikes=None,
    *,
    n=5,
    title="calcium traces C (each scaled to its peak) + spikes S (ticks)",
):
    """Stacked, peak-normalized calcium traces for the busiest ``n`` units.

    Each lane is one cell's trace scaled to its own peak (so dense bursts stay
    legible and per-cell brightness does not dominate the axis), offset vertically;
    when ``spikes`` is given, its event frames are drawn as ticks under each lane.
    "Busiest" is ranked by ``spikes`` if provided, else by ``C``. ``C``/``spikes``
    are ``(unit, frame)`` (``GroundTruth.C`` / ``.S``); ``t`` is the per-frame time
    axis. The standard trace view - true here, estimated-vs-true in a later notebook.
    """
    C = np.asarray(C)
    ax.clear()
    if len(C):
        rank = np.asarray(spikes) if spikes is not None else C
        for row, u in enumerate(np.argsort(rank.sum(axis=1))[-n:]):
            c = C[u] - C[u].min()
            c = c / (
                c.max() or 1.0
            )  # scale to its own peak -> a clean unit-height lane
            off = row * 1.15
            ax.plot(t, c + off, lw=0.9)
            if spikes is not None:
                spk = np.where(np.asarray(spikes)[u] > 0)[0]
                ax.plot(t[spk], np.full(spk.shape, off - 0.18), "|", color="k", ms=4)
    ax.set(title=title, xlabel="time (s)", ylabel="cell (offset)")
    ax.set_yticks([])


def plot_count_histogram(ax, counts, max_count, *, title="where the counts pile up"):
    """Log histogram of integer ADC counts, with the saturation ceiling marked.

    One bin per code from 0 to ``max_count`` (``2**bit_depth - 1``); the log y-axis
    spans the read-noise floor near 0, the body, and the saturation spike piling up
    at the ceiling (dashed line). ``counts`` is the digitized sensor frame
    (``Recording.observed[f]`` or a single ``photons_to_counts`` output).
    """
    ax.clear()
    ax.hist(
        np.asarray(counts).ravel(),
        bins=np.arange(0, max_count + 2) - 0.5,
        color="#2ca02c",
        log=True,
    )
    ax.axvline(max_count, color="#d62728", ls="--", lw=1.0, label="saturation")
    ax.set(xlabel="ADC count", ylabel="pixels (log)", title=title)
    ax.legend(fontsize=7, loc="upper right", frameon=False)


def _colorize_with_rings(movie, gt, picks, colors, vmax, downsample=2):
    """LUT-colourize a movie and overlay static coloured rings on the picked cells.

    Returns ``(n, h, w, 3)`` uint8 frames (downsampled by ``downsample``). The
    rings are the dilated boundary of each picked cell's observed footprint mask
    (:func:`minisim.metrics.footprint_mask`), painted identically on every frame.
    """
    md = np.ascontiguousarray(movie[:, ::downsample, ::downsample])
    rgb = _LUT[(np.clip(md / vmax, 0, 1) * 255).astype(np.uint8)]  # (n, h, w, 3)
    ring_any = np.zeros(md.shape[1:], bool)
    ring_rgb = np.zeros((*md.shape[1:], 3), np.uint8)
    for i, u in enumerate(picks):
        mask = footprint_mask(np.asarray(gt.A_observed[u])[::downsample, ::downsample])
        ring = binary_dilation(mask, iterations=2) & ~mask
        ring_rgb[ring] = tuple(int(255 * c) for c in to_rgb(colors[i]))
        ring_any |= ring
    rgb[:, ring_any] = ring_rgb[ring_any]  # rings are static across frames
    return rgb


def build_dashboard_frames(movie, gt, picks, colors, t, vmax, px_um, downsample=2):
    """Compose ``(N, H, Wtot, 3)`` frames: colourized movie | footprint + trace panel.

    No matplotlib redraw per frame (that lag is what killed the old scrubber): the
    movie is LUT-colourized with static rings on the picked cells, the right panel
    (footprint thumbnails + traces) is rendered ONCE, and per frame only a vertical
    time-cursor column is repainted.
    """
    mov_rgb = _colorize_with_rings(movie, gt, picks, colors, vmax, downsample)
    n, h, wm = mov_rgb.shape[:3]

    # right panel: footprint thumbnail (col 0) + trace (col 1) per cell, drawn once
    rfig = _agg_figure((5.0, h / 100.0))
    gs = rfig.add_gridspec(
        len(picks),
        2,
        width_ratios=[0.5, 4],
        wspace=0.08,
        hspace=0.3,
        left=0.015,
        right=0.985,
        top=0.93,
        bottom=0.13,
    )
    axts = []
    for i, u in enumerate(picks):
        # optical-center frame -> FOV pixel: the axis (0, 0) is the movie center.
        cy = (movie.shape[1] - 1) / 2.0 + gt.centers_um[u, 1] / px_um
        cx = (movie.shape[2] - 1) / 2.0 + gt.centers_um[u, 2] / px_um
        hw = 26
        y0, y1 = max(int(cy) - hw, 0), min(int(cy) + hw, movie.shape[1])
        x0, x1 = max(int(cx) - hw, 0), min(int(cx) + hw, movie.shape[2])
        axf = rfig.add_subplot(gs[i, 0])
        axf.imshow(np.asarray(gt.A_observed[u])[y0:y1, x0:x1], cmap=GCAMP)
        axf.set_xticks([])
        axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_color(colors[i])
            sp.set_linewidth(2.2)
        axt = rfig.add_subplot(gs[i, 1])
        axt.plot(t, np.asarray(gt.C[u]), color=colors[i], lw=0.9)
        axt.set_xlim(t[0], t[-1])
        axt.set_yticks([])
        axt.text(
            0.99,
            0.84,
            f"z={gt.centers_um[u, 0]:.0f}um",
            transform=axt.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="0.5",
        )
        axt.set_xlabel("time (s)", fontsize=9) if i == len(
            picks
        ) - 1 else axt.set_xticklabels([])
        if i == 0:
            axt.set_title("calcium traces C (cursor = current frame)", fontsize=9)
        axts.append(axt)
    right, xpix, row_top, row_bot = _cursor_panel(rfig, axts, t, h)

    frames = np.empty((n, h, wm + right.shape[1], 3), np.uint8)
    frames[:, :, :wm] = mov_rgb
    _paint_right_panel(frames, right, xpix, row_top, row_bot, wm)
    return frames


def _cursor_panel(rfig, axts, t, h):
    """Render a once-drawn matplotlib panel to RGB, height-matched to the movie.

    Returns ``(rgb, xpix, row_top, row_bot)`` so a caller can repaint a vertical
    time-cursor column per frame. Shared by all three dashboards.
    """
    rfig.canvas.draw()
    xpix = axts[0].transData.transform(np.column_stack([t, np.zeros_like(t)]))[:, 0]
    rgb = np.asarray(rfig.canvas.buffer_rgba())[:, :, :3].copy()
    hr, wr = rgb.shape[:2]
    row_top = int(hr - axts[0].get_window_extent().y1)  # display y is bottom-origin
    row_bot = int(hr - axts[-1].get_window_extent().y0)
    plt.close(rfig)
    if hr != h:  # match the movie-panel height so halves hstack
        fy = h / hr
        rgb = ndzoom(rgb, (fy, 1, 1), order=1)
        row_top, row_bot = int(row_top * fy), int(row_bot * fy)
    return rgb, np.clip(xpix.astype(int), 0, wr - 1), row_top, row_bot


def _paint_right_panel(frames, right, xpix, row_top, row_bot, x_off):
    """Place the once-rendered ``right`` panel at column ``x_off`` in every frame,
    then repaint only the vertical time-cursor column per frame.

    The static panel is broadcast into the output buffer once; per frame we touch
    just the thin cursor column directly in ``frames`` (no per-frame panel copy).
    Shared by all three dashboards.
    """
    frames[:, :, x_off : x_off + right.shape[1]] = right
    for k in range(len(frames)):
        frames[
            k, row_top:row_bot, x_off + max(xpix[k] - 1, 0) : x_off + xpix[k] + 1
        ] = (80, 80, 80)


def build_components_frames(spatial, temporal, population, t, downsample=2):
    """The background's OWN A.C, in colour: each component mapped to one RGB channel.

    The left movie's channel ``k = S_k(y,x) . T_k[t]``, so you read each component's
    spatial extent by hue and watch its channel brighten as its ``T_k`` rises
    (overlaps mix: red+green -> yellow). Same low-rank form as the cells, just
    smooth and diffuse. Right: the shared population driver ``P(t)`` on top, then
    each ``S_k`` thumbnail (tinted in its channel colour) next to its ``T_k`` trace.
    """
    nk = min(3, spatial.shape[0])
    s = np.ascontiguousarray(spatial[:nk, ::downsample, ::downsample]).astype(
        np.float32
    )
    tt = temporal[:nk].astype(np.float32)
    n, (h, wm) = tt.shape[1], s.shape[1:]
    chan = np.einsum("khw,kn->nhwk", s, tt)  # (n, h, w, nk): channel k = S_k . T_k[t]
    if nk < 3:
        chan = np.concatenate([chan, np.zeros((n, h, wm, 3 - nk), np.float32)], axis=-1)
    vmax = float(np.percentile(chan, 99.5)) + 1e-9
    left = (np.clip(chan / vmax, 0, 1) * 255).astype(np.uint8)  # (n, h, w, 3)

    rfig = _agg_figure((5.4, h / 100.0))
    gs = rfig.add_gridspec(
        nk + 1,
        2,
        width_ratios=[0.55, 4],
        hspace=0.3,
        wspace=0.08,
        left=0.01,
        right=0.985,
        top=0.92,
        bottom=0.12,
    )
    axp = rfig.add_subplot(gs[0, 1])
    axp.plot(t, population, color="0.15", lw=1.3)
    axp.set_xlim(t[0], t[-1])
    axp.set_yticks([])
    axp.set_xticklabels([])
    axp.set_title("population activity $P(t)$ drives the components below", fontsize=9)
    axts = [axp]
    for k in range(nk):
        axf = rfig.add_subplot(gs[k + 1, 0])
        axf.imshow(spatial[k], cmap=_RGB_CMAPS[k])
        axf.set_xticks([])
        axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_color(_RGB_TRACE[k])
            sp.set_linewidth(2.4)
        axt = rfig.add_subplot(gs[k + 1, 1])
        axt.plot(t, temporal[k], color=_RGB_TRACE[k], lw=0.9)
        axt.set_xlim(t[0], t[-1])
        axt.set_yticks([])
        axt.set_xlabel("time (s)", fontsize=9) if k == nk - 1 else axt.set_xticklabels(
            []
        )
        if k == 0:
            axt.set_title(
                "component $k$:  smooth field $S_k$  x  envelope $T_k$", fontsize=9
            )
        axts.append(axt)
    right, xpix, row_top, row_bot = _cursor_panel(rfig, axts, t, h)

    frames = np.empty((n, h, wm + right.shape[1], 3), np.uint8)
    frames[:, :, :wm] = left
    _paint_right_panel(frames, right, xpix, row_top, row_bot, wm)
    return frames


def build_neuropil_frames(
    clean, withbg, gt, picks, colors, t, vmax, px_um, downsample=2
):
    """The "add it in" reveal: clean render | render+neuropil, with naive-ROI traces.

    Both movies are LUT-colourized at a shared vmax with static rings on the picked
    cells; the trace panel shows, per cell, a NAIVE footprint-ROI mean
    (:func:`minisim.metrics.footprint_roi_trace`) of the rendered movie without vs
    with the haze. These ROI means are NOT the true C: they already fold in
    neighbour bleed; +haze adds the neuropil pedestal on top. Separating the true C
    from this mixture is demixing (final stage).
    """
    left = _colorize_with_rings(clean, gt, picks, colors, vmax, downsample)
    mid = _colorize_with_rings(withbg, gt, picks, colors, vmax, downsample)
    n, h, wm = left.shape[:3]

    rfig = _agg_figure((5.2, h / 100.0))
    gs = rfig.add_gridspec(
        len(picks), 1, hspace=0.3, left=0.02, right=0.86, top=0.93, bottom=0.13
    )
    axts = []
    for i, u in enumerate(picks):
        a = np.asarray(gt.A_observed[u])
        ax = rfig.add_subplot(gs[i, 0])
        ax.plot(
            t,
            footprint_roi_trace(withbg, a),
            color="0.55",
            lw=0.9,
            label="ROI +neuropil",
        )
        ax.plot(
            t,
            footprint_roi_trace(clean, a),
            color=colors[i],
            lw=1.0,
            label="ROI no haze",
        )
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([])
        ax.set_xlabel("time (s)", fontsize=9) if i == len(
            picks
        ) - 1 else ax.set_xticklabels([])
        if i == 0:
            ax.set_title(
                "naive footprint-ROI mean (not the true $C$): no haze vs +neuropil",
                fontsize=8.5,
            )
            ax.legend(fontsize=6, loc="upper right", framealpha=0.6)
        axts.append(ax)
    right, xpix, row_top, row_bot = _cursor_panel(rfig, axts, t, h)

    div = 2  # thin divider between the two movie panels
    frames = np.empty((n, h, wm + div + wm + right.shape[1], 3), np.uint8)
    frames[:, :, :wm] = left
    frames[:, :, wm : wm + div] = 60
    frames[:, :, wm + div : wm + div + wm] = mid
    _paint_right_panel(frames, right, xpix, row_top, row_bot, wm + div + wm)
    return frames


def build_naive_overlay_frames(
    movie,
    gt,
    picks,
    colors,
    t,
    vmax,
    px_um,
    naive_traces,
    own_traces,
    downsample=1,
    target_h=300,
):
    """Movie | per-cell (footprint + naive ROI vs the cell's own light) with a time cursor.

    The NB02 sibling of :func:`build_dashboard_frames`: instead of one true-``C`` trace
    per cell it overlays the naive footprint-ROI read-out (solid, the cell's colour)
    against the cell's *own light* (dashed black) so the cross-talk reads directly as the
    cursor sweeps - the naive trace bumps up on a neighbour's transient while the own-light
    line stays flat. ``naive_traces`` and ``own_traces`` are ``(len(picks), frames)`` arrays
    in the same units (sensor counts). The drawing machinery (rings, once-rendered panel,
    per-frame cursor column) is shared with the other dashboards.

    This notebook's recording is small (100 px) for speed, so the colourized movie is
    nearest-neighbour upscaled to ~``target_h`` px before the trace panel is height-matched
    to it - otherwise the frame degenerates into a short, very wide strip.
    """
    mov_rgb = _colorize_with_rings(movie, gt, picks, colors, vmax, downsample)
    scale = max(
        1, round(target_h / mov_rgb.shape[1])
    )  # crisp integer upscale (keep pixels sharp)
    if scale > 1:
        mov_rgb = np.repeat(np.repeat(mov_rgb, scale, axis=1), scale, axis=2)
    n, h, wm = mov_rgb.shape[:3]

    rfig = _agg_figure((5.4, h / 100.0))
    gs = rfig.add_gridspec(
        len(picks),
        2,
        width_ratios=[0.5, 4],
        wspace=0.08,
        hspace=0.3,
        left=0.015,
        right=0.985,
        top=0.92,
        bottom=0.13,
    )
    axts = []
    for i, u in enumerate(picks):
        # optical-center frame -> FOV pixel: the axis (0, 0) is the movie center.
        cy = (movie.shape[1] - 1) / 2.0 + gt.centers_um[u, 1] / px_um
        cx = (movie.shape[2] - 1) / 2.0 + gt.centers_um[u, 2] / px_um
        hw = 26
        y0, y1 = max(int(cy) - hw, 0), min(int(cy) + hw, movie.shape[1])
        x0, x1 = max(int(cx) - hw, 0), min(int(cx) + hw, movie.shape[2])
        axf = rfig.add_subplot(gs[i, 0])
        axf.imshow(np.asarray(gt.A_observed[u])[y0:y1, x0:x1], cmap=GCAMP)
        axf.set_xticks([])
        axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_color(colors[i])
            sp.set_linewidth(2.2)
        axt = rfig.add_subplot(gs[i, 1])
        axt.plot(t, naive_traces[i], color=colors[i], lw=0.9, label="naive ROI")
        axt.plot(t, own_traces[i], color="k", lw=0.9, ls="--", label="own light")
        axt.set_xlim(t[0], t[-1])
        axt.set_yticks([])
        axt.set_xlabel("time (s)", fontsize=9) if i == len(
            picks
        ) - 1 else axt.set_xticklabels([])
        if i == 0:
            axt.set_title(
                "naive ROI (solid) vs the cell's own light (dashed); gap = cross-talk",
                fontsize=8.5,
            )
            axt.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.6)
        axts.append(axt)
    right, xpix, row_top, row_bot = _cursor_panel(rfig, axts, t, h)

    frames = np.empty((n, h, wm + right.shape[1], 3), np.uint8)
    frames[:, :, :wm] = mov_rgb
    _paint_right_panel(frames, right, xpix, row_top, row_bot, wm)
    return frames


def interactive_panel(sliders, draw, canvas, ncols=2):
    """Wire every slider to redraw the SAME persistent canvas in place.

    ``draw`` reads the slider values itself and mutates the figure; we never
    re-display, which keeps redraws smooth and sidesteps VS Code's duplicate-output
    bug (no Output widget / re-display). If plots ever duplicate after reopening a
    notebook, run Command Palette -> "Developer: Reload Window".

    The canvas is drawn *after* ``display(canvas)``: an ipympl figure rendered
    before its view is attached (the natural order, draw-then-display) never pushes
    that first frame to the frontend, so in browser JupyterLab the panel comes up
    blank until you nudge a slider. Drawing once the view is live paints it
    immediately.
    """
    for s in sliders.values():
        if hasattr(
            s, "continuous_update"
        ):  # sliders only; some widgets lack this trait
            s.continuous_update = False
        s.style.description_width = "104px"
        s.layout.width = "340px"
    for s in sliders.values():
        s.observe(lambda _change: draw(), names="value")
    vals = list(sliders.values())
    per = -(-len(vals) // ncols)  # ceil
    display(HBox([VBox(vals[i * per : (i + 1) * per]) for i in range(ncols)]))
    display(canvas)
    draw()  # after the view is attached, so ipympl pushes the first frame


def show(fig):
    """Display a built *static* figure as an inline PNG.

    The figure-builder panels (e.g. :func:`optics_reveal_figure`) are static - they
    draw once at construction and are never interacted with. Under
    ``%matplotlib widget`` they are nonetheless ipympl figures, and ipympl does not
    reliably push the first frame to a browser JupyterLab view: a canvas drawn
    before its view attaches comes up blank, and even a post-attach draw is racy
    deep in a long notebook (and blanks again when windowing scrolls it off and
    back). Rasterizing to PNG and displaying that sidesteps the live canvas
    entirely, so the panel always paints, on every frontend. Close the figure so it
    does not linger as an unshown widget.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.get_dpi(), bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))

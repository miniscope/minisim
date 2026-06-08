"""Notebook-1-specific figure choreography for the anatomy notebook.

This is the bespoke teaching code that is *not* reusable across notebooks: the
Stage-1 "how miniscope imaging works" sandbox - a side-view scope schematic and a
disconnected five-cell optics demo. It ships beside ``01_anatomy.ipynb`` (copied
out by ``minisim-notebooks``) and is imported by the notebook as a sibling module,
so the Stage-1 cell stays a handful of physics knobs instead of ~130 lines of
matplotlib.

Two tiers, on purpose: the genuinely reusable, data-model-keyed plotters (the
A.C dashboards, ``plot_snr_vs_radius``, the GCaMP LUT) live in
:mod:`minisim.notebooks._support` so a later pipeline notebook can import them;
this file holds only the choreography unique to notebook 1. The physics is real
minisim throughout (:class:`~minisim.Optics` / :class:`~minisim.Tissue` /
:class:`~minisim.ImageSensor` and :func:`~minisim.steps.degrade_footprint`); only
the hand-placed five-cell layout and the schematic are illustrative.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import zoom
from scipy.signal import welch

from minisim import (
    ImageSensor,
    Optics,
    PlaceNeurons,
    Sensor,
    Tissue,
    detection_snr,
    sample_field_at,
)
from minisim.footprint import Footprint
from minisim.notebooks._support import (
    GCAMP,
    plot_count_histogram,
    plot_population,
    plot_snr_vs_radius,
    plot_traces,
)
from minisim.recording import DETECT_SNR_THRESHOLD
from minisim.scene import Scene
from minisim.steps import (
    bleaching_floor,
    bleaching_pool,
    calcium_kernel,
    dark_recovery,
    degrade_footprint,
    neuron_footprint,
    sample_neurons,
    smooth_spatial_field,
)
from minisim.steps.sensor import falloff_center_px, radial_falloff, radius_grid

# Sandbox geometry / tissue constants (a Miniscope-V4-like scope imaging GCaMP).
WD_UM = 700.0          # nominal front working distance
CHIP_UM = 900.0        # sandbox sensor chip extent
SOMA_UM, EMISSION_NM, BLUR_PER_UM = 6.0, 525.0, 0.05
# Round-trip scatter, asymmetric: the excitation leg (~470 nm in) is a diffuse
# widefield fluence that penetrates far (long MFP, barely dims); the emission leg
# (~525 nm out) is the image-forming signal at the scattering MFP and dominates.
# Effective MFP ~86 um.
MFP_EX_UM, MFP_EM_UM = 600.0, 100.0
N_TISSUE = 1.33        # tissue refractive index (DOF formula)
PX_REF_UM = 5.0 / 4.0  # = default pitch/mag (1.25 um/px); per-pixel light ~ (px_um/this)^2

# Five cells spanning a depth band around the 700 um focal anchor, at distinct
# lateral spots so they never overlap. 1 = shallowest, 5 = deepest.
CELLS = (
    {"name": "1", "y": -40.0, "x": -75.0, "offset": -50.0, "color": "#4daf4a"},
    {"name": "2", "y": 38.0, "x": -30.0, "offset": -25.0, "color": "#1f9e89"},
    {"name": "3", "y": -8.0, "x": 8.0, "offset": 0.0, "color": "#377eb8"},
    {"name": "4", "y": 40.0, "x": 48.0, "offset": 27.0, "color": "#e6a700"},
    {"name": "5", "y": -34.0, "x": 80.0, "offset": 55.0, "color": "#e41a1c"},
)

# 0.5 um/px reference grid: finer than the sensor ever samples (object-space pixel
# ~1.25 um at the defaults) and on par with the diffraction-limited spot the optics
# can't beat anyway -- a 1p miniscope is pixel-limited, never diffraction-limited --
# so the reference holds the "true" cell shape without ever being the bottleneck.
REF_PX_UM = 0.5
PATCH_HALF_UM = 64.0  # half-extent of each cell's patch (covers soma + dendrites)


def _dof_half_um(na: float) -> float:
    """In-focus half-depth; textbook DOF ~ n*lambda/NA^2, shrinks as NA rises."""
    return N_TISSUE * (EMISSION_NM / 1000.0) / na**2


def _add_centered(canvas: np.ndarray, patch: np.ndarray, cy: float, cx: float) -> None:
    """Accumulate ``patch`` into ``canvas`` centred at ``(cy, cx)``, clipping the edge."""
    ph, pw = patch.shape
    y0, x0 = int(round(cy - (ph - 1) / 2.0)), int(round(cx - (pw - 1) / 2.0))
    ys0, xs0 = max(y0, 0), max(x0, 0)
    ys1, xs1 = min(y0 + ph, canvas.shape[0]), min(x0 + pw, canvas.shape[1])
    if ys0 < ys1 and xs0 < xs1:
        canvas[ys0:ys1, xs0:xs1] += patch[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0]


class ImagingSandbox:
    """Stage-1 sandbox: real optics/tissue/sensor math on five hand-placed cells.

    A teaching *diagram*, deliberately DISCONNECTED from the committed recording:
    it owns a two-panel figure (left = side-view schematic of scope -> tissue ->
    cells with the focal surface + depth of field; right = what the image sensor
    sees) and redraws it from a slider dict. Construct it with the GCaMP variant and
    your sliders, then wire :attr:`draw` / :attr:`canvas` into
    ``interactive_panel``. Each redraw runs the simulator's real :class:`Optics`,
    :class:`Tissue`, :func:`degrade_footprint`, and
    :meth:`ImageSensor.photons_to_counts` on the five cells.

    The cell shapes are generated ONCE on a fine um grid (so a mag/pitch change only
    rescales them, never re-randomizes), keyed to the GCaMP variant; the slider knobs
    are NA / magnification / pixel pitch / tissue thickness / focus offset / exposure /
    read noise / field-curvature radius.
    """

    def __init__(
        self,
        sliders,
        *,
        morphology: str = "cytosolic",
        n_dendrites: int = 4,
        dendrite_len_um: float = 45.0,
        dendrite_width_um: float = 3.0,
    ) -> None:
        self.sliders = sliders
        self._patches = self._build_ref_patches(
            morphology, n_dendrites, dendrite_len_um, dendrite_width_um
        )
        # Build the figure ONCE; the fixed-range (0-255) colorbar is created once so
        # it never accumulates across redraws.
        self.fig, (self._axL, self._axR) = plt.subplots(1, 2, figsize=(11.5, 4.8))
        self.fig.subplots_adjust(left=0.07, right=0.87, wspace=0.30, top=0.84, bottom=0.12)
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        cax = self.fig.add_axes([0.89, 0.12, 0.015, 0.72])
        sm = ScalarMappable(norm=Normalize(0, 255), cmap=GCAMP)
        sm.set_array([])
        self.fig.colorbar(sm, cax=cax, label="ADC counts (8-bit)")

    @property
    def canvas(self):
        """The persistent figure canvas (hand this to ``interactive_panel``)."""
        return self.fig.canvas

    @staticmethod
    def _build_ref_patches(morphology, n_dendrites, dendrite_len_um, dendrite_width_um):
        """The five sharp reference footprints on the fixed fine grid (one per cell)."""
        rng = np.random.default_rng(0)  # fixed seed: identical shapes every build
        n = int(round(2 * PATCH_HALF_UM / REF_PX_UM))
        c = (n - 1) / 2.0  # each cell sits at the center of its own patch
        return [
            neuron_footprint(
                (n, n), (c, c), SOMA_UM / REF_PX_UM, 0.35, rng,
                morphology=morphology, n_dendrites=n_dendrites,
                dendrite_length_px=dendrite_len_um / REF_PX_UM,
                dendrite_width_px=dendrite_width_um / REF_PX_UM,
            )
            for _ in CELLS
        ]

    def _image_cells(self, na, magnification, pixel_pitch_um, tissue_thickness_um,
                     focus_offset_um, exposure, read_noise_e, field_curv_mm):
        """Image the five cells with the simulator's real optics/tissue/sensor."""
        optics = Optics(na=na, magnification=magnification, emission_nm=EMISSION_NM,
                        field_curvature_radius_um=field_curv_mm * 1000.0)
        tissue = Tissue(scatter_mfp_excitation_um=MFP_EX_UM, scatter_mfp_emission_um=MFP_EM_UM,
                        scatter_blur_per_um=BLUR_PER_UM)
        px_um = pixel_pitch_um / magnification
        n_px = int(np.clip(round(CHIP_UM / pixel_pitch_um), 48, 512))
        sensor = ImageSensor(n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=pixel_pitch_um,
                             quantum_efficiency=0.7, read_noise_e=read_noise_e,
                             gain_adu_per_e=1.0, bit_depth=8)
        c = (n_px - 1) / 2.0
        focal_dist = WD_UM + focus_offset_um
        dof = _dof_half_um(na)
        optical = np.zeros((n_px, n_px))
        info = []
        for cell, patch in zip(CELLS, self._patches, strict=True):
            dist = WD_UM + cell["offset"]                          # distance from scope
            path = max(tissue_thickness_um + cell["offset"], 0.0)  # tissue the light crosses
            # field curvature: off-axis cells focus shallower (no field flattener), so
            # each cell sees its own focal depth set by its radius from the optical axis
            r = np.hypot(cell["y"], cell["x"])
            focal_eff = focal_dist - optics.focal_curvature_shift_um(r)
            # the fixed shape, rescaled to the current pixel size (same cell, zoomed)
            sharp = np.clip(zoom(patch, REF_PX_UM / px_um, order=1), 0.0, 1.0)
            cy, cx = c + cell["y"] / px_um, c + cell["x"] / px_um
            placed = np.zeros((n_px, n_px))
            _add_centered(placed, sharp, cy, cx)
            defocus = optics.defocus_sigma_um(dist, focal_eff)
            sigma_um = np.hypot(optics.diffraction_sigma_um,
                                np.hypot(tissue.scatter_sigma_um(path), defocus))
            gain = tissue.attenuation(path) * optics.collection_efficiency
            optical += degrade_footprint(Footprint.from_dense(placed), sigma_um / px_um, gain).to_dense()
            info.append({**cell, "dist": dist, "in_focus": abs(dist - focal_eff) <= dof})
        # per-pixel light: a pixel integrates flux over its object area px_um^2 =
        # (pitch/mag)^2, so finer pitch OR higher mag dims each pixel (normalized to
        # the default px size, so the default state is unchanged).
        rng = np.random.default_rng(1)  # sensor shot/read noise (stable across redraws)
        counts = sensor.photons_to_counts(optical * exposure * (px_um / PX_REF_UM) ** 2, rng)
        meta = dict(px_um=px_um, n_px=n_px, fov_um=n_px * px_um, focal_dist=focal_dist,
                    surf=WD_UM - tissue_thickness_um, dof=dof, optics=optics)
        return counts, info, meta

    def draw(self) -> None:
        """Redraw both panels from the current slider values (no-arg, for the widget)."""
        v = {k: s.value for k, s in self.sliders.items()}
        counts, info, meta = self._image_cells(**v)
        axL, axR = self._axL, self._axR
        axL.clear()
        axR.clear()
        # -- left: side view (depth = distance from scope, 0 at top) --
        axL.set_xlim(-150, 150)
        axL.set_ylim(-90, 900)
        axL.invert_yaxis()
        axL.add_patch(Rectangle((-55, -85), 110, 70, color="0.25"))
        axL.text(0, -50, "miniscope", color="w", ha="center", va="center", fontsize=9)
        axL.axhline(0, color="0.25", lw=1.5)
        surf = meta["surf"]
        axL.add_patch(Rectangle((-150, surf), 300, 900 - surf, color="#f0c9a8", alpha=0.55, zorder=0))
        axL.text(-145, surf + 22, "tissue", color="#a0522d", fontsize=9, va="top")
        axL.axhline(surf, color="#a0522d", ls=":", lw=1)
        axL.add_patch(FancyArrowPatch((-120, 0), (-120, WD_UM), arrowstyle="<->",
                                      mutation_scale=10, color="0.4", lw=1))
        axL.text(-128, WD_UM / 2, "WD 700 um", rotation=90, ha="right", va="center",
                 color="0.4", fontsize=8)
        dof = meta["dof"]
        # field curvature: the in-focus surface is a shallow bowl (edges focus
        # shallower), not a flat plane. Draw its cross-section + the DOF band around it.
        xg = np.linspace(-150, 150, 121)
        opt = meta["optics"]
        fcurve = meta["focal_dist"] - np.array([opt.focal_curvature_shift_um(abs(x)) for x in xg])
        axL.fill_between(xg, fcurve - dof, fcurve + dof, color="#1f6fb2", alpha=0.16, zorder=1)
        axL.plot(xg, fcurve, color="#1f6fb2", ls="--", lw=1.4, zorder=2)
        axL.text(145, fcurve[-1], f"focal surface\n+/-{dof:.0f} um DOF", color="#1f6fb2",
                 ha="right", va="top", fontsize=8)
        for ci in info:  # id number in each dot (matches sensor panel); in-focus = white ring
            axL.scatter(ci["x"], ci["dist"], s=300, color=ci["color"], zorder=5,
                        edgecolor=("white" if ci["in_focus"] else "0.2"), linewidth=2.0)
            axL.text(ci["x"], ci["dist"], ci["name"], color="white", ha="center", va="center",
                     fontsize=9, weight="bold", zorder=6)
        axL.set(xlabel="lateral (um)", ylabel="distance from scope (um)",
                title="side view: scope -> tissue -> cells")
        # -- right: what the image sensor sees (colorbar is the fixed axis built once) --
        axR.imshow(counts, cmap=GCAMP, vmin=0, vmax=255, interpolation="nearest")
        for ci in info:
            cy = (meta["n_px"] - 1) / 2 + ci["y"] / meta["px_um"]
            cx = (meta["n_px"] - 1) / 2 + ci["x"] / meta["px_um"]
            if 0 <= cx < meta["n_px"] and 0 <= cy < meta["n_px"]:  # skip cells off the FOV
                axR.text(cx, cy - SOMA_UM / meta["px_um"] - 4, ci["name"], color=ci["color"],
                         ha="center", fontsize=8, weight="bold")
        axR.set(title=f"image sensor: {meta['n_px']}x{meta['n_px']} px  |  {meta['px_um']:.2f} um/px"
                      f"  |  FOV {meta['fov_um']:.0f} um\nNA {v['na']:g}: brightness NA^2="
                      f"{meta['optics'].collection_efficiency:.3f}, diffraction sigma "
                      f"{meta['optics'].diffraction_sigma_um * 1000:.0f} nm",
                xlabel="sensor px", ylabel="sensor px")
        self.fig.canvas.draw_idle()


class PlacementView:
    """Stage-2 placement explorer: top-down + side scatter of the sampled cell bodies.

    Owns the figure (top view, side view, depth colorbar); :attr:`draw` reads the slider
    dict and the GCaMP toggle, samples centres with :func:`~minisim.steps.sample_neurons`
    (the placement half of ``place_neurons`` - positions only, no footprints, so hundreds
    of cells redraw instantly), and renders via :func:`~minisim.notebooks._support.plot_population`.
    """

    DEPTH_MAX = 300.0  # fixed depth color/axis scale, so nothing jumps as you drag

    def __init__(self, sliders, morph_toggle, *, fov_um, seed):
        self.sliders = sliders
        self.morph = morph_toggle
        self.fov_um = fov_um
        self.seed = seed
        self.fig = plt.figure(figsize=(9.5, 6.6))
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1.5],
                                   hspace=0.32, left=0.1, right=0.86, top=0.92, bottom=0.1)
        self._ax_top = self.fig.add_subplot(gs[0, 0])
        self._ax_side = self.fig.add_subplot(gs[1, 0], sharex=self._ax_top)
        cax = self.fig.add_axes([0.88, 0.56, 0.015, 0.34])
        sm = ScalarMappable(norm=Normalize(0, self.DEPTH_MAX), cmap="viridis")
        sm.set_array([])
        self.fig.colorbar(sm, cax=cax, label="depth z (um)")

    @property
    def canvas(self):
        """The persistent figure canvas (hand this to ``interactive_panel``)."""
        return self.fig.canvas

    def draw(self):
        """Sample centres from the current sliders and redraw the top + side views."""
        v = {k: s.value for k, s in self.sliders.items()}
        depth_range = (v["depth_lo"], max(v["depth_hi"], v["depth_lo"]))
        spec = PlaceNeurons(density_per_mm3=v["density_per_mm3"], soma_radius_um=v["soma_radius_um"],
                            depth_range_um=depth_range, min_distance_um=v["min_distance_um"])
        fov_h, fov_w = self.fov_um
        centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(self.seed))
        plot_population(self._ax_top, self._ax_side, centers, self.fov_um,
                        depth_max=self.DEPTH_MAX, soma_radius_um=v["soma_radius_um"],
                        depth_range=depth_range, morph_label=self.morph.value)
        self.fig.canvas.draw_idle()


def _crop(a, cy, cx, half=22):
    h, w = a.shape
    return a[max(int(cy - half), 0) : min(int(cy + half), h),
             max(int(cx - half), 0) : min(int(cx + half), w)]


def optics_reveal_figure(gt, *, px_um, dof_um, morph_label=""):
    """Stage-2 reveal: the planted vs optically-degraded footprints, A_planted -> A_observed.

    Top row = full-FOV max-projections, each scaled to its own peak, so it reads as
    *shape* (which cells stay sharp), not brightness; bottom row = one cell up close,
    isolating the blur the optics add. The suptitle reports the true dimming (observed
    peak as a fraction of planted), the resolved "auto" focus depth, and the in-focus
    count. ``gt`` is the :class:`~minisim.GroundTruth` of a ``place_neurons`` + ``optics``
    build. Returns the figure (the caller displays its canvas).
    """
    a_planted, a_observed = gt.A_planted, gt.A_observed
    peak_ratio = float(a_observed.max() / a_planted.max())
    i_cell = int(np.argmax(a_observed.reshape(a_observed.shape[0], -1).max(axis=1)))
    cy, cx = gt.centers_um[i_cell, 1:] / px_um
    focal_um = float(gt.focal_depth_um)
    fig = plt.figure(figsize=(10, 7.6))
    if hasattr(fig.canvas, "header_visible"):
        fig.canvas.header_visible = False
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.5], hspace=0.28, wspace=0.12,
                          left=0.04, right=0.96, top=0.9, bottom=0.04)
    for col, (img, ttl) in enumerate([
            (a_planted.max(0), "A_planted - ground truth (sharp, full brightness)"),
            (a_observed.max(0), f"A_observed - through the optics (peak {peak_ratio * 100:.1f}% as bright)")]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap=GCAMP, vmin=0, vmax=(img.max() or 1.0), origin="lower")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for col, (img, ttl) in enumerate([
            (_crop(a_planted[i_cell], cy, cx), "one cell, planted (sharp)"),
            (_crop(a_observed[i_cell], cy, cx), "same cell, observed (blurred + dimmed)")]):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img, cmap=GCAMP, vmin=0, vmax=(img.max() or 1.0), origin="lower")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{gt.n_units} cells ({morph_label} GCaMP) | focus auto @ {focal_um:.0f} um "
                 f"(deeper than the {np.median(gt.depth_um):.0f} um median depth - field curvature) "
                 f"| DOF +/-{dof_um:.1f} um | in focus {int(gt.in_focus.sum())}/{gt.n_units}", fontsize=10)
    return fig


class ActivityPanel:
    """Stage-3 calcium-activity explorer: example traces + spikes, the kernel, the brightness spread.

    :attr:`draw` maps the sliders to a :class:`~minisim.CellActivity` via the notebook's
    ``activity_from_sliders`` (kept in the cell because the commit reuses it), previews a
    short ``cell_activity`` build, and renders three panels: the busiest traces with spike
    ticks (:func:`~minisim.notebooks._support.plot_traces`), the double-exponential kernel,
    and the per-cell brightness histogram. ``preview`` is the notebook's small-FOV builder.
    """

    def __init__(self, sliders, preview, activity_from_sliders):
        self.sliders = sliders
        self.preview = preview
        self.activity_from_sliders = activity_from_sliders
        self.fig = plt.figure(figsize=(10.5, 4.6))
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        gs = self.fig.add_gridspec(2, 2, width_ratios=[3, 1], hspace=0.55, wspace=0.3,
                                   left=0.07, right=0.97, top=0.9, bottom=0.14)
        self._ax_tr = self.fig.add_subplot(gs[:, 0])
        self._ax_k = self.fig.add_subplot(gs[0, 1])
        self._ax_b = self.fig.add_subplot(gs[1, 1])

    @property
    def canvas(self):
        """The persistent figure canvas (hand this to ``interactive_panel``)."""
        return self.fig.canvas

    def draw(self):
        """Preview a short activity build from the current sliders and redraw all three panels."""
        v = {k: s.value for k, s in self.sliders.items()}
        step3, tau_r, tau_d = self.activity_from_sliders(v)
        rec = self.preview(extra=step3, until="cell_activity", duration_s=100.0)
        g = rec.ground_truth
        t = np.arange(rec.spec.acquisition.n_frames) / rec.spec.acquisition.fps
        plot_traces(self._ax_tr, t, g.C, spikes=g.S, n=5)
        # the kernel: one transient's shape, drawn finely out to ~5*tau_d (independent of
        # the recording's frame rate). Dotted line = time-to-peak; grey line = half max.
        self._ax_k.clear()
        kfps = 200.0
        kk = calcium_kernel(tau_r, tau_d, kfps)
        tk = np.arange(len(kk)) / kfps
        self._ax_k.plot(tk, kk, color="#21a366", lw=1.5)
        self._ax_k.axvline(v["kernel_peak"], color="0.5", ls=":", lw=0.9)
        self._ax_k.axhline(0.5, color="0.8", lw=0.7, zorder=0)
        self._ax_k.set_title(f"kernel\n(τr {tau_r * 1000:.0f}, τd {tau_d * 1000:.0f} ms)", fontsize=9)
        self._ax_k.set_xlabel("t (s)")
        self._ax_k.set_yticks([0, 0.5, 1])
        # per-cell brightness gain: wide spread = a few bright cells over a dim majority.
        self._ax_b.clear()
        if g.n_units:
            self._ax_b.hist(g.amplitude_per_cell, bins=18, color="#4c72b0")
        self._ax_b.set(title="per-cell brightness", xlabel="gain", ylabel="cells")
        self.fig.canvas.draw_idle()


class BleachingSandbox:
    """Stage-6 photobleaching planning sandbox - DISCONNECTED, like Stage 1.

    Two panels from real bleaching physics (:func:`~minisim.steps.bleaching_pool` /
    :func:`~minisim.steps.bleaching_floor` / :func:`~minisim.steps.dark_recovery`):
    within-session decay toward the continuous-imaging floor ``B*``, and across-session
    recovery-or-ratchet over repeated dark gaps. Calibrated to CA1 GCaMP6f, but the exact
    rates are order-of-magnitude (see the notebook's grain-of-salt note). Light is
    dimensionless (1 = a typical continuous level).
    """

    Q = 6.3e-6  # bleach rate per s at unit light, baseline emission

    def __init__(self, sliders):
        self.sliders = sliders
        self.fig = plt.figure(figsize=(11, 4.2))
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        gs = self.fig.add_gridspec(1, 2, wspace=0.28, left=0.08, right=0.97, top=0.85, bottom=0.16)
        self._ax_a = self.fig.add_subplot(gs[0, 0])
        self._ax_b = self.fig.add_subplot(gs[0, 1])

    @property
    def canvas(self):
        """The persistent figure canvas (hand this to ``interactive_panel``)."""
        return self.fig.canvas

    @staticmethod
    def _emission(activity):
        return 1.0 + activity  # mean emission c: 1 (quiet, = the calibration limit) .. 2

    def _within(self, light, activity, turnover_h, minutes, dt=2.0):
        c = self._emission(activity)
        b = bleaching_pool(np.full(int(minutes * 60 / dt), c), self.Q * dt, turnover_h * 3600.0 / dt, light)
        b_star = bleaching_floor(self.Q, light, c, turnover_h * 3600.0)  # continuous-imaging floor
        return np.arange(b.size) * dt / 60.0, b, b_star

    def _across(self, light, activity, turnover_h, minutes, gap_days, n_sessions=5, dt=6.0):
        tau_s = turnover_h * 3600.0
        days, vals, b0, clock = [], [], 1.0, 0.0
        for _ in range(n_sessions):
            b = bleaching_pool(np.full(int(minutes * 60 / dt), self._emission(activity)),
                               self.Q * dt, tau_s / dt, light, b0=b0)
            days.append((clock + np.arange(b.size) * dt) / 86400.0)
            vals.append(b)
            clock += b.size * dt
            gap = np.linspace(0, gap_days * 86400.0, 60)  # dark gap: pure turnover recovery
            rec = dark_recovery(b[-1], gap, tau_s)
            days.append((clock + gap) / 86400.0)
            vals.append(rec)
            b0 = rec[-1]
            clock += gap_days * 86400.0
        return np.concatenate(days), np.concatenate(vals)

    def draw(self):
        """Redraw the within-session and across-session curves from the current sliders."""
        v = {k: s.value for k, s in self.sliders.items()}
        t, b, b_star = self._within(v["light"], v["activity"], v["turnover_h"], v["recording_min"])
        self._ax_a.clear()
        self._ax_a.plot(t, b, color="#d62728", lw=1.9)
        self._ax_a.axhline(b_star, color="#d62728", lw=0.9, ls=":", alpha=0.6)
        self._ax_a.text(t[-1], b_star, f" floor B*={b_star:.2f}", color="#d62728",
                        va="bottom", ha="right", fontsize=8)
        self._ax_a.set(xlabel="time (min)", ylabel="normalized fluorescence  B(t)",
                       ylim=(max(0.0, min(b_star, b.min()) - 0.03), 1.03))
        self._ax_a.set_title(f"within a session: {(1 - b[-1]) * 100:.0f}% bleached in "
                             f"{v['recording_min']:.0f} min", fontsize=10)
        d, bb = self._across(v["light"], v["activity"], v["turnover_h"], v["recording_min"], v["gap_days"])
        self._ax_b.clear()
        self._ax_b.plot(d, bb, color="#d62728", lw=1.4)
        self._ax_b.set(xlabel="time (days)", ylabel="normalized fluorescence  B(t)",
                       ylim=(max(0.0, bb.min() - 0.03), 1.02),
                       title="repeated imaging: recovery (or ratchet-down) across dark gaps")
        self.fig.canvas.draw_idle()


def bleaching_fade_figure(gt, neuropil_spec, acq, *, seed):
    """Stage-6 'activity-dependent fade': busy cells bleach fastest, so events fall faster than baseline.

    Reads clean ``C``, the per-cell intact-fluorophore envelope ``B`` (``gt.bleaching``),
    ``A_observed``, and the spikes ``S`` off the ground truth of an intensively-lit build.
    Panels: the footprint-weighted field-average baseline (slow, bulk decline) | active-cell
    event amplitudes at true spike frames with their mean-B envelope (steeper) | a begin/end
    frame pair holding the activity FIXED so only bleaching differs (one Stage-5 neuropil
    slice composited in for context). Returns the figure.
    """
    c_trace = np.asarray(gt.C, float)
    b_env = np.asarray(gt.bleaching, float)
    a_obs = np.asarray(gt.A_observed, float)
    spikes = np.asarray(gt.S, float)
    th = np.arange(c_trace.shape[1]) / acq.fps / 3600.0
    # field baseline = footprint-weighted mean intact fraction (slow, bulk decline)
    asum = a_obs.reshape(a_obs.shape[0], -1).sum(axis=1)
    base = (asum @ b_env) / asum.sum()
    # active cells bleach fastest; their mean B is the event-amplitude envelope
    active = [int(u) for u in np.argsort(c_trace.mean(axis=1))[::-1] if gt.detectable[u]][:40]
    env = b_env[active].mean(axis=0)
    pt, pa = [], []  # event amplitudes at true spike frames
    for u in active:
        fr = np.where(spikes[u] > 0)[0][::6]
        if fr.size:
            pt.append(th[fr])
            pa.append(b_env[u, fr] * (c_trace[u, fr] - 1.0))
    pt = np.concatenate(pt)
    pa = np.concatenate(pa)
    pa = pa / np.median(pa[pt < th[-1] * 0.1])
    # begin/end frames: hold the activity FIXED so only bleaching differs.
    pop_b = b_env.mean(axis=0)  # diffuse-background fade, starts at 1
    win = max(1, c_trace.shape[1] // 10)
    tstar = int(np.argmax(c_trace[:, :win].sum(axis=0)))  # a busy early moment
    h, w = acq.image_sensor.n_px_height, acq.image_sensor.n_px_width
    rng = np.random.default_rng(seed)
    haze = neuropil_spec.amplitude * np.mean(
        [smooth_spatial_field((h, w), acq.um_to_px(neuropil_spec.spatial_sigma_um), rng)
         for _ in range(neuropil_spec.n_components)], axis=0)

    def frame(cell_b, bg_b):  # same activity c_trace[:, tstar]
        cells = np.tensordot(c_trace[:, tstar] * cell_b, a_obs, axes=([0], [0]))
        return cells + haze * bg_b

    fb = frame(np.ones(c_trace.shape[0]), 1.0)  # beginning: fresh
    fe = frame(b_env[:, -1], pop_b[-1])         # end: identical activity, bleached
    vm = np.percentile(fb, 99.9)

    fig = plt.figure(figsize=(10.5, 6.6))
    if hasattr(fig.canvas, "header_visible"):
        fig.canvas.header_visible = False
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.4, wspace=0.22,
                          left=0.08, right=0.97, top=0.9, bottom=0.04)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(th, base, color="#ff7f0e", lw=2.2)
    ax1.set(xlabel="time (h)", ylabel="norm. baseline fluor.", ylim=(0.5, 1.03),
            title=f"field-average baseline: -{(1 - base[-1]) * 100:.0f}%")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(pt, pa, s=6, alpha=0.35, color="#2ca02c")
    ax2.plot(th, env, color="k", lw=2.4)
    ax2.set(xlabel="time (h)", ylabel="norm. event amplitude", ylim=(0.0, 2.0),
            title=f"active-cell event amplitudes: -{(1 - env[-1]) * 100:.0f}% (steeper)")
    for j, (img, ttl) in enumerate([(fb, "same moment, fresh"), (fe, "same moment, after bleaching")]):
        axi = fig.add_subplot(gs[1, j])
        axi.imshow(img, cmap=GCAMP, vmin=0, vmax=vm)
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_title(ttl, fontsize=11)
    fig.suptitle("Photobleaching fades active-cell events faster than the baseline", fontsize=12)
    return fig


def motion_diagnostics_figure(traj_fine, fps_fine, traj_long, fps_long, *,
                              amplitude_um, locomotion_freq_hz):
    """Stage-7 brain-motion diagnostics, read straight from the motion generator (no render).

    Three panels from two :func:`~minisim.steps.physical_brain_motion` trajectories (each
    ``(n, 2)`` ``(dy, dx)`` in um): the near-continuous ``traj_fine`` (mostly sloshing, a
    subtle stride rhythm) | the Welch-averaged dominant-axis spectrum of ``traj_long`` (red,
    a modest bump near the stride frequency, with the Nyquist line) | the 2-D per-frame shift
    cloud (Gaussian-like, dense at rest, thinning to the amplitude circle). Returns the figure.
    """
    t_fine = np.arange(traj_fine.shape[0]) / fps_fine
    freq, power = welch(traj_long[:, 0], fs=fps_long, nperseg=512)
    hist = np.histogram2d(traj_long[:, 1], traj_long[:, 0], bins=60, range=[(-12, 12), (-12, 12)])[0]
    fig = plt.figure(figsize=(11.0, 3.4))
    if hasattr(fig.canvas, "header_visible"):
        fig.canvas.header_visible = False
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.06, right=0.97, top=0.84, bottom=0.2)
    a0 = fig.add_subplot(gs[0, 0])
    a0.plot(t_fine, traj_fine[:, 1], color="#bbbbbb", lw=1.0, label="dx (cross axis)")
    a0.plot(t_fine, traj_fine[:, 0], color="#1f77b4", lw=1.0, label="dy (locomotion axis)")
    a0.set(xlim=(0, 5), xlabel="time (s)", ylabel="shift (um)",
           title="underlying motion: mostly sloshing, subtle rhythm")
    a0.legend(fontsize=8, loc="upper right", frameon=False)
    a1 = fig.add_subplot(gs[0, 1])
    a1.semilogy(freq, power / power.max(), color="#1f77b4", lw=1.2)
    a1.axvline(locomotion_freq_hz, color="#2ca02c", ls="--", lw=1.2, label="stride ~7 Hz")
    a1.axvline(fps_long / 2.0, color="#d62728", ls=":", lw=1.2, label=f"Nyquist @ {fps_long:.0f} fps")
    a1.set(xlim=(0, 11), ylim=(1e-4, 2), xlabel="frequency (Hz)", ylabel="rel. power",
           title="dominant-axis spectrum")
    a1.legend(fontsize=8, loc="upper right", frameon=False)
    a2 = fig.add_subplot(gs[0, 2])
    a2.imshow(hist.T, origin="lower", extent=(-12, 12, -12, 12), cmap="magma", aspect="equal")
    a2.set(xlabel="dx (um)", ylabel="dy (um)", title="per-frame shift distribution")
    a2.add_patch(plt.Circle((0, 0), amplitude_um, fill=False, ec="w", ls="--", lw=1.0))
    return fig


class IlluminationPanel:
    """Stage-8 illumination + vignette explorer (committed scope, render once then recompute).

    Renders the committed cell + tissue chain ONCE to a mean image; then each drag recomputes
    only the two cheap radial falloff fields and the per-cell SNR. Panels: the two falloffs
    combining centre->corner | the FOV dimmed by their product | per-cell SNR vs radius
    (:func:`~minisim.notebooks._support.plot_snr_vs_radius`) crossing the detection floor.
    """

    def __init__(self, sliders, acq, steps, seed, *, sensor_default_photons=300.0):
        self.sliders = sliders
        acq8 = acq.model_copy(update={"duration_s": 5.0})  # committed scope; short window
        self.acq = acq8
        sc = Scene.zeros(acq8, rng=np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        for s in steps:  # the committed cell+tissue chain (no fields)
            if s.domain in ("cell", "tissue"):
                s.build(acq8, rng)(sc)
        self.base = sc.movie.values.mean(axis=0)  # pre-falloff mean image (rendered once)
        self.sensor = next((s for s in steps if s.kind == "sensor"), None) \
            or Sensor(photons_per_unit=sensor_default_photons)
        self.hw = (acq8.image_sensor.n_px_height, acq8.image_sensor.n_px_width)
        self.ctr = falloff_center_px(self.hw, acq8, (0.0, 0.0))
        self.px = acq8.pixel_size_um
        self.qe, self.read = acq8.image_sensor.quantum_efficiency, acq8.image_sensor.read_noise_e
        cen = ((self.hw[0] - 1) / 2.0 * self.px, (self.hw[1] - 1) / 2.0 * self.px)  # FOV center, um
        self.incells = [c for c in sc.cells if c.in_focus and c.trace is not None]
        self.rmax = math.hypot((self.hw[0] - 1) / 2.0, (self.hw[1] - 1) / 2.0) * self.px
        self.rprof = np.linspace(0.0, self.rmax, 200)
        self.rad_cells = np.array([math.hypot(c.center_um[1] - cen[0], c.center_um[2] - cen[1])
                                   for c in self.incells])
        self.fig = plt.figure(figsize=(11.0, 3.7))
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        gs = self.fig.add_gridspec(1, 3, wspace=0.32, left=0.07, right=0.99, top=0.86, bottom=0.19)
        self._ap = self.fig.add_subplot(gs[0, 0])
        self._ai = self.fig.add_subplot(gs[0, 1])
        self._ad = self.fig.add_subplot(gs[0, 2])

    @property
    def canvas(self):
        """The persistent figure canvas (hand this to ``interactive_panel``)."""
        return self.fig.canvas

    def _profile(self, falloff, exponent):
        # the radial falloff as a function of distance, reaching `falloff` at the corner
        return 1.0 - (1.0 - falloff) * (self.rprof / self.rmax) ** exponent

    def _cell_snr(self, cell, field):
        # gain = optical brightness x illumination/vignette x exposure x QE; the field
        # sampling and shot+read SNR come from the library, matching finalize().
        g = ((cell.optical_brightness or 1.0)
             * sample_field_at(field, cell.center_um[1], cell.center_um[2], self.px)
             * self.sensor.photons_per_unit * self.qe)
        return float(detection_snr(cell.trace.max() - cell.trace.min(), cell.trace.min(), g, self.read))

    def draw(self):
        """Recompute the two falloff fields + per-cell SNR from the current sliders, and redraw."""
        v = {k: s.value for k, s in self.sliders.items()}
        field = (radial_falloff(self.hw, self.ctr, v["illum_corner"], v["illum_rolloff"])
                 * radial_falloff(self.hw, self.ctr, v["vig_corner"], v["vig_rolloff"]))
        self._ap.clear()
        self._ai.clear()
        ic, ir, vc, vr = v["illum_corner"], v["illum_rolloff"], v["vig_corner"], v["vig_rolloff"]
        self._ap.plot(self.rprof, self._profile(ic, ir), color="#1f77b4", lw=1.6, label="illumination (excitation)")
        self._ap.plot(self.rprof, self._profile(vc, vr), color="#d62728", lw=1.6, label="vignette (collection)")
        self._ap.plot(self.rprof, self._profile(ic, ir) * self._profile(vc, vr),
                      color="k", lw=2.2, label="observed = product")
        self._ap.set(xlabel="distance from center (um)", ylabel="relative brightness", ylim=(0, 1.05),
                     title="two falloffs combine (center -> corner)")
        self._ap.legend(fontsize=7, loc="lower left", frameon=False)
        self._ai.imshow(self.base * field, cmap=GCAMP, vmin=0, vmax=float(np.percentile(self.base, 99.5)))
        self._ai.set_xticks([])
        self._ai.set_yticks([])
        self._ai.set_title("FOV with falloff applied")
        snr = np.array([self._cell_snr(c, field) for c in self.incells])
        plot_snr_vs_radius(self._ad, self.rad_cells, snr, DETECT_SNR_THRESHOLD,
                           title="outer cells fall below the floor")
        self.fig.canvas.draw_idle()


class SensorPanel:
    """Stage-9 image-sensor explorer (committed chain, render once then re-digitize).

    Renders the committed cell + tissue chain ONCE to a clean intensity frame (illumination
    x vignette folded in); then each drag recomputes only the additive leakage glow + the
    digitization. Panels: the raw integer-count frame (with saturation %) | the count
    histogram (:func:`~minisim.notebooks._support.plot_count_histogram`) | per-cell SNR vs
    radius. The sensor draw is reseeded each redraw for a stable picture.
    """

    def __init__(self, sliders, acq, steps, seed):
        self.sliders = sliders
        self.seed = seed
        acq9 = acq.model_copy(update={"duration_s": 5.0})  # committed scope; short window
        self.acq = acq9
        sc = Scene.zeros(acq9, rng=np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        for s in steps:  # committed cell+tissue chain (incl. neuropil)
            if s.domain in ("cell", "tissue"):
                s.build(acq9, rng)(sc)
        mov = sc.movie.values
        self.frame = mov[int(np.argmax(mov.reshape(mov.shape[0], -1).sum(axis=1)))]  # liveliest frame
        self.hw = (acq9.image_sensor.n_px_height, acq9.image_sensor.n_px_width)
        self.px, self.qe = acq9.pixel_size_um, acq9.image_sensor.quantum_efficiency
        # committed illumination x vignette, applied to the intensity before digitizing
        self.field = np.ones(self.hw)
        for sp in (s for s in steps if s.kind in ("illumination_profile", "vignette")):
            self.field = self.field * radial_falloff(
                self.hw, falloff_center_px(self.hw, acq9, sp.center_offset_um), sp.falloff, sp.exponent)
        # leakage spatial shape: a central gaussian glow (stray excitation), scaled by the slider
        sig_px = acq9.um_to_px(0.25 * min(acq9.fov_um))
        self.glow = np.exp(-(radius_grid(self.hw, ((self.hw[0] - 1) / 2.0, (self.hw[1] - 1) / 2.0)) ** 2)
                           / (2.0 * sig_px ** 2))
        cells = [c for c in sc.cells if c.in_focus and c.trace is not None]
        cen = ((self.hw[0] - 1) / 2.0 * self.px, (self.hw[1] - 1) / 2.0 * self.px)
        self.rad = np.array([math.hypot(c.center_um[1] - cen[0], c.center_um[2] - cen[1]) for c in cells])
        self.fcell = np.array([self.field[int(np.clip(round(c.center_um[1] / self.px), 0, self.hw[0] - 1)),
                                          int(np.clip(round(c.center_um[2] / self.px), 0, self.hw[1] - 1))]
                               for c in cells])
        self.bright = np.array([(c.optical_brightness or 1.0) for c in cells])
        self.peak = np.array([float(c.trace.max() - c.trace.min()) for c in cells])
        self.baseln = np.array([max(float(c.trace.min()), 0.0) for c in cells])
        self.fig = plt.figure(figsize=(11.0, 3.7))
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        gs = self.fig.add_gridspec(1, 3, wspace=0.34, left=0.06, right=0.99, top=0.86, bottom=0.19)
        self._ai = self.fig.add_subplot(gs[0, 0])
        self._ah = self.fig.add_subplot(gs[0, 1])
        self._ad = self.fig.add_subplot(gs[0, 2])

    @property
    def canvas(self):
        """The persistent figure canvas (hand this to ``interactive_panel``)."""
        return self.fig.canvas

    def draw(self):
        """Re-apply leakage + digitize from the current sliders, and redraw all three panels."""
        v = {k: s.value for k, s in self.sliders.items()}
        photons_per_unit, rn, bd, leak = v["photons"], v["read_noise"], int(v["bit_depth"]), v["leak"]
        maxc = 2 ** bd - 1
        isensor = self.acq.image_sensor.model_copy(update={"read_noise_e": rn, "bit_depth": bd})
        photons = np.clip((self.frame * self.field + leak * self.glow) * photons_per_unit, 0.0, None)
        counts = isensor.photons_to_counts(photons, np.random.default_rng(self.seed))
        self._ai.clear()
        self._ad.clear()
        sat = float((counts >= maxc).mean()) * 100.0
        self._ai.imshow(counts, cmap=GCAMP, vmin=0, vmax=maxc)
        self._ai.set_xticks([])
        self._ai.set_yticks([])
        self._ai.set_title(f"raw counts ({bd}-bit, 0-{maxc}) | {sat:.1f}% saturated")
        plot_count_histogram(self._ah, counts, maxc)
        g = self.bright * self.fcell * photons_per_unit * self.qe
        snr = detection_snr(self.peak, self.baseln, g, rn)
        plot_snr_vs_radius(self._ad, self.rad, snr, DETECT_SNR_THRESHOLD,
                           title="exposure sets who clears the floor")
        self.fig.canvas.draw_idle()

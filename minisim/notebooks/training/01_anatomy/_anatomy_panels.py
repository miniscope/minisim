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

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import zoom

from minisim import ImageSensor, Optics, Tissue
from minisim.footprint import Footprint
from minisim.notebooks._support import GCAMP
from minisim.steps import degrade_footprint, neuron_footprint

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

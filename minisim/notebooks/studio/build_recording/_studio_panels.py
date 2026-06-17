"""The interactive slider panels for the ``build_recording`` studio.

The UI half of the studio: ipywidgets controls that read and write one shared,
carried-forward :class:`~_studio_config.StudioConfig` (the pure data half, in
``_studio_config.py``) and redraw a live matplotlib preview. Three panels, one
per stage of the workflow:

* :class:`AnatomyPanel` - every non-activity knob (optics, sensor, tissue, cell
  placement, and the optical confounds) plus preset buttons, previewing a single
  representative example frame.
* :class:`ActivityPanel` - the calcium model, previewing ground-truth traces.
* :class:`GeneratePanel` - duration / fps / seed / output, and the button that
  streams the full recording to disk.

Each panel is a small class owning its widgets and a persistent matplotlib
canvas; like the anatomy notebook's panels we redraw that one canvas in place
(no re-``display``), which keeps updates smooth and sidesteps VS Code's
duplicate-output bug. The physics is real minisim throughout - every preview runs
the same forward model :func:`minisim.simulate` runs - so what you tune is what
the generated file contains.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import (
    HTML,
    Accordion,
    BoundedFloatText,
    Button,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    IntSlider,
    IntText,
    Layout,
    Output,
    Text,
    ToggleButtons,
    VBox,
)

from minisim import simulate
from minisim.notebooks._support import GCAMP, plot_traces
from minisim.video import simulate_video

# Sibling modules are copied out together and run with the notebook directory on
# sys.path, so the bare import works at notebook runtime; fall back to the fully
# qualified package path for headless import (tests, the verify script).
try:  # notebook runtime (siblings on sys.path)
    from _studio_config import (
        ACTIVITY_PRESETS,
        CONFIG_FIELDS,
        STUDIO_PRESETS,
        StudioConfig,
    )
except ImportError:  # headless / packaged import
    from minisim.notebooks.studio.build_recording._studio_config import (
        ACTIVITY_PRESETS,
        CONFIG_FIELDS,
        STUDIO_PRESETS,
        StudioConfig,
    )

# Width budget for a slider's description label, so the long names line up.
_DESC_W = "140px"
_CTRL_W = "330px"


def _styled(w):
    """Apply the shared label/control widths to a widget and return it."""
    w.style.description_width = _DESC_W
    w.layout = Layout(width=_CTRL_W)
    return w


def _slider(cls, lo, hi, step, value, desc):
    """A FloatSlider/IntSlider that only fires on release (snappy heavy previews)."""
    return _styled(
        cls(min=lo, max=hi, step=step, value=value, description=desc, continuous_update=False)
    )


def _choice(widget, **kw):
    """A Dropdown/ToggleButtons/Checkbox with the shared description width."""
    return _styled(widget(**kw))


def _num_box(cls, value, desc, *, lo=None, hi=None, step=None):
    """A numeric entry box (BoundedFloatText/IntText), styled like the other controls.

    ``lo``/``hi`` are omitted for the unbounded IntText (seed); a number is typed
    rather than dragged, which suits a recording length or seed better than a slider.
    """
    kw: dict = {"value": value, "description": desc}
    if lo is not None:
        kw["min"] = lo
    if hi is not None:
        kw["max"] = hi
    if step is not None:
        kw["step"] = step
    return _styled(cls(**kw))


def _hide_canvas_chrome(fig) -> None:
    """Hide the ipympl figure toolbar/header chrome (no-op on non-widget backends).

    The ``header_visible``/``footer_visible``/``resizable`` traits exist only on the
    ipympl ``Canvas``; under the Agg backend (headless verify) they are absent, so
    each set is guarded.
    """
    for attr in ("header_visible", "footer_visible", "resizable"):
        if hasattr(fig.canvas, attr):
            setattr(fig.canvas, attr, False)


class AnatomyPanel:
    """Tune every non-activity knob against a live example frame.

    Exposes the optics, image sensor, tissue, cell-placement, and optical-confound
    fields of a :class:`StudioConfig` as a grouped accordion of sliders/toggles,
    with preset buttons that seed the whole panel from a named configuration. On
    any change it rebuilds a *fast preview* (the full forward pipeline on a
    cell-budgeted window at the true pixel scale - see
    :meth:`StudioConfig.preview_spec`), shows a max-projection of the result, and
    updates the FOV / cell-count / detectable readouts.

    The panel mutates the ``config`` it is handed, so the next panel sees the tuned
    values; nothing is copied. ``preview_seed`` fixes the preview RNG so dragging a
    slider back and forth returns to the same frame.
    """

    # Widget keys that do NOT map 1:1 to a StudioConfig field (handled explicitly in
    # _config_to_widgets/_apply). Everything else is a plain getattr/setattr.
    _SPECIAL = (
        "focus_auto", "focal_depth_um", "exposure_auto", "photons_per_unit",
        "field_curvature_on", "field_curvature_radius_um", "resolution",
    )

    # Dependent-control gating: each (controller, [dependents], enable_when) disables
    # the dependents whose stage/option is off, so the panel never shows a live slider
    # that feeds nothing (e.g. focal depth under auto-focus, or vessel sliders when
    # vasculature is off). `bool` enables on a truthy toggle; auto-focus inverts it.
    _GATES = (
        ("focus_auto", ["focal_depth_um"], lambda v: not v),
        ("exposure_auto", ["photons_per_unit"], lambda v: not v),
        ("field_curvature_on", ["field_curvature_radius_um"], bool),
        ("neuropil_enabled", ["neuropil_amplitude"], bool),
        ("vasculature_enabled", ["vessel_depth_um", "vessel_root_radius_um", "vessel_opacity",
                                 "vessel_n_roots", "vessel_branch_prob", "vessel_tortuosity_deg"], bool),
        ("illumination_enabled", ["illumination_falloff", "illumination_exponent"], bool),
        ("vignette_enabled", ["vignette_falloff", "vignette_exponent"], bool),
        ("leakage_enabled", ["leakage_level", "leakage_profile"], bool),
        ("motion_enabled", ["motion_model", "motion_amplitude_um", "max_shift_um",
                            "locomotion_freq_hz", "locomotion_axis"], bool),
    )

    def __init__(self, config: StudioConfig, *, cell_budget: int = 350, preview_seed: int = 0):
        self.config = config
        self.cell_budget = cell_budget
        self.preview_seed = preview_seed
        self._muting = False  # suppress per-widget redraws during a preset load

        self._build_widgets()
        with plt.ioff():
            self.fig = plt.figure(figsize=(5.4, 6.6))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.25], hspace=0.33)
        self.ax = self.fig.add_subplot(gs[0])       # max-projection preview
        self.ax_side = self.fig.add_subplot(gs[1])  # depth cross-section schematic
        # Reserve room on the right of the side-view axes for an outside legend (done
        # once: ax.clear() in draw keeps the axes position, so this never compounds).
        box = self.ax_side.get_position()
        self.ax_side.set_position([box.x0, box.y0, box.width * 0.78, box.height])
        _hide_canvas_chrome(self.fig)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for w in self._widgets.values():
            w.observe(self._on_change, names="value")
        self._sync_enabled()
        self.draw()

    # ----------------------------------------------------------------- widgets

    def _build_widgets(self) -> None:
        """Construct every control and the grouped accordion that lays them out."""
        c = self.config
        S, I = FloatSlider, IntSlider  # noqa: E741 - local aliases for the table below

        # field name -> widget. Field names match StudioConfig attributes 1:1 so the
        # read-back loop in _apply() is a plain getattr/setattr; the few non-1:1
        # controls (auto focus, square resolution) are handled explicitly there.
        w: dict[str, object] = {}

        # -- optics --
        w["na"] = _slider(S, 0.1, 0.6, 0.01, c.na, "NA")
        w["magnification"] = _slider(S, 1.0, 12.0, 0.1, c.magnification, "magnification")
        w["emission_nm"] = _slider(S, 480.0, 580.0, 1.0, c.emission_nm, "emission (nm)")
        w["focus_auto"] = _choice(
            Checkbox, value=(c.focal_depth_um == "auto"), description="auto focus"
        )
        w["focal_depth_um"] = _slider(
            S, 0.0, 400.0, 5.0, 150.0 if c.focal_depth_um == "auto" else float(c.focal_depth_um),
            "focal depth (um)",
        )
        w["field_curvature_on"] = _choice(
            Checkbox, value=(c.field_curvature_radius_um is not None), description="field curvature"
        )
        w["field_curvature_radius_um"] = _slider(
            S, 500.0, 8000.0, 100.0, c.field_curvature_radius_um or 2500.0, "curv radius (um)"
        )

        # -- image sensor --
        w["resolution"] = _slider(I, 64, 800, 8, c.n_px_height, "resolution (px/side)")
        w["pixel_pitch_um"] = _slider(S, 1.0, 8.0, 0.1, c.pixel_pitch_um, "pixel pitch (um)")
        w["quantum_efficiency"] = _slider(S, 0.3, 1.0, 0.05, c.quantum_efficiency, "QE")
        w["read_noise_e"] = _slider(S, 0.0, 10.0, 0.5, c.read_noise_e, "read noise (e-)")
        w["gain_adu_per_e"] = _slider(S, 0.1, 4.0, 0.1, c.gain_adu_per_e, "gain (ADU/e-)")
        w["bit_depth"] = _slider(I, 8, 12, 1, c.bit_depth, "bit depth")
        w["exposure_auto"] = _choice(
            Checkbox, value=(c.photons_per_unit == "auto"), description="auto exposure"
        )
        w["photons_per_unit"] = _slider(
            S, 10.0, 1000.0, 10.0,
            600.0 if c.photons_per_unit == "auto" else float(c.photons_per_unit), "exposure",
        )

        # -- tissue --
        w["scatter_mfp_emission_um"] = _slider(
            S, 30.0, 300.0, 5.0, c.scatter_mfp_emission_um, "emission MFP (um)"
        )
        w["scatter_mfp_excitation_um"] = _slider(
            S, 100.0, 1000.0, 20.0, c.scatter_mfp_excitation_um, "excite MFP (um)"
        )
        w["scatter_blur_per_um"] = _slider(
            S, 0.0, 0.15, 0.005, c.scatter_blur_per_um, "scatter blur/um"
        )

        # -- cells (placement only; activity is panel 2) --
        w["density_per_mm3"] = _slider(S, 2000.0, 120000.0, 1000.0, c.density_per_mm3, "density/mm3")
        w["depth_lo_um"] = _slider(S, 0.0, 400.0, 5.0, c.depth_lo_um, "depth lo (um)")
        w["depth_hi_um"] = _slider(S, 0.0, 400.0, 5.0, c.depth_hi_um, "depth hi (um)")
        w["soma_radius_um"] = _slider(S, 2.0, 12.0, 0.5, c.soma_radius_um, "soma r (um)")
        w["irregularity"] = _slider(S, 0.0, 1.0, 0.05, c.irregularity, "irregularity")
        w["morphology"] = _choice(
            ToggleButtons, options=["soma", "cytosolic"], value=c.morphology, description="GCaMP"
        )
        w["dendrite_length_um"] = _slider(
            S, 5.0, 60.0, 1.0, c.dendrite_length_um, "dendrite len (um)"
        )
        w["dendrite_width_um"] = _slider(S, 1.0, 6.0, 0.5, c.dendrite_width_um, "dendrite w (um)")
        w["min_distance_um"] = _slider(S, 0.0, 40.0, 1.0, c.min_distance_um, "min dist (um)")

        # -- neuropil --
        w["neuropil_enabled"] = _choice(Checkbox, value=c.neuropil_enabled, description="neuropil")
        w["neuropil_amplitude"] = _slider(
            S, 0.0, 3.0, 0.1, c.neuropil_amplitude, "neuropil amp"
        )

        # -- vasculature --
        w["vasculature_enabled"] = _choice(
            Checkbox, value=c.vasculature_enabled, description="vasculature"
        )
        w["vessel_depth_um"] = _slider(S, 0.0, 200.0, 5.0, c.vessel_depth_um, "vessel depth (um)")
        w["vessel_root_radius_um"] = _slider(
            S, 3.0, 40.0, 1.0, c.vessel_root_radius_um, "trunk r (um)"
        )
        w["vessel_opacity"] = _slider(S, 0.1, 1.0, 0.05, c.vessel_opacity, "opacity")
        w["vessel_n_roots"] = _slider(I, 1, 10, 1, c.vessel_n_roots, "n roots")
        w["vessel_branch_prob"] = _slider(S, 0.0, 0.5, 0.02, c.vessel_branch_prob, "branchiness")
        w["vessel_tortuosity_deg"] = _slider(
            S, 0.0, 25.0, 1.0, c.vessel_tortuosity_deg, "waviness (deg)"
        )

        # -- static illumination / vignette / leakage --
        w["illumination_enabled"] = _choice(
            Checkbox, value=c.illumination_enabled, description="illumination"
        )
        w["illumination_falloff"] = _slider(
            S, 0.0, 1.0, 0.05, c.illumination_falloff, "illum edge"
        )
        w["illumination_exponent"] = _slider(
            S, 0.5, 6.0, 0.5, c.illumination_exponent, "illum rolloff"
        )
        w["vignette_enabled"] = _choice(Checkbox, value=c.vignette_enabled, description="vignette")
        w["vignette_falloff"] = _slider(S, 0.0, 1.0, 0.05, c.vignette_falloff, "vignette corner")
        w["vignette_exponent"] = _slider(S, 0.5, 6.0, 0.5, c.vignette_exponent, "vignette rolloff")
        w["leakage_enabled"] = _choice(Checkbox, value=c.leakage_enabled, description="leakage glow")
        w["leakage_level"] = _slider(S, 0.0, 1.0, 0.02, c.leakage_level, "leakage level")
        w["leakage_profile"] = _choice(
            Dropdown, options=["uniform", "gaussian"], value=c.leakage_profile, description="glow shape"
        )

        # -- brain motion --
        w["motion_enabled"] = _choice(Checkbox, value=c.motion_enabled, description="brain motion")
        w["motion_model"] = _choice(
            Dropdown, options=["physical", "walk"], value=c.motion_model, description="motion model"
        )
        w["motion_amplitude_um"] = _slider(
            S, 0.0, 30.0, 0.5, c.motion_amplitude_um, "motion amp (um)"
        )
        w["max_shift_um"] = _slider(S, 1.0, 40.0, 1.0, c.max_shift_um, "max shift (um)")
        w["locomotion_freq_hz"] = _slider(S, 1.0, 12.0, 0.5, c.locomotion_freq_hz, "stride (Hz)")
        w["locomotion_axis"] = _choice(
            Dropdown, options=["y", "x"], value=c.locomotion_axis, description="motion axis"
        )

        self._widgets = w
        self._groups = {
            "Optics": ["na", "magnification", "emission_nm", "focus_auto", "focal_depth_um",
                       "field_curvature_on", "field_curvature_radius_um"],
            "Image sensor": ["resolution", "pixel_pitch_um", "quantum_efficiency", "read_noise_e",
                             "gain_adu_per_e", "bit_depth", "exposure_auto", "photons_per_unit"],
            "Tissue": ["scatter_mfp_emission_um", "scatter_mfp_excitation_um", "scatter_blur_per_um"],
            "Cells (placement)": ["density_per_mm3", "depth_lo_um", "depth_hi_um", "soma_radius_um",
                                  "irregularity", "morphology", "dendrite_length_um",
                                  "dendrite_width_um", "min_distance_um"],
            "Neuropil": ["neuropil_enabled", "neuropil_amplitude"],
            "Vasculature": ["vasculature_enabled", "vessel_depth_um", "vessel_root_radius_um",
                            "vessel_opacity", "vessel_n_roots", "vessel_branch_prob",
                            "vessel_tortuosity_deg"],
            "Illumination / vignette / glow": ["illumination_enabled", "illumination_falloff",
                                               "illumination_exponent", "vignette_enabled",
                                               "vignette_falloff", "vignette_exponent",
                                               "leakage_enabled", "leakage_level", "leakage_profile"],
            "Brain motion": ["motion_enabled", "motion_model", "motion_amplitude_um", "max_shift_um",
                             "locomotion_freq_hz", "locomotion_axis"],
        }
        # Guards that turn the panel's two implicit contracts into loud failures: every
        # non-special widget key must be a real config field, and the accordion groups
        # must list every widget exactly once.
        unknown = set(w) - set(self._SPECIAL) - set(CONFIG_FIELDS)
        assert not unknown, f"widget keys not in StudioConfig: {sorted(unknown)}"
        grouped = {k for keys in self._groups.values() for k in keys}
        assert grouped == set(w), f"groups/widgets mismatch: {grouped ^ set(w)}"

    def _preset_bar(self) -> HBox:
        """A row of buttons, one per studio preset, that seed the whole panel."""
        buttons = []
        for name in STUDIO_PRESETS:
            b = Button(description=name, layout=Layout(width="auto"))
            b.on_click(lambda _b, n=name: self._load_preset(n))
            buttons.append(b)
        return HBox([HTML("<b>preset:</b>"), *buttons])

    # ------------------------------------------------------------------- logic

    def _sync_enabled(self) -> None:
        """Grey out dependent sliders whose stage/option is currently off (see _GATES)."""
        for ctrl, deps, enable_when in self._GATES:
            disabled = not enable_when(self._widgets[ctrl].value)
            for key in deps:
                self._widgets[key].disabled = disabled

    def _on_change(self, _change) -> None:
        if not self._muting:
            self._sync_enabled()  # a toggle may have just gated some sliders
            self.draw()

    def _load_preset(self, name: str) -> None:
        """Load a preset into the shared config *in place*, then sync every widget.

        Copies the preset field-by-field onto the existing config object rather than
        rebinding ``self.config``, so the other panels (which hold the same object)
        see the preset too, and non-widget fields (e.g. front working distance) come
        along as well.
        """
        preset = STUDIO_PRESETS[name]()
        for field in CONFIG_FIELDS:
            setattr(self.config, field, getattr(preset, field))
        self._muting = True
        try:
            self._config_to_widgets(self.config)
        finally:
            self._muting = False
        self._sync_enabled()
        self.draw()

    def _config_to_widgets(self, c: StudioConfig) -> None:
        """Mirror a config's values back onto the widgets (after a preset load)."""
        w = self._widgets
        w["focus_auto"].value = c.focal_depth_um == "auto"
        if c.focal_depth_um != "auto":
            w["focal_depth_um"].value = float(c.focal_depth_um)
        w["exposure_auto"].value = c.photons_per_unit == "auto"
        if c.photons_per_unit != "auto":
            w["photons_per_unit"].value = float(c.photons_per_unit)
        w["field_curvature_on"].value = c.field_curvature_radius_um is not None
        if c.field_curvature_radius_um is not None:
            w["field_curvature_radius_um"].value = c.field_curvature_radius_um
        w["resolution"].value = c.n_px_height
        for key, widget in w.items():  # the rest map 1:1 to config fields
            if key not in self._SPECIAL:
                widget.value = getattr(c, key)

    def _apply(self) -> None:
        """Read every widget back into ``self.config``."""
        w = self._widgets
        c = self.config
        c.focal_depth_um = "auto" if w["focus_auto"].value else float(w["focal_depth_um"].value)
        c.photons_per_unit = (
            "auto" if w["exposure_auto"].value else float(w["photons_per_unit"].value)
        )
        c.field_curvature_radius_um = (
            float(w["field_curvature_radius_um"].value) if w["field_curvature_on"].value else None
        )
        c.n_px_height = c.n_px_width = int(w["resolution"].value)
        for key, widget in w.items():  # the rest map 1:1 to config fields
            if key not in self._SPECIAL:
                setattr(c, key, widget.value)

    def draw(self) -> None:
        """Rebuild the preview and readouts from the current widget values.

        Wrapped so an invalid combination (a spec validator raising - e.g. a soma
        wider than the FOV) shows its message in the canvas instead of killing the
        panel; advisory ``SpecWarning``s are silenced for the exploratory preview.
        """
        self._apply()
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = self.config.preview_spec(
                    cell_budget=self.cell_budget, seed=self.preview_seed
                )
                rec = simulate(spec)
        except Exception as err:
            self.ax.text(0.5, 0.5, f"invalid configuration:\n{err}", ha="center", va="center",
                         color="#b00020", fontsize=8, wrap=True, transform=self.ax.transAxes)
            self._draw_side_view()  # the geometry schematic still reads from the config
            self.fig.canvas.draw_idle()
            return

        obs = np.asarray(rec.observed, dtype=float)
        # A max-projection over the preview window: every cell that fires at least
        # once shows (a single instant would only catch the few active that frame),
        # and taking the max over time averages out the per-frame shot-noise grain so
        # the field reads clean - the standard calcium summary image. Auto-contrast
        # (2-99.7 percentile) keeps it visible at any exposure; the quantitative
        # brightness/SNR story is the detectable count in the readout.
        projection = obs.max(axis=0)
        lo, hi = (float(v) for v in np.percentile(projection, [2.0, 99.7]))
        self.ax.imshow(projection, cmap=GCAMP, vmin=lo, vmax=max(hi, lo + 1.0))
        self.ax.set_title(self._readout(rec, spec), fontsize=8.5, loc="left")
        self._draw_side_view(resolved_focal=float(rec.ground_truth.focal_depth_um))
        self.fig.canvas.draw_idle()

    def _draw_side_view(self, resolved_focal: float | None = None) -> None:
        """Depth cross-section across the full FOV width: where each layer sits in z.

        A schematic (pure geometry, no simulation) showing, down the optical axis:
        the cell depth band, the focal surface (a curve when field curvature is on,
        bowing shallower toward the edges), and the vasculature layer - all over the
        true full-FOV width. ``resolved_focal`` is the focus depth the optics step
        actually resolved (passed from the preview's ground truth); for "auto" with
        no preview it falls back to the middle of the cell band.
        """
        c = self.config
        ax = self.ax_side
        ax.clear()
        _, fov_w = c.fov_um
        lo, hi = c.depth_lo_um, c.depth_hi_um
        if resolved_focal is None:
            resolved_focal = (lo + hi) / 2.0 if c.focal_depth_um == "auto" else float(c.focal_depth_um)

        # z extent: cover every drawn feature, a little headroom below the deepest.
        depths = [hi, resolved_focal, lo]
        if c.vasculature_enabled:
            depths.append(c.vessel_depth_um)
        z_max = max(depths) * 1.15 + 5.0

        # cell depth band (shaded) + a non-flickering scatter of representative somata.
        # The dot count tracks the estimated full-FOV cell count, so the density slider
        # (and the depth band / FOV) visibly changes how packed the band looks; capped
        # so a very dense field stays legible and fast to draw.
        ax.axhspan(lo, max(hi, lo + 0.1), color="#2ca02c", alpha=0.13, lw=0)
        rng = np.random.default_rng(0)
        n_expected = c.estimated_cell_count()
        n_dots = int(min(n_expected, 1500))  # capped for legibility/speed at high density
        ax.scatter(rng.uniform(-fov_w / 2, fov_w / 2, n_dots),
                   rng.uniform(lo, max(hi, lo + 0.1), n_dots),
                   s=5, color="#2ca02c", alpha=0.45, edgecolors="none", label="cells")

        # focal surface: a field-curvature curve (shallower off-axis) or a flat plane.
        # x is the optical-center frame, so the field radius is |x| (the axis is 0).
        xs = np.linspace(-fov_w / 2, fov_w / 2, 120)
        optics = c.optics()
        z_focal = resolved_focal - np.array([optics.focal_curvature_shift_um(x) for x in xs])
        flabel = "focal (curved)" if c.field_curvature_radius_um else "focal plane"
        ax.plot(xs, z_focal, color="#1f77b4", lw=1.6, label=flabel)

        # vasculature layer.
        if c.vasculature_enabled:
            ax.axhline(c.vessel_depth_um, color="#8B0000", lw=1.4, ls="--", label="vessels")

        ax.axhline(0.0, color="0.4", lw=1.0)  # tissue surface
        ax.text(fov_w / 2 * 0.99, 1.0, "tissue surface", ha="right", va="top", fontsize=6.5, color="0.4")
        ax.set_xlim(-fov_w / 2, fov_w / 2)
        ax.set_ylim(z_max, -0.04 * z_max)  # depth increases downward; surface near top
        ax.set_xlabel("position across FOV (um)", fontsize=8)
        ax.set_ylabel("depth z (um)", fontsize=8)
        ax.tick_params(labelsize=7)
        # legend outside, to the right - it no longer covers the cell/tissue band.
        ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  framealpha=0.9, borderaxespad=0.0)
        ax.set_title(f"side view: ~{n_expected} expected neurons in FOV, across depth",
                     fontsize=8.5, loc="left")

    def _readout(self, rec, spec) -> str:
        """Two-line summary: full-FOV geometry/count, then the preview-window stats."""
        c = self.config
        fov_h, fov_w = c.fov_um
        n_full = c.estimated_cell_count()
        gt = rec.ground_truth
        n_pv = len(gt.centers_um)
        det = int(np.asarray(gt.detectable).sum()) if n_pv else 0
        ph, pw = spec.acquisition.image_sensor.n_px_height, spec.acquisition.image_sensor.n_px_width
        focal = "auto" if c.focal_depth_um == "auto" else f"{c.focal_depth_um:.0f}um"
        # Exposure mirrors focus: under "auto" show the value the Sensor step resolved
        # (recorded in ground truth), so the readout reports what auto-exposure picked.
        if c.photons_per_unit == "auto":
            resolved = gt.exposure_photons_per_unit
            exposure = f"auto (~{resolved:.0f})" if resolved is not None else "auto"
        else:
            exposure = f"{c.photons_per_unit:.0f}"
        # Line 1 is the density-driven number (full FOV); line 2 makes clear the preview
        # is a window auto-sized to a ~fixed cell budget, so its count stays ~constant as
        # density rises (the window shrinks instead) - the full file is the whole FOV.
        return (
            f"FOV {fov_h:.0f}x{fov_w:.0f} um  |  {c.pixel_size_um:.2f} um/px  |  "
            f"~{n_full} cells over full FOV  |  focus {focal}  |  exposure {exposure}\n"
            f"preview = {ph}x{pw}px window auto-sized to ~{self.cell_budget} cells "
            f"({n_pv} shown, {det} detectable); the generated file is the whole FOV"
        )

    # -------------------------------------------------------------------- view

    def ui(self) -> VBox:
        """The assembled widget: preset bar, the preview canvas, and the knob accordion."""
        accordion = Accordion(
            children=[VBox([self._widgets[k] for k in keys]) for keys in self._groups.values()]
        )
        for i, title in enumerate(self._groups):
            accordion.set_title(i, title)
        return VBox([self._preset_bar(), HBox([self.fig.canvas, accordion])])


class ActivityPanel:
    """Tune the calcium-activity model against a live ground-truth trace preview.

    Exposes the :class:`~minisim.CellActivity` fields (the two-state firing gate,
    its rates, the calcium kinetics, and the cell-to-cell brightness spread) and,
    on any change, redraws stacked ground-truth ``C`` traces with spike ticks for
    the busiest few cells. The preview runs only the cell-domain steps over a long
    window on a handful of cells (:meth:`StudioConfig.activity_preview_spec` +
    ``until="cell_activity"``), so it is fast even at minute scale and shows the
    *clean* ``C`` - measurement noise is a property of the sensor, added only in
    the generated file, never here.

    Like :class:`AnatomyPanel` it mutates the shared ``config`` in place, so panel 3
    generates exactly the dynamics previewed here. ``Quiet``/``Moderate``/``Active``
    preset buttons overlay just the firing parameters onto the current scope/region.
    """

    def __init__(
        self,
        config: StudioConfig,
        *,
        cell_budget: int = 30,
        duration_s: float = 60.0,
        fps: float = 20.0,
        preview_seed: int = 0,
    ):
        self.config = config
        self.cell_budget = cell_budget
        self.duration_s = duration_s
        self.fps = fps
        self.preview_seed = preview_seed
        self._muting = False

        self._build_widgets()
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(7.2, 4.2))
        _hide_canvas_chrome(self.fig)

        for w in self._widgets.values():
            w.observe(self._on_change, names="value")
        self.draw()

    def _build_widgets(self) -> None:
        c = self.config
        w: dict[str, object] = {}
        w["p_quiescent_to_active"] = _slider(
            FloatSlider, 0.0005, 0.03, 0.0005, c.p_quiescent_to_active, "burst onset p"
        )
        w["p_active_to_quiescent"] = _slider(
            FloatSlider, 0.05, 0.6, 0.01, c.p_active_to_quiescent, "burst end p"
        )
        w["active_rate_hz"] = _slider(
            FloatSlider, 20.0, 400.0, 5.0, c.active_rate_hz, "in-burst rate (Hz)"
        )
        w["quiescent_rate_hz"] = _slider(
            FloatSlider, 0.0, 3.0, 0.1, c.quiescent_rate_hz, "baseline rate (Hz)"
        )
        w["tau_rise_s"] = _slider(FloatSlider, 0.01, 0.3, 0.01, c.tau_rise_s, "tau rise (s)")
        w["tau_decay_s"] = _slider(FloatSlider, 0.1, 2.0, 0.05, c.tau_decay_s, "tau decay (s)")
        w["brightness_cv"] = _slider(FloatSlider, 0.0, 1.0, 0.05, c.brightness_cv, "bright spread")
        w["f0"] = _slider(FloatSlider, 0.0, 3.0, 0.1, c.f0, "baseline F0")
        self._widgets = w

    def _preset_bar(self) -> HBox:
        buttons = []
        for name in ACTIVITY_PRESETS:
            b = Button(description=name, layout=Layout(width="auto"))
            b.on_click(lambda _b, n=name: self._load_preset(n))
            buttons.append(b)
        return HBox([HTML("<b>activity:</b>"), *buttons])

    def _on_change(self, _change) -> None:
        if not self._muting:
            self.draw()

    def _load_preset(self, name: str) -> None:
        """Overlay an activity preset's firing parameters onto the widgets."""
        self._muting = True
        try:
            for field, value in ACTIVITY_PRESETS[name].items():
                self._widgets[field].value = value
        finally:
            self._muting = False
        self.draw()

    def _apply(self) -> None:
        for field, widget in self._widgets.items():
            setattr(self.config, field, widget.value)

    def draw(self) -> None:
        """Resimulate the cell-domain steps and redraw the trace preview."""
        self._apply()
        self.ax.clear()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = self.config.activity_preview_spec(
                    cell_budget=self.cell_budget, duration_s=self.duration_s,
                    fps=self.fps, seed=self.preview_seed,
                )
                rec = simulate(spec, until="cell_activity")
        except Exception as err:
            self.ax.text(0.5, 0.5, f"invalid configuration:\n{err}", ha="center", va="center",
                         color="#b00020", fontsize=8, wrap=True, transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return

        gt = rec.ground_truth
        C, S = np.asarray(gt.C, dtype=float), np.asarray(gt.S, dtype=float)
        if not len(C):
            self.ax.text(0.5, 0.5, "no cells placed", ha="center", va="center",
                         transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        t = np.arange(C.shape[1]) / self.fps
        plot_traces(self.ax, t, C, spikes=S, n=min(6, len(C)), title=self._readout(C, S))
        self.fig.canvas.draw_idle()

    def _readout(self, C, S) -> str:
        """Title line: clean ground-truth C, with the realized mean event rate."""
        minutes = self.duration_s / 60.0
        rate = (S.sum() / len(S) / minutes) if (len(S) and minutes) else 0.0
        return (
            f"clean ground-truth calcium C (peak-scaled) + spikes S (ticks) - "
            f"{len(C)} cells, ~{rate:.0f} events/cell/min  (noise enters at the sensor, not here)"
        )

    def ui(self) -> VBox:
        """The assembled widget: activity preset bar, trace canvas, and sliders."""
        return VBox([
            self._preset_bar(),
            self.fig.canvas,
            HBox([VBox(list(self._widgets.values())[:4]), VBox(list(self._widgets.values())[4:])]),
        ])


def _format_bytes(n: float) -> str:
    """Human-readable byte count (MB/GB), for the file-size estimate."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class GeneratePanel:
    """Set the recording length and write the full recording to disk.

    Reads the shared, fully-tuned :class:`StudioConfig` (whatever panels 1 and 2
    left), adds duration / fps / seed, and on the button writes the recording at
    the *full* FOV - no preview cropping. Two formats:

    * **zarr** (default) - the complete :class:`~minisim.Recording`: the movie *and*
      the ground truth (footprints ``A``, traces ``C``, spikes ``S``, positions,
      detectable flags, vessel mask) plus the spec, via
      :meth:`minisim.Recording.save`. This goes through the in-memory
      :func:`minisim.simulate`, so it holds the whole movie in RAM (the size
      estimate flags how much). Reload with ``minisim.Recording.load(path)``.
    * **avi** - a viewable grayscale movie only, written by the streaming
      :func:`minisim.simulate_video` (flat memory, any length), with the spec
      dropped beside it as ``<name>.spec.json`` since the video itself carries no
      ground truth.

    The live estimate shows the frame count and on-disk / in-RAM size so a length
    is chosen with eyes open; the generate runs in an output log with a progress
    bar and prints a summary (cells, detectable, path, size) when done.
    """

    def __init__(
        self,
        config: StudioConfig,
        *,
        duration_s: float = 120.0,
        fps: float = 20.0,
        seed: int = 42,
        out_path: str = "recording",
    ):
        self.config = config
        self.duration_s = _num_box(BoundedFloatText, duration_s, "duration (s)", lo=1.0, hi=7200.0, step=10.0)
        self.fps = _num_box(BoundedFloatText, fps, "fps", lo=1.0, hi=120.0, step=1.0)
        self.seed = _num_box(IntText, seed, "seed", step=1)
        self.out_path = _choice(Text, value=out_path, description="output path")
        self.fmt = _choice(
            Dropdown,
            options=[("zarr (movie + ground truth)", "zarr"), ("avi (movie only)", "avi")],
            value="zarr",
            description="format",
        )
        self.estimate = HTML()
        self.button = Button(description="Generate recording", button_style="success",
                             icon="play", layout=Layout(width="auto"))
        self.log = Output()
        self.button.on_click(self._on_generate)
        for w in (self.duration_s, self.fps, self.fmt):
            w.observe(lambda _c: self._refresh_estimate(), names="value")
        self._refresh_estimate()

    def _refresh_estimate(self) -> None:
        c = self.config
        n_frames = round(self.duration_s.value * self.fps.value)
        h, w = c.n_px_height, c.n_px_width
        if self.fmt.value == "zarr":  # float32 movie held in RAM, then on disk
            size = n_frames * h * w * 4
            note = f"~{_format_bytes(size)} on disk and in RAM (held by simulate())"
        else:  # streamed 8-bit grayscale
            size = n_frames * h * w
            note = f"~{_format_bytes(size)} on disk (streamed, flat memory)"
        warn = " <b style='color:#b06000'>- large; consider a shorter clip or avi</b>" if size > 2e9 else ""
        self.estimate.value = (
            f"{n_frames} frames @ {w}x{h}px - {note}{warn}"
        )

    def _on_generate(self, _b) -> None:
        self.button.disabled = True
        self.log.clear_output()
        with self.log:
            try:
                self._generate()
            except Exception as err:
                print(f"FAILED: {type(err).__name__}: {err}")
            finally:
                self.button.disabled = False

    def _generate(self) -> None:
        c = self.config
        dur, fps, seed = self.duration_s.value, self.fps.value, int(self.seed.value)
        spec = c.spec(duration_s=dur, fps=fps, seed=seed)
        n_frames = spec.acquisition.n_frames
        print(f"simulating {n_frames} frames @ {c.n_px_width}x{c.n_px_height}px, "
              f"{dur:.0f}s @ {fps:.0f}fps, seed {seed} ...")

        if self.fmt.value == "zarr":
            path = Path(self.out_path.value).with_suffix(".zarr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec = simulate(spec)
            rec.save(path)
            gt = rec.ground_truth
            n, det = len(gt.centers_um), int(np.asarray(gt.detectable).sum())
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"saved {path}  ({_format_bytes(size)})")
            print(f"  {n} cells, {det} detectable ({100 * det / max(n, 1):.0f}%); "
                  f"ground truth A/C/S + spec included")
            print(f"  reload: minisim.Recording.load({str(path)!r})")
        else:
            path = Path(self.out_path.value).with_suffix(".avi")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simulate_video(spec, path)
            spec_path = path.with_suffix(".spec.json")
            spec_path.write_text(spec.model_dump_json(indent=2))
            size = path.stat().st_size if path.exists() else 0
            print(f"saved {path}  ({_format_bytes(size)})")
            print(f"  movie only (no ground truth); spec written to {spec_path}")
            print(f"  re-simulate with ground truth: "
                  f"minisim.simulate(Spec.model_validate_json(open({str(spec_path)!r}).read()))")

    def ui(self) -> VBox:
        """The assembled widget: settings row, estimate, generate button, output log."""
        return VBox([
            HBox([self.duration_s, self.fps, self.seed]),
            HBox([self.out_path, self.fmt]),
            self.estimate,
            self.button,
            self.log,
        ])

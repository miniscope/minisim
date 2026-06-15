"""The shared, mutable configuration the three studio panels accumulate into.

This is the *pure* half of the ``build_recording`` studio - no ipywidgets, no
matplotlib - so it can be unit-tested headlessly and reused by the headless
verify script. :class:`StudioConfig` is a flat, mutable mirror of the knobs the
panels expose; its :meth:`~StudioConfig.spec` method assembles those values into
a real, frozen :class:`minisim.Spec` (the same object ``simulate`` consumes).
The UI half - the slider panels that read and write a ``StudioConfig`` live -
lives in the sibling ``_studio_panels.py``.

The flow the studio implements: panel 1 (anatomy/scope) and panel 2 (activity)
mutate one carried-forward ``StudioConfig``; panel 3 reads it, adds duration /
fps / seed, and generates the recording. Optional pipeline stages (neuropil,
vasculature, illumination, vignette, leakage, motion) are gated by ``*_enabled``
toggles so the config spans a bare cells-only movie up to the full forward chain.

:data:`STUDIO_PRESETS` seeds the whole config from a named, realistic starting
point (a generic scope, or a Miniscope V4 imaging a particular brain region).
The physical numbers behind those presets - the V4 optics/sensor and the
standard-region anatomy - are the single source of truth in
:mod:`minisim.presets`; the studio reads them from there and adds only its own
presentation choices (the vignette/glow look, neuropil/motion defaults), so the
notebook and the library presets can never drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Literal

from minisim import (
    Acquisition,
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
    Tissue,
    Vasculature,
    VesselLayer,
    Vignette,
    presets,
)
from minisim.spec import Output

# A focal depth is either a concrete depth into tissue (um) or "auto" (resolved to
# the median realized cell depth at the optics step).
FocalDepth = float | Literal["auto"]


@dataclass
class StudioConfig:
    """Every tunable the studio exposes, flat and mutable, plus optional-stage toggles.

    Field groups mirror the spec models the values flow into: ``Optics`` /
    ``ImageSensor`` / ``Tissue`` on the acquisition, then one ``PlaceNeurons``
    population, the ``CellActivity`` model, and the optional confound stages. The
    panels read and write these attributes directly; :meth:`spec` turns the whole
    thing into a validated :class:`minisim.Spec`. Defaults match the library's own
    defaults, so a bare ``StudioConfig()`` already builds a sensible recording.
    """

    # ---- optics (Optics) ----
    na: float = 0.45
    magnification: float = 8.0
    emission_nm: float = 525.0
    focal_depth_um: FocalDepth = "auto"
    field_curvature_radius_um: float | None = None

    # ---- image sensor hardware (ImageSensor) ----
    n_px_height: int = 256
    n_px_width: int = 256
    pixel_pitch_um: float = 3.0
    quantum_efficiency: float = 0.7
    read_noise_e: float = 2.0
    gain_adu_per_e: float = 1.0
    bit_depth: int = 8
    # exposure/flux scale (Sensor step, a scene property not sensor hardware). The
    # studio default is well above the library's 100: a usable recording is brightly
    # exposed (background well off the noise floor, cells clearly above shot noise),
    # whereas 100 left a deep field sitting at ~12 of 255 ADC counts - all noise.
    photons_per_unit: float = 600.0

    # ---- tissue scattering (Tissue) ----
    scatter_mfp_excitation_um: float = 600.0
    scatter_mfp_emission_um: float = 100.0
    scatter_blur_per_um: float = 0.05

    # ---- one neuron population (PlaceNeurons) ----
    density_per_mm3: float = 25000.0
    soma_radius_um: float = 7.0
    irregularity: float = 0.3
    morphology: Literal["soma", "cytosolic"] = "soma"
    dendrite_length_um: float = 24.0
    dendrite_width_um: float = 3.0
    depth_lo_um: float = 0.0
    depth_hi_um: float = 200.0
    min_distance_um: float = 0.0

    # ---- calcium activity (CellActivity) ----
    spike_sim_hz: float = 300.0
    p_quiescent_to_active: float = 0.005
    p_active_to_quiescent: float = 0.3
    active_rate_hz: float = 150.0
    quiescent_rate_hz: float = 0.6
    tau_rise_s: float = 0.05
    tau_decay_s: float = 0.5
    brightness_cv: float = 0.3
    f0: float = 1.0

    # ---- neuropil background (Neuropil) ----
    neuropil_enabled: bool = True
    neuropil_amplitude: float = 0.5
    neuropil_n_components: int = 3

    # ---- vasculature shadow (Vasculature / one VesselLayer) ----
    vasculature_enabled: bool = False
    vessel_depth_um: float = 30.0
    vessel_n_roots: int = 4
    vessel_root_radius_um: float = 22.0
    vessel_opacity: float = 0.85
    vessel_branch_prob: float = 0.2
    vessel_tortuosity_deg: float = 8.0

    # ---- static fields (IlluminationProfile / Vignette / Leakage) ----
    illumination_enabled: bool = True
    illumination_falloff: float = 0.7
    illumination_exponent: float = 2.0
    vignette_enabled: bool = True
    vignette_falloff: float = 0.5
    vignette_exponent: float = 2.0
    leakage_enabled: bool = True
    leakage_level: float = 0.1
    leakage_profile: Literal["uniform", "gaussian"] = "gaussian"

    # ---- brain motion (BrainMotion) ----
    motion_enabled: bool = True
    motion_model: Literal["physical", "walk"] = "physical"
    # A typical few-um head-fixed 1p excursion; max_shift (the hard clamp + margin
    # size) is set comfortably above it. On the small generic FOV this trips the
    # advisory "motion > 5% of FOV", which is honest there; the ~1 mm V4 FOVs are fine.
    motion_amplitude_um: float = 5.0
    max_shift_um: float = 10.0
    locomotion_freq_hz: float = 7.0
    locomotion_axis: Literal["y", "x"] = "y"

    # ---- informational ----
    front_working_distance_um: float | None = None

    # ----------------------------------------------------------------- builders

    def optics(self) -> Optics:
        """Assemble the :class:`~minisim.Optics` (also used by the side-view schematic)."""
        return Optics(
            na=self.na,
            magnification=self.magnification,
            emission_nm=self.emission_nm,
            field_curvature_radius_um=self.field_curvature_radius_um,
        )

    def acquisition(self, *, duration_s: float, fps: float) -> Acquisition:
        """Assemble the :class:`~minisim.Acquisition` (optics + sensor + tissue + sampling)."""
        return Acquisition(
            optics=self.optics(),
            image_sensor=ImageSensor(
                n_px_height=self.n_px_height,
                n_px_width=self.n_px_width,
                pixel_pitch_um=self.pixel_pitch_um,
                quantum_efficiency=self.quantum_efficiency,
                read_noise_e=self.read_noise_e,
                gain_adu_per_e=self.gain_adu_per_e,
                bit_depth=self.bit_depth,
            ),
            tissue=Tissue(
                scatter_mfp_excitation_um=self.scatter_mfp_excitation_um,
                scatter_mfp_emission_um=self.scatter_mfp_emission_um,
                scatter_blur_per_um=self.scatter_blur_per_um,
            ),
            fps=fps,
            duration_s=duration_s,
            focal_depth_in_tissue_um=self.focal_depth_um,
            front_working_distance_um=self.front_working_distance_um,
        )

    def place_neurons(self) -> PlaceNeurons:
        """The single neuron population the studio places."""
        return PlaceNeurons(
            density_per_mm3=self.density_per_mm3,
            soma_radius_um=self.soma_radius_um,
            irregularity=self.irregularity,
            morphology=self.morphology,
            dendrite_length_um=self.dendrite_length_um,
            dendrite_width_um=self.dendrite_width_um,
            depth_range_um=(self.depth_lo_um, self.depth_hi_um),
            min_distance_um=self.min_distance_um,
        )

    def cell_activity(self) -> CellActivity:
        """The calcium-activity model."""
        return CellActivity(
            spike_sim_hz=self.spike_sim_hz,
            p_quiescent_to_active=self.p_quiescent_to_active,
            p_active_to_quiescent=self.p_active_to_quiescent,
            active_rate_hz=self.active_rate_hz,
            quiescent_rate_hz=self.quiescent_rate_hz,
            tau_rise_s=self.tau_rise_s,
            tau_decay_s=self.tau_decay_s,
            brightness_cv=self.brightness_cv,
            f0=self.f0,
        )

    def steps(self) -> list:
        """The ordered pipeline steps, gating the optional confound stages by toggle.

        Always present: place_neurons -> cell_activity -> optics -> composite ->
        sensor (the minimal chain that yields real ADC counts). The neuropil,
        vasculature, illumination, vignette, leakage, and motion stages are added
        only when their ``*_enabled`` toggle is set. ``Spec`` re-sorts into the
        canonical order, so the order here does not matter.
        """
        steps: list = [self.place_neurons(), self.cell_activity(), CellOptics(), Composite()]
        if self.neuropil_enabled:
            steps.append(
                Neuropil(amplitude=self.neuropil_amplitude, n_components=self.neuropil_n_components)
            )
        if self.vasculature_enabled:
            steps.append(
                Vasculature(
                    enabled=True,
                    layers=[
                        VesselLayer(
                            depth_um=self.vessel_depth_um,
                            n_roots=self.vessel_n_roots,
                            root_radius_um=self.vessel_root_radius_um,
                            opacity=self.vessel_opacity,
                            branch_prob=self.vessel_branch_prob,
                            tortuosity_deg=self.vessel_tortuosity_deg,
                        )
                    ],
                )
            )
        if self.motion_enabled:
            steps.append(
                BrainMotion(
                    model=self.motion_model,
                    motion_amplitude_um=self.motion_amplitude_um,
                    max_shift_um=self.max_shift_um,
                    locomotion_freq_hz=self.locomotion_freq_hz,
                    locomotion_axis=self.locomotion_axis,
                )
            )
        if self.illumination_enabled:
            steps.append(
                IlluminationProfile(
                    falloff=self.illumination_falloff, exponent=self.illumination_exponent
                )
            )
        if self.vignette_enabled:
            steps.append(Vignette(falloff=self.vignette_falloff, exponent=self.vignette_exponent))
        if self.leakage_enabled:
            steps.append(Leakage(level=self.leakage_level, profile=self.leakage_profile))
        steps.append(Sensor(photons_per_unit=self.photons_per_unit))
        return steps

    def spec(
        self, *, duration_s: float, fps: float, seed: int, save_intermediates: bool = False
    ) -> Spec:
        """Assemble the full, validated :class:`minisim.Spec` for a recording.

        This is the single bridge from the studio's flat knobs to the frozen spec
        the engine runs. Construction runs every spec validator, so an impossible
        configuration (e.g. a soma larger than the FOV) raises here, at tune time,
        rather than at generate time.
        """
        return Spec(
            acquisition=self.acquisition(duration_s=duration_s, fps=fps),
            seed=seed,
            steps=self.steps(),
            output=Output(save_intermediates=save_intermediates),
        )

    # --------------------------------------------------- derived readouts / preview

    @property
    def pixel_size_um(self) -> float:
        """Object-space size of one pixel, um (sensor pitch / magnification)."""
        return self.pixel_pitch_um / self.magnification

    @property
    def fov_um(self) -> tuple[float, float]:
        """Full field of view ``(height, width)`` in um at the current settings."""
        px = self.pixel_size_um
        return (self.n_px_height * px, self.n_px_width * px)

    def estimated_cell_count(self) -> int:
        """Expected number of placed cells over the *full* FOV - analytic, no simulation.

        Mirrors the volumetric count the ``place_neurons`` step derives
        (``density x FOV-area x thickness``, the depth thickness floored at one soma
        diameter), so a panel can show a live "~N cells" readout without paying the
        cost of actually generating thousands of footprints. The realized count
        differs slightly (rounding / Poisson-disk thinning), so this is a ``~``.
        """
        fov_h, fov_w = self.fov_um
        thickness_um = max(self.depth_hi_um - self.depth_lo_um, 2.0 * self.soma_radius_um)
        volume_mm3 = (fov_h / 1000.0) * (fov_w / 1000.0) * (thickness_um / 1000.0)
        return round(self.density_per_mm3 * volume_mm3)

    def preview_window_px(self, *, cell_budget: int) -> tuple[int, int]:
        """Sensor size (h, w) for a fast preview holding ~``cell_budget`` cells.

        The live preview renders the *full forward pipeline at the true pixel
        scale*, but on a centered window small enough that footprint generation
        (the per-cell cost that dominates a dense field) stays cheap. When the full
        FOV already holds at most ``cell_budget`` cells the real size is used (the
        whole field previews); otherwise the window shrinks by the square root of
        the cell ratio (cells scale with area), floored at 64 px so a preview is
        never degenerate. Pixel size, optics, tissue and noise are untouched - only
        the framed area changes - so brightness, blur, crowding and the noise floor
        all read at their true scale.
        """
        full = self.estimated_cell_count()
        if full <= cell_budget:
            return (self.n_px_height, self.n_px_width)
        linear = math.sqrt(cell_budget / full)
        return (
            max(64, round(self.n_px_height * linear)),
            max(64, round(self.n_px_width * linear)),
        )

    def preview_spec(
        self, *, cell_budget: int = 400, duration_s: float = 3.0, fps: float = 20.0, seed: int = 0
    ) -> Spec:
        """A fast, full-pipeline :class:`~minisim.Spec` for the live example frame.

        Identical to :meth:`spec` except the sensor is cropped to
        :meth:`preview_window_px` (bounding the cell count, hence render time) and
        the duration is short - long enough that most cells fire at least once so a
        max-projection of the result shows the whole population, short enough to stay
        snappy (the per-cell footprint build, not the frame count, dominates cost).
        """
        ph, pw = self.preview_window_px(cell_budget=cell_budget)
        preview = replace(self, n_px_height=ph, n_px_width=pw)
        return preview.spec(duration_s=duration_s, fps=fps, seed=seed)

    def activity_preview_spec(
        self, *, cell_budget: int = 30, duration_s: float = 60.0, fps: float = 20.0, seed: int = 0
    ) -> Spec:
        """A :class:`~minisim.Spec` for previewing calcium *traces* (panel 2).

        The activity model is per-cell and independent of the optics, so the trace
        preview needs only a handful of cells but a realistically long window for
        bursts to develop: a small ``cell_budget`` (cheap footprint build) over tens
        of seconds. The caller runs it with ``simulate(spec, until="cell_activity")``
        so only the cell-domain steps execute - no rendering - making it fast even
        at minute-scale durations. The ground-truth ``C`` / ``S`` come back populated.
        """
        ph, pw = self.preview_window_px(cell_budget=cell_budget)
        preview = replace(self, n_px_height=ph, n_px_width=pw)
        return preview.spec(duration_s=duration_s, fps=fps, seed=seed)

    def copy(self) -> StudioConfig:
        """A shallow copy - the panels hand each other independent working configs."""
        return replace(self)


# Public field names, in declaration order: handy for a panel that wants to mirror
# every knob, or a test that wants to round-trip them.
CONFIG_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(StudioConfig))


# ---------------------------------------------------------------------------
# Presets - named, realistic starting points the panel buttons load.
# ---------------------------------------------------------------------------


def _generic() -> StudioConfig:
    """A neutral, generic 1-photon scope - the library defaults, motion + haze on."""
    return StudioConfig()


def _stamp_scope(cfg: StudioConfig, scope: presets.Scope) -> None:
    """Copy a :class:`minisim.presets.Scope`'s optics/sensor onto the flat ``cfg``."""
    cfg.na = scope.optics.na
    cfg.magnification = scope.optics.magnification
    cfg.emission_nm = scope.optics.emission_nm
    cfg.field_curvature_radius_um = scope.optics.field_curvature_radius_um
    cfg.n_px_height = scope.image_sensor.n_px_height
    cfg.n_px_width = scope.image_sensor.n_px_width
    cfg.pixel_pitch_um = scope.image_sensor.pixel_pitch_um
    cfg.front_working_distance_um = scope.front_working_distance_um
    cfg.focal_depth_um = scope.focal_depth_in_tissue_um


def _stamp_region(cfg: StudioConfig, region: presets.Region) -> None:
    """Copy a :class:`minisim.presets.Region`'s population/tissue/vessels onto ``cfg``."""
    pop = region.population
    cfg.density_per_mm3 = pop.density_per_mm3
    cfg.soma_radius_um = pop.soma_radius_um
    cfg.irregularity = pop.irregularity
    cfg.morphology = pop.morphology
    cfg.dendrite_length_um = pop.dendrite_length_um
    cfg.dendrite_width_um = pop.dendrite_width_um
    cfg.depth_lo_um, cfg.depth_hi_um = pop.depth_range_um
    cfg.min_distance_um = pop.min_distance_um
    cfg.scatter_mfp_excitation_um = region.tissue.scatter_mfp_excitation_um
    cfg.scatter_mfp_emission_um = region.tissue.scatter_mfp_emission_um
    cfg.scatter_blur_per_um = region.tissue.scatter_blur_per_um
    cfg.vasculature_enabled = region.vasculature is not None
    if region.vasculature is not None:
        layer = region.vasculature.layers[0]
        cfg.vessel_depth_um = layer.depth_um
        cfg.vessel_n_roots = layer.n_roots
        cfg.vessel_root_radius_um = layer.root_radius_um
        cfg.vessel_opacity = layer.opacity
        cfg.vessel_branch_prob = layer.branch_prob
        cfg.vessel_tortuosity_deg = layer.tortuosity_deg


def _v4_look(cfg: StudioConfig) -> None:
    """Add the V4's characteristic vignette + center-illumination glow (studio look).

    The V4 has moderate emission vignetting and a center-weighted excitation glow.
    These are *presentation* choices the studio layers on top of the physical
    :func:`minisim.presets.miniscope_v4` scope, not part of the scope's spec.
    """
    cfg.illumination_enabled = True
    cfg.illumination_falloff = 0.7
    cfg.vignette_enabled = True
    cfg.vignette_falloff = 0.6  # moderate (corner ~60% of center)
    cfg.leakage_enabled = True
    cfg.leakage_profile = "gaussian"
    cfg.leakage_level = 0.08  # gentle center glow; higher buries cells under the bloom


def _miniscope_v4_region(region: presets.Region) -> StudioConfig:
    """A studio config for the Miniscope V4 imaging a standard region preset."""
    cfg = StudioConfig(neuropil_amplitude=0.4)
    _stamp_scope(cfg, presets.miniscope_v4())
    _stamp_region(cfg, region)
    _v4_look(cfg)
    return cfg


def _miniscope_v4_ca1() -> StudioConfig:
    """Miniscope V4 imaging hippocampal CA1 (see :func:`minisim.presets.ca1`)."""
    return _miniscope_v4_region(presets.ca1())


def _miniscope_v4_cortex_l23() -> StudioConfig:
    """Miniscope V4 imaging neocortex L2/3 (see :func:`minisim.presets.cortex_l23`)."""
    return _miniscope_v4_region(presets.cortex_l23())


# name -> factory (factories, not instances, so each press yields a fresh config).
STUDIO_PRESETS: dict[str, Callable[[], StudioConfig]] = {
    "Generic 1p scope": _generic,
    "Miniscope V4 - CA1": _miniscope_v4_ca1,
    "Miniscope V4 - cortex L2/3": _miniscope_v4_cortex_l23,
}


# Activity-level presets for panel 2: a partial overlay of just the CellActivity
# fields (firing rates + a matching decay), applied onto whatever scope/region the
# anatomy panel left in place. "Moderate" is the library default (CaLab's moderate
# level); "Quiet" and "Active" bracket it with sparser/denser bursting.
ACTIVITY_PRESETS: dict[str, dict[str, float]] = {
    "Quiet": dict(
        p_quiescent_to_active=0.002, p_active_to_quiescent=0.4,
        active_rate_hz=80.0, quiescent_rate_hz=0.2, tau_decay_s=0.6,
    ),
    "Moderate": dict(
        p_quiescent_to_active=0.005, p_active_to_quiescent=0.3,
        active_rate_hz=150.0, quiescent_rate_hz=0.6, tau_decay_s=0.5,
    ),
    "Active": dict(
        p_quiescent_to_active=0.012, p_active_to_quiescent=0.2,
        active_rate_hz=250.0, quiescent_rate_hz=1.0, tau_decay_s=0.45,
    ),
}

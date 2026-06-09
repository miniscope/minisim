"""Tissue-domain steps: composite the cells, then add brain-bound field effects.

The tissue domain is the brain-frame stack - everything that moves rigidly with
the tissue under :mod:`~minisim` motion, as opposed to the
static optics/sensor fields (:mod:`minisim.steps.sensor`). It opens
with ``composite`` (the cell→image boundary) and then layers the diffuse/global
effects that ride on top of the cells:

* :class:`CompositeStep` (``cells_only``) - composite ``Σ footprint·trace`` into the
  movie; the first step to write ``scene.movie``.
* :class:`NeuropilStep` (``neuropil``) - additive diffuse background, a smooth
  spatial field modulated by a slow temporal envelope.
* :class:`VasculatureStep` (``vasculature``) - a dark, static absorbing-vessel
  mask multiplied into the movie (off by default), grown from depth-resolved
  branching vessel trees; the high-contrast landmark and a tunable confound.

All run before the motion boundary, so a later ``brain_motion`` step translates
the cells *and* these fields together (they are part of the brain frame), unlike
the static vignette/leakage applied after motion.

:class:`BleachingStep` (``bleaching``) also lives here but is a **cell-domain**
step: photobleaching is per-cell and activity-driven, so it runs *before* composite
and writes each cell's intact-fluorophore envelope rather than touching the movie
(see its docstring). It is kept in this module beside the composite/neuropil code it
coordinates with (composite emits ``C·B``; neuropil fades with the population ``B``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter

from minisim.footprint import RENDER_DTYPE, stack_dense
from minisim.scene import Scene
from minisim.steps.base import PipelineContext, Step

if TYPE_CHECKING:
    # Referenced only as string Generic bases (Step["Composite"] etc.), which ruff's
    # F401 misses; pyright needs them in scope to resolve the forward references.
    from minisim.spec import Bleaching, Composite, Neuropil, Vasculature  # noqa: F401

# Guards a divide-by-peak for a degenerate (flat) smooth field; far below any
# physically meaningful intensity.
_EPS = 1e-12

# Temporal fluctuation depth of the neuropil envelope, as the log-space sigma of
# its mean-1 lognormal modulation (see :func:`neuropil_envelope`). A fixed v1
# constant - slow background *drifts* by tens of percent rather than blinking;
# per-component variability could become a spec field later.
_NEUROPIL_FLUCT_LOG_STD = 0.4


class CompositeStep(Step["Composite"]):
    """Composite ``Σ_i footprint_i · trace_i`` additively into the movie.

    Each cell contributes its footprint scaled, frame by frame, by the light it
    actually *emits* - its calcium trace ``C`` times its bleaching envelope ``B``
    when ``bleaching`` has run (the trace is the clean calcium; ``B`` is the
    intact-fluorophore fraction that fades it), else just ``C``. The *observed*
    (optically degraded) footprint is used when present; until the ``optics`` step
    populates it, the *planted* (sharp) footprint is used - so the minimal
    chain renders sharp cells, and gains optical realism for free once optics
    lands, with no change here. Cells missing a footprint or a trace are skipped
    (e.g. before ``cell_activity`` has run), and an empty scene leaves the movie
    untouched. The composite is **additive** so later tissue effects (neuropil,
    etc.) accumulate onto the same movie.
    """

    name = "cells_only"
    domain = "tissue"

    def __call__(self, scene: Scene) -> None:
        footprints, traces = [], []
        for cell in scene.cells:
            # The observed (optically degraded) footprint is regenerated here from
            # the planted one rather than stored (see Cell.observed_footprint); it
            # falls back to the planted footprint until the optics step has run.
            footprint = cell.observed_footprint()
            if footprint is None or cell.trace is None:
                continue
            footprints.append(footprint)
            # The emitted trace: clean calcium, dimmed by bleaching when present.
            traces.append(cell.trace if cell.bleach is None else cell.trace * cell.bleach)
        if not footprints:
            return
        # Footprints are stored sparse (a small patch each); rebuild the dense
        # (unit, H, W) stack transiently for the BLAS contraction against the
        # traces -- faster than a per-cell loop, and the stack is freed at once.
        # A and C are float32 (RENDER_DTYPE) so the contraction is single-precision
        # (half the memory traffic on the large A); the float64 movie accumulates
        # the float32 result. The streaming writer composites identically.
        A = stack_dense(footprints, scene.canvas_shape)  # (unit, height, width)
        C = np.stack(traces).astype(RENDER_DTYPE)  # (unit, frame)
        contrib = np.tensordot(C, A, axes=([0], [0]))  # (frame, height, width)
        scene.movie.values[:] += contrib


# ---------------------------------------------------------------------------
# neuropil
# ---------------------------------------------------------------------------


def smooth_spatial_field(
    shape: tuple[int, int], sigma_px: float, rng: np.random.Generator
) -> np.ndarray:
    """A smooth, non-negative spatial field, peak-normalized to ``[0, 1]``.

    Low-pass-filtered white noise: ``gaussian_filter`` of a standard-normal field
    at ``sigma_px`` produces a blob with structure on the ``sigma_px`` length
    scale. Shifted to be non-negative (background light cannot be negative) and
    divided by its peak so ``amplitude`` carries the absolute level. A degenerate
    flat field falls back to all-ones.
    """
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma_px, mode="nearest")
    field -= field.min()
    peak = float(field.max())
    if peak <= _EPS:
        return np.ones(shape)
    return field / peak


def ou_process(n: int, tau_frames: float, rng: np.random.Generator) -> np.ndarray:
    """A stationary Ornstein–Uhlenbeck sequence: mean 0, unit variance, length ``n``.

    Discrete OU at a one-frame step: ``x[t] = a·x[t-1] + √(1-a²)·ε`` with
    ``a = exp(-1/τ_frames)`` the per-frame correlation. Larger ``τ_frames`` ⇒
    ``a → 1`` ⇒ slower drift. The noise term ``√(1-a²)`` fixes the stationary
    variance at 1, and the first sample is drawn from that stationary
    distribution, so the whole sequence is mean-0/unit-variance with correlation
    time ``τ_frames``. Sequential by construction (``x[t]`` depends on ``x[t-1]``)
    - an explicit loop, cheap at the recording lengths the simulator targets.
    """
    if n <= 0:
        return np.zeros(0)
    a = math.exp(-1.0 / tau_frames) if tau_frames > 0 else 0.0
    noise_scale = math.sqrt(1.0 - a * a)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = a * x[t - 1] + noise_scale * rng.standard_normal()
    return x


def population_envelope(
    traces: list[np.ndarray], tau_frames: float
) -> np.ndarray | None:
    """Mean-1 temporal driver from the surrounding population's calcium, or ``None``.

    The neuropil is the dendritic/axonal felt of the cells around the focal plane,
    so its brightness tracks *local population activity*: sum the per-cell traces
    into an aggregate ``g(t) = Σ_i C_i(t)``, then **causally** low-pass it with a
    one-pole exponential at ``tau_frames`` (``y[t] = a·y[t-1] + (1-a)·g[t]``,
    ``a = exp(-1/τ)``). Causal, not symmetric, because the felt *integrates and
    lags* activity - the haze swells after the population fires, never before.
    Normalized to mean 1 so the absolute level stays carried by ``amplitude``.

    Returns ``None`` when there is no signal to drive it (no cells, or all traces
    silent), so the caller falls back to a purely independent background rather
    than dividing by zero - the neuropil step must stay valid before
    ``cell_activity`` has run.
    """
    if not traces:
        return None
    g = np.sum(np.stack(traces), axis=0).astype(float)
    a = math.exp(-1.0 / tau_frames) if tau_frames > 0 else 0.0
    # The one-pole recurrence as an IIR filter (vectorized in C, the same idiom as
    # motion._lowpass / _integrate_dho). The initial state ``zi = a·g[0]`` seeds the
    # filter so smoothed[0] == g[0] (start at the first value, not the lfilter default).
    smoothed, _ = lfilter([1.0 - a], [1.0, -a], g, zi=[a * g[0]])
    mean = float(smoothed.mean())
    if mean <= _EPS:
        return None
    return smoothed / mean


def neuropil_envelope(
    n_frames: int, tau_frames: float, rng: np.random.Generator
) -> np.ndarray:
    """A slow, strictly positive temporal envelope with mean 1.

    Exponentiates a unit OU process into a lognormal modulation
    ``exp(σ·OU − σ²/2)`` (``σ = _NEUROPIL_FLUCT_LOG_STD``): always positive - so
    the additive background it scales stays non-negative without clipping - and
    mean exactly 1, so the neuropil's overall level is set by ``amplitude`` alone,
    not by the temporal fluctuation. The ``−σ²/2`` offset is the lognormal
    mean-1 correction, the same trick used for spike amplitudes in
    :mod:`minisim.steps.cell`.
    """
    s = _NEUROPIL_FLUCT_LOG_STD
    return np.exp(s * ou_process(n_frames, tau_frames, rng) - 0.5 * s * s)


def neuropil_components(
    spec, acq, cells, shape: tuple[int, int], n_frames: int, rng: np.random.Generator
):
    """The ``(spatial, temporal, population)`` neuropil components for a spec.

    The RNG-consuming generation half of :class:`NeuropilStep`, factored out so the
    step *and* the streaming video writer build the **identical** components from
    the same RNG draws (the draws run in a fixed order: all ``n_components`` spatial
    fields, then all ``n_components`` temporal envelopes - :func:`population_envelope`
    is deterministic). ``shape`` is the canvas ``(h, w)``; ``n_frames`` the recording
    length. The diffuse background fades with the population-average bleaching
    envelope when ``bleaching`` has run. Returns peak-normalized spatial fields
    ``(component, h, w)``, realized temporal envelopes ``(component, frame)``, and
    the population driver ``(frame,)`` (or ``None``).
    """
    sigma_px = acq.um_to_px(spec.spatial_sigma_um)
    drift_tau_frames = acq.s_to_frame(spec.temporal_tau_s)
    pop_tau_frames = acq.s_to_frame(spec.population_tau_s)
    spatial = np.stack(
        [smooth_spatial_field(shape, sigma_px, rng) for _ in range(spec.n_components)]
    )
    traces = [cell.trace for cell in cells if cell.trace is not None]
    population = population_envelope(traces, pop_tau_frames)
    c = spec.population_coupling if population is not None else 0.0
    temporal = np.stack([
        (1.0 - c) * neuropil_envelope(n_frames, drift_tau_frames, rng)
        + (c * population if population is not None else 0.0)
        for _ in range(spec.n_components)
    ])
    bleaches = [cell.bleach for cell in cells if cell.bleach is not None]
    if bleaches:
        temporal = temporal * np.mean(np.stack(bleaches), axis=0)[None, :]
    return spatial, temporal, population


class NeuropilStep(Step["Neuropil"]):
    """Additive diffuse background: ``amplitude · meanₖ(Sₖ(y,x) · Tₖ(t))``.

    Sums ``n_components`` diffuse sources, each a smooth spatial field
    :func:`smooth_spatial_field` (``[0, 1]``, structure on ``spatial_sigma_um``)
    modulated by a positive, mean-1 temporal envelope ``Tₖ``. The envelope is the
    biologically driven part: a convex blend, at ``population_coupling`` ``c``,

        ``Tₖ(t) = (1 − c)·OUₖ(t) + c·P(t)``

    of an independent slow drift ``OUₖ`` :func:`neuropil_envelope` (the unmodeled
    out-of-FOV/out-of-plane tissue) and the shared population driver ``P``
    :func:`population_envelope` (the local cells' aggregate calcium, lagged and
    smoothed - the dendritic felt brightening as the population fires). Both legs
    are positive and mean-1, so ``Tₖ`` is too: the absolute level stays set by
    ``amplitude`` (relative to the ``f0 = 1`` cell baseline) and the background is
    non-negative by construction. With no cells yet (``P`` is ``None``) it falls
    back to pure ``OUₖ``. This is the modeled diffuse mesh only - out-of-focus
    somata are a *separate* background that emerges for free from
    ``place_neurons`` + ``optics``.

    The diffuse fluorophore **bleaches with the cells**: when ``bleaching`` has
    run, the whole background is faded by the population-average intact fraction
    ``meanᵢ Bᵢ(t)`` - the felt is those same neurons' arbors, so it dims as they
    do (no separate neuropil pool). The fade is folded into the stored temporal
    envelopes, so they remain the true modulation applied.

    Records the spatial fields ``(component, height, width)``, the realized
    temporal envelopes ``(component, frame)``, and the population driver
    ``(frame,)`` to ground truth, so a background-removal stage can be scored
    against the true diffuse structure and its activity coupling.
    """

    name = "neuropil"
    domain = "tissue"
    consumes_rng = True  # smooth-field noise + OU drift in neuropil_components

    def __call__(self, scene: Scene) -> None:
        # Grid from the scene canvas (which a motion margin may enlarge beyond
        # the sensor) so the diffuse background covers the same tissue the cells
        # do and moves with it under motion.
        n_frames, h, w = scene.movie.values.shape
        spatial, temporal, population = neuropil_components(
            self.spec, self.acq, scene.cells, (h, w), n_frames, self.rng
        )
        # mean over components of the (frame, h, w) outer products, then scale.
        contrib = np.tensordot(temporal, spatial, axes=([0], [0])) / self.spec.n_components
        scene.movie.values[:] += self.spec.amplitude * contrib
        scene.truth.neuropil_spatial = spatial
        scene.truth.neuropil_temporal = temporal
        scene.truth.neuropil_population = population


# ---------------------------------------------------------------------------
# bleaching
# ---------------------------------------------------------------------------


def bleaching_pool(
    emission: np.ndarray,
    q: float,
    tau_turn_frames: float,
    intensity: float,
    b0: float = 1.0,
) -> np.ndarray:
    """Intact functional-fluorophore fraction ``B(t)`` under bleaching vs turnover.

    Photobleaching is a per-photon hazard: each excitation–emission cycle carries a
    small chance of permanently destroying the fluorophore, so intact protein is
    lost in proportion to how much it emits. Protein **turnover** opposes this,
    synthesizing fresh fluorophore back toward full expression. Per emitter,

        ``dB/dt = (1 − B)/τ_turn  −  q · intensity · emission(t) · B``

    starting at ``b0`` (1 = fresh). ``emission`` is the per-frame brightness drive
    (a cell's calcium trace; for a population, its aggregate), ``intensity`` the
    excitation level, ``q`` the bleach susceptibility (per frame, per unit
    emission·intensity), ``τ_turn`` the turnover time in frames. Integrated exactly
    per frame for piecewise-constant emission (unconditionally stable). With the
    light off (``emission`` or ``intensity`` 0) it relaxes back toward 1 - a dark
    recovery - so imaging sessions chain by passing the previous ending ``B`` as
    ``b0``. More active or more brightly lit emitters bleach faster and settle at a
    lower floor ``B* = k_turn / (k_turn + q·intensity·⟨emission⟩)``.
    """
    n = len(emission)
    out = np.empty(n)
    k_turn = 1.0 / tau_turn_frames if tau_turn_frames > 0 else 0.0
    b = float(b0)
    for t in range(n):
        out[t] = b
        decay = k_turn + q * intensity * float(emission[t])  # total per-frame rate
        if decay > 0:  # exact step toward the instantaneous equilibrium k_turn/decay
            b_eq = k_turn / decay
            b = b_eq + (b - b_eq) * math.exp(-decay)
    return out


def bleaching_floor(q: float, intensity: float, emission: float, tau_turn: float) -> float:
    """Steady-state intact fraction ``B*`` under constant illumination and emission.

    The fixed point of the :func:`bleaching_pool` ODE for a constant drive:
    bleaching at rate ``q·intensity·emission`` balances turnover at rate
    ``k_turn = 1/tau_turn``, giving

        ``B* = k_turn / (k_turn + q·intensity·emission)``

    the continuous-imaging floor a brightly-lit or busy emitter settles toward.
    A ratio of rates, so ``q`` and ``tau_turn`` need only share a time unit (both
    per-frame or both per-second). With the light off (``intensity`` or
    ``emission`` 0) the floor is 1 (turnover wins); ``tau_turn → ∞`` (no turnover)
    drives it to 0.
    """
    if q < 0 or intensity < 0 or emission < 0 or tau_turn < 0:
        raise ValueError(
            "bleaching_floor needs non-negative rates; got "
            f"q={q}, intensity={intensity}, emission={emission}, tau_turn={tau_turn}"
        )
    k_turn = 1.0 / tau_turn if tau_turn > 0 else 0.0
    denom = k_turn + q * intensity * emission
    # denom == 0 only when there is neither drive nor turnover: nothing destroys the
    # pool, so it stays fully intact at 1. With non-negative inputs denom is never < 0.
    return 1.0 if denom == 0 else k_turn / denom


def dark_recovery(b0: float, t, tau_turn: float):
    """Intact fraction recovering toward 1 over a dark gap of duration ``t``.

    The zero-input response of the :func:`bleaching_pool` ODE: with the light off
    (no emission), bleaching stops and turnover relaxes the pool back toward full
    expression with time constant ``tau_turn``,

        ``B(t) = 1 − (1 − b0)·exp(−t / tau_turn)``

    starting from ``b0`` at ``t = 0``. ``t`` may be a scalar or an array (same time
    unit as ``tau_turn``), so a whole recovery curve comes out at once. This is what
    chains imaging sessions: a long gap (``t ≫ tau_turn``) restores the signal, a
    short one lets the baseline ratchet down session to session.
    """
    if tau_turn <= 0:
        raise ValueError(f"dark_recovery requires tau_turn > 0; got {tau_turn}")
    return 1.0 - (1.0 - b0) * np.exp(-np.asarray(t, dtype=float) / tau_turn)


class BleachingStep(Step["Bleaching"]):
    """Per-cell, activity-driven fluorophore decay - bleaching fought by turnover.

    Gives each cell an intact-fluorophore envelope ``Bᵢ(t)`` from
    :func:`bleaching_pool`, driven by its own calcium trace (its emission) scaled by
    the excitation ``intensity``: busier, brighter-lit cells bleach faster and to a
    lower floor, while turnover pulls every cell back toward 1. A **cell-domain**
    step (it runs before ``composite``, like ``cell_activity`` and ``optics``): it
    writes ``cell.bleach`` rather than touching the movie, so the trace stays the
    clean calcium ``C`` and ``composite`` emits ``C·B``. The diffuse ``neuropil`` then
    fades with the population-average ``B``. ``finalize`` stacks the per-cell
    envelopes into ground truth ``(unit, frame)``, a scoreable confound.

    If an :class:`~minisim.spec.IlluminationProfile` is present, ``prepare`` pulls it
    from the :class:`~minisim.steps.base.PipelineContext` and the per-cell excitation
    dose is scaled by the illumination at the cell's **rest** lateral position, so
    brightly-lit center cells bleach faster than dim edge cells. (Motion's effect on a
    cell's dose as it
    jiggles through the gradient is second-order and ignored.) This is the one way
    the excitation-side illumination differs from the collection-side vignette.
    """

    name = "bleaching"
    domain = "cell"

    def __init__(self, spec, acq, rng) -> None:
        super().__init__(spec, acq, rng)
        # Optional IlluminationProfile spec, pulled from the PipelineContext in
        # prepare() when present, so the excitation dose varies across the FOV.
        # None -> spatially uniform dose.
        self.illumination = None

    def prepare(self, context: PipelineContext) -> None:
        self.illumination = context.illumination

    def __call__(self, scene: Scene) -> None:
        spec, acq = self.spec, self.acq
        q = spec.bleach_susceptibility / acq.fps  # per-second coefficient -> per-frame
        tau_frames = acq.s_to_frame(spec.turnover_tau_s)
        dose = self._illumination_dose(scene)
        for cell, illum in zip(scene.cells, dose, strict=True):
            if cell.trace is None:
                continue
            cell.bleach = bleaching_pool(
                cell.trace, q, tau_frames, spec.excitation_intensity * illum
            )

    def _illumination_dose(self, scene: Scene) -> np.ndarray:
        """Per-cell excitation scale from the illumination field at each rest position.

        All ones when no ``IlluminationProfile`` was injected. Otherwise the same
        :func:`~minisim.steps.sensor.radial_falloff` field the illumination step
        applies (sensor-FOV sized), sampled at each cell's clipped lateral pixel - so
        the dose a cell sees matches the brightness its image gets.
        """
        n = len(scene.cells)
        if self.illumination is None:
            return np.ones(n)
        from minisim.steps.sensor import falloff_center_px, radial_falloff

        acq = self.acq
        shape = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
        center = falloff_center_px(shape, acq, self.illumination.center_offset_um)
        field = radial_falloff(shape, center, self.illumination.falloff, self.illumination.exponent)
        px = acq.pixel_size_um
        # Cells live in canvas coords; the field is the sensor FOV, so subtract the
        # motion-margin offset before indexing - the same canvas -> FOV mapping
        # finalize() and _photon_budget_at() apply. Without motion the margin is 0.
        canvas_h, canvas_w = scene.canvas_shape
        margin_y_um = (canvas_h - shape[0]) // 2 * px
        margin_x_um = (canvas_w - shape[1]) // 2 * px
        ys = np.array([cell.center_um[1] for cell in scene.cells])
        xs = np.array([cell.center_um[2] for cell in scene.cells])
        iy = np.clip(np.round((ys - margin_y_um) / px), 0, shape[0] - 1).astype(int)
        ix = np.clip(np.round((xs - margin_x_um) / px), 0, shape[1] - 1).astype(int)
        return field[iy, ix]


# ---------------------------------------------------------------------------
# vasculature
# ---------------------------------------------------------------------------
#
# A blood vessel is an *absorber*, not an emitter: haemoglobin soaks up both the
# excitation going in (~470 nm) and the emission coming out (~525 nm), so a vessel
# casts a dark shadow on everything optically behind it. The vasculature effect is
# therefore a multiplicative mask M(y, x) in (0, 1] (1 = clear tissue, → 0 = opaque
# trunk) applied to the brain-frame movie. It is built in three pure stages, each
# individually testable: *grow* a branching vessel tree, *rasterize* it into a
# blood-path-length map, then map that to a transmission mask by Beer-Lambert.
#
# Everything here works in micrometres (the physical domain), except the rasterizer
# which bridges µm → the pixel grid; that keeps the absorption coefficient a true
# per-µm quantity, independent of magnification/pixel size.

# Murray's-law exponent: a parent vessel's cross-sectional "flow capacity" Σ rⁱ is
# conserved across a bifurcation with i ≈ 3 (minimum-work principle, Murray 1926).
# So a symmetric split shrinks each child to r·0.5^(1/3) ≈ 0.794·r - the natural
# trunk→capillary taper.
_MURRAY_EXPONENT = 3.0


@dataclass(frozen=True)
class VesselGrowth:
    """Growth knobs for one vessel tree - the runtime bundle the spec builds.

    Plain (non-pydantic) params consumed by :func:`grow_vessel_tree`; the future
    ``VesselLayer`` spec converts its µm/per-layer fields into one of these. All
    lengths are µm; angles radians.

    Attributes
    ----------
    r0_um
        Root (thickest) vessel radius - the trunk entering the field.
    min_radius_um
        A branch terminates once Murray's-law tapering drops it below this; the
        capillary floor.
    branch_prob
        Per-step probability of a bifurcation. A side branch peels off and the
        main vessel continues (asymmetric split, see ``branch_area_main``).
    tortuosity_rad
        Std of the Gaussian heading perturbation applied each step - sets how much
        a vessel wanders (0 = ruler-straight).
    step_per_radius
        Step length as a multiple of the current radius, so thick trunks advance
        in coarse strides and capillaries in fine ones (keeps the rasterized line
        smooth relative to its width).
    branch_area_main
        The main child's share of the conserved Σ r³ at a bifurcation, in (0.5, 1):
        0.5 is a symmetric split, → 1 a thin twig peeling off a barely-thinned
        trunk. The side child takes the rest.
    branch_angle_rad
        Base heading deviation at a bifurcation; the thinner child turns *more*
        (scaled by the radius ratio), the realistic "sharp little offshoot" look.
    max_segments
        Hard cap on segments per tree - a safety bound on total work, not a
        physical parameter (well above any realistic tree).
    """

    r0_um: float
    min_radius_um: float
    branch_prob: float
    tortuosity_rad: float
    step_per_radius: float = 2.0
    branch_area_main: float = 0.7
    branch_angle_rad: float = 0.5
    max_segments: int = 2000


def murray_children(r_parent: float, area_fraction: float) -> tuple[float, float]:
    """Child radii at a bifurcation under Murray's law: ``r1³ + r2³ = r_parent³``.

    ``area_fraction`` ``a`` in (0, 1) is the main child's share of the conserved
    cube-sum, so ``r1 = r_parent·a^(1/3)`` and ``r2 = r_parent·(1−a)^(1/3)``. At
    ``a = 0.5`` the split is symmetric (each child ``0.794·r_parent``); nearer 1 a
    thin twig peels off a trunk that barely thins. The exponent is
    :data:`_MURRAY_EXPONENT` (≈ 3, the minimum-work value). Returns
    ``(r_main, r_side)`` with ``r_main ≥ r_side`` whenever ``a ≥ 0.5``.
    """
    inv = 1.0 / _MURRAY_EXPONENT
    r_main = r_parent * area_fraction**inv
    r_side = r_parent * (1.0 - area_fraction) ** inv
    return r_main, r_side


def grow_vessel_tree(
    bounds_um: tuple[float, float],
    root_um: tuple[float, float],
    heading_rad: float,
    growth: VesselGrowth,
    rng: np.random.Generator,
) -> np.ndarray:
    """Grow one branching vessel tree; return its segments as ``(n, 5)`` µm rows.

    Stochastic recursive branching. A branch advances step by step from ``root_um``
    along ``heading_rad`` (image convention: ``dy = sin θ``, ``dx = cos θ``), the
    heading drifting by a Gaussian ``tortuosity_rad`` each step so the vessel curves
    rather than runs straight. With probability ``branch_prob`` per step it
    *bifurcates*: a side branch peels off (pushed onto a stack to grow later) while
    the main vessel continues, the two radii set by :func:`murray_children` and the
    thinner child deviating more. A branch ends when its radius falls below
    ``min_radius_um`` (it has tapered to a capillary) or it leaves ``bounds_um`` (it
    has crossed the canvas) - the boundary-crossing segment is kept, so the vessel
    reaches the edge. ``bounds_um`` is the canvas extent ``(H_um, W_um)``; roots are
    seeded at/over an edge by the caller so trees grow *into* the field, the way real
    vessels enter from the side.

    Each row is ``(y0, x0, y1, x1, radius)`` in µm - a straight sub-segment with a
    constant radius, the unit the rasterizer turns into a capsule. Returns
    ``(0, 5)`` if the root is already sub-threshold. Deterministic in ``rng``: the
    draws run step-then-branch in a fixed order, so the same seed yields the same
    tree (the streaming writer relies on this).
    """
    h_um, w_um = bounds_um
    segments: list[tuple[float, float, float, float, float]] = []
    # DFS over branches: each stack entry is one branch's start state. The main
    # vessel is grown inline; side branches are deferred so a single root expands
    # into the whole tree without recursion.
    stack: list[tuple[float, float, float, float]] = [
        (root_um[0], root_um[1], heading_rad, growth.r0_um)
    ]
    while stack and len(segments) < growth.max_segments:
        y, x, theta, r = stack.pop()
        while r >= growth.min_radius_um and len(segments) < growth.max_segments:
            step = growth.step_per_radius * r
            ny, nx = y + step * math.sin(theta), x + step * math.cos(theta)
            segments.append((y, x, ny, nx, r))
            y, x = ny, nx
            if not (0.0 <= y <= h_um and 0.0 <= x <= w_um):
                break  # left the canvas; keep the crossing segment, end the branch
            theta += float(rng.normal(0.0, growth.tortuosity_rad))
            if rng.random() < growth.branch_prob:
                r_main, r_side = murray_children(r, growth.branch_area_main)
                side = -1.0 if rng.random() < 0.5 else 1.0  # peel left or right
                # The thinner child turns more (its momentum is smaller): scale the
                # base angle by the radius ratio, so a fine offshoot leaves sharply
                # while the trunk barely bends.
                ratio = r / r_side if r_side > _EPS else 1.0
                stack.append((y, x, theta + side * growth.branch_angle_rad * ratio, r_side))
                theta -= side * growth.branch_angle_rad  # main bends slightly the other way
                r = r_main
    if not segments:
        return np.zeros((0, 5), dtype=float)
    return np.asarray(segments, dtype=float)


def rasterize_vessels(
    segments_um: np.ndarray, shape_px: tuple[int, int], pixel_size_um: float
) -> np.ndarray:
    """Rasterize vessel segments into a blood-path-length map ``L(y, x)``, in µm.

    Each segment is a capsule: a cylinder of radius ``r`` about its axis. A pixel at
    lateral distance ``d`` from the axis sees the light pass through a blood *chord*
    of length ``2·√(r² − d²)`` (the straight path through a circular cross-section),
    and 0 outside the radius - so the map peaks at ``2r`` over a thick trunk's spine
    and tapers smoothly to its edge, giving round soft borders before any optical
    blur. Overlapping segments **add** their chords (stacked vessels absorb more).

    Works per segment over its own bounding box (grown by the radius, clipped to the
    canvas), not the whole canvas, so cost scales with vessel area. ``segments_um``
    is the ``(n, 5)`` array from :func:`grow_vessel_tree` (µm); ``shape_px`` the
    output ``(H, W)`` and ``pixel_size_um`` the µm-per-pixel scale used to place the
    µm geometry on the grid and to convert the chord back to µm. An empty input
    yields an all-zero map.
    """
    h, w = int(shape_px[0]), int(shape_px[1])
    out = np.zeros((h, w), dtype=float)
    if segments_um.size == 0:
        return out
    inv = 1.0 / pixel_size_um
    for y0, x0, y1, x1, r_um in segments_um:
        # endpoints and radius in pixels
        py0, px0, py1, px1, r_px = y0 * inv, x0 * inv, y1 * inv, x1 * inv, r_um * inv
        ymin = max(int(math.floor(min(py0, py1) - r_px)), 0)
        ymax = min(int(math.ceil(max(py0, py1) + r_px)), h - 1)
        xmin = max(int(math.floor(min(px0, px1) - r_px)), 0)
        xmax = min(int(math.ceil(max(px0, px1) + r_px)), w - 1)
        if ymin > ymax or xmin > xmax:
            continue  # segment lies entirely off-canvas
        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        # distance from each pixel to the segment (not the infinite line): project
        # onto the axis, clamp the parameter to [0, 1] so the caps are round.
        vy, vx = py1 - py0, px1 - px0
        len2 = vy * vy + vx * vx
        wy, wx = yy - py0, xx - px0
        t = np.clip((wy * vy + wx * vx) / len2, 0.0, 1.0) if len2 > _EPS else 0.0
        dy, dx = wy - t * vy, wx - t * vx
        d2 = dy * dy + dx * dx
        inside = d2 < r_px * r_px
        chord_px = np.zeros_like(d2)
        chord_px[inside] = 2.0 * np.sqrt(r_px * r_px - d2[inside])
        out[ymin : ymax + 1, xmin : xmax + 1] += chord_px * pixel_size_um
    return out


def vessels_to_mask(
    path_length_um: np.ndarray, opacity: float, absorption_per_um: float
) -> np.ndarray:
    """Beer-Lambert transmission mask from a blood-path-length map - in ``[1−opacity, 1]``.

    Light crossing ``L`` µm of blood is attenuated by ``exp(−absorption_per_um·L)``
    (Beer-Lambert), so the absorbed fraction is ``1 − exp(−k·L)``, scaling from 0
    (clear tissue) toward 1 (an infinitely thick trunk). ``opacity`` caps that
    darkest absorption in (0, 1]: real vessels never read fully black, because
    out-of-plane and in-front fluorescence fills their shadow in, so

        ``M = 1 − opacity·(1 − exp(−k·L))``

    floors the transmission at ``1 − opacity``. ``k`` (``absorption_per_um``) sets
    how fast darkness ramps with thickness - the contrast between fine capillaries
    and thick trunks - while ``opacity`` sets the floor. With ``opacity = 0`` or an
    all-zero map the mask is all ones (no vessels), and the result multiplies the
    brain-frame movie.
    """
    absorbed = opacity * (1.0 - np.exp(-absorption_per_um * path_length_um))
    return 1.0 - absorbed


def _seed_edge_root(
    bounds_um: tuple[float, float], rng: np.random.Generator
) -> tuple[tuple[float, float], float]:
    """A root point on a random field edge with an inward heading.

    Real vessels enter the field from the side, so roots are seeded on one of the
    four ``bounds_um = (H_um, W_um)`` edges with a heading that points into the
    canvas (image convention ``dy = sin θ``, ``dx = cos θ``). The inward heading is
    drawn from a wedge so trees fan in at varied angles rather than all running
    straight across. Returns ``((y, x), heading_rad)``.
    """
    h_um, w_um = bounds_um
    side = int(rng.integers(4))
    if side == 0:  # top edge -> grow downward (dy > 0)
        return (0.0, float(rng.uniform(0.0, w_um))), float(rng.uniform(0.25, 0.75) * math.pi)
    if side == 1:  # bottom edge -> grow upward (dy < 0)
        return (h_um, float(rng.uniform(0.0, w_um))), float(rng.uniform(1.25, 1.75) * math.pi)
    if side == 2:  # left edge -> grow rightward (dx > 0)
        return (float(rng.uniform(0.0, h_um)), 0.0), float(rng.uniform(-0.4, 0.4) * math.pi)
    return (float(rng.uniform(0.0, h_um)), w_um), float(rng.uniform(0.6, 1.4) * math.pi)  # right -> left


def vasculature_mask_field(
    spec, acq, shape: tuple[int, int], focal_um: float, rng: np.random.Generator
) -> np.ndarray:
    """The composite vessel transmission mask ``M(y, x)`` in (0, 1] for a spec.

    The RNG-consuming generation half of :class:`VasculatureStep`, factored out so
    the step *and* the streaming video writer build the **identical** mask from the
    same RNG draws (the same pattern as :func:`neuropil_components`). For each
    :class:`~minisim.spec.VesselLayer`, in list order: seed ``n_roots`` edge roots
    and :func:`grow_vessel_tree` them, :func:`rasterize_vessels` the segments to a
    blood-path-length map, :func:`vessels_to_mask` that to a Beer-Lambert
    transmission mask, then **blur** the mask by the defocus + scatter σ at the
    layer's ``depth_um`` (the optics model: a vessel near ``focal_um`` stays sharp,
    one far from focus softens). The per-layer masks compose **multiplicatively**
    (stacked vessels absorb more), starting from an all-ones (clear) field.

    ``shape`` is the canvas ``(h, w)`` in pixels (margin-enlarged, so the mask covers
    the same tissue the cells do and moves with it under motion). The draws run in a
    fixed order - per layer, per root: edge/heading then the tree's own draws - so
    the same RNG yields the same mask. The blur uses the on-axis focal plane (field
    curvature over a whole layer is a second-order effect, ignored here, unlike the
    per-cell curvature in ``optics``).
    """
    h, w = int(shape[0]), int(shape[1])
    px = acq.pixel_size_um
    bounds_um = (h * px, w * px)
    mask = np.ones((h, w), dtype=float)
    for layer in spec.layers:
        growth = VesselGrowth(
            r0_um=layer.root_radius_um,
            min_radius_um=layer.min_radius_um,
            branch_prob=layer.branch_prob,
            tortuosity_rad=math.radians(layer.tortuosity_deg),
            step_per_radius=layer.step_per_radius,
            branch_area_main=layer.branch_area_main,
            branch_angle_rad=math.radians(layer.branch_angle_deg),
        )
        trees = []
        for _ in range(layer.n_roots):
            root, heading = _seed_edge_root(bounds_um, rng)
            tree = grow_vessel_tree(bounds_um, root, heading, growth, rng)
            if tree.size:
                trees.append(tree)
        if not trees:
            continue
        path_um = rasterize_vessels(np.vstack(trees), (h, w), px)
        layer_mask = vessels_to_mask(path_um, layer.opacity, layer.absorption_per_um)
        # Defocus + scatter blur at this layer's depth (diffraction is the floor):
        # the same σ the optics step uses, so a vessel's shadow softens with its
        # distance from focus exactly as a cell's footprint does.
        sigma_um = math.hypot(
            acq.optics.diffraction_sigma_um,
            acq.tissue.scatter_sigma_um(layer.depth_um),
            acq.optics.defocus_sigma_um(layer.depth_um, focal_um),
        )
        layer_mask = gaussian_filter(layer_mask, sigma_um / px, mode="nearest")
        mask *= layer_mask
    return mask


def vasculature_focal(scene: Scene, acq) -> float:
    """The focal depth (µm) the vessel blur uses - the optics step's value, or a fallback.

    Prefers the concrete plane the ``optics`` step resolved into
    ``scene.truth.focal_depth_um`` (so "auto" focus stays consistent with the cells).
    If ``optics`` did not run, falls back to :func:`resolve_focal_plane` on the cells
    (a numeric ``focal_depth_in_tissue_um`` as-is; "auto" → the geometric median
    depth, or the surface with no cells), so the step is still valid in a minimal
    chain. Both the step and the streaming writer call this, so they blur identically.
    """
    focal = scene.truth.focal_depth_um
    if focal is not None:
        return float(focal)
    from minisim.steps.cell import resolve_focal_plane

    return resolve_focal_plane(scene.cells, acq.focal_depth_in_tissue_um)


class VasculatureStep(Step["Vasculature"]):
    """Multiply a dark, static vessel-absorption mask into the brain-frame movie.

    Grows the spec's :class:`~minisim.spec.VesselLayer` trees into a single
    transmission mask :func:`vasculature_mask_field` and applies it multiplicatively
    (``movie *= M``): vessels absorb the light from everything optically behind them.
    A *tissue*-domain step run before ``brain_motion``, so the mask is fixed in the
    brain frame and the motion crop carries it rigidly with the cells - the static,
    high-contrast landmark motion correction registers against, and a tunable
    extraction confound. The mask is stored canvas-sized on ``scene.truth`` (it must
    align with the canvas-coordinate footprints); ``finalize`` crops it to the FOV
    for ``GroundTruth.vasculature_mask`` and records each cell's footprint-weighted
    occlusion as ``GroundTruth.vessel_overlap_fraction``.

    A no-op when ``enabled`` is False or ``layers`` is empty (the step ships off),
    leaving the movie and the ground-truth slot untouched. The blur depth comes from
    :func:`vasculature_focal` (the resolved focal plane).
    """

    name = "vasculature"
    domain = "tissue"
    consumes_rng = True  # vessel-tree growth in vasculature_mask_field

    def __call__(self, scene: Scene) -> None:
        spec = self.spec
        if not spec.enabled or not spec.layers:
            return  # off: no draws, no mask (the streaming writer mirrors this guard)
        _, h, w = scene.movie.values.shape
        focal = vasculature_focal(scene, self.acq)
        mask = vasculature_mask_field(spec, self.acq, (h, w), focal, self.rng)
        scene.movie.values[:] *= mask
        scene.truth.vasculature_mask = mask

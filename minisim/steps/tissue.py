"""Tissue-domain steps: composite the cells, then add brain-bound field effects.

The tissue domain is the brain-frame stack - everything that moves rigidly with
the tissue under :mod:`~minisim` motion, as opposed to the
static optics/sensor fields (:mod:`minisim.steps.sensor`). It opens
with ``render`` (the cell→image boundary) and then layers the diffuse/global
effects that ride on top of the cells:

* :class:`RenderStep` (``cells_only``) - composite ``Σ footprint·trace`` into the
  movie; the first step to write ``scene.movie``.
* :class:`NeuropilStep` (``neuropil``) - additive diffuse background, a smooth
  spatial field modulated by a slow temporal envelope.
* :class:`VasculatureStep` (``vasculature``) - honest no-op placeholder; the
  absorbing-vessel model is deferred to v1.1.

All run before the motion boundary, so a later ``brain_motion`` step translates
the cells *and* these fields together (they are part of the brain frame), unlike
the static vignette/leakage applied after motion.

:class:`BleachingStep` (``bleaching``) also lives here but is a **cell-domain**
step: photobleaching is per-cell and activity-driven, so it runs *before* render
and writes each cell's intact-fluorophore envelope rather than touching the movie
(see its docstring). It is kept in this module beside the render/neuropil code it
coordinates with (render emits ``C·B``; neuropil fades with the population ``B``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter

from minisim.footprint import stack_dense
from minisim.scene import Scene
from minisim.steps.base import PipelineContext, Step

if TYPE_CHECKING:
    # Referenced only as string Generic bases (Step["Render"] etc.), which ruff's
    # F401 misses; pyright needs them in scope to resolve the forward references.
    from minisim.spec import Bleaching, Neuropil, Render, Vasculature  # noqa: F401

# Guards a divide-by-peak for a degenerate (flat) smooth field; far below any
# physically meaningful intensity.
_EPS = 1e-12

# Temporal fluctuation depth of the neuropil envelope, as the log-space sigma of
# its mean-1 lognormal modulation (see :func:`neuropil_envelope`). A fixed v1
# constant - slow background *drifts* by tens of percent rather than blinking;
# per-component variability could become a spec field later.
_NEUROPIL_FLUCT_LOG_STD = 0.4


class RenderStep(Step["Render"]):
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
        A = stack_dense(footprints, scene.canvas_shape)  # (unit, height, width)
        C = np.stack(traces)  # (unit, frame)
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


class BleachingStep(Step["Bleaching"]):
    """Per-cell, activity-driven fluorophore decay - bleaching fought by turnover.

    Gives each cell an intact-fluorophore envelope ``Bᵢ(t)`` from
    :func:`bleaching_pool`, driven by its own calcium trace (its emission) scaled by
    the excitation ``intensity``: busier, brighter-lit cells bleach faster and to a
    lower floor, while turnover pulls every cell back toward 1. A **cell-domain**
    step (it runs before ``render``, like ``cell_activity`` and ``optics``): it
    writes ``cell.bleach`` rather than touching the movie, so the trace stays the
    clean calcium ``C`` and ``render`` emits ``C·B``. The diffuse ``neuropil`` then
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
        ys = np.array([cell.center_um[1] for cell in scene.cells])
        xs = np.array([cell.center_um[2] for cell in scene.cells])
        iy = np.clip(np.round(ys / px), 0, shape[0] - 1).astype(int)
        ix = np.clip(np.round(xs / px), 0, shape[1] - 1).astype(int)
        return field[iy, ix]


# ---------------------------------------------------------------------------
# vasculature (placeholder)
# ---------------------------------------------------------------------------


class VasculatureStep(Step["Vasculature"]):
    """Honest no-op placeholder - the absorbing-vessel model is deferred to v1.1.

    The dark, pulsating vasculature mask (a multiplicative absorber driven by slow
    dilation + cardiac motion) is registered in the v1 catalog so the spec surface
    is stable, but its body is not implemented yet. It leaves the scene untouched
    rather than raising, so a spec that lists ``vasculature`` runs end-to-end with
    the effect simply absent (the ground-truth slot stays ``None``).
    """

    name = "vasculature"
    domain = "tissue"

    def __call__(self, scene: Scene) -> None:
        return  # no-op: deferred to v1.1

"""The ``simulate()`` orchestrator - compose a ``Spec`` into a ``Recording``.

This is the package's headline entry point: it walks ``spec.steps`` in order,
building and running each against a shared ``Scene``, then hands the exhausted
scene to :func:`~minisim.recording.finalize`. The loop itself is
deliberately tiny - the same readable ``for step in spec.steps`` a learner
follows - with two responsibilities beyond running the steps:

* **Motion margin sizing.** If the spec contains a ``brain_motion`` step, the
  tissue canvas must be padded by ≥ the maximum shift so real off-FOV tissue
  moves into view (see :mod:`minisim.steps.motion`). ``simulate``
  computes that margin from the motion spec and allocates the padded ``Scene``,
  so callers never hand-size it.
* **Per-stage snapshots.** With ``Output.save_intermediates`` set, the working
  movie is captured after each *movie-affecting* (non-``cell``-domain) step,
  keyed by the step's stage ``name`` (``cells_only``, ``neuropil``, …,
  ``sensor``). Cell-domain steps fill per-cell records, not the movie, so
  snapshotting them would only duplicate the prior (often blank) frame.

``until=<stage name>`` stops the run right after that stage - the partial-build
path the training notebook uses to reveal the pipeline one effect at a time.
"""

from __future__ import annotations

import numpy as np

from minisim.perf import PerfTracker, measure
from minisim.recording import Recording, finalize
from minisim.scene import Scene
from minisim.spec import Acquisition, Spec
from minisim.steps.base import PipelineContext
from minisim.steps.sensor import combined_falloff_field


def simulate(
    spec: Spec, *, until: str | None = None, perf: PerfTracker | None = None
) -> Recording:
    """Run a full recording specification and return the typed ``Recording``.

    Seeds the RNG from ``spec.seed`` (so a spec + seed fully determines the
    output), sizes the motion margin, runs the steps in ``spec.steps`` order -
    already the canonical pipeline order, since ``Spec`` sorts the list on
    construction, so the order a caller listed them in is irrelevant - optionally
    snapshots each movie stage, and finalizes. ``until`` stops after the named
    stage (a ``step.name``, e.g. ``"vignette"``); an ``until`` that matches no
    step raises rather than silently running the whole pipeline.

    Pass a :class:`~minisim.perf.PerfTracker` as ``perf`` to record per-step (and
    ``finalize``) wall time; it is a no-op when ``None`` (the default), so an
    un-profiled run pays nothing for the instrumentation.
    """
    acq = spec.acquisition
    rng = np.random.default_rng(spec.seed)
    scene = Scene.zeros(acq, rng, margin_px=_motion_margin_px(spec, acq))
    context = build_context(spec, acq)

    stopped = False
    stage_names: list[str] = []
    for step_spec in spec.steps:
        step = step_spec.build(acq, rng)
        step.prepare(context)
        with measure(perf, step.name, domain=step.domain):
            step(scene)
        stage_names.append(step.name)
        if spec.output.save_intermediates and step.domain != "cell":
            scene.snapshots[step.name] = scene.movie.copy()
        if until is not None and step.name == until:
            stopped = True
            break

    if until is not None and not stopped:
        raise ValueError(
            f"until={until!r} matched no step in this spec; stage names are {stage_names}."
        )
    with measure(perf, "finalize"):
        return finalize(scene, spec)


def build_context(spec: Spec, acq: Acquisition) -> PipelineContext:
    """Resolve the :class:`~minisim.steps.base.PipelineContext` for a run.

    Looks up the sensor-domain specs that earlier cell-domain steps depend on (the
    illumination profile, the sensor spec, and the illumination × vignette photon
    budget) so each step can pull what it needs from the context in ``prepare``.
    Each slot falls back to ``None`` / a uniform field when its step is absent.

    Shared with the streaming writer (:func:`minisim.video._iter_count_frames`) so
    both paths resolve the same context and make identical focus/bleaching
    decisions, keeping the streamed counts bit-for-bit equal to ``simulate``.
    """
    illumination = next((s for s in spec.steps if s.kind == "illumination_profile"), None)
    vignette = next((s for s in spec.steps if s.kind == "vignette"), None)
    sensor_spec = next((s for s in spec.steps if s.kind == "sensor"), None)
    photon_field = combined_falloff_field(acq, illumination, vignette)
    return PipelineContext(
        illumination=illumination, sensor_spec=sensor_spec, photon_field=photon_field
    )


def _motion_margin_px(spec: Spec, acq: Acquisition) -> int:
    """Tissue-canvas margin (px) needed to keep the FOV crop on real tissue.

    Zero when the spec has no ``brain_motion`` step. Otherwise the maximum shift
    (the bound for a random walk, or the largest entry of an explicit trajectory)
    converted to pixels and rounded up, plus a one-pixel guard for the bilinear
    interpolation boundary. ``brain_motion`` then fails fast if a shift somehow
    exceeds this margin.
    """
    motion = next((s for s in spec.steps if s.kind == "brain_motion"), None)
    if motion is None:
        return 0
    if motion.trajectory_um is not None:
        max_um = max(
            (max(abs(dy), abs(dx)) for dy, dx in motion.trajectory_um), default=0.0
        )
    else:
        max_um = motion.max_shift_um
    px = acq.um_to_px(max_um)
    return int(np.ceil(px)) + 1 if px > 0 else 0

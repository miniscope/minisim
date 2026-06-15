"""Helpers for using minisim as test fixtures for an analysis pipeline.

minisim's second core use is supplying reproducible, ground-truth-carrying
recordings to the test suite of a calcium-imaging pipeline (minian, CaImAn,
suite2p, ...). The rest of the package gives you everything needed for that - a
typed :class:`~minisim.Spec`, :func:`~minisim.simulate`, and the recovery
:mod:`~minisim.metrics` - but assembling a small fixture and scoring a pipeline
against it both take several coordinated calls and a few conventions. This module
collapses each into one:

* :func:`make_recording` - a small, fast, deterministic :class:`~minisim.Recording`
  for CI, in one call. The same ``seed`` always yields the same recording.
* :func:`score` - run the common recovery scorecard (cell recall/precision, trace
  correlation, spike precision/recall, optional motion error) in one call,
  returning a :class:`Report`.

Both are thin wrappers over the public API; the a-la-carte metrics
(:func:`~minisim.hungarian_match`, :func:`~minisim.trace_pearson`, ...) remain the
foundation, for the cases this 90%-path does not cover.

**Dependency direction.** minisim never imports an analysis pipeline, so importing
``minisim.testing`` as a *test-only* dependency of such a pipeline cannot create an
import cycle. See the how-to guide ``docs/howto/use_in_test_suite.md`` for the
recommended ``pytest`` wiring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from minisim.metrics import (
    hungarian_match,
    shift_rmse,
    spike_precision_recall,
    trace_pearson,
)
from minisim.recording import GroundTruth, Recording
from minisim.simulate import simulate
from minisim.spec import (
    Acquisition,
    AnyStep,
    BrainMotion,
    CellActivity,
    CellOptics,
    Composite,
    ImageSensor,
    NeuronPopulation,
    Optics,
    Output,
    PlaceNeurons,
    Sensor,
    Spec,
)

__all__ = ["Estimate", "Report", "make_recording", "score"]


# ---------------------------------------------------------------------------
# make_recording - a one-call CI fixture
# ---------------------------------------------------------------------------


def _grid_positions_um(
    n: int, fov_um: float, depth_um: float, rng: np.random.Generator, *, margin_frac: float = 0.15
) -> list[tuple[float, float, float]]:
    """``n`` well-separated ``(z, y, x)`` soma centers on a jittered grid.

    Lays the cells on the smallest near-square grid that holds ``n`` of them,
    spanning the central ``1 - 2·margin_frac`` of the FOV (so footprints stay off
    the edge), all at the same depth ``z = depth_um``. A small deterministic jitter
    (seeded by ``rng``) keeps them off a perfect lattice without risking overlap.
    The optical-center frame puts ``(0, 0)`` on the axis, so positions are centered.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    usable = fov_um * (1.0 - 2.0 * margin_frac)
    # Cell-center coordinates evenly spaced across `usable`, centered on the axis.
    ys = np.linspace(-usable / 2.0, usable / 2.0, rows) if rows > 1 else np.array([0.0])
    xs = np.linspace(-usable / 2.0, usable / 2.0, cols) if cols > 1 else np.array([0.0])
    spacing = usable / max(rows, cols, 1)
    jitter = 0.15 * spacing
    out: list[tuple[float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            if len(out) >= n:
                break
            dy, dx = rng.uniform(-jitter, jitter, size=2)
            out.append((depth_um, float(ys[r] + dy), float(xs[c] + dx)))
    return out


def make_recording(
    *,
    n_cells: int = 6,
    n_px: int = 128,
    pixel_size_um: float = 1.0,
    duration_s: float = 2.0,
    fps: float = 20.0,
    seed: int = 0,
    depth_um: float = 50.0,
    soma_radius_um: float = 5.0,
    morphology: str = "soma",
    motion: bool = False,
    activity: CellActivity | None = None,
    sensor: Sensor | None = None,
    extra_steps: Sequence[AnyStep] = (),
    save_intermediates: bool = False,
) -> Recording:
    """A small, fast, deterministic recording with ground truth, for CI tests.

    Places ``n_cells`` well-separated somata on a jittered grid at a single
    ``depth_um`` and runs the minimal forward chain (``place_neurons →
    cell_activity → optics → composite → sensor``) on a square ``n_px × n_px``
    sensor at ``pixel_size_um`` per pixel. Cell count, positions, and every pixel
    are fully determined by the arguments, so the same ``seed`` (and the same
    arguments) always yields the same recording - the property a fixture needs.

    Defaults are tuned for CI: a 128 px FOV at 1 µm/px (128 µm), six cells at
    50 µm depth, two seconds at 20 fps, with a lively, brightly-exposed default
    activity/sensor so every placed cell reliably fires and is *detectable*
    (a flat or too-dim cell would otherwise be an unfair recall miss). Shrink
    ``n_px`` / ``duration_s`` for an even faster fixture, or raise ``n_cells`` for
    a denser one.

    Parameters
    ----------
    n_cells
        Number of somata to place (exactly this many; grid-arranged and separated).
    n_px
        Square sensor side, pixels (both height and width).
    pixel_size_um
        Object-space size of one pixel, µm. Realized via a sensor pitch of
        ``8.0`` µm and a matching magnification, so the FOV is ``n_px · pixel_size_um``.
    duration_s, fps, seed
        Sampling and the master RNG seed.
    depth_um
        Common depth of every soma below the tissue surface, µm. The focal plane
        is ``"auto"``, so it resolves here and the cells are in focus.
    soma_radius_um, morphology
        Cell shape: ``"soma"`` (body only, fast) or ``"cytosolic"`` (soma plus
        proximal dendrites).
    motion
        When ``True``, append a default :class:`~minisim.BrainMotion` step, so
        ``ground_truth.shifts`` is populated (lets you exercise motion correction).
    activity, sensor
        Override the :class:`~minisim.CellActivity` model or the
        :class:`~minisim.Sensor` exposure step; ``None`` uses defaults.
    extra_steps
        Additional steps to append (``Neuropil``, ``Vignette``, ``Leakage``, ...);
        the spec re-sorts into canonical order, so order here is free.
    save_intermediates
        Persist per-stage snapshots (see :class:`~minisim.Output`).

    Returns
    -------
    Recording
        ``rec.observed`` is the movie, ``rec.ground_truth`` the exact truth.
    """
    if n_cells < 1:
        raise ValueError(f"n_cells ({n_cells}) must be >= 1.")
    fov_um = n_px * pixel_size_um
    # Realize pixel_size_um as pitch / magnification with a fixed 8 µm pitch.
    pitch_um = 8.0
    magnification = pitch_um / pixel_size_um
    positions = _grid_positions_um(n_cells, fov_um, depth_um, np.random.default_rng(seed))
    acquisition = Acquisition(
        optics=Optics(magnification=magnification),
        image_sensor=ImageSensor(n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=pitch_um),
        fps=fps,
        duration_s=duration_s,
        focal_depth_in_tissue_um="auto",
    )
    steps: list[AnyStep] = [
        PlaceNeurons(
            populations=[
                NeuronPopulation(
                    positions_um=positions,
                    soma_radius_um=soma_radius_um,
                    morphology=morphology,  # type: ignore[arg-type]
                )
            ]
        ),
        # Lively + brightly exposed by default: every cell reliably fires a
        # transient in the short clip and clears the sensor noise floor, so the
        # cells are detectable (a flat or too-dim cell is not a fair recall miss).
        activity or CellActivity(p_quiescent_to_active=0.05),
        CellOptics(),
        Composite(),
        sensor or Sensor(photons_per_unit=600.0),
    ]
    if motion:
        steps.append(BrainMotion())
    steps.extend(extra_steps)
    spec = Spec(
        acquisition=acquisition,
        seed=seed,
        steps=steps,
        output=Output(save_intermediates=save_intermediates),
    )
    return simulate(spec)


# ---------------------------------------------------------------------------
# score - a one-call recovery scorecard
# ---------------------------------------------------------------------------


class Estimate(NamedTuple):
    """What a pipeline recovered, as the input to :func:`score`.

    ``A`` is the only required field. ``C`` / ``S`` / ``shifts`` are optional; when
    ``None``, the matching score in the :class:`Report` is ``nan`` (or ``None`` for
    motion). Arrays may be ``numpy`` or ``xarray`` (minian's CNMF returns
    ``xr.DataArray``); both are accepted.

    * ``A`` - spatial footprints, ``(n_units, height, width)``.
    * ``C`` - calcium traces, ``(n_units, frame)``.
    * ``S`` - deconvolved spikes, ``(n_units, frame)``.
    * ``shifts`` - estimated per-frame ``(dy, dx)`` motion, ``(frame, 2)``. A motion
      *correction* trajectory (the negation of the applied shift); :func:`score`
      negates it to compare against ``GroundTruth.shifts``.
    """

    A: object
    C: object | None = None
    S: object | None = None
    shifts: object | None = None


@dataclass(frozen=True)
class Report:
    """The recovery scorecard :func:`score` returns.

    Cell counts and spatial scores are always present; temporal scores are ``nan``
    when the matching estimate field was not supplied, and ``shift_rmse`` is
    ``None`` when motion truth or a motion estimate is absent.
    """

    n_true: int  # ground-truth cells scored against (after the detectable filter)
    n_est: int  # estimated cells supplied
    n_matched: int  # matched pairs clearing the IoU threshold (true positives)
    recall: float  # n_matched / n_true
    precision: float  # n_matched / n_est
    mean_iou: float  # mean spatial overlap over matched pairs
    trace_corr: float  # median Pearson r of matched traces (nan if no C / no match)
    spike_precision: float  # pooled spike precision (nan if no S)
    spike_recall: float  # pooled spike recall (nan if no S)
    shift_rmse: float | None  # motion RMSE in px (None if no motion truth/estimate)

    def summary(self) -> str:
        """A compact one-line-per-metric string, handy for a test failure message."""
        lines = [
            f"cells: recall={self.recall:.2f} precision={self.precision:.2f} "
            f"(matched {self.n_matched}/{self.n_true}, mean IoU {self.mean_iou:.2f})",
            f"traces: median r={self.trace_corr:.2f}",
            f"spikes: precision={self.spike_precision:.2f} recall={self.spike_recall:.2f}",
        ]
        if self.shift_rmse is not None:
            lines.append(f"motion: RMSE={self.shift_rmse:.2f} px")
        return "\n".join(lines)


def score(
    estimate: Estimate,
    ground_truth: GroundTruth,
    *,
    iou_threshold: float = 0.5,
    tol_frames: int = 2,
    restrict_to_detectable: bool = True,
) -> Report:
    """Score a pipeline's :class:`Estimate` against the ground truth, in one call.

    Runs the standard recovery pipeline: match estimated footprints to true ones by
    spatial overlap (:func:`~minisim.hungarian_match` against ``A_observed``, the
    recoverable target), then score the temporal recovery of the matched pairs. The
    conventions the a-la-carte recipe asks you to remember are applied here:

    * matches against ``A_observed`` (not the optics-free ``A_planted``);
    * scores recall over the **detectable** cells by default (the fair denominator;
      see :meth:`~minisim.GroundTruth.detectable_subset`);
    * reduces per-pair trace correlations with a nan-safe median;
    * treats an estimated motion trajectory as a *correction* (negated) when
      comparing to ``GroundTruth.shifts``.

    Parameters
    ----------
    estimate
        The pipeline output. Only :attr:`Estimate.A` is required.
    ground_truth
        The recording's ``ground_truth``.
    iou_threshold
        Minimum IoU for a matched pair to count as a true positive.
    tol_frames
        Spike-timing tolerance, frames (see
        :func:`~minisim.spike_precision_recall`).
    restrict_to_detectable
        Score against :meth:`~minisim.GroundTruth.detectable_subset` (default).
        Set ``False`` to score against every planted cell, detectable or not.

    Returns
    -------
    Report
        The scorecard. Unsupplied estimate fields score ``nan`` / ``None``.
    """
    gt = ground_truth.detectable_subset() if restrict_to_detectable else ground_truth
    match = hungarian_match(estimate.A, gt.A_observed)
    matched = match.matched_pairs(iou_threshold)

    if estimate.C is not None and match.pairing:
        r = trace_pearson(estimate.C, gt.C, match.pairing)
        trace_corr = float(np.nanmedian(r)) if r.size else float("nan")
    else:
        trace_corr = float("nan")

    if estimate.S is not None and match.pairing:
        spikes = spike_precision_recall(estimate.S, gt.S, match.pairing, tol_frames=tol_frames)
        spike_precision, spike_recall = spikes.precision, spikes.recall
    else:
        spike_precision = spike_recall = float("nan")

    rmse = (
        shift_rmse(estimate.shifts, gt.shifts, correction=True)
        if estimate.shifts is not None and gt.shifts is not None
        else None
    )

    return Report(
        n_true=match.n_true,
        n_est=match.n_est,
        n_matched=len(matched),
        recall=match.recall(iou_threshold),
        precision=match.precision(iou_threshold),
        mean_iou=match.mean_iou,
        trace_corr=trace_corr,
        spike_precision=spike_precision,
        spike_recall=spike_recall,
        shift_rmse=rmse,
    )

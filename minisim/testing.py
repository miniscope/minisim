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

Both are thin wrappers over the public API, built on the a-la-carte metrics
(:func:`~minisim.hungarian_match`, :func:`~minisim.trace_pearson`, ...), which stay
available for anything they do not cover.

**Dependency direction.** minisim depends only on its own stack, so a pipeline can
take ``minisim.testing`` as a *test-only* dependency. See the how-to guide
``docs/howto/use_in_test_suite.md`` for the recommended ``pytest`` wiring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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

# The default fixture exposure (photons per intensity unit). Chosen so the default
# scene's cells sit bright and clearly above the detection threshold with headroom
# below the 8-bit ADC ceiling. Fixed (not auto-leveled) so the fixture is reproducible
# and confound-invariant; override via the ``sensor`` argument for other regimes.
_DEFAULT_PHOTONS_PER_UNIT = 40.0


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
    morphology: Literal["soma", "cytosolic"] = "soma",
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
    50 µm depth, two seconds at 20 fps, with a lively default activity and a fixed,
    well-chosen exposure (the focal plane is ``"auto"``), so every placed cell reliably
    fires, is in focus, and is brightly but non-saturatingly exposed - hence
    *detectable* - with no manual tuning. Shrink ``n_px`` / ``duration_s`` for an even
    faster fixture, or raise ``n_cells`` for a denser one.

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
        :class:`~minisim.Sensor` exposure step; ``None`` uses defaults (the default
        sensor uses a fixed, well-exposed ``photons_per_unit``). Pass an explicit
        ``Sensor(photons_per_unit=...)`` for a dimmer or brighter recording.
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
                    morphology=morphology,
                )
            ]
        ),
        # Lively by default: every cell fires a transient in the short clip.
        activity or CellActivity(p_quiescent_to_active=0.05),
        CellOptics(),
        Composite(),
        # A fixed, well-chosen exposure (no auto-leveling): bright, clear, every cell
        # comfortably above the noise floor, with headroom below the ADC ceiling - and
        # the same for every scene, so the fixture is reproducible and a confound added
        # via extra_steps changes detectability only through its own physics, not through
        # a re-leveled exposure. Override with an explicit Sensor for a dimmer/brighter run.
        sensor or Sensor(photons_per_unit=_DEFAULT_PHOTONS_PER_UNIT),
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


_UNSET: object = object()  # distinguishes "not passed" from an explicit None


@dataclass(frozen=True, init=False)
class Estimate:
    """What a pipeline recovered, as the input to :func:`score`.

    The footprints are the only required field. Traces / spikes / shifts are
    optional; when omitted, the matching score in the :class:`Report` is ``nan``
    (or ``None`` for motion). Arrays may be ``numpy`` or ``xarray`` (minian's CNMF
    returns ``xr.DataArray``); both are accepted.

    Each field has two interchangeable spellings - the terse CNMF/minian symbol a
    pipeline already emits, and a spelled-out alias for anyone who does not speak
    that dialect. **Both work as keyword arguments**, and both are readable
    attributes::

        Estimate(A=A, C=C, S=S)                          # CNMF names
        Estimate(footprints=A, traces=C, spikes=S)       # spelled out

    * ``A`` / ``footprints`` - spatial footprints, ``(n_units, height, width)``.
    * ``C`` / ``traces`` - calcium traces, ``(n_units, frame)``.
    * ``S`` / ``spikes`` - deconvolved spikes, ``(n_units, frame)``.
    * ``shifts`` - estimated per-frame ``(dy, dx)`` motion, ``(frame, 2)``. A motion
      *correction* trajectory (the negation of the applied shift); :func:`score`
      negates it to compare against ``GroundTruth.shifts``.

    A frozen dataclass (not a ``NamedTuple``) on purpose: keyword construction with
    no positional contract, so a future field can be added without breaking callers
    that pin the current ones - the property a long-lived public scoring contract
    needs.
    """

    A: object
    C: object | None = None
    S: object | None = None
    shifts: object | None = None

    def __init__(
        self,
        A: object = _UNSET,
        C: object | None = None,
        S: object | None = None,
        shifts: object | None = None,
        *,
        footprints: object = _UNSET,
        traces: object | None = None,
        spikes: object | None = None,
    ) -> None:
        a = footprints if footprints is not _UNSET else A
        if a is _UNSET:
            raise TypeError(
                "Estimate requires the footprints (pass A=... or footprints=...)."
            )
        if footprints is not _UNSET and A is not _UNSET:
            raise TypeError("pass footprints= or A=, not both.")
        # Frozen dataclass: bypass the immutability guard to set the canonical fields.
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "C", traces if C is None else C)
        object.__setattr__(self, "S", spikes if S is None else S)
        object.__setattr__(self, "shifts", shifts)

    @property
    def footprints(self) -> object:
        """Spelled-out alias for :attr:`A` (spatial footprints)."""
        return self.A

    @property
    def traces(self) -> object | None:
        """Spelled-out alias for :attr:`C` (calcium traces)."""
        return self.C

    @property
    def spikes(self) -> object | None:
        """Spelled-out alias for :attr:`S` (deconvolved spikes)."""
        return self.S


@dataclass(frozen=True)
class Report:
    """The recovery scorecard :func:`score` returns.

    Cell counts and spatial scores are always present; temporal scores are ``nan``
    when the matching estimate field was not supplied, and ``shift_rmse`` is
    ``None`` when motion truth or a motion estimate is absent.

    **The recall denominator is reported explicitly**, because it is not obvious:
    by default :func:`score` scores against the *detectable* cells, so ``recall``
    is over ``n_true`` (= ``n_detectable``), which can be fewer than the cells that
    were planted. The three counts make that legible at a glance - ``recall = 1.0``
    with ``n_detectable=4 < n_requested=6`` means "recovered every detectable cell,
    but two planted cells were too dim to detect", not "recovered everything". See
    :func:`score`'s ``restrict_to_detectable``.
    """

    n_true: int  # ground-truth cells scored against = the recall denominator
    n_est: int  # estimated cells supplied
    n_matched: int  # matched pairs clearing the IoU threshold (true positives)
    recall: float  # n_matched / n_true
    precision: float  # n_matched / n_est
    f1: float  # harmonic mean of precision and recall (0 when both are 0)
    mean_iou: float  # mean spatial overlap over matched pairs
    trace_corr: float  # median Pearson r of matched traces (nan if no C / no match)
    spike_precision: float  # pooled spike precision (nan if no S)
    spike_recall: float  # pooled spike recall (nan if no S)
    shift_rmse: float | None  # motion RMSE in px (None if no motion truth/estimate)
    n_requested: int  # total ground-truth cells planted (the full population)
    n_detectable: int  # cells clearing the detection floor (the detectable subset)

    def summary(self) -> str:
        """A compact one-line-per-metric string, handy for a test failure message."""
        lines = [
            f"cells: recall={self.recall:.2f} precision={self.precision:.2f} "
            f"f1={self.f1:.2f} (matched {self.n_matched}/{self.n_true}, "
            f"mean IoU {self.mean_iou:.2f})",
            f"population: {self.n_detectable} detectable of {self.n_requested} "
            f"planted (recall denominator = {self.n_true})",
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
      see :meth:`~minisim.GroundTruth.detectable_subset`). The returned
      :class:`Report` always carries ``n_requested`` (cells planted) and
      ``n_detectable`` alongside ``n_true`` (the denominator used), so a high
      ``recall`` over a shrunken denominator can never be mistaken for "recovered
      everything";
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
    # Record the full population and the detectable count up front, so the Report
    # always shows what the recall denominator (match.n_true) was drawn from -
    # whether or not the detectable filter is applied.
    n_requested = ground_truth.n_units
    n_detectable = int(np.asarray(ground_truth.detectable).sum())
    gt = ground_truth.detectable_subset() if restrict_to_detectable else ground_truth
    match = hungarian_match(estimate.A, gt.A_observed)
    matched = match.matched_pairs(iou_threshold)
    n_matched = len(matched)
    recall = n_matched / match.n_true if match.n_true else 0.0
    precision = n_matched / match.n_est if match.n_est else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

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
        n_matched=n_matched,
        recall=recall,
        precision=precision,
        f1=f1,
        mean_iou=match.mean_iou,
        trace_corr=trace_corr,
        spike_precision=spike_precision,
        spike_recall=spike_recall,
        shift_rmse=rmse,
        n_requested=n_requested,
        n_detectable=n_detectable,
    )

"""Unit tests for the minisim.testing fixture + scoring helpers.

Covers ``make_recording`` (a one-call deterministic CI fixture: exact cell count,
seed reproducibility, the size/speed knobs, optional motion and snapshots) and
``score`` (the one-call recovery scorecard: a perfect estimate scores ~1, an empty
one scores 0, the detectable filter, and motion RMSE), plus the ``until=`` /
``stage()`` ``composite`` alias and the ``shift_rmse`` correction flag.
"""

import math

import numpy as np
import pytest

from minisim import (
    CellActivity,
    PlaceNeurons,
    Sensor,
    shift_rmse,
    simulate,
)
from minisim.testing import Estimate, Report, make_recording, score

# --- make_recording --------------------------------------------------------


def test_make_recording_places_exactly_n_cells():
    rec = make_recording(n_cells=4, n_px=96, duration_s=1.0)
    assert rec.ground_truth.n_units == 4
    assert rec.observed.shape == (20, 96, 96)  # 1.0 s x 20 fps, 96x96 sensor


def test_make_recording_is_deterministic_in_seed():
    a = make_recording(seed=7, n_cells=3, duration_s=1.0)
    b = make_recording(seed=7, n_cells=3, duration_s=1.0)
    np.testing.assert_array_equal(np.asarray(a.observed), np.asarray(b.observed))


def test_make_recording_different_seed_differs():
    a = make_recording(seed=1, n_cells=3, duration_s=1.0)
    b = make_recording(seed=2, n_cells=3, duration_s=1.0)
    assert not np.array_equal(np.asarray(a.observed), np.asarray(b.observed))


def test_make_recording_rejects_zero_cells():
    with pytest.raises(ValueError, match="n_cells"):
        make_recording(n_cells=0)


def test_make_recording_motion_populates_shifts():
    rec = make_recording(n_cells=3, duration_s=1.0, motion=True)
    assert rec.ground_truth.shifts is not None
    assert rec.ground_truth.shifts.shape == (20, 2)


def test_make_recording_no_motion_has_no_shifts():
    rec = make_recording(n_cells=3, duration_s=1.0)
    assert rec.ground_truth.shifts is None


def test_make_recording_save_intermediates_snapshots():
    rec = make_recording(n_cells=3, duration_s=1.0, save_intermediates=True)
    assert "cells_only" in rec.snapshots
    assert "sensor" in rec.snapshots


def test_make_recording_accepts_overrides():
    rec = make_recording(
        n_cells=2, duration_s=1.0, activity=CellActivity(active_rate_hz=200.0),
        sensor=Sensor(photons_per_unit=400.0),
    )
    assert rec.ground_truth.n_units == 2


# --- score: perfect and empty estimates ------------------------------------


def _perfect_estimate(rec) -> tuple[Estimate, object]:
    """Feed the detectable ground truth straight back as a flawless estimate."""
    det = rec.ground_truth.detectable_subset()
    return Estimate(A=det.A_observed, C=det.C, S=det.S), det


def test_score_perfect_estimate_recovers_everything():
    rec = make_recording(n_cells=6, duration_s=3.0, seed=0)
    est, det = _perfect_estimate(rec)
    report = score(est, rec.ground_truth)
    assert isinstance(report, Report)
    assert report.n_true == det.n_units
    assert report.recall == 1.0
    assert report.precision == 1.0
    assert report.mean_iou > 0.99
    # Identical traces/spikes for the active cells -> near-perfect temporal scores.
    assert report.trace_corr > 0.99
    assert report.spike_precision == 1.0
    assert report.spike_recall == 1.0


def test_score_empty_estimate_scores_zero():
    rec = make_recording(n_cells=4, duration_s=1.0)
    h, w = rec.ground_truth.fov_shape
    report = score(Estimate(A=np.zeros((0, h, w))), rec.ground_truth)
    assert report.n_est == 0
    assert report.recall == 0.0
    assert report.precision == 0.0
    assert math.isnan(report.trace_corr)  # no matched pairs
    assert report.shift_rmse is None  # no motion in this recording


def test_score_footprints_only_leaves_temporal_nan():
    rec = make_recording(n_cells=4, duration_s=1.0)
    det = rec.ground_truth.detectable_subset()
    report = score(Estimate(A=det.A_observed), rec.ground_truth)  # no C / S
    assert report.recall == 1.0
    assert math.isnan(report.trace_corr)
    assert math.isnan(report.spike_precision)


def test_score_restrict_to_detectable_changes_denominator():
    rec = make_recording(n_cells=6, duration_s=1.0, depth_um=120.0, seed=3)
    est, _ = _perfect_estimate(rec)
    restricted = score(est, rec.ground_truth, restrict_to_detectable=True)
    full = score(est, rec.ground_truth, restrict_to_detectable=False)
    # The detectable denominator is never larger than the full planted count.
    assert restricted.n_true <= full.n_true


def test_score_motion_rmse_perfect_correction_is_zero():
    rec = make_recording(n_cells=4, duration_s=1.0, motion=True, seed=1)
    det = rec.ground_truth.detectable_subset()
    # A perfect motion-correction estimate is the negation of the applied shift.
    est = Estimate(A=det.A_observed, shifts=-np.asarray(rec.ground_truth.shifts))
    report = score(est, rec.ground_truth)
    assert report.shift_rmse == pytest.approx(0.0, abs=1e-9)


def test_report_summary_is_a_string():
    rec = make_recording(n_cells=3, duration_s=1.0)
    est, _ = _perfect_estimate(rec)
    summary = score(est, rec.ground_truth).summary()
    assert isinstance(summary, str)
    assert "recall=" in summary


# --- shift_rmse correction flag --------------------------------------------


def test_shift_rmse_same_convention_is_zero():
    x = np.array([[1.0, -2.0], [0.5, 0.5]])
    assert shift_rmse(x, x) == pytest.approx(0.0)


def test_shift_rmse_correction_negates_estimate():
    truth = np.array([[1.0, -2.0], [0.5, 0.5]])
    correction = -truth  # what a correction pipeline would emit
    assert shift_rmse(correction, truth, correction=True) == pytest.approx(0.0)
    assert shift_rmse(correction, truth, correction=False) > 0.0


# --- composite / cells_only alias ------------------------------------------


def _minimal_spec(**output):
    from minisim import (
        Acquisition,
        CellOptics,
        Composite,
        ImageSensor,
        Optics,
        Output,
        Spec,
    )

    return Spec(
        acquisition=Acquisition(
            optics=Optics(magnification=8.0),
            image_sensor=ImageSensor(n_px_height=48, n_px_width=48, pixel_pitch_um=8.0),
            duration_s=1.0,
        ),
        seed=0,
        steps=[PlaceNeurons(), CellActivity(), CellOptics(), Composite(), Sensor()],
        output=Output(**output),
    )


def test_until_accepts_composite_kind_alias():
    rec = simulate(_minimal_spec(save_intermediates=True), until="composite")
    # Stops right after the composite step, whose stage name is "cells_only".
    assert set(rec.snapshots) == {"cells_only"}


def test_until_composite_matches_cells_only():
    by_kind = simulate(_minimal_spec(save_intermediates=True), until="composite")
    by_name = simulate(_minimal_spec(save_intermediates=True), until="cells_only")
    assert set(by_kind.snapshots) == set(by_name.snapshots)


def test_stage_resolves_composite_alias():
    rec = simulate(_minimal_spec(save_intermediates=True))
    np.testing.assert_array_equal(
        rec.stage("composite").values, rec.stage("cells_only").values
    )

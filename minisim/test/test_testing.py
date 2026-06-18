"""Unit tests for the minisim.testing fixture + scoring helpers.

Covers ``make_recording`` (a one-call deterministic CI fixture: exact cell count,
seed reproducibility, the size/speed knobs, optional motion and snapshots) and
``score`` (the one-call recovery scorecard: a perfect estimate scores ~1, an empty
one scores 0, the detectable filter, activity scoring, global-offset absorption,
and motion RMSE), plus the ``until=`` / ``stage()`` ``composite`` alias and the
``shift_rmse`` correction flag.
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
        n_cells=2,
        duration_s=1.0,
        activity=CellActivity(active_rate_hz=200.0),
        sensor=Sensor(photons_per_unit=400.0),
    )
    assert rec.ground_truth.n_units == 2


def test_make_recording_pins_focal_plane():
    # The default "auto" focus resolves the plane onto the cells; a fixed
    # focal_depth_um holds the plane still wherever the cells are placed - the setup a
    # recall-vs-depth sweep needs (auto-focus would refocus on each depth).
    auto = make_recording(n_cells=4, duration_s=1.0, depth_um=80.0, seed=0)
    pinned = make_recording(
        n_cells=4, duration_s=1.0, depth_um=80.0, focal_depth_um=0.0, seed=0
    )
    assert auto.ground_truth.focal_depth_um == 80.0
    assert pinned.ground_truth.focal_depth_um == 0.0


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
    assert report.mean_overlap > 0.99
    # Identical traces/activity for the active cells -> near-perfect temporal scores.
    assert report.trace_corr > 0.99
    assert report.activity_corr > 0.99
    assert report.activity_variance_explained > 0.99
    assert report.activity_scale == pytest.approx(1.0)  # identical -> unit gain


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
    assert math.isnan(report.activity_corr)
    assert math.isnan(report.activity_variance_explained)


def test_score_restrict_to_detectable_changes_denominator():
    rec = make_recording(n_cells=6, duration_s=1.0, depth_um=120.0, seed=3)
    est, _ = _perfect_estimate(rec)
    restricted = score(est, rec.ground_truth, restrict_to_detectable=True)
    full = score(est, rec.ground_truth, restrict_to_detectable=False)
    # The detectable denominator is never larger than the full planted count.
    assert restricted.n_true <= full.n_true


def test_report_surfaces_the_recall_denominator():
    # The point of n_requested / n_detectable: a high recall over a shrunken
    # denominator must be legible, not silent. Use a deep population so some cells
    # fall below the detection floor (n_detectable < n_requested).
    rec = make_recording(n_cells=6, duration_s=1.0, depth_um=120.0, seed=3)
    est, _ = _perfect_estimate(rec)
    report = score(est, rec.ground_truth, restrict_to_detectable=True)
    assert report.n_requested == rec.ground_truth.n_units == 6
    assert report.n_detectable == int(rec.ground_truth.detectable.sum())
    # Under the detectable filter, the recall denominator IS the detectable count.
    assert report.n_true == report.n_detectable
    # n_requested is invariant to the filter; the denominator is not.
    full = score(est, rec.ground_truth, restrict_to_detectable=False)
    assert full.n_requested == report.n_requested == 6
    assert full.n_true == 6
    assert report.summary().count("planted") == 1  # the population line is present


def test_report_f1_is_harmonic_mean():
    rec = make_recording(n_cells=5, duration_s=1.0, seed=0)
    est, _ = _perfect_estimate(rec)
    report = score(est, rec.ground_truth)
    expected = (
        2.0 * report.precision * report.recall / (report.precision + report.recall)
    )
    assert report.f1 == pytest.approx(expected)


def test_report_f1_is_zero_when_no_matches():
    rec = make_recording(n_cells=4, duration_s=1.0)
    h, w = rec.ground_truth.fov_shape
    report = score(Estimate(A=np.zeros((0, h, w))), rec.ground_truth)
    assert report.recall == 0.0
    assert report.precision == 0.0
    assert report.f1 == 0.0  # harmonic mean is defined as 0, never a 0/0 nan


# --- Estimate: the two interchangeable field spellings ---------------------


def test_estimate_accepts_both_field_spellings():
    rec = make_recording(n_cells=4, duration_s=1.0, seed=0)
    det = rec.ground_truth.detectable_subset()
    terse = Estimate(A=det.A_observed, C=det.C, S=det.S)
    spelled = Estimate(footprints=det.A_observed, traces=det.C, activity=det.S)
    # Both spellings populate the same canonical fields and read back either way.
    np.testing.assert_array_equal(terse.A, spelled.footprints)
    np.testing.assert_array_equal(terse.footprints, spelled.A)
    np.testing.assert_array_equal(terse.traces, spelled.C)
    np.testing.assert_array_equal(terse.activity, spelled.S)
    # The two estimates score identically.
    assert (
        score(terse, rec.ground_truth).recall == score(spelled, rec.ground_truth).recall
    )


def test_estimate_requires_footprints():
    with pytest.raises(TypeError, match="footprints"):
        Estimate()


def test_estimate_rejects_both_footprint_spellings_at_once():
    a = np.zeros((1, 8, 8))
    with pytest.raises(TypeError, match="not both"):
        Estimate(A=a, footprints=a)


def test_score_motion_rmse_perfect_correction_is_zero():
    rec = make_recording(n_cells=4, duration_s=1.0, motion=True, seed=1)
    det = rec.ground_truth.detectable_subset()
    # A perfect motion-correction estimate is the negation of the applied shift.
    est = Estimate(A=det.A_observed, shifts=-np.asarray(rec.ground_truth.shifts))
    report = score(est, rec.ground_truth)
    assert report.shift_rmse == pytest.approx(0.0, abs=1e-9)


def test_score_absorbs_global_footprint_offset_from_motion():
    # A pipeline whose footprints sit a constant (2, 1) px off (its registration
    # template differs from minisim's reference) but whose motion correction
    # over-shoots the true motion by that same constant. score() reads the offset
    # off the trajectories and re-aligns, so a perfect-but-shifted estimate still
    # recovers every cell.
    from minisim.metrics import _shift_stack

    rec = make_recording(n_cells=5, duration_s=2.0, motion=True, seed=2)
    gt = rec.ground_truth
    det = gt.detectable_subset()
    bias = (6, 4)
    A_off = _shift_stack(np.asarray(det.A_observed), *bias)
    est_shifts = -np.asarray(gt.shifts) + np.array(
        bias
    )  # correction over-shoots by bias
    est = Estimate(A=A_off, C=det.C, S=det.S, shifts=est_shifts)
    no_shift = score(est, gt, footprint_shift=None)
    aligned = score(est, gt)  # default "auto" -> trajectory-derived offset
    assert aligned.footprint_shift == (-6, -4)
    assert aligned.recall == 1.0
    # Re-aligning lifts the overlap the raw offset eroded.
    assert aligned.mean_overlap > no_shift.mean_overlap


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

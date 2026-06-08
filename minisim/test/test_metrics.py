"""Unit tests for the structural metrics oracle.

Synthetic, fully-controlled inputs with known overlaps and correlations: spatial
matching (Hungarian-IoU recall/precision/mean_iou, shuffled order, unequal
counts, partial overlap), temporal recovery (trace Pearson, spike P/R within and
beyond tolerance), motion shift RMSE, field correlation, and that xarray inputs
score identically to ndarrays.
"""

import numpy as np
import pytest
import xarray as xr

from minisim import (
    field_pearson,
    footprint_mask,
    footprint_roi_trace,
    hungarian_match,
    shift_rmse,
    spike_precision_recall,
    trace_pearson,
)


def _disk(h, w, cy, cx, r):
    """A simple filled disk footprint (uniform intensity) for controlled overlap."""
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(float)


# --- spatial matching ------------------------------------------------------


def test_identical_footprints_match_perfectly():
    A = np.stack([_disk(32, 32, 8, 8, 4), _disk(32, 32, 20, 22, 5)])
    m = hungarian_match(A, A)
    assert m.recall(0.5) == 1.0
    assert m.precision(0.5) == 1.0
    assert m.mean_iou == pytest.approx(1.0)
    assert set(m.pairing) == {(0, 0), (1, 1)}


def test_disjoint_footprints_do_not_match():
    A_est = np.stack([_disk(32, 32, 4, 4, 3)])
    A_true = np.stack([_disk(32, 32, 28, 28, 3)])
    m = hungarian_match(A_est, A_true)
    assert m.pairing == ()
    assert m.recall(0.5) == 0.0
    assert m.precision(0.5) == 0.0
    assert m.mean_iou == 0.0


def test_match_recovers_shuffled_order():
    a, b, c = _disk(40, 40, 8, 8, 4), _disk(40, 40, 20, 20, 4), _disk(40, 40, 32, 10, 4)
    A_true = np.stack([a, b, c])
    A_est = np.stack([c, a, b])  # permuted
    m = hungarian_match(A_est, A_true)
    assert set(m.pairing) == {(0, 2), (1, 0), (2, 1)}
    assert m.recall(0.5) == 1.0


def test_unequal_counts_split_recall_and_precision():
    a, b = _disk(32, 32, 8, 8, 4), _disk(32, 32, 22, 22, 4)
    A_true = np.stack([a, b])
    A_est = np.stack([a])  # found only one of two true cells
    m = hungarian_match(A_est, A_true)
    assert m.recall(0.5) == 0.5  # 1 of 2 true cells recovered
    assert m.precision(0.5) == 1.0  # the 1 estimate is correct


def test_partial_overlap_iou_is_thresholdable():
    # two identical-size disks offset so their IoU sits between 0 and 1
    A_true = np.stack([_disk(48, 48, 24, 20, 6)])
    A_est = np.stack([_disk(48, 48, 24, 26, 6)])
    m = hungarian_match(A_est, A_true)
    (iou,) = [m.iou_matrix[i, j] for i, j in m.pairing]
    assert 0.0 < iou < 1.0
    assert m.recall(iou_threshold=iou - 0.01) == 1.0  # counts above its IoU
    assert m.recall(iou_threshold=iou + 0.01) == 0.0  # drops below


def test_unsupported_metric_raises():
    A = np.stack([_disk(16, 16, 8, 8, 3)])
    with pytest.raises(ValueError, match="metric"):
        hungarian_match(A, A, metric="dice")


def test_energy_frac_validated():
    A = np.stack([_disk(16, 16, 8, 8, 3)])
    with pytest.raises(ValueError, match="energy_frac"):
        hungarian_match(A, A, energy_frac=1.5)


def test_xarray_input_matches_ndarray():
    A = np.stack([_disk(32, 32, 8, 8, 4), _disk(32, 32, 22, 22, 5)])
    xA = xr.DataArray(A, dims=["unit_id", "height", "width"])
    assert hungarian_match(xA, xA).mean_iou == pytest.approx(hungarian_match(A, A).mean_iou)


# --- temporal recovery -----------------------------------------------------


def test_trace_pearson_per_pair():
    t = np.linspace(0, 4 * np.pi, 200)
    C_true = np.stack([np.sin(t), np.cos(t)])
    C_est = np.stack([np.sin(t), -np.cos(t)])  # unit 0 identical, unit 1 anti-correlated
    r = trace_pearson(C_est, C_true, [(0, 0), (1, 1)])
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(-1.0)


def test_trace_pearson_constant_is_nan():
    C_true = np.stack([np.zeros(50)])
    C_est = np.stack([np.arange(50.0)])
    assert np.isnan(trace_pearson(C_est, C_true, [(0, 0)])[0])


def test_spike_exact_match_is_perfect():
    S = np.zeros((1, 100))
    S[0, [10, 40, 70]] = 1.0
    score = spike_precision_recall(S, S, [(0, 0)])
    assert score == (1.0, 1.0)


def test_spike_within_tolerance_detected_beyond_missed():
    S_true = np.zeros((1, 100))
    S_true[0, [10, 50]] = 1.0
    S_est = np.zeros((1, 100))
    S_est[0, [11, 60]] = 1.0  # 11 within tol of 10; 60 too far from 50
    score = spike_precision_recall(S_est, S_true, [(0, 0)], tol_frames=2)
    assert score.recall == pytest.approx(0.5)  # 1 of 2 true spikes found
    assert score.precision == pytest.approx(0.5)  # 1 of 2 est spikes correct


def test_spike_extra_estimates_lower_precision():
    S_true = np.zeros((1, 100))
    S_true[0, [20]] = 1.0
    S_est = np.zeros((1, 100))
    S_est[0, [20, 60, 80]] = 1.0  # one correct, two spurious
    score = spike_precision_recall(S_est, S_true, [(0, 0)])
    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(1 / 3)


# --- per-effect fields -----------------------------------------------------


def test_shift_rmse_known_and_zero():
    true = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert shift_rmse(true, true) == 0.0
    est = np.array([[0.0, 0.0], [0.0, 0.0]])
    # squared errors: 0,0,9,16 -> mean 6.25 -> rmse 2.5
    assert shift_rmse(est, true) == pytest.approx(2.5)


def test_field_pearson_scale_invariant():
    rng = np.random.default_rng(0)
    f = rng.random((16, 16))
    assert field_pearson(f, f) == pytest.approx(1.0)
    assert field_pearson(3.0 * f + 2.0, f) == pytest.approx(1.0)  # affine -> still 1


def test_field_pearson_constant_is_nan():
    assert np.isnan(field_pearson(np.ones((8, 8)), np.arange(64.0).reshape(8, 8)))


# --- naive footprint ROI ---------------------------------------------------


def test_footprint_mask_thresholds_relative_to_peak():
    a = _disk(20, 20, 10, 10, 4) * 10.0  # uniform disk at intensity 10
    a[10, 10] = 100.0  # a hot center pixel
    mask = footprint_mask(a, rel=0.18)
    # every disk pixel (10) clears 0.18*100=18? no -> only the hot pixel does.
    assert mask[10, 10]
    assert mask.sum() == 1
    # a gentler threshold keeps the whole disk; an all-zero footprint masks nothing.
    assert footprint_mask(a, rel=0.05).sum() == int((a > 0).sum())
    assert not footprint_mask(np.zeros((8, 8))).any()


def test_footprint_roi_trace_averages_over_the_mask():
    a = _disk(16, 16, 8, 8, 3)
    mask = footprint_mask(a)
    # a movie that is a constant value per frame -> ROI mean equals that value.
    vals = np.array([1.0, 5.0, 2.0])
    movie = vals[:, None, None] * np.ones((3, 16, 16))
    np.testing.assert_allclose(footprint_roi_trace(movie, a), vals)
    # the ROI is exactly the mask mean, frame by frame, on an arbitrary movie.
    rng = np.random.default_rng(0)
    movie = rng.random((4, 16, 16))
    np.testing.assert_allclose(
        footprint_roi_trace(movie, a), movie[:, mask].mean(axis=1)
    )
    # an empty footprint yields zeros, not a divide-by-zero.
    np.testing.assert_array_equal(footprint_roi_trace(movie, np.zeros((16, 16))), np.zeros(4))

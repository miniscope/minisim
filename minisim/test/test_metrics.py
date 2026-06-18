"""Unit tests for the structural metrics oracle.

Synthetic, fully-controlled inputs with known overlaps and correlations: spatial
matching (Hungarian recall/precision/mean overlap, shuffled order, unequal counts,
partial overlap, weighted metrics that see pixel weights, global-shift recovery),
temporal recovery (trace Pearson, scale-aware activity similarity), motion shift
RMSE (with origin alignment), field correlation, and that xarray inputs score
identically to ndarrays.
"""

import numpy as np
import pytest
import xarray as xr

from minisim import (
    activity_similarity,
    field_pearson,
    footprint_mask,
    footprint_roi_trace,
    global_shift_from_trajectories,
    hungarian_match,
    shift_rmse,
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
    assert m.recall(threshold=iou - 0.01) == 1.0  # counts above its IoU
    assert m.recall(threshold=iou + 0.01) == 0.0  # drops below


def test_unsupported_metric_raises():
    A = np.stack([_disk(16, 16, 8, 8, 3)])
    with pytest.raises(ValueError, match="metric"):
        hungarian_match(A, A, metric="dice")


def test_energy_frac_validated():
    A = np.stack([_disk(16, 16, 8, 8, 3)])
    with pytest.raises(ValueError, match="energy_frac"):
        hungarian_match(A, A, energy_frac=1.5)


def test_full_energy_mask_keeps_whole_support_under_huge_dynamic_range():
    # energy_frac=1.0 must keep the entire support, even when the footprint spans many
    # orders of magnitude (a bright core over a near-zero blur tail). Re-scaling the
    # weight profile (raising to a power) leaves the support unchanged, so the binary
    # IoU between the original and every rescaled version must stay exactly 1.0.
    # Regression: a separately-summed total vs the descending cumulative sum used to
    # collapse the mask to the bright core for some scalings, giving IoU ~0.08.
    yy, xx = np.ogrid[:48, :48]
    true = np.exp(
        -((yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 5.0**2)
    )  # peak 1 -> tail ~1e-10
    support = true > 0
    for gamma in np.linspace(1.0, 0.0, 11):
        est = np.where(support, true**gamma, 0.0)  # same support, flattened profile
        m = hungarian_match(est[None], true[None], metric="iou", energy_frac=1.0)
        assert m.similarity_matrix[0, 0] == pytest.approx(1.0)


def test_xarray_input_matches_ndarray():
    A = np.stack([_disk(32, 32, 8, 8, 4), _disk(32, 32, 22, 22, 5)])
    xA = xr.DataArray(A, dims=["unit_id", "height", "width"])
    assert hungarian_match(xA, xA).mean_iou == pytest.approx(
        hungarian_match(A, A).mean_iou
    )


# --- weighted footprint similarity -----------------------------------------


def test_weighted_metrics_perfect_on_identical():
    A = np.stack([_disk(32, 32, 8, 8, 4), _disk(32, 32, 20, 22, 5)])
    for metric in ("cosine", "weighted_jaccard"):
        m = hungarian_match(A, A, metric=metric)
        assert m.metric == metric
        assert m.mean_similarity == pytest.approx(1.0)
        assert set(m.pairing) == {(0, 0), (1, 1)}


def test_weighted_metric_sees_pixel_weights_that_binary_iou_misses():
    # Same lit pixels, different intensity profile: a flat disk vs. the same disk
    # with one very bright pixel. Binary IoU over the full support cannot tell them
    # apart; the weighted metrics, comparing the intensity profile, can.
    support = _disk(24, 24, 12, 12, 5)
    A_true = support[None] * 1.0
    spiked = support * 1.0
    spiked[12, 12] = 30.0
    A_est = spiked[None]
    # Full-energy binary masks are identical (same support) -> IoU == 1.
    m_iou = hungarian_match(A_est, A_true, energy_frac=1.0)
    (iou,) = [m_iou.iou_matrix[i, j] for i, j in m_iou.pairing]
    assert iou == pytest.approx(1.0)
    # The weighted metrics drop below 1, penalizing the weight mismatch.
    (cos,) = [
        m.similarity_matrix[i, j]
        for m in [hungarian_match(A_est, A_true, metric="cosine")]
        for i, j in m.pairing
    ]
    (wj,) = [
        m.similarity_matrix[i, j]
        for m in [hungarian_match(A_est, A_true, metric="weighted_jaccard")]
        for i, j in m.pairing
    ]
    assert cos < 0.99
    assert wj < 0.99


# --- global translational shift --------------------------------------------


def test_auto_shift_recovers_global_translation():
    # A rigid (+3, +2) offset of every footprint: unrecoverable as-is, perfect once
    # the global shift is found by overlap maximization.
    A_true = np.stack([_disk(48, 48, 12, 12, 4), _disk(48, 48, 30, 28, 5)])
    A_est = np.stack([_disk(48, 48, 15, 14, 4), _disk(48, 48, 33, 30, 5)])
    plain = hungarian_match(A_est, A_true)
    shifted = hungarian_match(A_est, A_true, shift="auto")
    assert shifted.shift == (-3, -2)  # shift applied to A_est to align onto A_true
    assert shifted.recall(0.5) == 1.0
    assert shifted.mean_iou > plain.mean_iou


def test_auto_shift_recovers_translation_with_high_dynamic_range_footprints():
    # Regression: real (blurred) footprints span many orders of magnitude - bright
    # cores over a near-zero tail - and overlap into a broad summed image. The earlier
    # phase correlation whitened the raw intensity sum, which amplified that near-zero
    # tail to equal weight and lost the peak, so a uniform offset went unrecovered.
    # Registering the binary supports finds it. Build several wide overlapping
    # Gaussians (peak 1 down to ~1e-9 tails) and shift them all by a known offset.
    from minisim.metrics import _shift_stack

    yy, xx = np.ogrid[:64, :64]
    centers = [(20, 22), (24, 40), (40, 26), (44, 44)]
    A_true = np.stack(
        [
            np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0**2)))
            for cy, cx in centers
        ]
    )
    assert (
        A_true.sum(0).max() / A_true.sum(0)[A_true.sum(0) > 0].min() > 1e6
    )  # huge range
    A_est = _shift_stack(A_true, 6, 5)
    plain = hungarian_match(A_est, A_true)
    shifted = hungarian_match(A_est, A_true, shift="auto")
    assert plain.recall(0.5) < 0.5  # the raw offset wrecks the match
    assert shifted.shift == (-6, -5)  # ...and auto recovers it exactly
    assert shifted.recall(0.5) == 1.0


def test_explicit_shift_is_applied_and_recorded():
    A_true = np.stack([_disk(48, 48, 12, 12, 4), _disk(48, 48, 30, 28, 5)])
    A_est = np.stack([_disk(48, 48, 15, 14, 4), _disk(48, 48, 33, 30, 5)])
    m = hungarian_match(A_est, A_true, shift=(-3, -2))
    assert m.shift == (-3, -2)
    assert m.recall(0.5) == 1.0


def test_auto_shift_robust_to_population_mismatch():
    # An estimate carrying an extra (false-positive) footprint must not invent a
    # shift: the shared mass dominates the overlap, so the offset stays (0, 0).
    A_true = np.stack([_disk(48, 48, 12, 12, 4), _disk(48, 48, 30, 28, 5)])
    extra = _disk(48, 48, 40, 8, 4)
    A_est = np.stack([A_true[0], A_true[1], extra])
    m = hungarian_match(A_est, A_true, shift="auto")
    assert m.shift == (0, 0)


def test_global_shift_from_trajectories_reads_off_the_constant_offset():
    # A correction trajectory that over-shoots the (zero) applied motion by a
    # constant (2, 1) places the recovered footprints at +(2, 1); the shift to
    # re-align them is the negation.
    correction = np.tile([2.0, 1.0], (8, 1))
    applied = np.zeros((8, 2))
    assert global_shift_from_trajectories(correction, applied) == (-2, -1)


# --- temporal recovery -----------------------------------------------------


def test_trace_pearson_per_pair():
    t = np.linspace(0, 4 * np.pi, 200)
    C_true = np.stack([np.sin(t), np.cos(t)])
    C_est = np.stack(
        [np.sin(t), -np.cos(t)]
    )  # unit 0 identical, unit 1 anti-correlated
    r = trace_pearson(C_est, C_true, [(0, 0), (1, 1)])
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(-1.0)


def test_trace_pearson_constant_is_nan():
    C_true = np.stack([np.zeros(50)])
    C_est = np.stack([np.arange(50.0)])
    assert np.isnan(trace_pearson(C_est, C_true, [(0, 0)])[0])


def test_activity_is_scale_invariant():
    # The deconvolved activity differs from the truth by an unknown amplitude
    # factor; the correlation must ignore it and the recovered scale must report it.
    rng = np.random.default_rng(0)
    S_true = np.abs(rng.standard_normal((1, 200)))
    S_est = S_true / 3.0  # estimate is the truth scaled down by 3
    act = activity_similarity(S_est, S_true, [(0, 0)])
    assert act.correlation[0] == pytest.approx(1.0)
    assert act.variance_explained[0] == pytest.approx(1.0)
    assert act.scale[0] == pytest.approx(3.0)  # true = 3 * est


def test_activity_is_not_binarized():
    # Two estimates with identical *timing* but different amplitudes score
    # differently (a thresholded spike metric would call them identical).
    S_true = np.zeros((1, 100))
    S_true[0, [10, 50, 80]] = [1.0, 3.0, 2.0]
    graded = np.zeros((1, 100))
    graded[0, [10, 50, 80]] = [1.0, 3.0, 2.0]
    flat = np.zeros((1, 100))
    flat[0, [10, 50, 80]] = 1.0  # same frames, wrong amplitudes
    a_graded = activity_similarity(graded, S_true, [(0, 0)])
    a_flat = activity_similarity(flat, S_true, [(0, 0)])
    assert a_graded.variance_explained[0] == pytest.approx(1.0)
    assert a_flat.variance_explained[0] < a_graded.variance_explained[0]


def test_activity_anticorrelated_explains_nothing():
    # A non-negative scale cannot fit an anti-correlated estimate, so it explains
    # no variance (scale clamps to 0) even though the shape correlation is negative.
    S_true = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    S_est = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    act = activity_similarity(S_est, S_true, [(0, 0)])
    assert act.correlation[0] < 0.0
    assert act.scale[0] == 0.0
    assert act.variance_explained[0] == pytest.approx(0.0)


def test_activity_constant_truth_is_nan():
    S_true = np.zeros((1, 50))
    S_est = np.abs(np.arange(50.0))[None]
    act = activity_similarity(S_est, S_true, [(0, 0)])
    assert np.isnan(act.correlation[0])
    assert np.isnan(act.variance_explained[0])


# --- per-effect fields -----------------------------------------------------


def test_shift_rmse_known_and_zero():
    true = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert shift_rmse(true, true) == 0.0
    est = np.array([[0.0, 0.0], [0.0, 0.0]])
    # squared errors: 0,0,9,16 -> mean 6.25 -> rmse 2.5
    assert shift_rmse(est, true) == pytest.approx(2.5)


def test_shift_rmse_align_removes_constant_origin_offset():
    # A trajectory that tracks the motion perfectly but sits a constant (2, -1) off
    # (a different registration template): raw RMSE is non-zero, aligned RMSE is 0.
    true = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, -2.0]])
    est = true + np.array([2.0, -1.0])
    assert shift_rmse(est, true) > 0.0
    assert shift_rmse(est, true, align=True) == pytest.approx(0.0)


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
    # all-negative is a distinct peak<=0 path (peak = max < 0): still all-False, same shape.
    neg = footprint_mask(-np.ones((5, 5)))
    assert not neg.any() and neg.shape == (5, 5)


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
    np.testing.assert_array_equal(
        footprint_roi_trace(movie, np.zeros((16, 16))), np.zeros(4)
    )


# --- edge cases: empty / degenerate / robustness ---------------------------
# The metrics are an oracle other code asserts against, so their behavior on empty,
# all-zero, and adversarial inputs must be defined, not "happens to not crash".


@pytest.mark.parametrize("metric", ["iou", "cosine", "weighted_jaccard"])
def test_match_empty_estimate_stack(metric):
    # A pipeline that recovered nothing: a (0, H, W) stack must score, not raise
    # (this reshape was a real bug for the weighted metrics).
    A_true = np.stack([_disk(16, 16, 8, 8, 3)])
    m = hungarian_match(np.zeros((0, 16, 16)), A_true, metric=metric)
    assert m.pairing == ()
    assert m.n_est == 0 and m.n_true == 1
    assert m.recall() == 0.0 and m.precision() == 0.0 and m.mean_similarity == 0.0


@pytest.mark.parametrize("metric", ["iou", "cosine", "weighted_jaccard"])
def test_match_both_stacks_empty(metric):
    m = hungarian_match(np.zeros((0, 8, 8)), np.zeros((0, 8, 8)), metric=metric)
    assert m.pairing == () and m.recall() == 0.0 and m.precision() == 0.0


@pytest.mark.parametrize("metric", ["iou", "cosine", "weighted_jaccard"])
def test_match_all_zero_footprint_does_not_match(metric):
    # A recovered-nothing row (all zeros) has zero similarity to everything and is
    # dropped from the pairing rather than matched at "0".
    A_true = np.stack([_disk(16, 16, 8, 8, 3)])
    m = hungarian_match(np.zeros((1, 16, 16)), A_true, metric=metric)
    assert m.pairing == () and m.mean_similarity == 0.0


@pytest.mark.parametrize("metric", ["cosine", "weighted_jaccard"])
def test_weighted_disjoint_footprints_score_zero(metric):
    A_true = np.stack([_disk(32, 32, 6, 6, 3)])
    A_est = np.stack([_disk(32, 32, 26, 26, 3)])
    m = hungarian_match(A_est, A_true, metric=metric)
    assert m.pairing == () and m.mean_similarity == 0.0


def test_match_negative_values_are_clipped():
    # Negative footprint pixels are clipped to zero before scoring, so a negative
    # "background" does not perturb the similarity.
    A = np.stack([_disk(16, 16, 8, 8, 3)])
    neg = np.where(A > 0, A, -5.0)
    assert hungarian_match(neg, A, metric="cosine").mean_similarity == pytest.approx(
        1.0
    )


def test_auto_shift_no_mass_is_zero():
    # An all-zero estimate carries no mass to align; auto must return (0, 0), not nan.
    A_true = np.stack([_disk(16, 16, 8, 8, 3)])
    assert hungarian_match(np.zeros((1, 16, 16)), A_true, shift="auto").shift == (0, 0)


def test_auto_shift_not_adopted_when_it_cannot_help():
    # Disjoint, far-apart footprints: no within-bound shift creates overlap, so the
    # guard leaves the shift at (0, 0) rather than inventing one.
    A_true = np.stack([_disk(40, 40, 10, 10, 4)])
    A_est = np.stack([_disk(40, 40, 32, 32, 4)])
    assert hungarian_match(A_est, A_true, shift="auto").shift == (0, 0)


def test_auto_shift_survives_a_false_positive():
    # The regression that motivated phase-correlation + the adopt-only-if-it-helps
    # guard: a real global offset plus a spurious footprint must still recover the
    # true shift, not drift to the search-window edge.
    from minisim.metrics import _shift_stack

    true = np.stack(
        [_disk(64, 64, 16, 16, 5), _disk(64, 64, 44, 20, 5), _disk(64, 64, 28, 48, 5)]
    )
    est = np.concatenate([_shift_stack(true, 2, 3), _disk(64, 64, 56, 8, 4)[None]])
    m = hungarian_match(est, true, shift="auto")
    assert m.shift == (-2, -3)
    assert m.recall(0.5) == 1.0


def test_explicit_shift_is_trusted_unbounded_and_unguarded():
    # An explicit shift is applied as given - beyond the auto search bound, and even
    # if it hurts (the caller knows their convention); only "auto" is guarded.
    A_true = np.stack([_disk(48, 48, 8, 8, 4)])
    A_est = np.stack([_disk(48, 48, 40, 40, 4)])  # 32 px away, past the 25% auto bound
    far = hungarian_match(A_est, A_true, shift=(-32, -32))
    assert far.shift == (-32, -32) and far.recall(0.5) == 1.0
    hurt = hungarian_match(
        A_true, A_true, shift=(20, 0)
    )  # explicit, harmful, still applied
    assert hurt.shift == (20, 0)
    assert hungarian_match(A_true, A_true, shift="auto").shift == (
        0,
        0,
    )  # auto declines


def test_trace_pearson_empty_pairing_is_empty():
    assert trace_pearson(np.zeros((1, 5)), np.zeros((1, 5)), []).size == 0


def test_activity_empty_pairing_is_empty():
    act = activity_similarity(np.zeros((1, 5)), np.zeros((1, 5)), [])
    assert (
        act.correlation.size == 0
        and act.scale.size == 0
        and act.variance_explained.size == 0
    )


def test_activity_constant_estimate_explains_nothing():
    S_true = np.array([[0.0, 1.0, 0.0, 2.0, 0.0, 1.0]])
    S_est = np.ones((1, 6))  # constant: no shape to correlate, no variance to explain
    act = activity_similarity(S_est, S_true, [(0, 0)])
    assert np.isnan(act.correlation[0])
    assert act.scale[0] == 0.0 and act.variance_explained[0] == pytest.approx(0.0)


def test_shift_rmse_single_frame():
    assert shift_rmse(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])) == 0.0
    # one frame: removing its mean residual always zeroes the aligned error.
    assert shift_rmse(
        np.array([[5.0, 5.0]]), np.zeros((1, 2)), align=True
    ) == pytest.approx(0.0)


def test_global_shift_rounds_and_honors_correction_flag():
    # correction=True (default) negates the estimate; the per-axis mean is rounded.
    correction = np.tile([2.4, -1.6], (5, 1))
    assert global_shift_from_trajectories(correction, np.zeros((5, 2))) == (-2, 2)
    # correction=False compares the trajectories as given.
    assert global_shift_from_trajectories(
        np.tile([1.0, 0.0], (4, 1)), np.zeros((4, 2)), correction=False
    ) == (1, 0)

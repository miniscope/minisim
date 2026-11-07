"""Realistic (``"miniscope"``) rigid motion and the non-rigid deformation field."""

from __future__ import annotations

import numpy as np
import pytest

from minisim import (
    BrainMotion,
    Spec,
    load_ground_truth,
    patch_shift_rmse,
    presets,
    simulate,
    write_miniscope_daq,
)
from minisim.cli import build_parser, spec_from_args
from minisim.cli import main as cli_main
from minisim.steps.motion import (
    brain_motion_fields,
    draw_bursts,
    evaluate_nonrigid,
    grid_interp_matrix,
    miniscope_brain_motion,
    nonrigid_nodes,
    shift_and_crop,
)
from minisim.test.test_video import _spec as _small_spec
from minisim.video import _iter_count_frames

FPS = 20.0
PX_UM = 4.8 / 2.9  # Miniscope V4 pixel size
N_20MIN = int(20 * 60 * FPS)


def _acq(duration_s=60.0):
    return presets.build_spec(
        presets.miniscope_v4(), presets.ca1(), duration_s=duration_s
    ).acquisition


def _rigid(seed, n=N_20MIN, **over):
    spec = BrainMotion(model="miniscope", **over)
    acq = presets.build_spec(
        presets.miniscope_v4(), presets.ca1(), duration_s=n / FPS
    ).acquisition
    return brain_motion_fields(spec, acq, n, np.random.default_rng(seed))[0]


@pytest.fixture(scope="module")
def trajectories():
    return [_rigid(s) for s in range(4)]


def test_std_per_axis_matches_targets(trajectories):
    std = np.mean([t.std(axis=0) for t in trajectories], axis=0)
    assert std[0] == pytest.approx(0.9, rel=0.2)  # dy
    assert std[1] == pytest.approx(1.1, rel=0.2)  # dx
    assert std[1] > std[0]  # anisotropic, x-leaning


def test_dominant_axis_orientation(trajectories):
    ax = []
    for t in trajectories:
        _w, v = np.linalg.eigh(np.cov(t.T))
        ax.append(np.abs(v[:, 1]))  # (|dy|, |dx|) of the major axis
    ax = np.mean(ax, axis=0)
    assert ax[0] == pytest.approx(0.49, abs=0.12)
    assert ax[1] == pytest.approx(0.87, abs=0.08)


def test_spectrum_band_fractions(trajectories):
    edges = [0.1, 0.5, 1.0, 2.0, 5.0]
    target = [0.39, 0.63, 0.76, 0.86, 0.96]
    cum = np.zeros(5)
    for t in trajectories:
        f = np.fft.rfftfreq(len(t), 1 / FPS)
        p = (np.abs(np.fft.rfft(t - t.mean(0), axis=0)) ** 2).sum(1)
        cum += [p[(f > 0) & (f <= e)].sum() / p[1:].sum() for e in edges]
    cum /= len(trajectories)
    np.testing.assert_allclose(cum, target, atol=0.06)


def test_frame_to_frame_steps_and_tails(trajectories):
    for t in trajectories:
        step = np.diff(t, axis=0)
        assert 0.4 < step.std() < 0.65
        assert np.percentile(np.hypot(*step.T), 99) < 2.0
        assert np.hypot(*t.T).max() < 9.0  # inside the default 15 µm clamp


def test_frame_zero_is_reference_and_drift_is_slow(trajectories):
    for t in trajectories:
        np.testing.assert_array_equal(t[0], 0.0)
    # Drift alone (no noise-like parts): ~1 px over 20 min at the default speed.
    spec = BrainMotion(model="miniscope")
    drift_px = _acq().um_to_px(spec.drift_um_per_min) * 20
    assert drift_px == pytest.approx(1.0, rel=0.1)


def test_burst_rate_and_shape():
    t = np.arange(N_20MIN) / FPS
    counts, peaks = [], []
    for s in range(5):
        burst, n = draw_bursts(
            t, 0.05, (1.0, 3.0), (2.0, 4.0), np.random.default_rng(s)
        )
        counts.append(n)
        peaks.append(np.abs(burst).max())
    assert np.mean(counts) / (t[-1] + 3.0) == pytest.approx(0.05, rel=0.15)
    assert 1.5 < min(peaks) and max(peaks) < 8.0  # a few overlapping at most
    # A lone burst peaks inside the configured band and lasts 1-3 s.
    burst, n = draw_bursts(
        np.arange(400) / FPS, 0.0, (1.0, 3.0), (2.0, 4.0), np.random.default_rng(0)
    )
    assert n == 0 and not burst.any()


def test_events_exceed_threshold_at_reasonable_rate(trajectories):
    # Episodes of |shift - running median| > 1.5 px (merged within 1 s): several per
    # minute, as in real data (hundreds per 20 min).
    from scipy.ndimage import median_filter

    t = trajectories[0]
    dev = np.hypot(*(t - median_filter(t, size=(int(10 * FPS), 1))).T)
    above = dev > 1.5
    starts = np.flatnonzero(above[1:] & ~above[:-1])
    events = 1 + np.sum(np.diff(starts) > FPS) if len(starts) else 0
    assert 100 < events < 900


def test_deterministic_and_seed_sensitive():
    a, b, c = _rigid(1, n=2000), _rigid(1, n=2000), _rigid(2, n=2000)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_short_recordings_do_not_break():
    kw = dict(
        rms_major_px=1.0,
        axis_ratio=0.7,
        axis_deg=30.0,
        burst_rate_hz=0.05,
        burst_duration_s=(1.0, 3.0),
        burst_peak_px=(2.0, 4.0),
        drift_px_per_s=0.0,
        max_px=9.0,
    )
    for n in (0, 1, 2, 5, 40):
        tr = miniscope_brain_motion(n, FPS, rng=np.random.default_rng(0), **kw)
        assert tr.shape == (n, 2) and np.isfinite(tr).all()


def test_default_spec_serialization_is_unchanged():
    # Existing specs (no miniscope / non-rigid use) serialize without the new fields,
    # so their cache keys are untouched.
    dumped = BrainMotion().model_dump()
    assert "nonrigid" not in dumped and "burst_rate_hz" not in dumped
    assert "nonrigid" in BrainMotion(nonrigid=True).model_dump()
    assert "burst_rate_hz" in BrainMotion(model="miniscope").model_dump()
    spec = BrainMotion(model="miniscope", nonrigid=True, burst_rate_hz=0.2)
    assert BrainMotion.model_validate_json(spec.model_dump_json()) == spec


# --- non-rigid -----------------------------------------------------------------


def _field(seed=0, n=6000, grid=(4, 4)):
    spec = BrainMotion(model="miniscope", nonrigid=True, nonrigid_grid=grid)
    acq = presets.build_spec(
        presets.miniscope_v4(), presets.ca1(), duration_s=n / FPS
    ).acquisition
    shifts, field = brain_motion_fields(spec, acq, n, np.random.default_rng(seed))
    return shifts, field


def test_nonrigid_off_returns_none_and_keeps_rng_stream():
    spec = BrainMotion(model="miniscope")
    acq = _acq()
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    _s, nr = brain_motion_fields(spec, acq, 200, r1)
    assert nr is None
    # RNG state after rigid-only equals a direct rigid draw (no extra consumption).
    from minisim.steps.motion import _rigid_shifts

    _rigid_shifts(spec, acq, 200, r2)
    assert r1.integers(1 << 30) == r2.integers(1 << 30)


def test_nonrigid_shape_deterministic_zero_mean():
    s, f = _field(grid=(4, 4), n=400)
    assert f.shape == (400, 4, 4, 2)
    s2, f2 = _field(grid=(4, 4), n=400)
    np.testing.assert_array_equal(f, f2)
    np.testing.assert_array_equal(s, s2)
    # The rigid trajectory carries the global motion: the field's spatial mean is small
    # next to its spread (edge weighting leaves a little).
    assert np.abs(f.mean(axis=(1, 2))).mean() < 0.5 * f.std()
    assert _field(grid=(6, 5), n=100)[1].shape == (100, 6, 5, 2)


def test_nonrigid_amplitude_and_coupling():
    s, f = _field(n=8000)
    fov = (600, 600)
    ys = xs = np.arange(0, 600, 12)
    dense = evaluate_nonrigid(f[::10], fov, ys, xs)
    vec_rms = np.sqrt((dense**2).sum(-1).mean(axis=(1, 2)))
    m = np.hypot(*(s - s.mean(0))[::10].T)
    overall = np.sqrt((vec_rms**2).mean())
    assert 0.35 < overall < 0.65  # target ~0.4-0.5 px RMS
    quiet = np.sqrt((vec_rms[m < 1.0] ** 2).mean())
    busy = np.sqrt((vec_rms[m > 2.5] ** 2).mean())
    assert 0.3 < quiet < 0.55
    assert busy > 1.5 * quiet  # amplitude tracks the rigid magnitude
    assert busy < 1.4
    # Coupling also holds frame-to-frame (smoothed): correlation of rms with motion.
    assert np.corrcoef(vec_rms, m)[0, 1] > 0.4


def test_nonrigid_smooth_in_space_and_time():
    _s, f = _field(n=2000)
    dense = evaluate_nonrigid(f[:50], (600, 600), np.arange(600), np.arange(600))
    # Spatially smooth: neighbouring-pixel differences are tiny vs the field spread.
    assert np.abs(np.diff(dense, axis=1)).max() < 0.05 * 0.5 + 0.02
    assert np.abs(np.diff(dense, axis=2)).max() < 0.05
    # Temporally correlated: lag-1 autocorrelation of node series is high.
    x = f[:, 1, 1, 0]
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > 0.8


def test_interpolation_exact_at_nodes_and_linear_in_values():
    ys, xs = nonrigid_nodes((600, 600), (4, 4))
    assert ys[0] == 0 and ys[-1] == 599
    m = grid_interp_matrix(ys, ys)
    np.testing.assert_allclose(m, np.eye(4), atol=1e-12)
    _, f = _field(n=10)
    full = evaluate_nonrigid(f, (600, 600), ys, xs)
    np.testing.assert_allclose(full, f, atol=1e-10)


def test_shift_and_crop_nonrigid_matches_rigid_for_zero_field():
    rng = np.random.default_rng(0)
    canvas = rng.random((3, 40, 40))
    shifts = np.array([[0.0, 0.0], [1.3, -0.7], [-2.0, 1.1]])
    rigid = shift_and_crop(canvas, shifts, (32, 32))
    zero = shift_and_crop(canvas, shifts, (32, 32), np.zeros((3, 4, 4, 2)))
    np.testing.assert_allclose(zero, rigid, atol=1e-5)


def test_shift_and_crop_nonrigid_applies_local_shift():
    # A uniform field of (0, 1.0) must equal a rigid dx=1 shift.
    rng = np.random.default_rng(1)
    canvas = rng.random((2, 40, 40))
    field = np.zeros((2, 3, 3, 2))
    field[..., 1] = 1.0
    out = shift_and_crop(canvas, np.zeros((2, 2)), (32, 32), field)
    ref = shift_and_crop(canvas, np.tile([0.0, 1.0], (2, 1)), (32, 32))
    np.testing.assert_allclose(out, ref, atol=1e-5)


# --- end to end ----------------------------------------------------------------


def _motion_spec(nonrigid=True, model="miniscope"):
    base = _small_spec(motion=False, duration_s=3.0)
    motion = BrainMotion(
        model=model,
        nonrigid=nonrigid,
        motion_rms_um=1.2,
        max_shift_um=10.0,
        burst_min_peak_um=1.0,
        burst_max_peak_um=2.0,
    )
    return Spec(
        acquisition=base.acquisition,
        seed=base.seed,
        steps=[*base.steps, motion],
        output=base.output,
    )


@pytest.mark.parametrize("nonrigid", [False, True])
def test_streamed_equals_in_memory(nonrigid):
    spec = _motion_spec(nonrigid)
    rec = simulate(spec)
    sink: dict = {}
    streamed = np.stack([f for _, f in _iter_count_frames(spec, 7, truth_sink=sink)])
    np.testing.assert_array_equal(streamed, rec.observed)
    gt, sgt = rec.ground_truth, sink["ground_truth"]
    np.testing.assert_array_equal(gt.shifts, sgt.shifts)
    if nonrigid:
        np.testing.assert_array_equal(gt.nonrigid_field, sgt.nonrigid_field)
        np.testing.assert_array_equal(gt.nonrigid_nodes_y, sgt.nonrigid_nodes_y)
        assert gt.nonrigid_field.shape[0] == streamed.shape[0]
    else:
        assert gt.nonrigid_field is None and sgt.nonrigid_field is None


def test_ground_truth_displacement_helper():
    rec = simulate(_motion_spec(True))
    gt = rec.ground_truth
    h, w = gt.fov_shape
    pts_y, pts_x = np.array([0, h - 1, h // 2]), np.array([0, w - 1, w // 2])
    d = gt.displacement(pts_y, pts_x)
    assert d.shape == (gt.shifts.shape[0], 3, 2)
    # At a control node the total is exactly rigid + that node's displacement.
    node = gt.nonrigid_field[:, 0, 0, :] + gt.shifts
    np.testing.assert_allclose(d[:, 0, :], node, atol=1e-10)
    np.testing.assert_allclose(
        gt.displacement(pts_y, pts_x, frames=slice(2, 5)), d[2:5]
    )
    # Rigid-only recording: displacement is just the shifts.
    rigid = simulate(_motion_spec(False)).ground_truth
    np.testing.assert_allclose(rigid.displacement([1], [1])[:, 0, :], rigid.shifts)
    # Subsetting detectable cells carries the field.
    assert gt.detectable_subset().nonrigid_field is not None


def test_patch_shift_rmse():
    gt = simulate(_motion_spec(True)).ground_truth
    h, w = gt.fov_shape
    centers = np.array([[h * a, w * b] for a in (0.25, 0.75) for b in (0.25, 0.75)])
    truth = gt.displacement(centers[:, 0], centers[:, 1])
    assert patch_shift_rmse(-truth, gt, centers) == pytest.approx(0.0, abs=1e-12)
    assert patch_shift_rmse(-(truth + 0.5), gt, centers) == pytest.approx(
        np.sqrt(2) * 0.5 / np.sqrt(2), rel=1e-9
    )
    assert patch_shift_rmse(-(truth + 0.5), gt, centers, align=True) < 1e-9
    # A rigid-only estimate misses the local structure.
    rigid_est = np.repeat(-gt.shifts[:, None, :], 4, axis=1)
    assert patch_shift_rmse(rigid_est, gt, centers) > 0.05
    with pytest.raises(ValueError):
        patch_shift_rmse(-truth[:, :2], gt, centers)


def test_daq_writes_nonrigid_ground_truth(tmp_path):
    spec = _motion_spec(True)
    folder = write_miniscope_daq(spec, tmp_path / "d", progress=False)
    z = np.load(folder.ground_truth_dir / "arrays.npz")
    ref = simulate(spec).ground_truth
    np.testing.assert_allclose(z["nonrigid_field"], ref.nonrigid_field)
    np.testing.assert_allclose(z["nonrigid_nodes_x"], ref.nonrigid_nodes_x)
    gt, _ = load_ground_truth(folder.ground_truth_dir / "ground_truth.zarr")
    np.testing.assert_allclose(gt.nonrigid_field, ref.nonrigid_field)
    np.testing.assert_allclose(gt.shifts, ref.shifts)


def test_nonrigid_margin_is_enlarged_and_checked():
    # Margin covers rigid clamp + non-rigid clip, so the crop never exposes the edge.
    simulate(_motion_spec(True))  # would raise on a margin violation


def test_cli_motion_miniscope_nonrigid():
    args = build_parser().parse_args(
        ["generate", "--motion", "miniscope", "--nonrigid", "--out", "x"]
    )
    spec = spec_from_args(args)
    motion = next(s for s in spec.steps if s.kind == "brain_motion")
    assert motion.model == "miniscope" and motion.nonrigid
    args = build_parser().parse_args(["generate", "--nonrigid", "--out", "x"])
    with pytest.raises(ValueError):
        spec_from_args(args)
    assert cli_main(["generate", "--nonrigid", "--out", "x", "--quiet"]) == 1


def test_presets_realistic_motion():
    m = presets.realistic_motion()
    assert m.model == "miniscope" and m.nonrigid
    assert not presets.realistic_motion(nonrigid=False).nonrigid
    assert presets.realistic_motion(burst_rate_hz=0.2).burst_rate_hz == 0.2
    spec = presets.build_spec(
        presets.generic_1p(), presets.ca1(), duration_s=1, extra_steps=[m]
    )
    assert any(s.kind == "brain_motion" for s in spec.steps)

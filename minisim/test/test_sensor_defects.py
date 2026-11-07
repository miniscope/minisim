"""SensorDefects: hot pixels, stuck pixels, PRNU, column/row fixed-pattern offsets."""

from __future__ import annotations

import numpy as np
import pytest

from minisim import (
    Acquisition,
    ImageSensor,
    Leakage,
    Optics,
    Recording,
    Sensor,
    SensorDefects,
    Spec,
    load_ground_truth,
    simulate,
    write_miniscope_daq,
)
from minisim.steps.sensor import sensor_defect_maps
from minisim.testing import make_recording
from minisim.video import _iter_count_frames

N = 64


def _acq(n=N, duration_s=1.0, **sensor_kw):
    return Acquisition(
        fps=20.0,
        duration_s=duration_s,
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=n, n_px_width=n, pixel_pitch_um=8.0, **sensor_kw
        ),
    )


def _flat_spec(defects=None, ppu=100.0, level=0.5, seed=1, n=N, duration_s=1.0, **kw):
    """No cells: a uniform glow through the sensor, so defect effects are measurable."""
    steps = [Leakage(profile="uniform", level=level), Sensor(photons_per_unit=ppu)]
    if defects is not None:
        steps.append(defects)
    return Spec(acquisition=_acq(n, duration_s, **kw), seed=seed, steps=steps)


def _mean_image(spec):
    return np.asarray(simulate(spec).observed).mean(axis=0)


# --- off by default / bit-identical -------------------------------------------------


def test_absent_step_leaves_default_output_untouched():
    # A spec without the step has no defect fields at all.
    gt = simulate(_flat_spec()).ground_truth
    for name in ("hot_pixel_mask", "prnu_gain", "fpn_column", "fpn_row"):
        assert getattr(gt, name) is None


def test_all_zero_defects_equal_no_defects_bit_for_bit():
    zero = SensorDefects(hot_pixel_fraction=0.0, prnu_sd=0.0, column_fpn_sd=0.0)
    a = simulate(_flat_spec()).observed
    b = simulate(_flat_spec(zero)).observed
    np.testing.assert_array_equal(a, b)


def test_defects_do_not_perturb_cells_or_motion():
    base = make_recording(n_cells=4, n_px=N, duration_s=1.0, seed=3, motion=True)
    with_d = make_recording(
        n_cells=4,
        n_px=N,
        duration_s=1.0,
        seed=3,
        motion=True,
        extra_steps=[SensorDefects()],
    )
    for name in ("C", "S", "shifts", "centers_um"):
        np.testing.assert_array_equal(
            getattr(base.ground_truth, name), getattr(with_d.ground_truth, name)
        )
    np.testing.assert_array_equal(
        base.ground_truth.A_observed, with_d.ground_truth.A_observed
    )
    assert not np.array_equal(base.observed, with_d.observed)  # but pixels differ


def test_requires_a_sensor_step():
    with pytest.raises(ValueError, match="requires a 'sensor' step"):
        Spec(acquisition=_acq(), steps=[SensorDefects()])


# --- maps -----------------------------------------------------------------------------


def test_hot_pixel_count_and_magnitude_distribution():
    spec = SensorDefects(n_hot_pixels=300)
    acq = _acq(256)
    m = sensor_defect_maps(spec, acq, 0)
    assert int(m.hot_mask.sum()) == 300
    vals = m.hot_excess_counts[m.hot_mask]
    assert vals.min() >= spec.hot_pixel_min_counts - 1e-9
    assert vals.max() <= spec.hot_pixel_max_counts + 1e-9
    # Pareto tail: P(>25)/P(>10) = (2.5)^-shape ~ 0.087 (clipping barely matters).
    frac = (vals > 25).mean()
    assert frac == pytest.approx(2.5**-spec.hot_pixel_tail_shape, abs=0.04)


def test_default_fraction_matches_reference_density():
    # ~209 px above +10 in a 600x600 frame -> default fraction 5.8e-4.
    spec = SensorDefects()
    big = Acquisition(
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(n_px_height=600, n_px_width=600, pixel_pitch_um=8.0),
        fps=20.0,
        duration_s=0.1,
    )
    m = sensor_defect_maps(spec, big, 0)
    assert abs(int(m.hot_mask.sum()) - 209) <= 3
    assert 5 <= (m.hot_excess_counts > 25).sum() <= 40  # reference: 18


def test_maps_are_seeded_and_effects_independent():
    acq = _acq(128)
    a = sensor_defect_maps(SensorDefects(), acq, 5)
    b = sensor_defect_maps(SensorDefects(), acq, 5)
    c = sensor_defect_maps(SensorDefects(), acq, 6)
    np.testing.assert_array_equal(a.hot_excess_counts, b.hot_excess_counts)
    assert not np.array_equal(a.hot_excess_counts, c.hot_excess_counts)
    # Changing the hot-pixel parameters leaves the PRNU / column maps untouched.
    d = sensor_defect_maps(SensorDefects(n_hot_pixels=5), acq, 5)
    np.testing.assert_array_equal(a.prnu_gain, d.prnu_gain)
    np.testing.assert_array_equal(a.fpn_column, d.fpn_column)
    # An explicit seed overrides the spec seed.
    e = sensor_defect_maps(SensorDefects(seed=9), acq, 5)
    f = sensor_defect_maps(SensorDefects(seed=9), acq, 77)
    np.testing.assert_array_equal(e.prnu_gain, f.prnu_gain)


# --- effects on the movie -------------------------------------------------------------


def test_hot_pixels_raise_the_mean_image_by_their_excess():
    d = SensorDefects(n_hot_pixels=20, prnu_sd=0.0, column_fpn_sd=0.0)
    base = _mean_image(_flat_spec(duration_s=4.0))
    hot = _mean_image(_flat_spec(d, duration_s=4.0))
    gt = simulate(_flat_spec(d)).ground_truth
    assert gt.hot_pixel_mask.sum() == 20
    diff = (hot - base)[gt.hot_pixel_mask]
    np.testing.assert_allclose(diff, gt.hot_pixel_excess[gt.hot_pixel_mask], atol=1.5)
    # and nothing else changed beyond noise
    assert abs((hot - base)[~gt.hot_pixel_mask].mean()) < 0.1


def test_column_and_row_fpn_shift_column_and_row_means():
    d = SensorDefects(
        hot_pixel_fraction=0.0, prnu_sd=0.0, column_fpn_sd=3.0, row_fpn_sd=2.0
    )
    spec = _flat_spec(d, duration_s=2.0)
    gt = simulate(spec).ground_truth
    base = _mean_image(_flat_spec(duration_s=2.0))
    delta = _mean_image(spec) - base
    np.testing.assert_allclose(
        delta.mean(axis=0), gt.fpn_column + gt.fpn_row.mean(), atol=0.5
    )
    np.testing.assert_allclose(
        delta.mean(axis=1), gt.fpn_row + gt.fpn_column.mean(), atol=0.5
    )
    assert gt.fpn_column.std() == pytest.approx(3.0, rel=0.4)


def test_prnu_scales_the_signal_per_pixel():
    d = SensorDefects(hot_pixel_fraction=0.0, prnu_sd=0.05, column_fpn_sd=0.0)
    spec = _flat_spec(d, duration_s=4.0, ppu=200.0, level=0.4)
    gt = simulate(spec).ground_truth
    base = _mean_image(_flat_spec(duration_s=4.0, ppu=200.0, level=0.4))
    ratio = _mean_image(spec) / base
    np.testing.assert_allclose(ratio, gt.prnu_gain, atol=0.03)
    assert gt.prnu_gain.std() == pytest.approx(0.05, rel=0.15)


def test_stuck_pixels_read_their_value():
    d = SensorDefects(
        hot_pixel_fraction=0.0,
        prnu_sd=0.0,
        column_fpn_sd=0.0,
        stuck_pixel_fraction=0.01,
        stuck_pixel_value=200.0,
    )
    rec = simulate(_flat_spec(d))
    gt = rec.ground_truth
    assert gt.stuck_pixel_mask.sum() == int(round(0.01 * N * N))
    stuck_vals = np.asarray(rec.observed)[:, gt.stuck_pixel_mask]
    assert (stuck_vals == 200.0).all()
    # default value is full scale for the sensor's bit depth
    d2 = SensorDefects(stuck_pixel_fraction=0.01)
    gt2 = simulate(_flat_spec(d2)).ground_truth
    assert (
        np.asarray(simulate(_flat_spec(d2)).observed)[:, gt2.stuck_pixel_mask] == 255
    ).all()


# --- ground truth, streaming, persistence --------------------------------------------


def test_ground_truth_fields_present_and_consistent():
    gt = simulate(
        _flat_spec(SensorDefects(n_hot_pixels=7, stuck_pixel_fraction=0.005))
    ).ground_truth
    assert gt.hot_pixel_mask.shape == (N, N) and gt.hot_pixel_mask.dtype == bool
    assert gt.prnu_gain.shape == (N, N)
    assert gt.fpn_column.shape == (N,) and gt.fpn_row.shape == (N,)
    assert (gt.hot_pixel_excess > 0).sum() == 7


def test_streaming_matches_simulate_with_defects():
    rec = make_recording(
        n_cells=3,
        n_px=N,
        duration_s=1.0,
        seed=4,
        motion=True,
        extra_steps=[
            SensorDefects(
                n_hot_pixels=10, column_fpn_sd=2.0, stuck_pixel_fraction=0.003
            )
        ],
    )
    streamed = np.stack([f for _, f in _iter_count_frames(rec.spec, 6)])
    np.testing.assert_array_equal(streamed, np.asarray(rec.observed))
    sink: dict = {}
    for _ in _iter_count_frames(rec.spec, 6, truth_sink=sink):
        break
    for name in ("hot_pixel_mask", "prnu_gain", "fpn_column", "stuck_pixel_mask"):
        np.testing.assert_array_equal(
            getattr(sink["ground_truth"], name), getattr(rec.ground_truth, name)
        )


def test_recording_save_load_round_trip(tmp_path):
    rec = simulate(_flat_spec(SensorDefects(n_hot_pixels=9)))
    rec.save(tmp_path / "r.zarr")
    back = Recording.load(tmp_path / "r.zarr").ground_truth
    for name in (
        "hot_pixel_mask",
        "hot_pixel_excess",
        "prnu_gain",
        "fpn_column",
        "fpn_row",
    ):
        np.testing.assert_array_equal(
            getattr(back, name), getattr(rec.ground_truth, name)
        )


def test_daq_ground_truth_carries_defect_maps(tmp_path):
    spec = make_recording(
        n_cells=3, n_px=N, duration_s=1.0, extra_steps=[SensorDefects(n_hot_pixels=6)]
    ).spec
    f = write_miniscope_daq(spec, tmp_path, progress=False)
    z = np.load(f.ground_truth_dir / "arrays.npz")
    assert z["hot_pixel_mask"].sum() == 6 and z["prnu_gain"].shape == (N, N)
    gt, _ = load_ground_truth(f.ground_truth_dir / "ground_truth.zarr")
    np.testing.assert_array_equal(gt.hot_pixel_mask, z["hot_pixel_mask"])


def test_photons_to_counts_offset_none_is_unchanged():
    sensor = ImageSensor(n_px_height=8, n_px_width=8)
    photons = np.full((8, 8), 30.0)
    a = sensor.photons_to_counts(photons, np.random.default_rng(0))
    b = sensor.photons_to_counts(photons, np.random.default_rng(0), offset_adu=None)
    np.testing.assert_array_equal(a, b)

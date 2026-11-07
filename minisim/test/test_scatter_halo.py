"""The heavy-tailed scatter PSF: ``(1-f)·G(σ_core) + f·G(σ_halo)``.

Defaults (``Tissue.scatter_halo_fraction = 0``) must change nothing; with ``f > 0`` the
observed footprint must follow the analytic two-component PSF, conserve the footprint
integral (the wing is redistributed from the core), keep the peak-brightness scalar
consistent with the rendered peak, and stream bit-for-bit like every other effect.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from minisim import (
    Acquisition,
    CellActivity,
    CellOptics,
    Composite,
    ImageSensor,
    Optics,
    PlaceNeurons,
    Sensor,
    Spec,
    Tissue,
    simulate,
)
from minisim.footprint import Footprint, degrade_footprint
from minisim.video import _iter_count_frames


def _delta(n: int, y: int, x: int) -> Footprint:
    d = np.zeros((n, n))
    d[y, x] = 1.0
    return Footprint.from_dense(d)


# --- spec: defaults are invisible -------------------------------------------------


def test_default_tissue_serializes_without_halo_fields():
    dumped = Tissue().model_dump()
    assert "scatter_halo_fraction" not in dumped
    assert "scatter_halo_blur_per_um" not in dumped
    assert (
        Tissue().model_dump_json()
        == Tissue(scatter_halo_fraction=0.0).model_dump_json()
    )


def test_active_halo_changes_serialization_and_cache_key():
    base = _spec(Tissue())
    halo = _spec(Tissue(scatter_halo_fraction=0.2))
    assert halo.model_dump()["acquisition"]["tissue"]["scatter_halo_fraction"] == 0.2
    assert base.cache_key() != halo.cache_key()
    # a customized rate alone (fraction still 0) is a different spec, too
    assert _spec(Tissue(scatter_halo_blur_per_um=0.5)).cache_key() != base.cache_key()


def test_halo_spec_round_trips_json():
    spec = _spec(Tissue(scatter_halo_fraction=0.3, scatter_halo_blur_per_um=0.4))
    assert Spec.model_validate_json(spec.model_dump_json()) == spec


# --- degrade_footprint ------------------------------------------------------------


def test_zero_fraction_is_bit_identical_to_single_gaussian():
    rng = np.random.default_rng(0)
    planted = np.zeros((90, 90))
    planted[30:44, 40:50] = rng.random((14, 10))
    fp = Footprint.from_dense(planted)
    ref = 0.7 * gaussian_filter(planted, sigma=3.3, mode="constant")
    base = degrade_footprint(fp, 3.3, 0.7)
    for kw in ({}, {"halo_sigma_px": 25.0, "halo_fraction": 0.0}):
        out = degrade_footprint(fp, 3.3, 0.7, **kw)
        np.testing.assert_array_equal(out.patch, base.patch)
        assert out.offset == base.offset
    np.testing.assert_allclose(base.to_dense(dtype=float), ref, atol=1e-6)


def test_radial_profile_matches_analytic_two_component_psf():
    n, c = 401, 200
    sc, sh, f, gain = 4.0, 24.0, 0.3, 2.0
    out = degrade_footprint(_delta(n, c, c), sc, gain, sh, f).to_dense(dtype=float)
    yy, xx = np.mgrid[:n, :n]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    analytic = gain * (
        (1 - f) * np.exp(-r2 / (2 * sc**2)) / (2 * np.pi * sc**2)
        + f * np.exp(-r2 / (2 * sh**2)) / (2 * np.pi * sh**2)
    )
    near = r2 <= (3 * sh) ** 2  # well inside the 4σ truncation, where the pdf applies
    np.testing.assert_allclose(out[near], analytic[near], rtol=2e-2, atol=1e-7)
    # the heavy tail: halo-dominated at 3σ_core, core-dominated at the centre
    assert out[c, c + 20] / out[c, c] > 1e-3
    assert out[c, c + 20] > 20 * gain * (1 - f) * np.exp(-(20**2) / (2 * sc**2)) / (
        2 * np.pi * sc**2
    )


def test_integral_is_conserved_and_halo_is_truncated():
    n, c = 401, 200
    sh = 20.0
    for f in (0.1, 0.45):
        fp = degrade_footprint(_delta(n, c, c), 3.0, 1.5, sh, f)
        assert float(fp.patch.sum()) == pytest.approx(1.5, rel=1e-3)
        ph, pw = fp.patch.shape
        # window = 4σ halo reach + small pad; never grows past the canvas
        assert ph <= 2 * (math.ceil(4 * sh) + 1) + 1
        assert pw <= n and ph <= n


def test_halo_near_canvas_edge_loses_light_like_core():
    fp = degrade_footprint(_delta(120, 3, 60), 3.0, 1.0, 20.0, 0.4)
    assert fp.patch.sum() < 0.95  # light past the edge is lost, as for the core
    ref = 0.6 * gaussian_filter(
        _delta(120, 3, 60).to_dense(dtype=float), 3.0, mode="constant"
    )
    ref += 0.4 * gaussian_filter(
        _delta(120, 3, 60).to_dense(dtype=float), 20.0, mode="constant"
    )
    np.testing.assert_allclose(fp.to_dense(dtype=float), ref, atol=2e-6)


# --- optics accounting ------------------------------------------------------------


def _acq(tissue: Tissue, **kw) -> Acquisition:
    return Acquisition(
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(n_px_height=64, n_px_width=64, pixel_pitch_um=8.0),
        tissue=tissue,
        **kw,
    )


def test_cell_optics_unchanged_at_zero_fraction():
    a0 = _acq(Tissue())
    a1 = _acq(Tissue(scatter_halo_fraction=0.0, scatter_halo_blur_per_um=0.9))
    for z, focal in ((0.0, 0.0), (60.0, 60.0), (150.0, 90.0)):
        assert a0.cell_optics(z, focal) == a1.cell_optics(z, focal)


def test_brightness_equals_rendered_point_source_peak():
    f, z, focal = 0.35, 120.0, 70.0
    acq = _acq(Tissue(scatter_halo_fraction=f))
    sigma_px, brightness = acq.cell_optics(z, focal)
    halo_px, frac = acq.halo_optics(z, focal)
    assert frac == f and halo_px > sigma_px
    gain = acq.tissue.attenuation(z) * acq.optics.collection_efficiency
    peak = degrade_footprint(
        _delta(401, 200, 200), sigma_px, gain, halo_px, f
    ).patch.max()
    sigma0_px = (
        math.hypot(acq.optics.diffraction_sigma_um, acq.tissue.scatter_sigma_um(z))
        / acq.pixel_size_um
    )
    # brightness is the peak relative to an unblurred-by-defocus single Gaussian σ_0
    assert brightness == pytest.approx(peak * 2 * np.pi * sigma0_px**2, rel=5e-3)


def test_halo_dims_peak_but_not_integral_or_surface_conservation():
    z, focal = 100.0, 100.0
    b0 = _acq(Tissue()).cell_optics(z, focal)[1]
    b1 = _acq(Tissue(scatter_halo_fraction=0.4)).cell_optics(z, focal)[1]
    assert b0 * 0.55 < b1 < b0  # redistributed: peak falls by ~(1-f), not more
    # integral of a cell is gain·Σplanted regardless of f (checked on footprints above)


# --- pipeline: recording, ground truth, streaming ---------------------------------


def _spec(tissue: Tissue, n_px=48, seed=3, sensor=True) -> Spec:
    acq = Acquisition(
        fps=20.0,
        duration_s=1.5,
        focal_depth_in_tissue_um="auto",
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=8
        ),
        tissue=tissue,
    )
    steps = [
        PlaceNeurons(
            density_per_mm3=40000.0, soma_radius_um=5.0, depth_range_um=(0.0, 60.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Composite(),
    ]
    if sensor:
        steps.append(Sensor(photons_per_unit=120.0))
    return Spec(acquisition=acq, seed=seed, steps=steps)


HALO = Tissue(scatter_halo_fraction=0.3, scatter_halo_blur_per_um=0.6)


def test_zero_fraction_recording_is_bit_identical_and_has_no_halo_truth():
    base = simulate(_spec(Tissue()))
    explicit = simulate(_spec(Tissue(scatter_halo_fraction=0.0)))
    assert (
        _spec(Tissue()).cache_key()
        == _spec(Tissue(scatter_halo_fraction=0.0)).cache_key()
    )
    np.testing.assert_array_equal(base.observed, explicit.observed)
    np.testing.assert_array_equal(
        base.ground_truth.A_observed, explicit.ground_truth.A_observed
    )
    assert base.ground_truth.observed_halo_fraction is None
    assert base.ground_truth.observed_halo_sigma_px is None


def test_halo_truth_is_stored_and_a_observed_matches_movie():
    spec = _spec(HALO, sensor=False)
    rec = simulate(spec)
    gt = rec.ground_truth
    assert gt.observed_halo_fraction is not None
    assert np.all(gt.observed_halo_fraction == 0.3)
    assert np.all(gt.observed_halo_sigma_px >= gt.observed_sigma_px)
    # No sensor / neuropil: the movie is exactly Σ A_observed · C (float32 render).
    movie = np.tensordot(gt.C, gt.A_observed, axes=([0], [0]))
    np.testing.assert_allclose(rec.observed, movie, rtol=2e-4, atol=1e-4 * movie.max())
    # the halo is visible: A_observed differs from the same scene with no wing
    flat = simulate(_spec(Tissue(), sensor=False)).ground_truth
    assert not np.allclose(gt.A_observed, flat.A_observed)
    np.testing.assert_array_equal(gt.A_planted, flat.A_planted)  # same cells


def test_halo_ground_truth_survives_save_load_and_subset(tmp_path):
    from minisim import load_ground_truth, save_ground_truth

    rec = simulate(_spec(HALO))
    save_ground_truth(rec.ground_truth, rec.spec, tmp_path / "gt.zarr")
    gt, _ = load_ground_truth(tmp_path / "gt.zarr")
    np.testing.assert_array_equal(
        gt.observed_halo_sigma_px, rec.ground_truth.observed_halo_sigma_px
    )
    np.testing.assert_allclose(gt.A_observed, rec.ground_truth.A_observed)
    sub = rec.ground_truth.detectable_subset()
    assert len(sub.observed_halo_fraction) == int(rec.ground_truth.detectable.sum())
    np.testing.assert_allclose(
        sub.A_observed, rec.ground_truth.A_observed[rec.ground_truth.detectable]
    )


def test_streamed_frames_match_simulate_with_halo():
    spec = _spec(HALO)
    streamed = np.stack([f for _, f in _iter_count_frames(spec, 8)])
    np.testing.assert_array_equal(streamed, simulate(spec).observed)


def test_streamed_daq_ground_truth_equals_simulate_with_halo(tmp_path):
    from minisim import load_ground_truth, write_miniscope_daq

    spec = _spec(HALO)
    out = write_miniscope_daq(spec, tmp_path / "s", progress=False)
    gt, _ = load_ground_truth(out.ground_truth_dir / "ground_truth.zarr")
    ref = simulate(spec).ground_truth
    for name in ("observed_halo_sigma_px", "observed_halo_fraction", "detectable"):
        np.testing.assert_array_equal(getattr(gt, name), getattr(ref, name))
    np.testing.assert_allclose(gt.A_observed, ref.A_observed)


# --- performance ------------------------------------------------------------------


def _render_time(tissue: Tissue) -> float:
    acq = Acquisition(
        fps=30.0,
        duration_s=10.0,  # 300 frames
        focal_depth_in_tissue_um="auto",
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=256, n_px_width=256, pixel_pitch_um=8.0, bit_depth=8
        ),
        tissue=tissue,
    )
    spec = Spec(
        acquisition=acq,
        seed=1,
        steps=[
            PlaceNeurons(
                density_per_mm3=30000.0,
                soma_radius_um=5.0,
                depth_range_um=(50.0, 150.0),
            ),
            CellActivity(),
            CellOptics(),
            Composite(),
            Sensor(photons_per_unit=120.0),
        ],
    )
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        rec = simulate(spec)
        _ = rec.ground_truth.A_observed  # on-demand regeneration is part of the cost
        best = min(best, time.perf_counter() - t0)
    return best


def test_halo_render_time_stays_within_budget():
    base = _render_time(Tissue())
    halo = _render_time(Tissue(scatter_halo_fraction=0.3))
    # Measured ~1.1x locally; the 2x bound is the stated budget.
    assert halo < 2.0 * base, f"halo {halo:.2f}s vs base {base:.2f}s"

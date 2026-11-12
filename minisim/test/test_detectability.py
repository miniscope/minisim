"""Unit tests for the matched-filter detectability criterion.

``matched_filter_snr`` is checked against hand-computed values on fixtures where the
answer is known analytically (single-pixel and flat footprints, uniform fields), and
its building blocks (``transient_energy``, ``detection_noise_fields``,
``spatial_snr2``, the focus scan's Gaussian closed form) are checked one by one. The
physics the change is about is pinned too: defocus lowers the matched-filter SNR
only as ``1/σ`` (light is conserved and pooled) where the old peak test fell as
``1/σ²``, and more transients raise the SNR as ``√n``.
"""

import numpy as np
import pytest

from minisim import (
    DETECT_SNR_THRESHOLD,
    Acquisition,
    Bleaching,
    CellActivity,
    Composite,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    PlaceNeurons,
    Sensor,
    Spec,
    Vignette,
    detection_snr,
    finalize,
    matched_filter_snr,
    peak_snr,
)
from minisim.footprint import Footprint, degrade_footprint
from minisim.recording import (
    detection_noise_fields,
    spatial_snr2,
    transient_energy,
)
from minisim.scene import Cell, Scene
from minisim.steps.cell import gaussian_spatial_snr2

PPU, QE, READ = 50.0, 0.8, 3.0


def _acq(n_px=16, duration_s=1.0, fps=20.0):
    return Acquisition(
        fps=fps,
        duration_s=duration_s,
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=n_px,
            n_px_width=n_px,
            pixel_pitch_um=8.0,
            quantum_efficiency=QE,
            read_noise_e=READ,
            bit_depth=12,
        ),
    )


def _spec(acq, *extra, sensor=True):
    steps = [PlaceNeurons(depth_range_um=(0.0, 0.0)), CellActivity(), Composite()]
    steps += list(extra)
    if sensor:
        steps.append(Sensor(photons_per_unit=PPU))
    return Spec(acquisition=acq, steps=steps)


def _patch(shape, y0, x0, patch):
    canvas = np.zeros(shape)
    ph, pw = np.shape(patch)
    canvas[y0 : y0 + ph, x0 : x0 + pw] = patch
    return Footprint.from_dense(canvas)


def _trace(nf, events=1, amp=4.0, base=1.0):
    trace = np.full(nf, base)
    for i in range(events):
        trace[(i + 1) * nf // (events + 1)] = base + amp
    return trace


def _gt(acq, spec, cells, **truth):
    """Finalize a hand-built scene; ``truth`` sets per-effect fields (e.g. leakage)."""
    scene = Scene.zeros(acq)
    scene.cells += cells
    for name, value in truth.items():
        setattr(scene.truth, name, value)
    return finalize(scene, spec).ground_truth


# --- analytic fixtures -------------------------------------------------------


def test_single_pixel_cell_matches_hand_computed_snr():
    # One lit pixel of weight a, one transient of height 4 over baseline 1. Signal
    # energy Σ Δ² = 16; the pixel's noise is shot on its mean light plus read noise.
    acq = _acq()
    nf = acq.n_frames
    a = 0.5
    trace = _trace(nf)
    gt = _gt(
        acq,
        _spec(acq),
        [
            Cell(
                center_um=(0, 0, 0),
                footprint_planted=_patch((16, 16), 8, 8, [[a]]),
                trace=trace,
            )
        ],
    )
    g = PPU * QE
    mean_e = a * trace.mean() * g
    expected = np.sqrt(16.0 * (g * a) ** 2 / (mean_e + READ**2))
    snr = matched_filter_snr(gt, _spec(acq))
    assert snr[0] == pytest.approx(expected, rel=1e-6)


def test_flat_footprint_pools_pixels_as_sqrt_n():
    # A flat n-pixel footprint at read-noise-limited light: SNR = √n × the
    # single-pixel SNR (the spatial matched filter pools the pixels).
    acq = _acq()
    nf = acq.n_frames
    trace = _trace(nf, base=0.0, amp=1.0)  # no baseline light -> read noise only

    def snr_for(patch):
        cell = Cell(
            center_um=(0, 0, 0),
            footprint_planted=_patch((16, 16), 4, 4, patch),
            trace=trace,
        )
        return matched_filter_snr(_gt(acq, _spec(acq), [cell]), _spec(acq))[0]

    one = snr_for([[0.01]])
    nine = snr_for(np.full((3, 3), 0.01))
    # 1e-2 of light gives a negligible shot term next to READ² = 9.
    assert nine / one == pytest.approx(3.0, rel=1e-3)


def test_snr_grows_as_sqrt_of_transient_count():
    acq = _acq(duration_s=4.0)
    nf = acq.n_frames
    fp = _patch((16, 16), 8, 8, [[0.01]])  # dim: read-noise-limited
    one = Cell(
        center_um=(0, 0, 0), footprint_planted=fp, trace=_trace(nf, events=1, base=0.0)
    )
    four = Cell(
        center_um=(0, 0, 0), footprint_planted=fp, trace=_trace(nf, events=4, base=0.0)
    )
    s1 = matched_filter_snr(_gt(acq, _spec(acq), [one]), _spec(acq))[0]
    s4 = matched_filter_snr(_gt(acq, _spec(acq), [four]), _spec(acq))[0]
    # the shot term grows a little with the extra mean light, so just under 2
    assert 1.99 < s4 / s1 <= 2.0


def test_defocus_lowers_snr_as_one_over_sigma_not_sigma_squared():
    # Same planted point source, two PSF widths, read-noise-limited: the integral
    # is conserved, so Σa² ∝ 1/σ² and the matched-filter SNR ∝ 1/σ; the peak (the
    # old test) falls as 1/σ².
    acq = _acq(n_px=64)
    nf = acq.n_frames
    trace = _trace(nf, base=0.0, amp=1e-3)
    point = _patch((64, 64), 32, 32, [[1.0]])
    snrs, peaks = [], []
    for sigma in (2.0, 4.0):
        cell = Cell(
            center_um=(0, 0, 0),
            footprint_planted=point,
            trace=trace,
            observed_sigma_px=sigma,
            observed_gain=1.0,
        )
        snrs.append(matched_filter_snr(_gt(acq, _spec(acq), [cell]), _spec(acq))[0])
        peaks.append(float(degrade_footprint(point, sigma, 1.0).patch.max()))
    assert snrs[0] / snrs[1] == pytest.approx(2.0, rel=0.02)
    assert peaks[0] / peaks[1] == pytest.approx(4.0, rel=0.02)


def test_other_light_raises_the_shot_noise():
    # A neighbour's light and the leakage glow under a cell add shot noise only
    # (their mean), lowering its SNR.
    acq = _acq()
    nf = acq.n_frames
    trace = _trace(nf)
    cell = Cell(
        center_um=(0, 0, 0),
        footprint_planted=_patch((16, 16), 8, 8, [[1.0]]),
        trace=trace,
    )
    alone = matched_filter_snr(_gt(acq, _spec(acq), [cell]), _spec(acq))[0]
    leak_spec = _spec(acq, Leakage(profile="uniform", level=5.0))
    leak = np.full((16, 16), 5.0)
    glow = matched_filter_snr(_gt(acq, leak_spec, [cell], leakage=leak), leak_spec)[0]
    g = PPU * QE
    expected = np.sqrt(16.0 * g**2 / ((trace.mean() + 5.0) * g + READ**2))
    assert glow == pytest.approx(expected, rel=1e-6)
    assert glow < alone


def test_noise_fields_walk_the_step_order():
    # Vignette before leakage dims the cell light but not the glow added after it;
    # the cell gain carries the vignette.
    acq = _acq()
    nf = acq.n_frames
    trace = _trace(nf)
    cell = Cell(
        center_um=(0, 0, 0),
        footprint_planted=_patch((16, 16), 8, 8, [[1.0]]),
        trace=trace,
    )
    spec = _spec(acq, Vignette(falloff=0.5), Leakage(profile="uniform", level=2.0))
    vig = np.linspace(0.5, 1.0, 256).reshape(16, 16)
    gt = _gt(acq, spec, [cell], vignette=vig, leakage=np.full((16, 16), 2.0))
    gain_e, mean_e = detection_noise_fields(gt, spec)
    v = gt.vignette[8, 8]
    g = PPU * QE
    assert gain_e[8, 8] == pytest.approx(v * g)
    assert mean_e[8, 8] == pytest.approx((trace.mean() * v + 2.0) * g)
    assert mean_e[0, 0] == pytest.approx(2.0 * g)


def test_neuropil_mean_enters_the_noise_floor():
    acq = _acq()
    spec = Spec(
        acquisition=acq,
        steps=[
            PlaceNeurons(density_per_mm3=200000.0, depth_range_um=(0.0, 0.0)),
            CellActivity(),
            Composite(),
            Neuropil(amplitude=0.5),
            Sensor(photons_per_unit=PPU),
        ],
    )
    from minisim import simulate

    gt = simulate(spec.model_copy(update={"seed": 1})).ground_truth
    _, mean_e = detection_noise_fields(gt, spec)
    np_mean = (
        0.5 * np.tensordot(gt.neuropil_temporal.mean(1), gt.neuropil_spatial, 1) / 3
    )
    cells = np.zeros(gt.fov_shape)
    emit = gt.C if gt.bleaching is None else gt.C * gt.bleaching
    for k, fp in enumerate(gt.observed_footprints()):
        if not fp.is_empty:
            y0, x0 = fp.offset
            ph, pw = fp.patch.shape
            cells[y0 : y0 + ph, x0 : x0 + pw] += emit[k].mean() * fp.patch
    np.testing.assert_allclose(mean_e, (cells + np_mean) * PPU * QE, rtol=1e-6)


def test_bleaching_fades_the_transient_energy():
    C = np.array([[1.0, 3.0, 1.0, 3.0]])
    B = np.array([[1.0, 1.0, 0.5, 0.5]])
    assert transient_energy(C)[0] == pytest.approx(8.0)
    assert transient_energy(C, B)[0] == pytest.approx(4.0 + 1.0)
    assert transient_energy(np.zeros((2, 0))).tolist() == [0.0, 0.0]


def test_spatial_snr2_weights_each_pixel_by_its_noise():
    fp = Footprint.from_dense(np.array([[1.0, 2.0], [0.0, 0.0]]))
    gain = np.full((2, 2), 2.0)
    mean = np.array([[1.0, 3.0], [0.0, 0.0]])
    # Σ (g a)² / (mean + r²) with r = 1: 4/2 + 16/4
    assert spatial_snr2([fp], gain, mean, 1.0)[0] == pytest.approx(2.0 + 4.0)
    empty = Footprint.from_dense(np.zeros((2, 2)))
    assert spatial_snr2([empty], gain, mean, 1.0)[0] == 0.0


# --- flag rules ------------------------------------------------------------------


def test_detectable_is_the_threshold_on_the_snr():
    acq = _acq()
    nf = acq.n_frames
    trace = _trace(nf)
    g = PPU * QE

    def snr_at(a):
        return np.sqrt(16.0 * (g * a) ** 2 / (a * trace.mean() * g + READ**2))

    # Find footprint weights just above and below the threshold by bisection on
    # the analytic single-pixel SNR.
    lo, hi = 1e-6, 1e3
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if snr_at(mid) < DETECT_SNR_THRESHOLD else (lo, mid)
    cells = [
        Cell(
            center_um=(0, 0, 0),
            footprint_planted=_patch((16, 16), 4, 4, [[hi * 1.01]]),
            trace=trace,
        ),
        Cell(
            center_um=(0, 0, 0),
            footprint_planted=_patch((16, 16), 12, 12, [[lo * 0.99]]),
            trace=trace,
        ),
    ]
    # Far apart, so neither cell's light reaches the other's pixel.
    gt = _gt(acq, _spec(acq), cells)
    assert gt.detectable.tolist() == [True, False]


def test_no_trace_is_never_detectable_and_no_sensor_falls_back_to_in_focus():
    acq = _acq()
    nf = acq.n_frames
    bright = _patch((16, 16), 8, 8, [[100.0]])
    cells = [
        Cell(center_um=(0, 0, 0), footprint_planted=bright, trace=None, in_focus=True),
        Cell(
            center_um=(0, 0, 0),
            footprint_planted=bright,
            trace=_trace(nf),
            in_focus=True,
        ),
        Cell(
            center_um=(0, 0, 0),
            footprint_planted=bright,
            trace=_trace(nf),
            in_focus=False,
        ),
    ]
    with_sensor = _gt(acq, _spec(acq), cells)
    assert with_sensor.detectable.tolist() == [False, True, True]
    no_sensor_spec = _spec(acq, sensor=False)
    assert matched_filter_snr(with_sensor, no_sensor_spec) is None
    without = _gt(acq, no_sensor_spec, cells)
    assert without.detectable.tolist() == [False, True, False]


def test_bleaching_spec_runs_through_finalize():
    # A full chain with bleaching still yields a finite SNR per cell.
    acq = _acq(duration_s=2.0)
    spec = Spec(
        acquisition=acq,
        steps=[
            PlaceNeurons(density_per_mm3=200000.0, depth_range_um=(0.0, 0.0)),
            CellActivity(),
            Bleaching(bleach_susceptibility=1e-2),
            Composite(),
            Sensor(photons_per_unit=PPU),
        ],
    )
    from minisim import simulate

    gt = simulate(spec.model_copy(update={"seed": 2})).ground_truth
    snr = matched_filter_snr(gt, spec)
    assert np.isfinite(snr).all() and (snr > 0).all()
    assert gt.detectable.tolist() == (snr >= DETECT_SNR_THRESHOLD).tolist()


# --- diagnostics and the focus scan's closed form --------------------------------


def test_peak_snr_is_the_old_formula_and_detection_snr_aliases_it():
    assert detection_snr is peak_snr
    assert peak_snr(2.0, 0.0, 3.0, 6.0) == pytest.approx(1.0)
    assert peak_snr(1.0, 4.0, 2.0, 1.0) == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("alpha", [0.0, 1e-6, 0.3, 5.0])
def test_gaussian_closed_form_matches_the_pixel_sum(alpha):
    # Σ a²/(αa + β) over a finely sampled 2-D Gaussian equals the closed form.
    peak, s, beta = 3.0, 6.0, 2.0
    y, x = np.mgrid[-60:61, -60:61]
    a = peak * np.exp(-(x**2 + y**2) / (2 * s**2))
    exact = (a * a / (alpha * a + beta)).sum()
    approx = gaussian_spatial_snr2(peak, s**2, alpha, beta)
    assert float(approx) == pytest.approx(exact, rel=1e-3)

"""Unit tests for the typed output + finalize() transform.

Covers ``finalize(scene, spec)`` turning an exhausted ``Scene`` into a
``Recording``/``GroundTruth``: array shapes and dtypes, the planted-vs-observed
footprint split, FOV cropping of margin cells under motion, the per-effect
ground-truth fields (present vs ``None``), the detectability rule, and
``detectable_subset()``. The ``simulate()`` orchestrator is tested separately.
"""

import numpy as np
import pytest

from minisim import (
    Acquisition,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    Composite,
    GroundTruth,
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
    sample_field_at,
)
from minisim.footprint import Footprint, FootprintStack
from minisim.recording import _vessel_overlap
from minisim.scene import Cell, Scene


def _acq(n_px=20, fps=20.0, duration_s=1.0, bit_depth=8, **kw):
    """Small scene at a clean 1.0 µm/px scale (pitch 8 / mag 8)."""
    kw.setdefault("optics", Optics(magnification=8.0))
    kw.setdefault(
        "image_sensor",
        ImageSensor(n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=bit_depth),
    )
    return Acquisition(fps=fps, duration_s=duration_s, **kw)


def _run(acq, steps, seed=0, margin_px=0):
    """Run a step list against a fresh (optionally margined) scene."""
    rng = np.random.default_rng(seed)
    scene = Scene.zeros(acq, margin_px=margin_px)
    for sspec in steps:
        sspec.build(acq, rng)(scene)
    return scene


def _dot(shape, iy, ix, value=1.0):
    """A single-lit-pixel footprint as a Footprint (the form cells now hold)."""
    fp = np.zeros(shape)
    fp[iy, ix] = value
    return Footprint.from_dense(fp)


# --- shapes / types --------------------------------------------------------


def test_finalize_produces_typed_recording():
    acq = _acq(n_px=24, duration_s=1.0)
    steps = [
        PlaceNeurons(density_per_mm3=250000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Composite(),
        Sensor(photons_per_unit=100.0),
    ]
    rec = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps))
    gt = rec.ground_truth
    n = gt.n_units
    assert n > 0
    assert gt.A_planted.shape == (n, 24, 24)
    assert gt.A_observed.shape == (n, 24, 24)
    assert gt.C.shape == (n, acq.n_frames)
    assert gt.S.shape == (n, acq.n_frames)
    assert gt.centers_um.shape == (n, 3)
    assert gt.in_focus.shape == (n,) and gt.detectable.shape == (n,)
    # observed is the integer-count movie downcast to the store dtype.
    assert rec.observed.shape == (acq.n_frames, 24, 24)
    assert rec.observed.dtype == np.float32
    np.testing.assert_array_equal(rec.observed, np.round(rec.observed))


def test_observed_footprint_differs_from_planted_under_optics():
    acq = _acq(n_px=40, optics=Optics(magnification=8.0), focal_depth_in_tissue_um=0.0)
    steps = [
        PlaceNeurons(density_per_mm3=62500.0, soma_radius_um=4.0, depth_range_um=(80.0, 120.0)),
        CellActivity(active_rate_hz=5.0),
        CellOptics(),
        Composite(),
        Sensor(),
    ]
    gt = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps)).ground_truth
    # Deep cells: scatter + defocus make the observed footprint dimmer/broader.
    assert not np.allclose(gt.A_planted, gt.A_observed)
    assert gt.A_observed.sum() < gt.A_planted.sum()


def test_regenerate_observed_handles_a_cell_without_optics_params():
    # A heterogeneous stack: optics ran for the population (so observed_sigma_px is
    # not None), but one cell carries NaN sigma (e.g. added after the optics step).
    # That cell must fall back to its sharp planted footprint; the others degrade.
    n_frames = 3
    planted = FootprintStack.from_footprints(
        [_dot((24, 24), 8, 8), _dot((24, 24), 16, 16)], (24, 24)
    )
    gt = GroundTruth(
        planted=planted,
        fov_offset=(0, 0),
        fov_shape=(24, 24),
        observed_sigma_px=np.array([np.nan, 2.0]),  # cell 0 has no optics params
        observed_gain=np.array([np.nan, 0.5]),
        C=np.zeros((2, n_frames)),
        S=np.zeros((2, n_frames)),
        centers_um=np.zeros((2, 3)),
        amplitude_per_cell=np.ones(2),
        in_focus=np.array([True, True]),
        detectable=np.array([True, True]),
    )
    a_planted, a_observed = gt.A_planted, gt.A_observed
    # Cell 0 (NaN sigma) -> sharp: observed == planted, bit for bit.
    np.testing.assert_array_equal(a_observed[0], a_planted[0])
    # Cell 1 (real sigma) -> degraded: blurred/scaled, no longer the sharp footprint.
    assert not np.allclose(a_observed[1], a_planted[1])


# --- per-effect fields -----------------------------------------------------


def test_per_effect_fields_are_none_when_steps_absent():
    acq = _acq()
    steps = [
        PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)),
        CellActivity(tau_decay_s=0.4),
        Composite(),
        Sensor(),
    ]
    gt = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps)).ground_truth
    assert gt.shifts is None
    assert gt.vignette is None
    assert gt.leakage is None
    assert gt.bleaching is None
    assert gt.neuropil_spatial is None and gt.neuropil_temporal is None


def test_per_effect_fields_present_for_full_pipeline():
    acq = _acq(n_px=32, duration_s=1.0)
    max_shift_um = 3.0
    margin = int(np.ceil(acq.um_to_px(max_shift_um))) + 1
    steps = [
        PlaceNeurons(density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 100.0)),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        Bleaching(),
        CellOptics(),
        Composite(),
        Neuropil(n_components=2),
        BrainMotion(model="walk", walk_step_um=0.3, max_shift_um=max_shift_um),
        Vignette(falloff=0.6),
        Leakage(profile="gaussian", level=0.1),
        Sensor(),
    ]
    gt = finalize(
        _run(acq, steps, margin_px=margin), Spec(acquisition=acq, steps=steps)
    ).ground_truth
    assert gt.shifts.shape == (acq.n_frames, 2)
    assert gt.vignette.shape == (32, 32)  # FOV-sized
    assert gt.leakage.shape == (32, 32)
    assert gt.bleaching.shape == (gt.n_units, acq.n_frames)  # per-cell envelopes
    assert gt.neuropil_temporal.shape == (2, acq.n_frames)
    assert gt.neuropil_spatial.shape == (2, 32, 32)  # cropped to FOV grid


# --- FOV cropping of margin cells ------------------------------------------


def test_finalize_drops_pure_margin_cells():
    acq = _acq(n_px=20)
    margin = 6
    canvas = 20 + 2 * margin
    nf = acq.n_frames
    scene = Scene.zeros(acq, margin_px=margin)
    # one cell at the FOV center (canvas center pixel = margin + 10), one entirely
    # in the top margin (rows < margin). Footprint pixels are explicit; center_um is
    # the optical-center frame (origin = FOV center), invariant to the margin.
    scene.cells += [
        Cell(
            center_um=(0.0, 0.0, 0.0),
            footprint_planted=_dot((canvas, canvas), margin + 10, margin + 10),
            trace=np.ones(nf),
        ),
        Cell(
            center_um=(0.0, -100.0, 0.0),  # dropped anyway; coordinate is moot
            footprint_planted=_dot((canvas, canvas), 2, 10),
            trace=np.ones(nf),
        ),
    ]
    minimal = [PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)), Composite()]
    gt = finalize(scene, Spec(acquisition=acq, steps=minimal)).ground_truth
    assert gt.n_units == 1  # the margin-only cell was dropped
    assert gt.A_planted.shape == (1, 20, 20)
    # surviving cell's center is in the optical-center frame: the FOV center is (0, 0).
    np.testing.assert_allclose(gt.centers_um[0], [0.0, 0.0, 0.0])


# --- detectability ---------------------------------------------------------


def _detect_spec(acq):
    return Spec(
        acquisition=acq,
        steps=[
            PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
            CellActivity(tau_decay_s=0.4),
            Composite(),
            Sensor(photons_per_unit=100.0),
        ],
    )


def test_detectable_reflects_brightness_and_focus():
    acq = _acq(n_px=16)
    nf = acq.n_frames
    trace = np.ones(nf)
    trace[nf // 2] = 5.0  # baseline 1, transient peak 5
    scene = Scene.zeros(acq)
    scene.cells += [
        # bright, in focus -> well above the sensor floor
        Cell(center_um=(0.0, 8.0, 8.0), footprint_planted=_dot((16, 16), 8, 8),
             trace=trace, in_focus=True, optical_brightness=1.0),
        # in focus but optically ~dark -> below the floor
        Cell(center_um=(0.0, 4.0, 4.0), footprint_planted=_dot((16, 16), 4, 4),
             trace=trace, in_focus=True, optical_brightness=1e-3),
        # bright but out of focus -> gated out by in_focus
        Cell(center_um=(0.0, 12.0, 12.0), footprint_planted=_dot((16, 16), 12, 12),
             trace=trace, in_focus=False, optical_brightness=1.0),
    ]
    gt = finalize(scene, _detect_spec(acq)).ground_truth
    assert gt.detectable.tolist() == [True, False, False]
    # detectable is always a subset of in_focus.
    assert not (gt.detectable & ~gt.in_focus).any()


def test_detectable_subset_keeps_only_detectable_units():
    acq = _acq(n_px=16)
    nf = acq.n_frames
    trace = np.ones(nf)
    trace[nf // 2] = 5.0
    scene = Scene.zeros(acq)
    scene.cells += [
        Cell(center_um=(0.0, 8.0, 8.0), footprint_planted=_dot((16, 16), 8, 8),
             trace=trace, in_focus=True, optical_brightness=1.0),
        Cell(center_um=(0.0, 4.0, 4.0), footprint_planted=_dot((16, 16), 4, 4),
             trace=trace, in_focus=True, optical_brightness=1e-3),
    ]
    gt = finalize(scene, _detect_spec(acq)).ground_truth
    sub = gt.detectable_subset()
    assert sub.n_units == 1
    assert sub.detectable.all()
    assert isinstance(sub, GroundTruth)


def test_vessel_occlusion_dims_detectability_and_records_overlap():
    # A bright, in-focus cell is detectable in the clear; drop an opaque vessel over
    # it and its peak no longer clears the floor, while its footprint-weighted
    # occlusion is recorded as the scoreable confound axis.
    acq = _acq(n_px=16)
    nf = acq.n_frames
    trace = np.ones(nf)
    trace[nf // 2] = 5.0
    spec = _detect_spec(acq)

    def _scene():
        s = Scene.zeros(acq)
        s.cells.append(
            Cell(center_um=(0.0, 0.0, 0.0), footprint_planted=_dot((16, 16), 8, 8),
                 trace=trace, in_focus=True, optical_brightness=1.0)
        )
        return s

    clear = finalize(_scene(), spec).ground_truth
    assert clear.detectable.tolist() == [True]
    assert clear.vessel_overlap_fraction is None  # vasculature step absent -> no field

    occluded = _scene()
    mask = np.ones((16, 16))
    mask[8, 8] = 1e-3  # near-opaque vessel right over the cell
    occluded.truth.vasculature_mask = mask
    gt = finalize(occluded, spec).ground_truth
    assert gt.detectable.tolist() == [False]  # vessel knocked it below the floor
    assert gt.vessel_overlap_fraction[0] == pytest.approx(1.0 - 1e-3)


def test_vessel_overlap_is_footprint_weighted():
    # _vessel_overlap weights transmission by the footprint, so a vessel covering
    # only part of a multi-pixel footprint occludes only that fraction.
    fp = Footprint.from_dense(np.array([[1.0, 3.0], [0.0, 0.0]]))  # weights 1 and 3
    mask = np.array([[0.0, 1.0], [1.0, 1.0]])  # opaque over the weight-1 pixel only
    # surviving = (1*0 + 3*1)/(1+3) = 0.75 -> overlap 0.25
    assert _vessel_overlap(fp, mask) == pytest.approx(0.25)
    assert _vessel_overlap(fp, None) == 0.0  # no mask -> no occlusion
    assert _vessel_overlap(None, mask) == 0.0  # no footprint -> no occlusion


# --- snapshots / stage -----------------------------------------------------


def test_stage_raises_for_absent_snapshot():
    acq = _acq()
    steps = [PlaceNeurons(depth_range_um=(0.0, 0.0)), Composite(), Sensor()]
    rec = finalize(_run(acq, steps), Spec(acquisition=acq, steps=steps))
    assert rec.snapshots == {}  # save_intermediates defaulted off
    with pytest.raises(KeyError, match="save_intermediates"):
        rec.stage("observed")


# --- shared detectability primitives ---------------------------------------


def test_detection_snr_formula_and_edges():
    # SNR = peak*gain / sqrt(max(baseline,0)*gain + read^2). With read noise only
    # (zero baseline) it is just peak*gain/read.
    assert detection_snr(2.0, 0.0, 3.0, 6.0) == pytest.approx(2.0 * 3.0 / 6.0)
    # shot noise on the baseline adds in quadrature under the root.
    snr = detection_snr(1.0, 4.0, 2.0, 1.0)
    assert snr == pytest.approx((1.0 * 2.0) / np.sqrt(4.0 * 2.0 + 1.0))
    # vectorizes; a zero noise floor with signal is inf, with no signal is 0.
    out = detection_snr(np.array([1.0, 0.0]), 0.0, np.array([1.0, 1.0]), 0.0)
    assert out[0] == np.inf and out[1] == 0.0


def test_sample_field_at_clamps_and_handles_none():
    field = np.arange(12.0).reshape(3, 4)  # value == 4*row + col at 1 um/px
    # optical-center frame: (0, 0) µm is the field center pixel ((h-1)/2, (w-1)/2).
    assert sample_field_at(field, 0.0, 0.0, 1.0) == field[1, 2]
    # positions past the edge clamp to the nearest pixel rather than wrapping/erroring.
    assert sample_field_at(field, 99.0, 99.0, 1.0) == field[2, 3]
    assert sample_field_at(field, -99.0, -99.0, 1.0) == field[0, 0]
    # a non-unit pixel size scales the position; an absent field is the identity 1.0.
    assert sample_field_at(field, 2.0, 0.0, 2.0) == field[2, 2]
    assert sample_field_at(None, 0.0, 0.0, 1.0) == 1.0

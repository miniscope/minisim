"""Unit tests for the simulate() orchestrator.

Covers the compose loop end-to-end: a recording from a full ``Spec``, seed
determinism, automatic motion-margin sizing (so motion specs need no hand-sized
canvas), ``until=`` partial builds, and the per-stage snapshot keys.
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
    IlluminationProfile,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Sensor,
    Spec,
    Vignette,
    simulate,
)
from minisim.steps import combined_falloff_field


def _acq(n_px=64, fps=20.0, duration_s=1.0, bit_depth=8):
    """64×64 sensor at a clean 1.0 µm/px scale (pitch 8 / mag 8)."""
    return Acquisition(
        fps=fps,
        duration_s=duration_s,
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=bit_depth
        ),
    )


def _minimal_spec(**output_kw):
    acq = _acq()
    return Spec(
        acquisition=acq,
        seed=7,
        steps=[
            PlaceNeurons(
                density_per_mm3=312500.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)
            ),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Composite(),
            Sensor(photons_per_unit=100.0),
        ],
        output=Output(**output_kw),
    )


def _full_spec(**output_kw):
    acq = _acq()
    return Spec(
        acquisition=acq,
        seed=11,
        steps=[
            PlaceNeurons(
                density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 100.0)
            ),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            Bleaching(),
            CellOptics(),
            Composite(),
            Neuropil(n_components=2),
            BrainMotion(
                model="walk", walk_step_um=0.3, max_shift_um=2.0
            ),  # ≤ 5% of the 64 µm FOV
            Vignette(falloff=0.6),
            Leakage(profile="gaussian", level=0.1),
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(**output_kw),
    )


def test_simulate_runs_minimal_spec_end_to_end():
    rec = simulate(_minimal_spec())
    acq = rec.spec.acquisition
    assert rec.observed.shape == (acq.n_frames, 64, 64)
    assert rec.observed.dtype == np.float32
    np.testing.assert_array_equal(rec.observed, np.round(rec.observed))  # counts
    assert 0.0 <= rec.observed.min() and rec.observed.max() <= 255.0
    assert rec.ground_truth.n_units > 0


def test_simulate_is_deterministic_given_seed():
    a = simulate(_minimal_spec())
    b = simulate(_minimal_spec())
    np.testing.assert_array_equal(a.observed, b.observed)
    assert a.ground_truth.n_units == b.ground_truth.n_units
    np.testing.assert_array_equal(a.ground_truth.A_observed, b.ground_truth.A_observed)


def test_simulate_auto_sizes_motion_margin():
    # A motion spec runs with no hand-sized canvas: simulate() pads it, and the
    # observed movie comes back cropped to the sensor FOV with shifts recorded.
    rec = simulate(_full_spec())
    acq = rec.spec.acquisition
    assert rec.observed.shape == (acq.n_frames, 64, 64)
    assert rec.ground_truth.shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(rec.ground_truth.shifts[0], [0.0, 0.0])


def test_simulate_until_stops_after_named_stage():
    rec = simulate(_full_spec(save_intermediates=True), until="cells_only")
    # only render (and the cell steps before it) ran -> just the cells_only snapshot
    assert set(rec.snapshots) == {"cells_only"}
    # sensor never ran, so observed is the raw render movie (not integer counts)
    assert not np.allclose(rec.observed, np.round(rec.observed))


def test_simulate_until_cell_domain_skips_the_movie():
    # Stopping after a cell-domain step (optics) runs no pixel-writing step, so no
    # movie buffer is allocated: observed comes back empty (0, H, W) while the
    # per-cell ground truth is fully populated. This is the fast path the anatomy
    # notebook's until="optics" cells rely on (no (n_frames, H, W) allocation).
    rec = simulate(_minimal_spec(), until="optics")
    acq = rec.spec.acquisition
    assert rec.observed.shape == (0, 64, 64)
    assert rec.ground_truth.n_units > 0
    assert rec.ground_truth.C.shape[1] == acq.n_frames


def test_simulate_physical_motion_runs_end_to_end():
    # The default (physical) brain-motion model runs through the orchestrator: the
    # margin is auto-sized from max_shift_um, the movie comes back cropped to the
    # sensor FOV, and the recorded shifts start at the reference and stay bounded.
    acq = _acq(duration_s=3.0)  # 60 frames, 64 µm FOV
    spec = Spec(
        acquisition=acq,
        seed=5,
        steps=[
            PlaceNeurons(
                density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 80.0)
            ),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Composite(),
            BrainMotion(motion_amplitude_um=2.0, max_shift_um=3.0),  # physical default
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(),
    )
    rec = simulate(spec)
    assert rec.observed.shape == (acq.n_frames, 64, 64)
    shifts = rec.ground_truth.shifts
    np.testing.assert_array_equal(shifts[0], [0.0, 0.0])
    assert np.hypot(shifts[:, 0], shifts[:, 1]).max() <= 3.0 + 1e-9
    assert np.abs(shifts).max() > 0.0  # the brain really moved


def test_simulate_records_both_illumination_and_vignette_fields():
    # Illumination profile + vignette both run as static sensor-frame fields; each is
    # recorded separately, and detectability reflects their combined photon budget.
    acq = _acq()
    spec = Spec(
        acquisition=acq,
        seed=7,
        steps=[
            PlaceNeurons(
                density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 60.0)
            ),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Composite(),
            IlluminationProfile(falloff=0.5),
            Vignette(falloff=0.6),
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(),
    )
    rec = simulate(spec)
    gt = rec.ground_truth
    assert gt.illumination.shape == (64, 64)
    assert gt.vignette.shape == (64, 64)
    assert gt.illumination.max() == pytest.approx(
        1.0, abs=2e-3
    )  # bright center (even grid)
    assert gt.illumination[0, 0] == pytest.approx(
        0.5
    )  # edge excitation, farthest corner
    assert gt.vignette[0, 0] == pytest.approx(0.6)  # corner collection loss


def test_combined_falloff_field_matches_the_applied_illumination_vignette_product():
    # The spec-only field auto-focus consumes up front must equal what the sensor
    # steps actually apply (the recorded illumination × vignette), or focus would
    # be chosen against a different photon budget than the pipeline produces.
    acq = _acq()
    illum = IlluminationProfile(falloff=0.5, exponent=1.7, center_offset_um=(8.0, -4.0))
    vig = Vignette(falloff=0.6, exponent=2.3)
    spec = Spec(
        acquisition=acq,
        seed=7,
        steps=[
            PlaceNeurons(
                density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 60.0)
            ),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Composite(),
            illum,
            vig,
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(),
    )
    gt = simulate(spec).ground_truth
    predicted = combined_falloff_field(acq, illum, vig)
    np.testing.assert_allclose(
        predicted, gt.illumination * gt.vignette, rtol=0, atol=1e-12
    )


def test_simulate_records_the_resolved_auto_focus_depth():
    # "auto" focus is resolved once at the optics step; the chosen plane is recorded
    # so it is observable. With no field curvature the effective depths are the cell
    # depths, so the yield-maximizing plane lands within the placed layer.
    acq = _acq()  # flat field (no curvature) -> effective depth == cell depth
    spec = Spec(
        acquisition=acq,
        seed=7,
        steps=[
            PlaceNeurons(
                density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 60.0)
            ),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Composite(),
            Sensor(photons_per_unit=120.0),
        ],
        output=Output(),
    )
    gt = simulate(spec).ground_truth
    assert gt.focal_depth_um is not None
    assert gt.depth_um.min() <= gt.focal_depth_um <= gt.depth_um.max()


def test_simulate_save_intermediates_records_movie_stage_names():
    rec = simulate(_full_spec(save_intermediates=True))
    assert set(rec.snapshots) == {
        "cells_only",
        "neuropil",
        "brain_motion",
        "vignette",
        "leakage",
        "sensor",
    }
    # cell-domain steps are not snapshotted (they don't touch the movie)
    assert "place_neurons" not in rec.snapshots
    assert "optics" not in rec.snapshots
    assert "bleaching" not in rec.snapshots  # cell-domain: writes per-cell envelopes
    # the sensor-stage snapshot is the observed movie
    np.testing.assert_array_equal(rec.stage("sensor").values, rec.observed)


def test_simulate_without_intermediates_keeps_no_snapshots():
    rec = simulate(_minimal_spec())
    assert rec.snapshots == {}


def test_simulate_rejects_unknown_until_stage():
    with pytest.raises(ValueError, match="until="):
        simulate(_minimal_spec(), until="not_a_stage")

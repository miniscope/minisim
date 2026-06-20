"""Unit tests for the scope/region presets and the ``build_spec`` composer.

Covers the confirmed Miniscope V4 numbers and its ~1 mm FOV, the standard-region
anatomy, that ``build_spec`` composes any scope × region into a runnable ``Spec``
(default chain, population override, extra steps, vasculature toggle), and that
the ``build_recording`` studio presets stay in lock-step with this single source
of truth.
"""

import numpy as np
import pytest

from minisim import (
    BrainMotion,
    NeuronPopulation,
    Sensor,
    Spec,
    build_spec,
    presets,
    simulate,
    sweep,
)


def test_miniscope_v4_numbers_and_fov():
    scope = presets.miniscope_v4()
    assert scope.optics.na == 0.3
    assert scope.optics.magnification == 2.9
    assert scope.image_sensor.n_px_height == 608
    assert scope.image_sensor.pixel_pitch_um == 4.8
    assert scope.image_sensor.bit_depth == 8
    assert scope.front_working_distance_um == 700.0
    # magnification is set so the sensor sees the V4's ~1.0 mm field of view
    h_um, w_um = scope.fov_um
    assert h_um == pytest.approx(1006.3, abs=1.0)
    assert w_um == pytest.approx(1006.3, abs=1.0)


def test_region_anatomy():
    ca1 = presets.ca1()
    assert ca1.population.depth_range_um == (140.0, 160.0)
    assert ca1.population.morphology == "cytosolic"
    assert ca1.vasculature is not None and ca1.vasculature.enabled
    assert ca1.neuropil is not None
    cortex = presets.cortex_l23()
    assert cortex.population.depth_range_um == (100.0, 200.0)
    # cortex vessels are thicker / higher-contrast than CA1's
    assert (
        cortex.vasculature.layers[0].root_radius_um
        > ca1.vasculature.layers[0].root_radius_um
    )
    assert cortex.vasculature.layers[0].opacity > ca1.vasculature.layers[0].opacity
    # cortex L2/3 carries more prominent neuropil haze than CA1's thin soma band
    assert cortex.neuropil is not None
    assert cortex.neuropil.amplitude > ca1.neuropil.amplitude


def test_build_spec_default_chain():
    spec = build_spec(presets.miniscope_v4(), presets.ca1(), duration_s=1.0)
    assert isinstance(spec, Spec)
    kinds = [s.kind for s in spec.steps]
    # forward chain + the region's neuropil/vasculature + the V4 scope's static fields
    assert kinds == [
        "place_neurons",
        "cell_activity",
        "optics",
        "composite",
        "neuropil",
        "vasculature",
        "illumination_profile",
        "vignette",
        "leakage",
        "sensor",
    ]
    # the scope/region flowed into the acquisition
    assert spec.acquisition.optics.na == 0.3
    assert spec.acquisition.image_sensor.n_px_height == 608
    # the scope's exposure became the sensor's photons_per_unit (V4 = 600)
    sensor = next(s for s in spec.steps if s.kind == "sensor")
    assert sensor.photons_per_unit == 600.0


def test_build_spec_population_override_places_exact_cells():
    # two overlapping somata near the optical axis (optical-center frame), at
    # explicit positions instead of density sampling
    overlapping = NeuronPopulation(
        positions_um=[(150.0, 0.0, -5.0), (150.0, 0.0, 5.0)],
        soma_radius_um=5.0,
        morphology="cytosolic",
    )
    spec = build_spec(
        presets.miniscope_v4(),
        presets.ca1(),
        duration_s=1.0,
        populations=[overlapping],
    )
    place = next(s for s in spec.steps if s.kind == "place_neurons")
    assert place.populations == [overlapping]


def test_build_spec_extra_steps_and_canonical_order():
    spec = build_spec(
        presets.generic_1p(),
        presets.cortex_l23(),
        duration_s=1.0,
        extra_steps=[BrainMotion(motion_amplitude_um=2.0)],
    )
    # brain_motion was appended after sensor but Spec re-sorts to canonical order
    kinds = [s.kind for s in spec.steps]
    assert "brain_motion" in kinds
    assert kinds.index("brain_motion") < kinds.index("sensor")


def test_build_spec_can_drop_vasculature():
    spec = build_spec(
        presets.miniscope_v4(), presets.ca1(), duration_s=1.0, include_vasculature=False
    )
    assert "vasculature" not in [s.kind for s in spec.steps]


def test_build_spec_can_drop_neuropil():
    spec = build_spec(
        presets.miniscope_v4(), presets.ca1(), duration_s=1.0, include_neuropil=False
    )
    assert "neuropil" not in [s.kind for s in spec.steps]


def test_build_spec_can_drop_scope_fields():
    spec = build_spec(
        presets.miniscope_v4(),
        presets.ca1(),
        duration_s=1.0,
        include_scope_fields=False,
    )
    kinds = [s.kind for s in spec.steps]
    assert "illumination_profile" not in kinds
    assert "vignette" not in kinds
    assert "leakage" not in kinds


def test_generic_scope_adds_no_static_fields():
    # the generic scope sets no illumination/vignette/leakage, so build_spec leaves
    # the chain flat-field even with include_scope_fields on (the default)
    scope = presets.generic_1p()
    assert scope.illumination is None
    assert scope.vignette is None
    assert scope.leakage is None
    spec = build_spec(scope, presets.ca1(), duration_s=1.0)
    kinds = [s.kind for s in spec.steps]
    assert "illumination_profile" not in kinds
    assert "vignette" not in kinds
    assert "leakage" not in kinds
    # and the generic scope's exposure (library default) reaches the sensor
    sensor = next(s for s in spec.steps if s.kind == "sensor")
    assert sensor.photons_per_unit == scope.photons_per_unit == 100.0


def test_build_spec_simulates_end_to_end():
    # a small generic scope keeps this fast; explicit cells near the optical axis
    # (optical-center frame) so it is deterministic
    cells = NeuronPopulation(
        positions_um=[(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)], soma_radius_um=4.0
    )
    spec = build_spec(
        presets.generic_1p(),
        presets.cortex_l23(),
        duration_s=1.0,
        fps=20.0,
        seed=3,
        populations=[cells],
        sensor=Sensor(photons_per_unit=300.0),
    )
    rec = simulate(spec)
    h = spec.acquisition.image_sensor.n_px_height
    w = spec.acquisition.image_sensor.n_px_width
    assert rec.observed.shape == (spec.acquisition.n_frames, h, w)
    assert rec.ground_truth.n_units == 2
    np.testing.assert_array_equal(
        rec.observed, np.round(rec.observed)
    )  # integer counts


def test_positions_um_round_trip_to_centers_um_under_motion():
    # The optical-center frame exists so that the input positions_um come back
    # unchanged as GroundTruth.centers_um - including under a motion margin, which
    # grows the canvas (the frame is margin-invariant). This is the whole reason
    # for the frame, so lock it: place exact cells near the axis, add brain motion,
    # and check they round-trip. (generic_1p's ~96 um FOV keeps the cells central.)
    positions = [(120.0, 0.0, -5.0), (120.0, 0.0, 5.0)]
    cells = NeuronPopulation(positions_um=positions, soma_radius_um=4.0)
    spec = build_spec(
        presets.generic_1p(),
        presets.cortex_l23(),
        duration_s=1.0,
        fps=20.0,
        seed=1,
        populations=[cells],
        extra_steps=[BrainMotion(motion_amplitude_um=3.0)],
    )
    rec = simulate(spec)
    assert rec.ground_truth.n_units == len(positions)
    # order-independent compare (lexsort by z, y, x) so this does not depend on the
    # cell ordering finalize happens to emit
    got = np.asarray(rec.ground_truth.centers_um, dtype=float)
    want = np.asarray(positions, dtype=float)
    got = got[np.lexsort((got[:, 2], got[:, 1], got[:, 0]))]
    want = want[np.lexsort((want[:, 2], want[:, 1], want[:, 0]))]
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_build_spec_is_a_valid_sweep_base():
    base = build_spec(presets.miniscope_v4(), presets.ca1(), duration_s=1.0)
    specs = list(
        sweep(base, {"acquisition.focal_depth_in_tissue_um": [140.0, 150.0, 160.0]})
    )
    assert [s.acquisition.focal_depth_in_tissue_um for s in specs] == [
        140.0,
        150.0,
        160.0,
    ]


def test_studio_presets_match_library_presets():
    # the studio's V4 region presets read their physical numbers from minisim.presets,
    # so the two must agree (single source of truth).
    from minisim.notebooks.studio.build_recording._studio_config import STUDIO_PRESETS

    cfg = STUDIO_PRESETS["Miniscope V4 - CA1"]()
    scope, region = presets.miniscope_v4(), presets.ca1()
    assert cfg.na == scope.optics.na
    assert cfg.magnification == scope.optics.magnification
    assert cfg.n_px_height == scope.image_sensor.n_px_height
    assert cfg.pixel_pitch_um == scope.image_sensor.pixel_pitch_um
    assert cfg.bit_depth == scope.image_sensor.bit_depth
    assert cfg.front_working_distance_um == scope.front_working_distance_um
    # the scope's static field signature + exposure flow through too
    assert cfg.photons_per_unit == scope.photons_per_unit
    assert cfg.illumination_falloff == scope.illumination.falloff
    assert cfg.vignette_falloff == scope.vignette.falloff
    assert cfg.leakage_level == scope.leakage.level
    assert cfg.density_per_mm3 == region.population.density_per_mm3
    assert (cfg.depth_lo_um, cfg.depth_hi_um) == region.population.depth_range_um
    assert cfg.neuropil_enabled == (region.neuropil is not None)
    assert cfg.neuropil_amplitude == region.neuropil.amplitude
    assert cfg.vessel_root_radius_um == region.vasculature.layers[0].root_radius_um
    assert cfg.vessel_opacity == region.vasculature.layers[0].opacity

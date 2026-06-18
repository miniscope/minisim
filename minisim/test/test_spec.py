"""Unit tests for the ``minisim`` spec surface.

Covers the model contract (``extra="forbid"``, ``frozen``), the unit
conversions on ``Acquisition``/``Optics``, JSON round-tripping through the
static ``AnyStep`` discriminated union, ``cache_key`` behavior, and the
cross-field validators (hard fails and advisory warnings).
"""

import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

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
    NeuronPopulation,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Sensor,
    Spec,
    SpecWarning,
    Tissue,
    Vasculature,
    Vignette,
)
from minisim.steps import Step


def _minimal_steps():
    """A short, in-order, individually-valid step list for a tiny FOV."""
    return [
        PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
        CellActivity(tau_decay_s=0.4),
        CellOptics(),
        Composite(),
        Sensor(),
    ]


def _tiny_acquisition(**kw):
    """32×32 sensor → 12 µm FOV at the default 0.375 µm pixel size."""
    kw.setdefault("fps", 20.0)
    kw.setdefault("duration_s", 25.0)
    kw.setdefault("image_sensor", ImageSensor(n_px_height=32, n_px_width=32))
    return Acquisition(**kw)


def _valid_spec(**kw):
    kw.setdefault("acquisition", _tiny_acquisition())
    kw.setdefault("steps", _minimal_steps())
    return Spec(**kw)


# --- model contract --------------------------------------------------------


def test_extra_forbidden():
    with pytest.raises(ValidationError):
        Optics(nope=1)


def test_models_are_frozen():
    opt = Optics()
    with pytest.raises(ValidationError):
        opt.na = 0.9


def test_defaults_construct():
    spec = _valid_spec()
    assert spec.seed == 42
    assert isinstance(spec.acquisition.optics, Optics)
    assert isinstance(spec.acquisition.image_sensor, ImageSensor)
    assert isinstance(spec.acquisition.tissue, Tissue)
    assert isinstance(spec.output, Output)


def test_sensor_hardware_lives_on_image_sensor():
    # Hardware moved off the Sensor step onto Acquisition.image_sensor; the step
    # keeps only the (non-hardware) exposure scale.
    assert (
        ImageSensor(read_noise_e=3.0, quantum_efficiency=0.8, bit_depth=12).bit_depth
        == 12
    )
    assert Sensor(photons_per_unit=80.0).photons_per_unit == pytest.approx(80.0)
    with pytest.raises(ValidationError):
        Sensor(read_noise_e=3.0)  # hardware no longer accepted on the step


def test_sensor_photons_per_unit_rejects_nonpositive():
    assert Sensor().photons_per_unit == pytest.approx(100.0)  # numeric default
    with pytest.raises(ValidationError):
        Sensor(photons_per_unit=0.0)
    with pytest.raises(ValidationError):
        Sensor(photons_per_unit=-5.0)


# --- unit conversions ------------------------------------------------------


def test_pixel_size_um():
    # Pixel size is the joint optics×sensor quantity, owned by Acquisition.
    acq = Acquisition(
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(pixel_pitch_um=3.0),
    )
    assert acq.pixel_size_um == pytest.approx(0.375)


def test_n_frames_rounds():
    assert Acquisition(fps=20.0, duration_s=25.0).n_frames == 500
    assert Acquisition(fps=30.0, duration_s=1.05).n_frames == round(31.5)  # 32


def test_fov_um_derived():
    acq = _tiny_acquisition()
    assert acq.fov_um == pytest.approx((12.0, 12.0))


def test_um_to_px_and_s_to_frame():
    acq = _tiny_acquisition()
    assert acq.um_to_px(0.75) == pytest.approx(2.0)
    assert acq.s_to_frame(1.0) == pytest.approx(20.0)


# --- discriminated union + serialization -----------------------------------


def test_json_round_trip_preserves_step_types():
    spec = _valid_spec()
    restored = Spec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert isinstance(restored.steps[0], PlaceNeurons)
    assert isinstance(restored.steps[-1], Sensor)


def test_union_discriminates_on_kind():
    spec = Spec.model_validate(
        {
            "acquisition": _tiny_acquisition().model_dump(),
            "steps": [{"kind": "composite"}, {"kind": "sensor"}],
        }
    )
    assert isinstance(spec.steps[0], Composite)
    assert isinstance(spec.steps[1], Sensor)


# --- cache key -------------------------------------------------------------


def test_cache_key_stable_and_content_sensitive():
    a = _valid_spec()
    b = _valid_spec()
    assert a.cache_key() == b.cache_key()
    assert _valid_spec(seed=43).cache_key() != a.cache_key()


# --- validators: hard fails ------------------------------------------------


def test_duplicate_kind_fails():
    with pytest.raises(ValidationError, match="unique"):
        _valid_spec(steps=[Composite(), Composite()])


def test_soma_larger_than_fov_fails():
    # 20 µm radius → 40 µm diameter, far larger than the 12 µm FOV.
    with pytest.raises(ValidationError, match="FOV"):
        _valid_spec(steps=[PlaceNeurons(soma_radius_um=20.0), *_minimal_steps()[1:]])


def test_oversized_soma_in_any_population_fails():
    # The FOV check is over the *largest* soma across populations: a tiny first
    # layer doesn't excuse a giant second one.
    oversized = PlaceNeurons(
        populations=[
            NeuronPopulation(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
            NeuronPopulation(soma_radius_um=20.0, depth_range_um=(0.0, 0.0)),
        ]
    )
    with pytest.raises(ValidationError, match="FOV"):
        _valid_spec(steps=[oversized, *_minimal_steps()[1:]])


def test_unresolvable_decay_fails():
    acq = _tiny_acquisition(fps=1.0)
    steps = [
        PlaceNeurons(soma_radius_um=3.0),
        CellActivity(tau_decay_s=0.5),
        Composite(),
    ]
    with pytest.raises(ValidationError, match="unresolvable"):
        Spec(acquisition=acq, steps=steps)


def test_steps_are_canonicalized_regardless_of_listing_order():
    # The order steps are listed in carries no meaning: the engine runs them in the
    # canonical pipeline order, so a shuffled list (here render and sensor placed
    # before their producers) is silently normalized rather than rejected. The Scene
    # data-dependencies - cell_activity needs place_neurons, composite needs both -
    # are all satisfied once canonicalized.
    steps = [
        Composite(),
        Sensor(),
        CellActivity(tau_decay_s=0.4),
        PlaceNeurons(soma_radius_um=3.0),
    ]
    spec = _valid_spec(steps=steps)
    assert [s.kind for s in spec.steps] == [
        "place_neurons",
        "cell_activity",
        "composite",
        "sensor",
    ]


def test_listing_order_does_not_affect_identity():
    # Two specs differing only in how their steps were listed canonicalize to the
    # same thing, so they compare and cache equal.
    steps = [
        PlaceNeurons(soma_radius_um=3.0),
        CellActivity(tau_decay_s=0.4),
        Composite(),
    ]
    a = _valid_spec(steps=steps)
    b = _valid_spec(steps=list(reversed(steps)))
    assert a.steps == b.steps
    assert a.cache_key() == b.cache_key()


def test_absent_prerequisite_is_allowed_for_partial_pipelines():
    # A partial pipeline (cells rendered, no cell_activity / motion / sensor) is
    # valid: requires enforces order-when-present, not completeness - so a few
    # stages can be run to make targeted test data. place_neurons precedes render.
    Spec(
        acquisition=_tiny_acquisition(),
        steps=[PlaceNeurons(soma_radius_um=3.0), Composite()],
    )


# --- validators: advisory warnings -----------------------------------------


def test_out_of_order_domains_are_silently_canonicalized():
    # Steps given out of cell->tissue->motion->sensor order are reordered, not warned
    # about: a sensor listed before the render lands after it in the stored spec, and
    # no SpecWarning is raised for the (now meaningless) listing order.
    steps = [PlaceNeurons(soma_radius_um=3.0), Sensor(), Composite()]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = _valid_spec(steps=steps)
    assert not any(isinstance(w.message, SpecWarning) for w in caught)
    assert [s.kind for s in spec.steps] == ["place_neurons", "composite", "sensor"]


def test_order_steps_is_idempotent_and_keeps_unknown_kinds_last():
    from minisim.spec import order_steps

    canonical = [
        PlaceNeurons(),
        CellActivity(),
        Bleaching(),
        CellOptics(),
        Composite(),
        Sensor(),
    ]
    kinds = [s.kind for s in canonical]
    assert [
        s.kind for s in order_steps(canonical)
    ] == kinds  # already-canonical unchanged
    assert [
        s.kind for s in order_steps(list(reversed(canonical)))
    ] == kinds  # shuffled -> canonical


def test_pipeline_order_matches_catalog_and_respects_requires():
    # Drift guard: the canonical order must stay a permutation of the step catalog
    # (so a newly-added kind can't be silently dropped to the end) and a topological
    # extension of every step's declared `requires` (so canonicalizing never produces
    # an order that _check_step_dependencies would reject).
    from minisim.spec import _KIND_RANK, _PIPELINE_ORDER
    from minisim.steps import STEP_FOR_KIND

    assert set(_PIPELINE_ORDER) == set(STEP_FOR_KIND)
    for spec_cls in (CellActivity, CellOptics, Composite, Neuropil, Bleaching):
        for required_kind in spec_cls.requires:
            assert _KIND_RANK[required_kind] < _KIND_RANK[spec_cls().kind]


def test_focal_plane_out_of_range_warns():
    acq = _tiny_acquisition(focal_depth_in_tissue_um=500.0)
    steps = [PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 200.0)), Composite()]
    with pytest.warns(SpecWarning, match="focal"):
        Spec(acquisition=acq, steps=steps)


def test_auto_focal_plane_does_not_warn(recwarn):
    _valid_spec()  # default acquisition → focal_depth_in_tissue_um="auto"
    assert not [w for w in recwarn.list if issubclass(w.category, SpecWarning)]


def test_large_motion_warns():
    steps = [
        PlaceNeurons(soma_radius_um=3.0, depth_range_um=(0.0, 0.0)),
        CellActivity(tau_decay_s=0.4),
        CellOptics(),
        Composite(),
        BrainMotion(max_shift_um=50.0),  # ≫ 5% of the 12 µm FOV
        Sensor(),
    ]
    with pytest.warns(SpecWarning, match="Motion"):
        _valid_spec(steps=steps)


# --- the whole step catalog builds -----------------------------------------


def test_every_step_kind_builds():
    # Every spec in the v1 catalog now returns an executable Step - the full
    # forward pipeline is wired. None falls back to the base
    # NotImplementedError, and each step self-describes its name and domain.
    acq = _tiny_acquisition()
    rng = np.random.default_rng(0)
    specs = [
        PlaceNeurons(),
        CellActivity(),
        CellOptics(),
        Composite(),
        Neuropil(),
        Vasculature(),
        Bleaching(),
        BrainMotion(),
        IlluminationProfile(),
        Vignette(),
        Leakage(),
        Sensor(),
    ]
    for spec in specs:
        step = spec.build(acq, rng)
        assert isinstance(step, Step)
        assert step.name and step.domain


def test_step_registry_matches_spec_catalog():
    # The declarative spec→step table must stay 1:1 with the StepSpec kinds: every
    # concrete spec resolves to a step, and no registry entry is orphaned. Guards
    # the table in steps/__init__ from drifting out of sync with the catalog.
    from minisim.spec import StepSpec
    from minisim.steps import STEP_FOR_KIND

    catalog_kinds = {s.model_fields["kind"].default for s in StepSpec.__subclasses__()}
    assert set(STEP_FOR_KIND) == catalog_kinds
    for step_cls in STEP_FOR_KIND.values():
        assert issubclass(step_cls, Step)


# --- Layer-2 physics helpers ------------------------------------------------
# Diffraction


def test_diffraction_sigma_matches_closed_form():
    opt = Optics(na=0.45, emission_nm=525.0)
    assert opt.diffraction_sigma_um == pytest.approx(0.21 * 0.525 / 0.45)


def test_diffraction_sigma_decreases_with_na():
    assert Optics(na=0.6).diffraction_sigma_um < Optics(na=0.3).diffraction_sigma_um


# Defocus


def test_defocus_zero_at_focal_plane():
    assert Optics(na=0.45).defocus_sigma_um(80.0, 80.0) == 0.0


def test_defocus_symmetric_and_grows_with_distance():
    opt = Optics(na=0.45)
    assert opt.defocus_sigma_um(60.0, 80.0) == pytest.approx(
        opt.defocus_sigma_um(100.0, 80.0)
    )
    assert opt.defocus_sigma_um(120.0, 80.0) > opt.defocus_sigma_um(100.0, 80.0)


def test_defocus_grows_with_na():
    assert Optics(na=0.6).defocus_sigma_um(120.0, 80.0) > Optics(
        na=0.3
    ).defocus_sigma_um(120.0, 80.0)


# Scatter / attenuation


def test_attenuation_monotonic_and_bounded():
    t = Tissue(scatter_mfp_excitation_um=90.0, scatter_mfp_emission_um=110.0)
    assert t.attenuation(0.0) == pytest.approx(1.0)
    assert t.attenuation(200.0) < t.attenuation(50.0) < t.attenuation(0.0)
    assert 0.0 < t.attenuation(500.0) <= 1.0


def test_round_trip_mfp_is_reciprocal_sum_and_steeper_than_either_leg():
    # The excitation + emission legs multiply, so the effective MFP is the
    # reciprocal sum (1/mfp_eff = 1/mfp_ex + 1/mfp_em) - shorter than either leg.
    t = Tissue(scatter_mfp_excitation_um=90.0, scatter_mfp_emission_um=110.0)
    assert t.scatter_mfp_um == pytest.approx(1.0 / (1.0 / 90.0 + 1.0 / 110.0))
    assert t.scatter_mfp_um < min(
        t.scatter_mfp_excitation_um, t.scatter_mfp_emission_um
    )
    # Round trip == the two single-leg exponentials applied in series.
    z = 75.0
    leg_product = math.exp(-z / 90.0) * math.exp(-z / 110.0)
    assert t.attenuation(z) == pytest.approx(leg_product)


def test_scatter_sigma_monotonic_and_zero_at_surface():
    t = Tissue(scatter_blur_per_um=0.02)
    assert t.scatter_sigma_um(0.0) == 0.0
    assert t.scatter_sigma_um(50.0) < t.scatter_sigma_um(200.0)
    assert t.scatter_sigma_um(100.0) == pytest.approx(2.0)


# Combined per-cell optics


def test_cell_optics_in_focus_surface_cell_is_undegraded():
    # z=0, focal=0: no scatter, no defocus → the only light-loss is the NA²
    # collection efficiency; σ = diffraction only.
    acq = _tiny_acquisition()
    sigma_px, brightness = acq.cell_optics(0.0, 0.0)
    assert brightness == pytest.approx(acq.optics.collection_efficiency)
    assert sigma_px == pytest.approx(
        acq.optics.diffraction_sigma_um / acq.pixel_size_um
    )


def test_cell_optics_brightness_scales_with_na_squared():
    # Light collection ∝ NA²: at the surface (no scatter/defocus) the per-cell
    # brightness is exactly the collection efficiency, so doubling NA quadruples it.
    lo = _tiny_acquisition(optics=Optics(magnification=8.0, na=0.2))
    hi = _tiny_acquisition(optics=Optics(magnification=8.0, na=0.4))
    _, b_lo = lo.cell_optics(0.0, 0.0)
    _, b_hi = hi.cell_optics(0.0, 0.0)
    assert b_lo == pytest.approx(0.2**2)
    assert b_hi == pytest.approx(0.4**2)
    assert b_hi / b_lo == pytest.approx(4.0)


def test_cell_optics_defocus_conserves_integrated_intensity():
    # Integrated intensity ∝ σ_tot²·brightness = sigma_px²·brightness (px² is a
    # constant factor). Defocus broadens σ and drops the peak but leaves the
    # integral untouched - only attenuation removes light - so the product is
    # invariant to the focal plane for a fixed depth.
    acq = _tiny_acquisition()
    z = 60.0
    integrals = []
    for focal in (0.0, 30.0, 60.0, 120.0, 200.0):
        sigma_px, brightness = acq.cell_optics(z, focal)
        integrals.append(sigma_px**2 * brightness)
    assert integrals == pytest.approx([integrals[0]] * len(integrals))
    # ...and that conserved integral equals the pure (defocus-free) value: the
    # defocus-free peak (σ_0) times the two flat light-losses, scatter attenuation
    # and the NA² collection efficiency.
    sigma_0_px = (
        math.hypot(acq.optics.diffraction_sigma_um, acq.tissue.scatter_sigma_um(z))
        / acq.pixel_size_um
    )
    assert integrals[0] == pytest.approx(
        sigma_0_px**2 * acq.tissue.attenuation(z) * acq.optics.collection_efficiency
    )


def test_cell_optics_depth_blurs_and_dims():
    # Deeper cell: scatter broadens σ and attenuation drops brightness.
    acq = _tiny_acquisition()
    shallow_sigma, shallow_b = acq.cell_optics(10.0, 10.0)  # in focus
    deep_sigma, deep_b = acq.cell_optics(180.0, 180.0)  # in focus, but deep
    assert deep_sigma > shallow_sigma
    assert deep_b < shallow_b


# Sensor model


def _flat_sensor_acq(**sensor_kw):
    sensor_kw.setdefault("n_px_height", 64)
    sensor_kw.setdefault("n_px_width", 64)
    return ImageSensor(**sensor_kw)


def test_photons_to_counts_are_integer_valued_and_clipped():
    sensor = _flat_sensor_acq(bit_depth=8, read_noise_e=2.0)
    rng = np.random.default_rng(0)
    photons = np.full((64, 64), 50.0)
    counts = sensor.photons_to_counts(photons, rng)
    assert np.all(counts == np.floor(counts))  # integer-valued
    assert counts.min() >= 0.0
    assert counts.max() <= 2**8 - 1
    # Saturation: a huge photon flux clips to the ADC ceiling.
    saturated = sensor.photons_to_counts(np.full((64, 64), 1e6), rng)
    assert np.all(saturated == 2**8 - 1)


def test_photons_to_counts_poisson_mean():
    # With 12-bit headroom (no clipping), mean counts ≈ photons·QE·gain.
    sensor = _flat_sensor_acq(
        quantum_efficiency=0.7, gain_adu_per_e=1.0, read_noise_e=2.0, bit_depth=12
    )
    rng = np.random.default_rng(1)
    photons = np.full((256, 256), 100.0)
    counts = sensor.photons_to_counts(photons, rng)
    assert counts.mean() == pytest.approx(100.0 * 0.7, abs=1.0)


def test_photons_to_counts_read_noise_adds_variance():
    # Shot noise alone vs shot + read noise: the latter has larger variance.
    rng = np.random.default_rng(2)
    photons = np.full((256, 256), 100.0)
    quiet = _flat_sensor_acq(read_noise_e=0.0, bit_depth=12).photons_to_counts(
        photons, rng
    )
    noisy = _flat_sensor_acq(read_noise_e=10.0, bit_depth=12).photons_to_counts(
        photons, rng
    )
    assert noisy.var() > quiet.var()


def test_photons_to_counts_gain_scales_counts():
    rng = np.random.default_rng(3)
    photons = np.full((256, 256), 100.0)
    low = _flat_sensor_acq(
        gain_adu_per_e=1.0, read_noise_e=2.0, bit_depth=16
    ).photons_to_counts(photons, rng)
    high = _flat_sensor_acq(
        gain_adu_per_e=4.0, read_noise_e=2.0, bit_depth=16
    ).photons_to_counts(photons, rng)
    assert high.mean() == pytest.approx(4.0 * low.mean(), rel=0.05)


def test_spec_internals_are_submodule_only():
    # StepSpec (the closed-catalog base) and order_steps (Spec applies it itself)
    # are implementation details, reachable via minisim.spec but off the top-level
    # surface. AnyStep stays public: it is the type annotating a step list.
    import minisim

    for name in ("StepSpec", "order_steps"):
        assert not hasattr(minisim, name), (
            f"{name} leaked back onto the top-level surface"
        )
        assert name not in minisim.__all__
    from minisim.spec import (  # noqa: F401  (reachable via submodule)
        StepSpec,
        order_steps,
    )

    assert "AnyStep" in minisim.__all__ and hasattr(minisim, "AnyStep")

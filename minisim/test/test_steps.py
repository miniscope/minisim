"""Unit tests for the executable step chain.

Covers the steps that turn a blank ``Scene`` into a digitized recording - the
minimal chain ``place_neurons`` → ``cell_activity`` → ``composite`` → ``sensor``,
the ``optics`` degradation and planted/observed split, and the field
effects ``neuropil`` / ``bleaching`` / ``vignette`` / ``leakage`` plus the
absorbing ``vasculature`` mask. Each step is exercised in isolation
against a hand-built scene (the primary test substrate) as well as in the
end-to-end chain, with ``brain_motion`` covered in its own section below.
"""

import numpy as np
import pytest
import xarray as xr
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
    PlaceNeurons,
    Sensor,
    Spec,
    Vasculature,
    VesselLayer,
    Vignette,
)
from minisim.footprint import Footprint
from minisim.scene import Cell, Scene
from minisim.steps import (
    BleachingStep,
    CellOpticsStep,
    CompositeStep,
    IlluminationProfileStep,
    LeakageStep,
    NeuropilStep,
    SensorStep,
    VesselGrowth,
    VignetteStep,
    bleaching_floor,
    bleaching_pool,
    bounded_random_walk,
    calcium_kernel,
    combined_falloff_field,
    dark_recovery,
    degrade_footprint,
    grow_vessel_tree,
    kernel_timing,
    murray_children,
    neuron_footprint,
    ou_process,
    physical_brain_motion,
    population_envelope,
    radial_falloff,
    rasterize_vessels,
    resolve_exposure,
    resolve_focal_plane,
    sample_neurons,
    shift_and_crop,
    spike_activity_params,
    tau_from_kernel_timing,
    vasculature_mask_field,
    vessels_to_mask,
)


def _acq(n_px=50, fps=20.0, duration_s=2.0, bit_depth=8, **kw):
    """A small scene with a clean 1.0 µm/px scale (pitch 8 µm / magnification 8).

    50 px → a 50 µm FOV (area 2.5e-3 mm²), so an (unphysically high, on purpose)
    density makes the cell count a clean integer for assertions.
    """
    kw.setdefault("optics", Optics(magnification=8.0))
    kw.setdefault(
        "image_sensor",
        ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=bit_depth
        ),
    )
    return Acquisition(fps=fps, duration_s=duration_s, **kw)


# --- place_neurons ----------------------------------------------------------


def test_place_neurons_count_from_density_and_fov():
    # 50 µm FOV → 2.5e-3 mm²; depth (0,0) floors to one soma diameter (2·5 = 10 µm
    # → 1e-2 mm), so volume = 2.5e-5 mm³ and 200000/mm³ → exactly 5 cells.
    acq = _acq()
    step = PlaceNeurons(
        density_per_mm3=200000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(0))
    scene = Scene.zeros(acq)
    step(scene)
    assert len(scene.cells) == 5


def test_place_neurons_centers_in_bounds():
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    step = PlaceNeurons(density_per_mm3=28571.0, depth_range_um=(10.0, 80.0)).build(
        acq, np.random.default_rng(1)
    )
    scene = Scene.zeros(acq)
    step(scene)
    for cell in scene.cells:
        z, y, x = cell.center_um
        assert 10.0 <= z <= 80.0
        # optical-center frame: lateral coords span [-fov/2, +fov/2] about the axis.
        assert -fov_h / 2.0 <= y <= fov_h / 2.0
        assert -fov_w / 2.0 <= x <= fov_w / 2.0


def test_place_neurons_footprint_is_peak_normalized():
    acq = _acq()
    step = PlaceNeurons(
        density_per_mm3=200000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(2))
    scene = Scene.zeros(acq)
    step(scene)
    for cell in scene.cells:
        fp = cell.footprint_planted.to_dense()
        assert fp.shape == (50, 50)
        assert fp.max() == pytest.approx(1.0)
        assert fp.min() >= 0.0
        assert (fp > 0).sum() > 0


def test_irregularity_zero_is_a_clean_binary_disk():
    fp = neuron_footprint(
        (40, 40),
        (20.0, 20.0),
        radius_px=8.0,
        irregularity=0.0,
        rng=np.random.default_rng(3),
    )
    # A clean disk is binary (0 outside, 1 inside) and roughly the disk's area.
    assert set(np.unique(fp)).issubset({0.0, 1.0})
    assert fp.sum() == pytest.approx(np.pi * 8.0**2, rel=0.1)


def _placed_scene(acq, seed=4):
    scene = Scene.zeros(acq)
    PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)).build(
        acq, np.random.default_rng(seed)
    )(scene)
    return scene


def test_brightness_gain_is_per_cell_and_scales_the_whole_trace():
    # cell_activity draws a per-cell expression gain (mean 1, spread brightness_cv)
    # and scales each cell's *whole* trace by it -- baseline included -- so a bright
    # cell is brighter everywhere.
    acq = _acq()
    scene = _placed_scene(acq)
    spec = CellActivity(brightness_cv=0.5)
    spec.build(acq, np.random.default_rng(4))(scene)
    amps = np.array([c.amplitude for c in scene.cells])
    assert (amps > 0).all() and amps.std() > 0  # a real cell-to-cell spread
    # the convolved calcium is non-negative, so the trace never dips below the
    # cell's scaled baseline gain*f0 -- the gain is applied to the whole trace.
    for cell in scene.cells:
        assert cell.trace.min() >= cell.amplitude * spec.f0 - 1e-6


def test_brightness_cv_zero_makes_every_cell_equally_bright():
    acq = _acq()
    scene = _placed_scene(acq)
    CellActivity(brightness_cv=0.0).build(acq, np.random.default_rng(4))(scene)
    amps = np.array([c.amplitude for c in scene.cells])
    np.testing.assert_array_equal(amps, np.ones_like(amps))


def test_min_distance_is_respected():
    acq = _acq()
    step = PlaceNeurons(
        density_per_mm3=142857.0, depth_range_um=(0.0, 0.0), min_distance_um=8.0
    ).build(acq, np.random.default_rng(5))
    scene = Scene.zeros(acq)
    step(scene)
    centers = np.array([c.center_um for c in scene.cells])
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            assert np.linalg.norm(centers[i] - centers[j]) >= 8.0


def test_place_neurons_is_reproducible():
    acq = _acq()
    spec = PlaceNeurons(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0))
    scenes = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        spec.build(acq, np.random.default_rng(7))(scene)
        scenes.append([c.center_um for c in scene.cells])
    assert scenes[0] == scenes[1]


def test_sample_neurons_matches_the_full_step():
    # The extracted distribution sampler must reproduce, draw-for-draw, the
    # centers the fused PlaceNeuronsStep would have placed - same spec, same FOV,
    # same seeded rng. (This is the contract the notebook relies on: visualizing
    # sample_neurons() shows the *real* placement, not an approximation.)
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    spec = PlaceNeurons(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0))
    centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(7))

    scene = Scene.zeros(acq)
    spec.build(acq, np.random.default_rng(7))(scene)
    step_centers = [c.center_um for c in scene.cells]

    assert centers == step_centers


def test_sample_neurons_count_from_density_and_fov():
    # No footprints stamped, but the volumetric count rule still gives a clean 5:
    # 2.5e-3 mm² × floor(2·7 = 14 µm) = 3.5e-5 mm³, × 142857/mm³ = 5.
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    spec = PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0))
    centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(0))
    assert len(centers) == 5


def test_flat_spec_resolves_to_one_population_identically():
    # A flat PlaceNeurons is its own single population: resolving it must reproduce
    # the same placement as the equivalent explicit one-population spec, draw for
    # draw (so the flat form stays a faithful shorthand, not a separate code path).
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    flat = PlaceNeurons(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0), soma_radius_um=6.0)
    listed = PlaceNeurons(
        populations=[
            NeuronPopulation(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0), soma_radius_um=6.0)
        ]
    )
    a = sample_neurons(flat, fov_h, fov_w, np.random.default_rng(7))
    b = sample_neurons(listed, fov_h, fov_w, np.random.default_rng(7))
    assert a == b


def test_multiple_populations_concatenate_and_keep_per_population_shape():
    # Two layers placed together: the cell count is the sum of the per-population
    # volumetric counts, and each cell carries its own population's morphology
    # (only the cytosolic layer grows dendrites, so it lights more pixels per soma).
    acq = _acq()
    spec = PlaceNeurons(
        populations=[
            NeuronPopulation(  # thin soma-only band: 50 µm FOV, depth (0,0) → 5 cells
                morphology="soma", density_per_mm3=200000.0, soma_radius_um=5.0,
                depth_range_um=(0.0, 0.0),
            ),
            NeuronPopulation(  # planar cytosolic layer at depth 30 → another 5
                morphology="cytosolic", density_per_mm3=200000.0, soma_radius_um=5.0,
                depth_range_um=(30.0, 30.0),
            ),
        ]
    )
    scene = Scene.zeros(acq)
    spec.build(acq, np.random.default_rng(0))(scene)
    assert len(scene.cells) == 10
    shallow = [c for c in scene.cells if c.center_um[0] == 0.0]
    deep = [c for c in scene.cells if c.center_um[0] == 30.0]
    assert len(shallow) == 5 and len(deep) == 5


def test_sample_neurons_matches_the_full_step_multi_population():
    # The sample_neurons()==step contract must hold across populations too: the step
    # samples every population *before* it stamps (footprint stamping consumes rng),
    # so the distribution sampler reproduces the placement draw-for-draw.
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    spec = PlaceNeurons(
        populations=[
            NeuronPopulation(density_per_mm3=40000.0, depth_range_um=(0.0, 50.0)),
            NeuronPopulation(
                morphology="cytosolic", density_per_mm3=30000.0, depth_range_um=(80.0, 150.0)
            ),
        ]
    )
    centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(3))
    scene = Scene.zeros(acq)
    spec.build(acq, np.random.default_rng(3))(scene)
    assert centers == [c.center_um for c in scene.cells]


def test_explicit_positions_are_placed_verbatim():
    # positions_um places exactly those (z, y, x) centers, consuming no rng and
    # ignoring the distribution fields - the same coordinates come back as centers.
    acq = _acq()
    fov_h, fov_w = acq.fov_um
    pos = [(20.0, 10.0, 10.0), (60.0, 25.0, 40.0), (5.0, 49.0, 1.0)]
    spec = PlaceNeurons(positions_um=pos, density_per_mm3=999999.0)  # density ignored
    centers = sample_neurons(spec, fov_h, fov_w, np.random.default_rng(0))
    assert centers == pos
    scene = Scene.zeros(acq)
    spec.build(acq, np.random.default_rng(0))(scene)
    assert [c.center_um for c in scene.cells] == pos


def test_explicit_and_sampled_populations_mix():
    # One density-sampled layer plus one explicit-position layer in a single spec:
    # the explicit cells land verbatim at the tail, after the sampled ones.
    acq = _acq()
    explicit = [(40.0, 12.0, 12.0), (40.0, 24.0, 24.0)]
    spec = PlaceNeurons(
        populations=[
            NeuronPopulation(density_per_mm3=200000.0, soma_radius_um=5.0, depth_range_um=(0.0, 0.0)),
            NeuronPopulation(positions_um=explicit, morphology="soma"),
        ]
    )
    scene = Scene.zeros(acq)
    spec.build(acq, np.random.default_rng(0))(scene)
    assert len(scene.cells) == 5 + 2
    assert [c.center_um for c in scene.cells][-2:] == explicit


def test_populations_and_step_level_fields_cannot_mix():
    # Setting populations *and* a step-level population field is ambiguous (which
    # wins?), so it is rejected at construction with a pointed message.
    with pytest.raises(ValueError, match="both populations and step-level"):
        PlaceNeurons(soma_radius_um=8.0, populations=[NeuronPopulation()])


def test_empty_populations_list_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        PlaceNeurons(populations=[])


def test_populations_spec_survives_serialization_round_trip():
    # A populations-based step must round-trip through model_dump()/model_validate()
    # unchanged - a dump marks every field "set", so the both-set guard must compare
    # against defaults, not model_fields_set, or sweep()/cache reload would reject it.
    step = PlaceNeurons(populations=[NeuronPopulation(soma_radius_um=5.0, positions_um=[(1.0, 2.0, 3.0)])])
    restored = PlaceNeurons.model_validate(step.model_dump())
    assert restored.populations == step.populations


def test_cytosolic_morphology_adds_dendrites_beyond_soma():
    # Same seed, same soma: cytosolic lights strictly more pixels (the dendrites)
    # and reaches farther from the center than the soma-only variant, while the
    # peak stays at the soma (1.0) and the dendrites are graded dimmer.
    shape, center, radius = (80, 80), (40.0, 40.0), 6.0
    soma = neuron_footprint(
        shape, center, radius, irregularity=0.0, rng=np.random.default_rng(0)
    )
    cyto = neuron_footprint(
        shape,
        center,
        radius,
        irregularity=0.0,
        rng=np.random.default_rng(0),
        morphology="cytosolic",
        dendrite_length_px=18.0,
        dendrite_width_px=2.5,
    )
    assert cyto.max() == pytest.approx(1.0)  # still peak-normalized at the soma
    assert (cyto > 0).sum() > (soma > 0).sum()  # dendrites light extra pixels
    yy, xx = np.indices(shape)
    rr = np.hypot(yy - center[0], xx - center[1])
    assert rr[cyto > 0].max() > rr[soma > 0].max() + 2.0  # reach beyond the soma
    assert 0.0 < cyto[cyto > 0].min() < 1.0  # dendrites graded dimmer than soma


def test_cytosolic_dendrite_count_is_random_per_cell():
    # The count is drawn per cell (not a spec input), so different RNG states give
    # different dendrite layouts - what makes cells look distinct.
    shape, center, radius = (96, 96), (48.0, 48.0), 6.0
    fps = [
        neuron_footprint(shape, center, radius, irregularity=0.0,
                         rng=np.random.default_rng(s), morphology="cytosolic",
                         dendrite_length_px=18.0, dendrite_width_px=2.5)
        for s in range(8)
    ]
    counts = {int((fp > 0).sum()) for fp in fps}
    assert len(counts) >= 5  # cells differ from one another (varied arbors)
    # Same seed reproduces the same arbor exactly (determinism for streaming/cache).
    again = neuron_footprint(shape, center, radius, irregularity=0.0,
                             rng=np.random.default_rng(3), morphology="cytosolic",
                             dendrite_length_px=18.0, dendrite_width_px=2.5)
    np.testing.assert_array_equal(again, fps[3])


def test_soma_morphology_is_unchanged_by_dendrite_params():
    # The default "soma" variant ignores the dendrite params and matches a
    # soma-only footprint bit-for-bit (cytosolic only *adds* after the soma, so
    # the soma's RNG draw is never perturbed).
    shape, center, radius = (60, 60), (30.0, 30.0), 5.0
    a = neuron_footprint(
        shape, center, radius, irregularity=0.3, rng=np.random.default_rng(11)
    )
    b = neuron_footprint(
        shape,
        center,
        radius,
        irregularity=0.3,
        rng=np.random.default_rng(11),
        morphology="soma",
        dendrite_length_px=18.0,
        dendrite_width_px=2.5,
    )
    np.testing.assert_array_equal(a, b)


# --- cell_activity ---------------------------------------------------------


def test_calcium_kernel_shape():
    k = calcium_kernel(tau_rise_s=0.05, tau_decay_s=0.5, fps=20.0)
    assert k.max() == pytest.approx(1.0)
    assert (k >= 0).all()
    assert k[0] == pytest.approx(0.0, abs=1e-9)  # k(0) = 1 - 1 = 0
    assert k[-1] < k.max()  # has decayed by the end of the window


def test_calcium_kernel_requires_rise_faster_than_decay():
    with pytest.raises(ValueError, match="tau_rise_s"):
        calcium_kernel(tau_rise_s=0.5, tau_decay_s=0.5, fps=20.0)


@pytest.mark.parametrize("tr,td", [(0.05, 0.5), (0.02, 0.3), (0.1, 0.8), (0.08, 0.4)])
def test_kernel_timing_round_trips_with_tau(tr, td):
    # (tau_rise, tau_decay) -> (t_peak, fwhm) -> (tau_rise, tau_decay) is identity.
    t_peak, fwhm = kernel_timing(tr, td)
    tr2, td2 = tau_from_kernel_timing(t_peak, fwhm)
    assert tr2 == pytest.approx(tr, rel=1e-3)
    assert td2 == pytest.approx(td, rel=1e-3)


def test_kernel_timing_matches_a_finely_sampled_kernel():
    # The analytic peak time and FWHM must match what a densely sampled kernel shows.
    tr, td, fps = 0.05, 0.5, 2000.0
    t_peak, fwhm = kernel_timing(tr, td)
    k = calcium_kernel(tr, td, fps)
    t = np.arange(len(k)) / fps
    assert t[k.argmax()] == pytest.approx(t_peak, abs=1.0 / fps)
    above = t[k >= 0.5]  # kernel is peak-normalized to 1, so half max = 0.5
    assert (above.max() - above.min()) == pytest.approx(fwhm, abs=2.0 / fps)


def test_tau_from_kernel_timing_clamps_impossible_ratio():
    # t_peak/fwhm above the kernel's achievable range clamps to the alpha-function
    # limit (tau_rise -> tau_decay) instead of failing.
    tr, td = tau_from_kernel_timing(t_peak_s=0.3, fwhm_s=0.35)
    assert tr == pytest.approx(td, rel=1e-3)


def test_spike_activity_params_hit_calab_levels():
    # activity 0/0.5/1 reproduce CaLab's sparse/moderate/dense SPIKE_ACTIVITY_LEVELS.
    assert spike_activity_params(0.0) == pytest.approx((0.002, 0.4, 90.0, 0.3))
    assert spike_activity_params(0.5) == pytest.approx((0.005, 0.3, 150.0, 0.6))
    assert spike_activity_params(1.0) == pytest.approx((0.01, 0.2, 210.0, 1.5))


def test_spike_activity_params_are_monotonic_in_density():
    # Denser activity: bursts start more often (p_q2a up), last longer (p_a2q down),
    # fire harder (active_rate up), and the background rate rises.
    sparse, dense = spike_activity_params(0.2), spike_activity_params(0.8)
    assert dense[0] > sparse[0]   # p_quiescent_to_active
    assert dense[1] < sparse[1]   # p_active_to_quiescent
    assert dense[2] > sparse[2]   # active_rate_hz
    assert dense[3] > sparse[3]   # quiescent_rate_hz
    assert spike_activity_params(2.0) == spike_activity_params(1.0)  # clamps


def test_cell_activity_sets_trace_and_spikes():
    acq = _acq(duration_s=5.0)
    scene = Scene.zeros(acq)
    PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)).build(
        acq, np.random.default_rng(9)
    )(scene)
    CellActivity(active_rate_hz=5.0, tau_decay_s=0.4).build(
        acq, np.random.default_rng(9)
    )(scene)
    for cell in scene.cells:
        assert cell.trace.shape == (acq.n_frames,)
        assert cell.spikes.shape == (acq.n_frames,)
        assert (cell.spikes >= 0).all()
        np.testing.assert_array_equal(
            cell.spikes, np.round(cell.spikes)
        )  # integer counts


def test_noise_free_trace_never_dips_below_baseline():
    # With trace_noise=0 the trace is f0 + (nonneg amplitudes) ⊛ (nonneg kernel),
    # so it can never fall below f0.
    acq = _acq(duration_s=5.0)
    scene = Scene.zeros(acq)
    scene.cells.append(Cell(center_um=(0.0, 25.0, 25.0)))
    # brightness_cv=0 keeps the per-cell gain at 1, so the baseline stays exactly f0
    # and this isolates the kernel/amplitude non-negativity invariant.
    CellActivity(f0=1.0, trace_noise=0.0, active_rate_hz=5.0, brightness_cv=0.0).build(
        acq, np.random.default_rng(10)
    )(scene)
    assert scene.cells[0].trace.min() >= 1.0 - 1e-9


def test_cell_activity_is_reproducible():
    acq = _acq(duration_s=5.0)
    traces = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        scene.cells.append(Cell(center_um=(0.0, 25.0, 25.0)))
        CellActivity().build(acq, np.random.default_rng(11))(scene)
        traces.append(scene.cells[0].trace)
    np.testing.assert_array_equal(traces[0], traces[1])


def test_cell_activity_opens_at_steady_state_no_cold_ramp():
    # The burn-in lead-in means a trace starts with a realistic decaying history, so
    # the population-mean opens at its stationary level instead of ramping up from
    # baseline over a few tau_decay (a raw history-free convolution would start at f0).
    acq = _acq(n_px=80, duration_s=15.0)
    scene = Scene.zeros(acq)
    PlaceNeurons(density_per_mm3=6e5, depth_range_um=(0.0, 0.0), soma_radius_um=4.0).build(
        acq, np.random.default_rng(3)
    )(scene)
    CellActivity(active_rate_hz=5.0, tau_decay_s=0.5, brightness_cv=0.0).build(
        acq, np.random.default_rng(3)
    )(scene)
    m = np.stack([c.trace for c in scene.cells]).mean(axis=0)  # population mean trace
    early = m[: int(0.5 * acq.fps)].mean()       # first 0.5 s
    steady = m[int(5.0 * acq.fps):].mean()       # well past any ramp
    assert steady > 1.3                          # a real calcium baseline exists to ramp to
    assert 0.85 * steady < early < 1.15 * steady  # ...yet the opening is already there


# --- render ----------------------------------------------------------------


def test_render_is_the_footprint_trace_outer_sum():
    acq = _acq(n_px=8, duration_s=0.15)  # 3 frames
    scene = Scene.zeros(acq)
    fp1 = np.zeros((8, 8))
    fp1[2, 3] = 1.0
    fp2 = np.zeros((8, 8))
    fp2[5, 6] = 1.0
    tr1 = np.array([1.0, 2.0, 3.0])
    tr2 = np.array([4.0, 5.0, 6.0])
    scene.cells += [
        Cell(center_um=(0.0, 0.0, 0.0), footprint_planted=Footprint.from_dense(fp1), trace=tr1),
        Cell(center_um=(0.0, 0.0, 0.0), footprint_planted=Footprint.from_dense(fp2), trace=tr2),
    ]
    CompositeStep(Composite(), acq, np.random.default_rng(0))(scene)
    np.testing.assert_allclose(scene.movie.values[:, 2, 3], tr1)
    np.testing.assert_allclose(scene.movie.values[:, 5, 6], tr2)
    # Pixels with no cell stay zero.
    assert scene.movie.values[:, 0, 0].sum() == 0.0


def test_render_regenerates_observed_footprint_from_optics_scalars():
    # Once optics has set the per-cell blur scalars, render composites the observed
    # (regenerated) footprint -- gain·(planted ⊛ Gaussian(sigma)) -- not the raw
    # planted dot. The observed footprint is not stored; render rebuilds it.
    acq = _acq(n_px=8, duration_s=0.05)  # 1 frame
    scene = Scene.zeros(acq)
    planted = np.zeros((8, 8))
    planted[4, 4] = 1.0
    scene.cells.append(
        Cell(
            center_um=(0.0, 0.0, 0.0),
            footprint_planted=Footprint.from_dense(planted),
            observed_sigma_px=1.0,
            observed_gain=0.5,
            trace=np.array([2.0]),
        )
    )
    CompositeStep(Composite(), acq, np.random.default_rng(0))(scene)
    expected = degrade_footprint(
        Footprint.from_dense(planted), 1.0, 0.5
    ).to_dense(dtype=float) * 2.0
    np.testing.assert_allclose(scene.movie.values[0], expected, rtol=1e-5)
    assert scene.movie.values[0, 4, 4] < 2.0 * 0.5  # blurred: peak below gain·trace


def test_render_empty_scene_leaves_movie_untouched():
    acq = _acq(n_px=8, duration_s=0.15)
    scene = Scene.zeros(acq)
    CompositeStep(Composite(), acq, np.random.default_rng(0))(scene)
    assert (scene.movie.values == 0.0).all()


# --- sensor ----------------------------------------------------------------


def test_sensor_counts_are_integer_and_within_adc_range():
    acq = _acq(n_px=16, duration_s=0.5, bit_depth=8)
    scene = Scene.zeros(acq)
    scene.movie.values[:] = 1.5  # uniform positive intensity
    SensorStep(Sensor(photons_per_unit=100.0), acq, np.random.default_rng(0))(scene)
    counts = scene.movie.values
    np.testing.assert_array_equal(counts, np.round(counts))  # integer-valued
    assert counts.min() >= 0.0
    assert counts.max() <= 255.0  # 2^8 - 1


def test_sensor_mean_counts_increase_with_exposure():
    acq = _acq(n_px=16, duration_s=0.5, bit_depth=12)  # headroom so we don't saturate
    means = []
    for ppu in (20.0, 80.0):
        scene = Scene.zeros(acq)
        scene.movie.values[:] = 1.0
        SensorStep(Sensor(photons_per_unit=ppu), acq, np.random.default_rng(0))(scene)
        means.append(scene.movie.values.mean())
    assert means[1] > means[0]


def test_sensor_is_reproducible():
    acq = _acq(n_px=16, duration_s=0.5)
    outs = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        scene.movie.values[:] = 1.0
        SensorStep(Sensor(), acq, np.random.default_rng(3))(scene)
        outs.append(scene.movie.values.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


# --- auto exposure --------------------------------------------------------


def _cell_with_trace_peak(acq, peak, y_um=0.0, x_um=0.0):
    """A cell with a unit-peak (optics-free) footprint and a trace peaking at ``peak``."""
    shape = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
    fp = Footprint.from_dense(
        neuron_footprint(
            shape, acq.um_to_index(y_um, x_um, shape), acq.um_to_px(4.0), 0.0,
            np.random.default_rng(0),
        )
    )
    trace = np.ones(acq.n_frames)
    trace[acq.n_frames // 2] = peak
    return Cell(center_um=(0.0, y_um, x_um), footprint_planted=fp, trace=trace)


def test_resolve_exposure_passes_numeric_through():
    acq = _acq(n_px=16, duration_s=0.5)
    cell = _cell_with_trace_peak(acq, peak=5.0)
    assert resolve_exposure([cell], acq, Sensor(photons_per_unit=77.0), None) == 77.0


def test_resolve_exposure_auto_lands_brightest_peak_near_target():
    acq = _acq(n_px=16, duration_s=0.5, bit_depth=8)
    cell = _cell_with_trace_peak(acq, peak=5.0)  # observed footprint peak 1.0, trace peak 5.0
    ppu = resolve_exposure([cell], acq, Sensor(photons_per_unit="auto"), None)
    sensor = acq.image_sensor
    # The brightest *mean* pixel = footprint_peak(1) * trace_peak(5) * ppu * QE * gain.
    mean_peak_counts = 1.0 * 5.0 * ppu * sensor.quantum_efficiency * sensor.gain_adu_per_e
    full_scale = 2 ** sensor.bit_depth - 1
    assert mean_peak_counts == pytest.approx(0.85 * full_scale, rel=1e-6)


def test_resolve_exposure_auto_scales_with_the_brightest_cell():
    # Two cells of different peak brightness: exposure is set by the brighter one, so
    # doubling the bright cell's peak halves the chosen exposure.
    acq = _acq(n_px=16, duration_s=0.5)
    dim = _cell_with_trace_peak(acq, peak=2.0, y_um=-10.0)
    bright = _cell_with_trace_peak(acq, peak=8.0, y_um=10.0)
    brighter = _cell_with_trace_peak(acq, peak=16.0, y_um=10.0)
    ppu_a = resolve_exposure([dim, bright], acq, Sensor(photons_per_unit="auto"), None)
    ppu_b = resolve_exposure([dim, brighter], acq, Sensor(photons_per_unit="auto"), None)
    assert ppu_b == pytest.approx(ppu_a / 2.0, rel=1e-6)


def test_resolve_exposure_auto_falls_back_without_active_cells():
    acq = _acq(n_px=16, duration_s=0.5)
    # No activity-bearing cells (no traces) -> the documented fallback exposure.
    from minisim.steps.sensor import _EXPOSURE_FALLBACK_PHOTONS

    assert resolve_exposure([], acq, Sensor(photons_per_unit="auto"), None) == _EXPOSURE_FALLBACK_PHOTONS


def test_auto_exposure_end_to_end_is_bright_but_not_saturating():
    # The whole point for fixtures: "auto" yields bright, clear dynamics that do not
    # saturate, with the resolved exposure recorded in ground truth. Explicit cell
    # positions (shallow, in focus) keep the placement deterministic.
    acq = _acq(n_px=32, duration_s=1.0, bit_depth=8)
    population = NeuronPopulation(positions_um=[(10.0, -15.0, -15.0), (10.0, 15.0, 15.0)])
    spec = Spec(
        acquisition=acq,
        seed=1,
        steps=[PlaceNeurons(populations=[population]),
               CellActivity(p_quiescent_to_active=0.05),
               CellOptics(), Composite(), Sensor(photons_per_unit="auto")],
    )
    from minisim import simulate

    rec = simulate(spec)
    full_scale = 2 ** acq.image_sensor.bit_depth - 1
    assert rec.ground_truth.n_units == 2
    assert rec.ground_truth.exposure_photons_per_unit is not None
    assert rec.observed.max() <= full_scale  # never exceeds the ADC ceiling
    # Not *meaningfully* saturated: "auto" targets the mean peak below the rail, so at
    # most a negligible noise-tail fraction can touch it (no flat-topped dynamics).
    assert float((rec.observed >= full_scale).mean()) < 0.005
    assert rec.observed.max() >= 0.8 * full_scale  # but is genuinely bright


# --- optics -----------------------------------------------------------


def _cell_with_footprint(acq, z, radius_um=4.0):
    """A single centered cell carrying a clean planted disk at depth ``z``."""
    h, w = acq.image_sensor.n_px_height, acq.image_sensor.n_px_width
    fp = Footprint.from_dense(
        neuron_footprint(
            (h, w), (h / 2, w / 2), acq.um_to_px(radius_um), 0.0, np.random.default_rng(0)
        )
    )
    y_um, x_um = (h / 2) * acq.pixel_size_um, (w / 2) * acq.pixel_size_um
    return Cell(center_um=(z, y_um, x_um), footprint_planted=fp)


def test_degrade_footprint_blur_conserves_sum_and_drops_peak():
    fp = Footprint.from_dense(
        neuron_footprint(
            (64, 64),
            (32.0, 32.0),
            radius_px=6.0,
            irregularity=0.0,
            rng=np.random.default_rng(0),
        )
    )
    out = degrade_footprint(fp, sigma_px=2.0, gain=1.0)
    assert out.patch.sum() == pytest.approx(
        fp.patch.sum(), rel=1e-3
    )  # convolution conserves integral
    assert out.patch.max() < fp.patch.max()  # ...but spreads light, so the peak drops


def test_degrade_footprint_gain_scales_integral():
    fp = Footprint.from_dense(
        neuron_footprint(
            (64, 64),
            (32.0, 32.0),
            radius_px=6.0,
            irregularity=0.0,
            rng=np.random.default_rng(0),
        )
    )
    full = degrade_footprint(fp, sigma_px=2.0, gain=1.0)
    half = degrade_footprint(fp, sigma_px=2.0, gain=0.5)
    assert half.patch.sum() == pytest.approx(0.5 * full.patch.sum())


def test_resolve_focal_plane_auto_is_median_and_numeric_passes_through():
    acq = _acq()  # focal_depth_in_tissue_um defaults to "auto"
    cells = [_cell_with_footprint(acq, z) for z in (0.0, 50.0, 100.0, 150.0, 200.0)]
    assert resolve_focal_plane(cells, acq.focal_depth_in_tissue_um) == 100.0
    assert resolve_focal_plane([], acq.focal_depth_in_tissue_um) == 0.0  # empty -> surface
    numeric = _acq(focal_depth_in_tissue_um=42.0)
    assert resolve_focal_plane(cells, numeric.focal_depth_in_tissue_um) == 42.0


def test_resolve_focal_plane_auto_accounts_for_field_curvature():
    # All cells at the same depth but spread laterally. Defocus is linear in the
    # effective depth z + shift(r), so the min-total-defocus focus is the MEDIAN
    # effective depth. With curvature, off-axis cells read deeper, so auto sits
    # deeper than the plain median z -- and falls back to median z without optics.
    acq = _acq(n_px=200, optics=Optics(magnification=8.0, field_curvature_radius_um=600.0))
    radii = (0.0, 20.0, 40.0, 60.0, 80.0)
    # Optical axis is the frame origin (0, 0), so field radius is just |x| here.
    cells = [Cell(center_um=(100.0, 0.0, r)) for r in radii]
    expected = float(np.median([100.0 + acq.optics.focal_curvature_shift_um(r) for r in radii]))

    assert resolve_focal_plane(cells, "auto") == 100.0  # no optics -> plain median z
    curv = resolve_focal_plane(cells, "auto", acq.optics)
    assert curv == pytest.approx(expected)
    assert curv > 100.0  # curvature pulls the focus deeper to recover off-axis cells
    flat = Optics(magnification=8.0)  # field_curvature_radius_um=None
    assert resolve_focal_plane(cells, "auto", flat) == 100.0  # flat field -> median z


def _scored_cell(z, y_um=25.0, x_um=25.0, lo=0.0, hi=100.0):
    """A bare cell at depth ``z`` carrying a 2-frame trace (baseline ``lo``, peak ``hi``)."""
    return Cell(center_um=(z, y_um, x_um), trace=np.array([lo, hi]))


def test_resolve_focal_plane_auto_maximizes_detectable_yield_not_median():
    # A tight shallow cluster (5 cells in 4 µm) plus 5 sparse deeper cells. The
    # median depth sits in the empty gap (~42 µm), but the focus that recovers the
    # MOST cells is the one parked on the cluster -- yield, not the median.
    acq = _acq(focal_depth_in_tissue_um="auto")  # na 0.45 -> DOF ~3.4 µm
    cluster = [_scored_cell(z) for z in (20.0, 21.0, 22.0, 23.0, 24.0)]
    sparse = [_scored_cell(z) for z in (60.0, 80.0, 100.0, 120.0, 140.0)]
    cells = cluster + sparse
    sensor = Sensor(photons_per_unit=300.0)  # bright -> every in-focus cell detectable

    assert resolve_focal_plane(cells, "auto", acq.optics) == 42.0  # geometric median
    focus = resolve_focal_plane(cells, "auto", acq.optics, acq=acq, sensor_spec=sensor)
    assert 20.0 <= focus <= 24.0  # parked on the dense cluster
    assert focus < 40.0  # ...nowhere near the median gap


def test_resolve_focal_plane_auto_focuses_shallower_when_scatter_dims_deep_cells():
    # Equal-size shallow and deep clusters, symmetric so the median sits between
    # them (~74 µm). Scatter attenuation dims the deep cluster below the SNR floor,
    # so only the shallow cells are recoverable -- auto focus moves shallow.
    acq = _acq(focal_depth_in_tissue_um="auto")
    shallow = [_scored_cell(z, lo=20.0, hi=28.0) for z in (22.0, 24.0, 26.0)]
    deep = [_scored_cell(z, lo=20.0, hi=28.0) for z in (122.0, 124.0, 126.0)]
    cells = shallow + deep
    sensor = Sensor(photons_per_unit=70.0)  # tuned: shallow clears the floor, deep does not

    assert resolve_focal_plane(cells, "auto", acq.optics) == 74.0  # geometric median
    focus = resolve_focal_plane(cells, "auto", acq.optics, acq=acq, sensor_spec=sensor)
    assert focus < 50.0  # pulled to the recoverable shallow cluster, off the median
    assert 20.0 <= focus <= 28.0


def test_resolve_focal_plane_auto_shifts_when_vignette_removes_edge_cells():
    # Field curvature makes off-axis (corner) cells focus deeper in effective depth
    # than the on-axis cells, and they OUTNUMBER them -- so without vignetting the
    # yield-optimal focus sits on the edge group. A strong vignette dims those
    # corner cells below the floor, dropping them from the vote, so the focus
    # snaps back to the on-axis cluster. The plane moves once you account for it.
    acq = _acq(n_px=200, optics=Optics(magnification=8.0, field_curvature_radius_um=600.0))
    # Optical-center frame (n_px=200, 1 µm/px): the axis is (0, 0); the FOV corner
    # is ~(-90, -90), a large field radius.
    center = [Cell(center_um=(80.0, 0.0, 0.0), trace=np.array([20.0, 28.0])) for _ in range(5)]
    edge = [Cell(center_um=(80.0, -90.0, -90.0), trace=np.array([20.0, 28.0])) for _ in range(8)]
    cells = center + edge
    sensor = Sensor(photons_per_unit=200.0)
    strong = combined_falloff_field(acq, None, Vignette(falloff=0.01, exponent=1.0))

    focus_none = resolve_focal_plane(
        cells, "auto", acq.optics, acq=acq, sensor_spec=sensor, photon_field=None
    )
    focus_vig = resolve_focal_plane(
        cells, "auto", acq.optics, acq=acq, sensor_spec=sensor, photon_field=strong
    )
    assert focus_none > 90.0  # edge group (deeper effective depth) wins the count
    assert focus_vig < 84.0  # vignette kills the edge cells -> back to the center cluster
    assert focus_vig < focus_none - 5.0


def test_optics_in_focus_surface_cell_is_barely_degraded():
    # z=0, focal auto -> 0: no scatter, no defocus, atten=1 -> only diffraction
    # blur and the flat NA² collection efficiency dim the footprint.
    acq = _acq(n_px=64)
    collection = acq.optics.collection_efficiency
    scene = Scene.zeros(acq)
    scene.cells.append(_cell_with_footprint(acq, z=0.0))
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    cell = scene.cells[0]
    assert cell.in_focus is True
    assert cell.optical_brightness == pytest.approx(collection)
    # the sum-normalized PSF conserves the integral, so the only change is the
    # NA² collection loss: observed integral == planted integral × collection.
    assert cell.observed_footprint().patch.sum() == pytest.approx(
        cell.footprint_planted.patch.sum() * collection, rel=1e-2
    )
    assert cell.detectable is None  # deferred to finalize


def test_optics_deeper_cell_is_broader_and_dimmer():
    # Both in focus (focal == z, defocus 0), so the difference is pure depth:
    # scatter broadens the footprint and attenuation removes light.
    def run(z):
        acq = _acq(n_px=80, optics=Optics(magnification=8.0), focal_depth_in_tissue_um=z)
        scene = Scene.zeros(acq)
        scene.cells.append(_cell_with_footprint(acq, z=z))
        CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
        return scene.cells[0]

    shallow, deep = run(10.0), run(180.0)
    assert deep.optical_brightness < shallow.optical_brightness  # attenuation
    assert deep.observed_footprint().patch.sum() < shallow.observed_footprint().patch.sum()
    assert (
        deep.observed_footprint().patch.max() < shallow.observed_footprint().patch.max()
    )  # broader + dimmer


def test_optics_defocus_conserves_observed_integral():
    # Fixed depth, sweep the focal plane: defocus broadens the footprint but
    # (being a convolution) conserves its integral; attenuation(z) is fixed.
    z, sums = 50.0, []
    for focal in (48.0, 50.0, 52.0):
        acq = _acq(n_px=80, optics=Optics(magnification=8.0), focal_depth_in_tissue_um=focal)
        scene = Scene.zeros(acq)
        scene.cells.append(_cell_with_footprint(acq, z=z, radius_um=3.0))
        CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
        sums.append(scene.cells[0].observed_footprint().patch.sum())
    assert sums == pytest.approx([sums[0]] * 3, rel=1e-2)


def test_optics_in_focus_flag_respects_depth_of_field():
    acq = _acq(
        n_px=64,
        optics=Optics(magnification=8.0, depth_of_field_um=15.0),
        focal_depth_in_tissue_um=80.0,
    )
    scene = Scene.zeros(acq)
    for z in (70.0, 80.0, 96.0):  # within DOF, at plane, just outside DOF
        scene.cells.append(_cell_with_footprint(acq, z=z))
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    assert [c.in_focus for c in scene.cells] == [True, True, False]


def test_focal_curvature_shift_um():
    flat = Optics(magnification=8.0)  # field_curvature_radius_um defaults to None
    assert flat.focal_curvature_shift_um(250.0) == 0.0  # flat field: no shift
    curved = Optics(magnification=8.0, field_curvature_radius_um=2500.0)
    assert curved.focal_curvature_shift_um(0.0) == 0.0  # on-axis: no shift
    s250 = curved.focal_curvature_shift_um(250.0)
    assert s250 == pytest.approx(250.0**2 / (2 * 2500.0), rel=1e-2)  # ~ r²/2R ≈ 12.5
    assert curved.focal_curvature_shift_um(500.0) > s250  # grows with field radius
    with pytest.raises(ValidationError, match="field_curvature_radius_um"):
        Optics(field_curvature_radius_um=0.0)  # must be > 0 or None


def test_resolved_depth_of_field_um():
    # a numeric value is used as-is
    assert Optics(na=0.3, depth_of_field_um=12.0).resolved_depth_of_field_um == 12.0
    # "auto" (the default) derives the DOF from NA: n·λ/NA², falling as 1/NA²
    o30 = Optics(na=0.30, emission_nm=525.0)  # depth_of_field_um defaults to "auto"
    assert o30.resolved_depth_of_field_um == pytest.approx(1.33 * 0.525 / 0.30**2, rel=1e-6)
    assert Optics(na=0.45).resolved_depth_of_field_um < o30.resolved_depth_of_field_um
    with pytest.raises(ValidationError, match="depth_of_field_um"):
        Optics(depth_of_field_um=0.0)  # must be > 0 or "auto"


def test_field_curvature_blurs_off_axis_cells():
    # Two cells at the same depth (the central focal plane): on-axis vs near the
    # FOV corner. With curvature the corner cell focuses shallower, so it falls
    # out of focus and its peak drops, while the on-axis cell stays sharp.
    npx = 200
    acq = _acq(
        n_px=npx,
        optics=Optics(
            magnification=8.0,
            depth_of_field_um=5.0,
            field_curvature_radius_um=600.0,
        ),
        focal_depth_in_tissue_um=100.0,
    )
    z = 100.0

    def cell_at(y_um, x_um):
        fp = Footprint.from_dense(
            neuron_footprint(
                (npx, npx), acq.um_to_index(y_um, x_um, (npx, npx)), acq.um_to_px(4.0), 0.0,
                np.random.default_rng(0),
            )
        )
        return Cell(center_um=(z, y_um, x_um), footprint_planted=fp)

    center = cell_at(0.0, 0.0)        # on axis (r = 0)
    corner = cell_at(-90.0, -90.0)    # near a corner (large r)
    scene = Scene.zeros(acq)
    scene.cells += [center, corner]
    CellOpticsStep(CellOptics(), acq, np.random.default_rng(0))(scene)
    assert center.in_focus is True
    assert corner.in_focus is False  # off-axis sagitta pushes it past the DOF
    assert corner.observed_footprint().patch.max() < center.observed_footprint().patch.max()


def test_optics_makes_render_use_the_degraded_footprint():
    # Composite a deep cell from its planted footprint, then again after optics:
    # the optically degraded composite is dimmer (blurred + attenuated).
    acq = _acq(n_px=40, duration_s=1.0)
    rng = np.random.default_rng(1)
    scene = Scene.zeros(acq)
    PlaceNeurons(
        density_per_mm3=250000.0, soma_radius_um=4.0, depth_range_um=(120.0, 120.0)
    ).build(acq, rng)(scene)
    CellActivity(active_rate_hz=5.0).build(acq, rng)(scene)

    CompositeStep(Composite(), acq, rng)(scene)  # observed still None -> uses planted
    planted_peak = scene.movie.values.max()

    scene.movie.values[:] = 0.0
    CellOpticsStep(CellOptics(), acq, rng)(scene)
    CompositeStep(Composite(), acq, rng)(scene)  # now regenerates the observed footprint
    observed_peak = scene.movie.values.max()

    assert all(c.observed_sigma_px is not None for c in scene.cells)
    assert observed_peak < planted_peak


def test_optics_chain_with_sensor_runs_end_to_end():
    acq = _acq(n_px=40, duration_s=1.5, bit_depth=8)
    rng = np.random.default_rng(99)
    scene = Scene.zeros(acq)
    steps = [
        PlaceNeurons(
            density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        CellOptics(),
        Composite(),
        Sensor(photons_per_unit=120.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)
    assert all(c.observed_sigma_px is not None for c in scene.cells)
    movie = scene.movie.values
    np.testing.assert_array_equal(movie, np.round(movie))
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.max() > 0.0 and movie.var() > 0.0


# --- the minimal chain -----------------------------------------------------


def test_minimal_chain_place_activity_render_sensor():
    acq = _acq(n_px=40, duration_s=2.0, bit_depth=8)
    rng = np.random.default_rng(2026)
    scene = Scene.zeros(acq)
    steps = [
        PlaceNeurons(
            density_per_mm3=375000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        Composite(),
        Sensor(photons_per_unit=100.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)

    assert len(scene.cells) > 0
    movie = scene.movie.values
    assert movie.shape == (acq.n_frames, 40, 40)
    np.testing.assert_array_equal(movie, np.round(movie))  # digitized counts
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.max() > 0.0  # cells produced signal
    assert movie.var() > 0.0  # spatial/temporal structure, not a flat field


# --- neuropil ---------------------------------------------------------


def test_ou_process_is_stationary_and_correlated():
    # Mean ~0, unit variance, and a high lag-1 correlation for a slow tau.
    slow = ou_process(20000, tau_frames=50.0, rng=np.random.default_rng(0))
    assert abs(slow.mean()) < 0.1
    assert slow.std() == pytest.approx(1.0, rel=0.1)
    slow_ac = np.corrcoef(slow[1:], slow[:-1])[0, 1]
    assert slow_ac > 0.9  # a ≈ exp(-1/50) ≈ 0.98
    # A fast tau decorrelates frame-to-frame.
    fast = ou_process(20000, tau_frames=0.2, rng=np.random.default_rng(1))
    assert np.corrcoef(fast[1:], fast[:-1])[0, 1] < slow_ac


def test_neuropil_adds_nonnegative_background():
    acq = _acq(n_px=40, duration_s=1.0)
    scene = Scene.zeros(acq)
    NeuropilStep(Neuropil(amplitude=0.5), acq, np.random.default_rng(0))(scene)
    movie = scene.movie.values
    assert movie.min() >= 0.0  # additive light, never negative
    assert movie.mean() > 0.0  # background was actually added


def test_neuropil_records_smooth_spatial_and_temporal_ground_truth():
    acq = _acq(n_px=40, duration_s=2.0)
    scene = Scene.zeros(acq)
    NeuropilStep(
        Neuropil(n_components=4, spatial_sigma_um=10.0, temporal_tau_s=10.0),
        acq,
        np.random.default_rng(1),
    )(scene)
    spatial = scene.truth.neuropil_spatial
    temporal = scene.truth.neuropil_temporal
    assert spatial.shape == (4, 40, 40)
    assert temporal.shape == (4, acq.n_frames)
    # Spatial fields are non-negative, peak-normalized, and smooth (adjacent-pixel
    # variation well below the field's overall spread - unlike white noise).
    field = spatial[0]
    assert field.min() >= 0.0 and field.max() == pytest.approx(1.0)
    assert np.abs(np.diff(field, axis=0)).mean() < field.std()
    # Temporal envelopes are strictly positive (the lognormal guarantee).
    assert (temporal > 0).all()


def test_neuropil_is_reproducible():
    acq = _acq(n_px=32, duration_s=1.0)
    outs = []
    for _ in range(2):
        scene = Scene.zeros(acq)
        Neuropil().build(acq, np.random.default_rng(3))(scene)
        outs.append(scene.movie.values.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


def test_population_envelope_is_mean_one_lagged_and_smoothed():
    # Two cells whose summed activity is a single sharp spike on a flat baseline.
    n = 200
    a = np.full(n, 0.1)
    a[100] = 5.0  # a one-frame burst
    b = np.full(n, 0.1)
    env = population_envelope([a, b], tau_frames=10.0)
    assert env.shape == (n,)
    assert env.min() >= 0.0
    assert env.mean() == pytest.approx(1.0)  # normalized
    # The one-pole low-pass spreads the spike forward in time: the post-spike
    # frame is elevated (a lag the raw aggregate, flat after frame 100, lacks)
    # and the response is smoother than the input step.
    assert env[101] > env[99]
    raw = (a + b) / (a + b).mean()
    assert np.abs(np.diff(env)).sum() < np.abs(np.diff(raw)).sum()


def test_population_envelope_returns_none_without_signal():
    assert population_envelope([], tau_frames=10.0) is None  # no cells
    assert population_envelope([np.zeros(50)], tau_frames=10.0) is None  # all silent


def test_neuropil_temporal_couples_to_population_activity():
    # A scene with cells carrying a shared activity bump in the middle third.
    acq = _acq(n_px=24, duration_s=10.0)
    n = acq.n_frames
    bump = np.full(n, 0.2)
    bump[n // 3 : 2 * n // 3] = 4.0
    aggregate = bump * 3  # three identical cells

    def _background_per_frame(coupling):
        scene = Scene.zeros(acq)
        scene.cells += [Cell(center_um=(0.0, 12.0, 12.0), trace=bump.copy()) for _ in range(3)]
        NeuropilStep(
            Neuropil(amplitude=1.0, n_components=3, population_coupling=coupling),
            acq, np.random.default_rng(0),
        )(scene)
        return scene.movie.values.mean(axis=(1, 2)), scene.truth.neuropil_population

    coupled, pop = _background_per_frame(1.0)
    independent, _ = _background_per_frame(0.0)
    # The stored driver is the (mean-1) aggregate low-passed at population_tau_s.
    assert pop is not None and pop.shape == (n,)
    assert pop == pytest.approx(population_envelope([aggregate], acq.s_to_frame(1.5)))
    # Fully coupled: the diffuse background tracks population activity closely;
    # fully independent: it does not (the OU drift is blind to the cells).
    assert np.corrcoef(coupled, pop)[0, 1] > 0.95
    assert abs(np.corrcoef(independent, pop)[0, 1]) < 0.6


def test_neuropil_envelope_mix_stays_positive_with_cells():
    # With cells present and partial coupling, the realized temporal envelopes
    # are still strictly positive (the additive background never clips), exactly
    # as in the no-cell path -- so amplitude alone sets the absolute level.
    acq = _acq(n_px=20, duration_s=5.0)
    scene = Scene.zeros(acq)
    scene.cells += [
        Cell(
            center_um=(0.0, 10.0, 10.0),
            trace=np.abs(ou_process(acq.n_frames, 5.0, np.random.default_rng(i))) + 0.1,
        )
        for i in range(4)
    ]
    NeuropilStep(
        Neuropil(n_components=3, population_coupling=0.7),
        acq, np.random.default_rng(1),
    )(scene)
    temporal = scene.truth.neuropil_temporal
    assert temporal.shape == (3, acq.n_frames)
    assert (temporal > 0).all()
    assert scene.movie.values.min() >= 0.0


# --- bleaching (per-cell pool: bleaching vs turnover) ----------------------


def test_bleaching_pool_decays_under_drive_and_recovers_in_dark():
    # Constant emission with negligible turnover: the intact fraction decays
    # monotonically from 1 (a per-photon hazard).
    n = 200
    lit = bleaching_pool(np.ones(n), q=0.02, tau_turn_frames=1e9, intensity=1.0)
    assert lit[0] == pytest.approx(1.0)
    assert (np.diff(lit) <= 1e-12).all()  # monotonically non-increasing
    assert lit[-1] < 0.1
    # Light off (no emission) with turnover: the pool recovers back toward 1.
    dark = bleaching_pool(np.zeros(n), q=0.02, tau_turn_frames=50.0, intensity=1.0, b0=0.3)
    assert dark[0] == pytest.approx(0.3)
    assert (np.diff(dark) >= -1e-12).all()  # monotonically non-decreasing
    assert dark[-1] > 0.95  # recovered


def test_bleaching_pool_more_emission_or_brighter_bleaches_more():
    n = 300
    base = bleaching_pool(np.full(n, 1.0), q=0.01, tau_turn_frames=1e9, intensity=1.0)
    busier = bleaching_pool(np.full(n, 3.0), q=0.01, tau_turn_frames=1e9, intensity=1.0)
    brighter = bleaching_pool(np.full(n, 1.0), q=0.01, tau_turn_frames=1e9, intensity=3.0)
    assert busier[-1] < base[-1]      # a more active cell fades more
    assert brighter[-1] < base[-1]    # ...as does a more brightly-lit one
    # With turnover the decay settles at a floor B* = k_turn/(k_turn + q*I*emission).
    settled = bleaching_pool(np.full(2000, 1.0), q=0.01, tau_turn_frames=100.0, intensity=1.0)
    k = 1.0 / 100.0
    assert settled[-1] == pytest.approx(k / (k + 0.01 * 1.0 * 1.0), rel=0.02)


def test_bleaching_floor_is_the_pool_fixed_point():
    # bleaching_floor must equal the value the integrator settles at under a long
    # constant drive (the steady state it documents), so the notebook's planning
    # curves agree with the engine.
    q, tau, intensity, emission = 0.01, 100.0, 1.0, 1.0
    settled = bleaching_pool(np.full(4000, emission), q=q, tau_turn_frames=tau, intensity=intensity)
    assert bleaching_floor(q, intensity, emission, tau) == pytest.approx(settled[-1], rel=1e-3)
    # Light off (or no turnover) are the two limits: turnover wins -> 1; no turnover -> 0.
    assert bleaching_floor(q, intensity=0.0, emission=emission, tau_turn=tau) == pytest.approx(1.0)
    assert bleaching_floor(q, intensity, emission, tau_turn=float("inf")) == pytest.approx(0.0)
    # tau_turn == 0 (no turnover) is the same limit, and still equals the pool: under
    # any positive drive the integrator settles at 0, and the degenerate no-drive /
    # no-turnover case (denom == 0) falls back to the fully-intact 1.0.
    settled0 = bleaching_pool(np.full(4000, emission), q=q, tau_turn_frames=0.0, intensity=intensity)
    assert bleaching_floor(q, intensity, emission, tau_turn=0.0) == pytest.approx(settled0[-1], abs=1e-9)
    assert bleaching_floor(q, intensity=0.0, emission=0.0, tau_turn=0.0) == pytest.approx(1.0)
    # negative rates are non-physical and rejected rather than silently returning 1.0.
    for bad in (dict(q=-q), dict(intensity=-1.0), dict(emission=-1.0), dict(tau_turn=-1.0)):
        with pytest.raises(ValueError, match="non-negative"):
            bleaching_floor(**{"q": q, "intensity": intensity, "emission": emission, "tau_turn": tau, **bad})


def test_dark_recovery_is_the_pool_zero_input_response():
    # dark_recovery must trace the integrator with the light off (no emission),
    # frame for frame, starting from b0.
    n, tau, b0 = 200, 50.0, 0.3
    pool = bleaching_pool(np.zeros(n), q=0.02, tau_turn_frames=tau, intensity=1.0, b0=b0)
    closed = dark_recovery(b0, np.arange(n), tau)
    np.testing.assert_allclose(closed, pool, atol=1e-9)
    assert dark_recovery(b0, 0.0, tau) == pytest.approx(b0)  # starts at b0
    assert dark_recovery(b0, 1e9, tau) == pytest.approx(1.0)  # fully recovered after long dark
    assert np.ndim(dark_recovery(b0, 10.0, tau)) == 0  # scalar t -> 0-d scalar back
    # a non-positive turnover time is rejected (would divide by zero / diverge).
    for bad_tau in (0.0, -1.0):
        with pytest.raises(ValueError, match="tau_turn > 0"):
            dark_recovery(b0, np.arange(n), bad_tau)


def test_bleaching_step_sets_per_cell_envelope_and_render_dims_over_time():
    acq = _acq(duration_s=10.0)
    scene = Scene.zeros(acq)
    PlaceNeurons(density_per_mm3=142857.0, depth_range_um=(0.0, 0.0)).build(
        acq, np.random.default_rng(4)
    )(scene)
    CellActivity(active_rate_hz=5.0, tau_decay_s=0.4, brightness_cv=0.0).build(
        acq, np.random.default_rng(4)
    )(scene)
    # Exaggerated susceptibility (at unit intensity) so the fade is unambiguous.
    BleachingStep(
        Bleaching(bleach_susceptibility=0.05, excitation_intensity=1.0),
        acq, np.random.default_rng(4),
    )(scene)
    for cell in scene.cells:
        assert cell.bleach is not None
        assert cell.bleach[0] == pytest.approx(1.0)
        assert cell.bleach[-1] < cell.bleach[0]  # faded by the end
    CompositeStep(Composite(), acq, np.random.default_rng(4))(scene)
    brightness = scene.movie.values.sum(axis=(1, 2))
    assert brightness[-int(acq.fps):].mean() < brightness[: int(acq.fps)].mean()


# --- vignette ---------------------------------------------------------


# --- illumination profile + vignette (Stage 8) -----------------------------


def test_radial_falloff_is_one_at_center_falloff_at_corner_and_monotonic():
    field = radial_falloff((51, 51), (25.0, 25.0), falloff=0.4, exponent=2.0)
    assert field[25, 25] == pytest.approx(1.0)  # bright center
    assert field[0, 0] == pytest.approx(0.4)  # farthest corner == falloff
    # strictly dimmer moving out along a row from the center
    row = field[25, 25:]
    assert np.all(np.diff(row) < 0)


def test_illumination_profile_is_radial_and_time_invariant():
    acq = _acq(n_px=51, duration_s=1.0)  # odd -> clean center pixel at (25, 25)
    scene = Scene.ones(acq)
    IlluminationProfileStep(
        IlluminationProfile(falloff=0.5, exponent=2.0), acq, np.random.default_rng(0)
    )(scene)
    field = scene.truth.illumination
    assert field.shape == (51, 51)
    assert field[25, 25] == pytest.approx(1.0)
    assert field[0, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(scene.movie.values[0], field)  # all-ones movie == field
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time


def test_illumination_drives_bleaching_faster_at_the_bright_center():
    # With an illumination profile injected, a cell at the bright center receives a
    # larger excitation dose and bleaches more (lower end B) than an *identically
    # active* cell at a dim edge; without it (uniform dose) the two are equal. Both
    # cells are given the same fixed trace so only the illumination dose differs.
    acq = _acq(n_px=64, duration_s=120.0, fps=10.0)
    px = acq.pixel_size_um
    trace = np.full(acq.n_frames, 1.5)  # identical steady emission for both cells

    def end_B(with_illum):
        scene = Scene.zeros(acq, rng=np.random.default_rng(0))
        scene.cells = [
            Cell(center_um=(0.0, 0.0, 0.0), trace=trace.copy()),              # optical axis
            Cell(center_um=(0.0, -30 * px, -30 * px), trace=trace.copy()),    # near a corner
        ]
        step = BleachingStep(Bleaching(excitation_intensity=10.0), acq, np.random.default_rng(2))
        if with_illum:
            step.illumination = IlluminationProfile(falloff=0.2, exponent=2.0)
        step(scene)
        return scene.cells[0].bleach[-1], scene.cells[1].bleach[-1]

    c_on, e_on = end_B(True)
    c_off, e_off = end_B(False)
    assert c_on < e_on  # center bleaches more than edge when illumination present
    assert e_on > e_off  # the dim edge bleaches less than the uniform-dose baseline
    assert c_off == pytest.approx(e_off, rel=1e-9)  # equal without illumination (same trace)


def test_bleaching_illumination_dose_is_margin_invariant():
    # The illumination field is sensor-FOV sized; cells live in the optical-center
    # frame whose origin is the FOV center, which is invariant to the motion margin.
    # A cell at the optical axis (0, 0) therefore gets the same bright-center dose -
    # and bleaches to the same end-B - with or without a margin (no canvas->FOV
    # bookkeeping needed any more; the frame handles it by construction).
    acq = _acq(n_px=64, duration_s=120.0, fps=10.0)
    trace = np.full(acq.n_frames, 1.5)

    def center_end_B(margin_px):
        scene = Scene.zeros(acq, rng=np.random.default_rng(0), margin_px=margin_px)
        scene.cells = [Cell(center_um=(0.0, 0.0, 0.0), trace=trace.copy())]
        step = BleachingStep(Bleaching(excitation_intensity=10.0), acq, np.random.default_rng(2))
        step.illumination = IlluminationProfile(falloff=0.2, exponent=2.0)
        step(scene)
        return scene.cells[0].bleach[-1]

    # Same physical FOV position -> same dose -> same end-B regardless of margin.
    assert center_end_B(0) == pytest.approx(center_end_B(12), rel=1e-9)


def test_vignette_is_radial_and_time_invariant():
    acq = _acq(n_px=51, duration_s=1.0)  # odd -> a clean center pixel at (25, 25)
    scene = Scene.ones(acq)
    VignetteStep(
        Vignette(falloff=0.5, exponent=2.0), acq, np.random.default_rng(0)
    )(scene)
    field = scene.truth.vignette
    assert field.shape == (51, 51)
    assert field[25, 25] == pytest.approx(1.0)  # bright center
    assert field[0, 0] == pytest.approx(0.5)  # farthest corner == falloff
    # On an all-ones movie the movie equals the field, and is identical per frame.
    np.testing.assert_allclose(scene.movie.values[0], field)
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time: every frame identical


def test_vignette_center_offset_moves_the_bright_spot():
    acq = _acq(n_px=51, duration_s=0.1)  # 1 µm/px, so +12 µm == +12 px in y
    scene = Scene.ones(acq)
    VignetteStep(
        Vignette(falloff=0.4, center_offset_um=(12.0, 0.0)),
        acq,
        np.random.default_rng(0),
    )(scene)
    peak_row, peak_col = np.unravel_index(scene.truth.vignette.argmax(), (51, 51))
    assert peak_row > 25  # shifted down from the FOV center row
    assert peak_col == pytest.approx(25, abs=1)  # unshifted in x


# --- leakage ----------------------------------------------------------


def test_leakage_uniform_adds_level_everywhere():
    acq = _acq(n_px=16, duration_s=1.0)
    scene = Scene.zeros(acq)
    LeakageStep(
        Leakage(profile="uniform", level=0.2), acq, np.random.default_rng(0)
    )(scene)
    np.testing.assert_allclose(scene.movie.values, 0.2)
    np.testing.assert_allclose(scene.truth.leakage, 0.2)
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time: every frame identical


def test_leakage_gaussian_peaks_at_center():
    acq = _acq(n_px=51, duration_s=0.1)
    scene = Scene.zeros(acq)
    LeakageStep(
        Leakage(profile="gaussian", level=0.3, sigma_um=10.0),
        acq,
        np.random.default_rng(0),
    )(scene)
    field = scene.truth.leakage
    assert field[25, 25] == pytest.approx(0.3)  # central glow == level
    assert field[0, 0] < field[25, 25]  # dimmer at the corner
    movie = scene.movie.values
    assert (movie == movie[0]).all()  # static in time: every frame identical


# --- vasculature: disabled is a no-op ---------------------------------


def test_vasculature_disabled_is_a_noop():
    # The default Vasculature is enabled=False with no layers, so it draws no RNG
    # and leaves the movie and the ground-truth slot untouched (the streaming
    # writer mirrors this guard).
    acq = _acq(n_px=16, duration_s=0.5)
    scene = Scene.ones(acq)
    Vasculature().build(acq, np.random.default_rng(0))(scene)
    assert (scene.movie.values == 1.0).all()  # scene untouched
    assert scene.truth.vasculature_mask is None  # no ground-truth contribution


# --- vasculature model core: grow / rasterize / mask ------------------


def _growth(**kw):
    """A VesselGrowth with small, test-friendly defaults; override per test."""
    base = dict(r0_um=8.0, min_radius_um=1.0, branch_prob=0.1, tortuosity_rad=0.1)
    base.update(kw)
    return VesselGrowth(**base)


def test_murray_children_conserves_cube_sum():
    # Murray's law: r1**3 + r2**3 == r_parent**3 for any split fraction.
    for a in (0.5, 0.7, 0.9):
        r1, r2 = murray_children(10.0, a)
        assert r1**3 + r2**3 == pytest.approx(10.0**3)
        assert r1 >= r2  # the "main" child (a >= 0.5) is the thicker one
    # symmetric split shrinks each child to 0.5**(1/3) of the parent
    r1, r2 = murray_children(10.0, 0.5)
    assert r1 == pytest.approx(r2) == pytest.approx(10.0 * 0.5 ** (1 / 3))


def test_grow_vessel_tree_is_deterministic_in_the_rng():
    bounds, root = (200.0, 200.0), (100.0, 0.0)
    a = grow_vessel_tree(bounds, root, 0.0, _growth(), np.random.default_rng(7))
    b = grow_vessel_tree(bounds, root, 0.0, _growth(), np.random.default_rng(7))
    assert np.array_equal(a, b)  # same seed -> identical tree (streaming relies on it)


def test_grow_vessel_tree_unbranched_is_a_constant_radius_path():
    # branch_prob=0 -> no bifurcation -> no Murray taper: one straight-ish path,
    # every segment at the root radius, starting at the root.
    bounds, root, r0 = (200.0, 200.0), (100.0, 0.0), 6.0
    segs = grow_vessel_tree(
        bounds, root, 0.0, _growth(r0_um=r0, branch_prob=0.0), np.random.default_rng(1)
    )
    assert segs.shape[0] >= 1
    assert np.allclose(segs[:, 4], r0)  # radius column constant (no branching)
    assert segs[0, 0] == pytest.approx(root[0]) and segs[0, 1] == pytest.approx(root[1])
    # segments chain end-to-end: each start == previous end
    assert np.allclose(segs[1:, :2], segs[:-1, 2:4])


def test_grow_vessel_tree_branching_tapers_and_terminates_at_capillary_floor():
    bounds, root = (400.0, 400.0), (200.0, 0.0)
    segs = grow_vessel_tree(
        bounds, root, 0.0,
        _growth(r0_um=10.0, min_radius_um=2.0, branch_prob=0.4, tortuosity_rad=0.2),
        np.random.default_rng(3),
    )
    radii = segs[:, 4]
    assert radii.max() == pytest.approx(10.0)  # the trunk is the thickest
    assert radii.min() >= 2.0  # branches stop at the capillary floor, never thinner
    assert (radii < 10.0).any()  # tapering did happen (children below the trunk)


def test_grow_vessel_tree_respects_the_segment_cap():
    # A dense brancher in a big canvas would grow forever without the cap.
    segs = grow_vessel_tree(
        (1000.0, 1000.0), (500.0, 0.0), 0.0,
        _growth(r0_um=20.0, min_radius_um=0.1, branch_prob=0.9, max_segments=50),
        np.random.default_rng(0),
    )
    assert segs.shape[0] <= 50


def test_rasterize_vessels_chord_peaks_at_diameter_and_is_local():
    # One horizontal segment of radius 5 µm at 1 µm/px: path length on its axis is
    # the full chord 2r = 10 µm, and pixels well outside the radius are untouched.
    seg = np.array([[20.0, 5.0, 20.0, 45.0, 5.0]])  # y const, x 5->45, r=5
    L = rasterize_vessels(seg, (40, 50), pixel_size_um=1.0)
    assert L[20, 25] == pytest.approx(10.0, abs=1e-6)  # 2r on the spine
    assert L[20, 25] >= L[18, 25] >= L[14, 25]  # tapers off the axis
    assert L[14, 25] == 0.0  # 6 µm off a 5 µm vessel -> outside, no blood
    assert L[0, 0] == 0.0  # far corner untouched (rasterized locally)


def test_rasterize_vessels_overlap_adds_path_length():
    # Two crossing vessels: where they overlap the light crosses both -> chords add.
    horiz = np.array([[20.0, 0.0, 20.0, 40.0, 4.0]])
    vert = np.array([[0.0, 20.0, 40.0, 20.0, 4.0]])
    both = rasterize_vessels(np.vstack([horiz, vert]), (40, 40), 1.0)
    one = rasterize_vessels(horiz, (40, 40), 1.0)
    assert both[20, 20] == pytest.approx(2 * one[20, 20])  # crossing doubles it


def test_rasterize_vessels_empty_input_is_blank():
    L = rasterize_vessels(np.zeros((0, 5)), (10, 12), 1.0)
    assert L.shape == (10, 12) and not L.any()


def test_vessels_to_mask_beer_lambert_floor_and_clear_tissue():
    L = np.array([[0.0, 1.0, 100.0]])
    m = vessels_to_mask(L, opacity=0.8, absorption_per_um=0.2)
    assert m[0, 0] == pytest.approx(1.0)  # no blood -> fully transmissive
    assert m[0, 2] == pytest.approx(1.0 - 0.8, abs=1e-3)  # thick trunk -> 1-opacity floor
    assert 1.0 - 0.8 < m[0, 1] < 1.0  # a thin vessel sits between floor and clear
    # opacity=0 disables the effect regardless of path length
    assert np.allclose(vessels_to_mask(L, 0.0, 0.2), 1.0)


def test_vessels_to_mask_darker_with_more_absorption():
    L = np.full((4, 4), 5.0)
    weak = vessels_to_mask(L, 1.0, 0.05)
    strong = vessels_to_mask(L, 1.0, 0.5)
    assert (strong < weak).all()  # higher k -> darker shadow at fixed thickness


# --- vasculature spec + mask field + step ----------------------------


def test_vessel_layer_rejects_capillary_floor_above_trunk():
    with pytest.raises(ValidationError, match="min_radius_um"):
        VesselLayer(root_radius_um=5.0, min_radius_um=6.0)


def test_vasculature_mask_field_is_deterministic_and_composes_layers():
    acq = _acq(n_px=48, duration_s=0.2)
    one = [VesselLayer(depth_um=10.0, n_roots=2, root_radius_um=8.0)]
    two = [*one, VesselLayer(depth_um=40.0, n_roots=2, root_radius_um=6.0)]
    spec1, spec2 = Vasculature(enabled=True, layers=one), Vasculature(enabled=True, layers=two)
    m1a = vasculature_mask_field(spec1, acq, (48, 48), 30.0, np.random.default_rng(5))
    m1b = vasculature_mask_field(spec1, acq, (48, 48), 30.0, np.random.default_rng(5))
    np.testing.assert_array_equal(m1a, m1b)  # same seed -> identical mask
    assert ((m1a > 0) & (m1a <= 1.0)).all() and (m1a < 1.0).any()  # a valid, non-trivial mask
    m2 = vasculature_mask_field(spec2, acq, (48, 48), 30.0, np.random.default_rng(5))
    assert m2.mean() <= m1a.mean()  # a second absorbing layer only darkens


def test_vasculature_step_applies_a_dark_static_mask_and_records_it():
    acq = _acq(n_px=48, duration_s=0.3)
    scene = Scene.ones(acq)
    spec = Vasculature(
        enabled=True, layers=[VesselLayer(depth_um=10.0, n_roots=3, root_radius_um=8.0)]
    )
    spec.build(acq, np.random.default_rng(0))(scene)
    mask = scene.truth.vasculature_mask
    assert mask is not None and mask.shape == scene.canvas_shape
    assert ((mask > 0) & (mask <= 1.0)).all() and (mask < 1.0).any()  # vessels darkened pixels
    # the movie started at ones, so a multiplicative static mask leaves every frame
    # equal to the mask itself (and identical frame-to-frame: spatially static).
    movie = scene.movie.values
    for t in range(movie.shape[0]):
        np.testing.assert_allclose(movie[t], mask)


def test_vasculature_step_disabled_or_layerless_is_noop():
    acq = _acq(n_px=24, duration_s=0.2)
    for spec in (
        Vasculature(enabled=False, layers=[VesselLayer()]),  # master switch off
        Vasculature(enabled=True, layers=[]),  # on but nothing to grow
    ):
        scene = Scene.ones(acq)
        spec.build(acq, np.random.default_rng(0))(scene)
        assert (scene.movie.values == 1.0).all()
        assert scene.truth.vasculature_mask is None


# --- scene-grid plumbing -----------------------------------


def _oversized_scene(acq, canvas):
    """A scene whose movie canvas is larger than the sensor - the shape a motion
    margin will produce. Built by hand here, before Scene grows margin
    support, to prove the steps read their grid from the scene, not the sensor."""
    movie = xr.DataArray(
        np.zeros((acq.n_frames, canvas, canvas)),
        dims=("frame", "height", "width"),
    )
    scene = Scene.zeros(acq, rng=np.random.default_rng(0))
    scene.movie = movie  # hand-set an oversized canvas (canvas_shape reads it back)
    return scene


def test_steps_fill_the_scene_canvas_not_the_sensor_dims():
    # Sensor is 20×20 but the canvas is 30×30; place_neurons and neuropil must
    # honor the canvas (so off-FOV tissue exists to move in under motion).
    acq = _acq(n_px=20, duration_s=1.0)
    scene = _oversized_scene(acq, canvas=30)

    PlaceNeurons(
        density_per_mm3=250000.0, soma_radius_um=4.0, depth_range_um=(0.0, 0.0)
    ).build(acq, np.random.default_rng(0))(scene)
    assert scene.cells, "expected cells placed across the larger canvas"
    assert all(c.footprint_planted.canvas_shape == (30, 30) for c in scene.cells)

    NeuropilStep(Neuropil(), acq, np.random.default_rng(0))(scene)
    assert scene.truth.neuropil_spatial.shape[1:] == (30, 30)


# --- the field chain -------------------------------------------------------


def test_field_chain_runs_end_to_end_and_records_ground_truth():
    acq = _acq(n_px=40, duration_s=1.5, bit_depth=8)
    rng = np.random.default_rng(7)
    scene = Scene.zeros(acq)
    steps = [
        PlaceNeurons(
            density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        Bleaching(),  # cell-domain: before render, sets each cell's bleach envelope
        CellOptics(),
        Composite(),
        Neuropil(amplitude=0.3),
        Vignette(falloff=0.6),
        Leakage(profile="gaussian", level=0.1),
        Sensor(photons_per_unit=120.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)

    movie = scene.movie.values
    np.testing.assert_array_equal(movie, np.round(movie))  # digitized counts
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.var() > 0.0
    # Every field step left its ground-truth contribution.
    assert scene.truth.neuropil_spatial is not None
    assert all(c.bleach is not None for c in scene.cells)  # bleaching ran per-cell
    assert scene.truth.vignette is not None
    assert scene.truth.leakage is not None


# --- brain_motion -----------------------------------------------------


def test_bounded_random_walk_starts_at_zero_and_stays_bounded():
    walk = bounded_random_walk(500, step_px=1.0, max_px=4.0, rng=np.random.default_rng(0))
    assert walk.shape == (500, 2)
    np.testing.assert_array_equal(walk[0], [0.0, 0.0])  # frame 0 is the reference
    mags = np.hypot(walk[:, 0], walk[:, 1])
    assert mags.max() <= 4.0 + 1e-9  # never leaves the radius-max_px disk


def test_physical_motion_starts_at_origin_and_stays_bounded():
    traj = physical_brain_motion(
        2000, fps=50.0, locomotion_freq_hz=7.0, resonance_freq_hz=6.0,
        damping_ratio=0.5, locomotion_fraction=0.6, locomotion_axis=0,
        amplitude_px=10.0, max_px=12.0, rng=np.random.default_rng(0),
    )
    assert traj.shape == (2000, 2)
    np.testing.assert_array_equal(traj[0], [0.0, 0.0])  # frame 0 is the reference
    assert np.hypot(traj[:, 0], traj[:, 1]).max() <= 12.0 + 1e-9  # clamped to the disk


def test_physical_motion_amplitude_tracks_the_target():
    # The oscillator is linear in the drive, so the trajectory is scaled exactly so
    # the 99th-percentile displacement radius equals the requested amplitude (the
    # extreme excursion), with the bulk of frames well inside it. Clamp well clear at
    # max_px=30, so it never bites.
    traj = physical_brain_motion(
        3000, fps=50.0, locomotion_freq_hz=7.0, resonance_freq_hz=6.0,
        damping_ratio=0.5, locomotion_fraction=0.25, locomotion_axis=0,
        amplitude_px=10.0, max_px=30.0, rng=np.random.default_rng(1),
    )
    radius = np.hypot(traj[:, 0], traj[:, 1])
    assert np.percentile(radius, 99) == pytest.approx(10.0, rel=1e-6)
    assert np.median(radius) < 6.0  # most frames move much less than the extreme


def test_physical_motion_spectrum_peaks_at_the_locomotion_frequency():
    # The dominant axis carries the locomotion rhythm: its power spectrum (above the
    # slow-drift bins) peaks at locomotion_freq_hz. Sample at 50 fps so 7 Hz is well
    # below Nyquist and the line is cleanly resolved.
    fps, f_loco = 50.0, 7.0
    traj = physical_brain_motion(
        4096, fps=fps, locomotion_freq_hz=f_loco, resonance_freq_hz=f_loco,
        damping_ratio=0.4, locomotion_fraction=0.9, locomotion_axis=0,
        amplitude_px=10.0, max_px=40.0, rng=np.random.default_rng(2),
    )
    freqs = np.fft.rfftfreq(traj.shape[0], d=1.0 / fps)
    power = np.abs(np.fft.rfft(traj[:, 0] - traj[:, 0].mean())) ** 2
    band = freqs > 1.0  # ignore DC and the slow-drift bins
    assert freqs[band][np.argmax(power[band])] == pytest.approx(f_loco, abs=0.5)


def test_physical_motion_cross_axis_lacks_the_locomotion_peak():
    # Only the dominant (y) axis is driven by locomotion; the cross (x) axis sees
    # broadband noise only, so its power at the stride frequency is far weaker.
    fps, f_loco = 50.0, 7.0
    traj = physical_brain_motion(
        4096, fps=fps, locomotion_freq_hz=f_loco, resonance_freq_hz=f_loco,
        damping_ratio=0.4, locomotion_fraction=0.9, locomotion_axis=0,
        amplitude_px=10.0, max_px=40.0, rng=np.random.default_rng(3),
    )
    freqs = np.fft.rfftfreq(traj.shape[0], d=1.0 / fps)
    bin_loco = int(np.argmin(np.abs(freqs - f_loco)))
    p_y = np.abs(np.fft.rfft(traj[:, 0] - traj[:, 0].mean())) ** 2
    p_x = np.abs(np.fft.rfft(traj[:, 1] - traj[:, 1].mean())) ** 2
    assert p_y[bin_loco] > 20.0 * p_x[bin_loco]


def test_physical_motion_is_deterministic_given_seed():
    kw = dict(
        locomotion_freq_hz=7.0, resonance_freq_hz=6.0, damping_ratio=0.5,
        locomotion_fraction=0.6, locomotion_axis=0, amplitude_px=10.0, max_px=20.0,
    )
    a = physical_brain_motion(1000, fps=30.0, rng=np.random.default_rng(7), **kw)
    b = physical_brain_motion(1000, fps=30.0, rng=np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(a, b)


def test_brain_motion_physical_is_the_default_and_records_bounded_shifts():
    # The spec default (no model=) is the physical oscillator; it records per-frame
    # pixel shifts starting at the origin and bounded by max_shift_um.
    acq = _acq(n_px=64, duration_s=4.0)  # 1 µm/px, 80 frames
    scene = Scene.zeros(acq, margin_px=12)
    BrainMotion(motion_amplitude_um=6.0, max_shift_um=9.0).build(
        acq, np.random.default_rng(4)
    )(scene)
    shifts = scene.truth.shifts
    assert shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(shifts[0], [0.0, 0.0])
    assert np.hypot(shifts[:, 0], shifts[:, 1]).max() <= 9.0 + 1e-9


def test_shift_and_crop_recenters_a_padded_frame():
    # A 4 px frame padded to 8 px, shifted back by the margin, crops to the FOV.
    canvas = np.zeros((1, 8, 8))
    canvas[0, 5, 5] = 1.0
    out = shift_and_crop(canvas, np.array([[-2.0, -2.0]]), fov_shape=(4, 4))
    assert out.shape == (1, 4, 4)
    assert out[0, 1, 1] == pytest.approx(1.0)  # (5,5) - margin(2) - shift(2) -> (1,1)


def test_explicit_trajectory_moves_content_by_the_given_shift():
    acq = _acq(n_px=20, duration_s=0.1)  # 2 frames, 1 µm/px
    margin = 6
    scene = Scene.zeros(acq, margin_px=margin)
    c = (20 + 2 * margin) // 2  # canvas center == FOV center
    scene.movie.values[:, c, c] = 9.0
    BrainMotion(trajectory_um=[(0.0, 0.0), (3.0, 2.0)]).build(
        acq, np.random.default_rng(0)
    )(scene)
    fov0, fov1 = scene.movie.values[0], scene.movie.values[1]
    fc = 10  # FOV center (20×20)
    assert fov0[fc, fc] == pytest.approx(9.0)  # frame 0 unshifted
    assert fov1[fc + 3, fc + 2] == pytest.approx(9.0)  # frame 1 moved down 3, right 2
    assert fov1[fc, fc] < 1e-6


def test_motion_brings_offscreen_tissue_into_view():
    # A bright spot in the top margin (outside the FOV) is cropped away at rest
    # and brought into view by a downward shift - real tissue, not a fill.
    acq = _acq(n_px=20, duration_s=0.1)  # 2 frames
    margin = 6
    scene = Scene.zeros(acq, margin_px=margin)
    scene.movie.values[:, margin - 3, margin + 10] = 5.0  # 3 px above the FOV top
    BrainMotion(trajectory_um=[(0.0, 0.0), (4.0, 0.0)]).build(
        acq, np.random.default_rng(0)
    )(scene)
    assert scene.movie.values[0].max() < 1e-6  # off-FOV at rest
    assert scene.movie.values[1].max() == pytest.approx(5.0)  # shifted into view


def test_brain_motion_records_shifts_in_pixels():
    acq = _acq(n_px=16, duration_s=0.5)  # 10 frames
    scene = Scene.zeros(acq, margin_px=6)
    BrainMotion(model="walk", walk_step_um=0.5, max_shift_um=4.0).build(
        acq, np.random.default_rng(2)
    )(scene)
    shifts = scene.truth.shifts
    assert shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(shifts[0], [0.0, 0.0])
    assert np.hypot(shifts[:, 0], shifts[:, 1]).max() <= 4.0 + 1e-9  # px, bounded


def test_static_field_is_invariant_under_motion():
    # vignette is fixed to the sensor: the field it writes is byte-identical
    # whether or not the brain moved first (the reference-frame invariant).
    acq = _acq(n_px=24, duration_s=0.5)
    s0 = Scene.ones(acq)
    VignetteStep(Vignette(falloff=0.5), acq, np.random.default_rng(0))(s0)

    s1 = Scene.ones(acq, margin_px=4)
    BrainMotion(model="walk", walk_step_um=0.5, max_shift_um=3.0).build(
        acq, np.random.default_rng(1)
    )(s1)
    VignetteStep(Vignette(falloff=0.5), acq, np.random.default_rng(2))(s1)

    assert s1.truth.vignette.shape == (24, 24)  # cropped to the sensor FOV
    np.testing.assert_array_equal(s0.truth.vignette, s1.truth.vignette)


def test_brain_motion_rejects_wrong_length_trajectory():
    acq = _acq(n_px=16, duration_s=0.5)  # 10 frames
    scene = Scene.zeros(acq, margin_px=4)
    with pytest.raises(ValueError, match="trajectory_um"):
        BrainMotion(trajectory_um=[(0.0, 0.0), (1.0, 1.0)]).build(
            acq, np.random.default_rng(0)
        )(scene)


def test_brain_motion_rejects_insufficient_margin():
    acq = _acq(n_px=16, duration_s=0.1)  # 2 frames
    scene = Scene.zeros(acq, margin_px=1)  # only 1 px of tissue margin
    with pytest.raises(ValueError, match="margin"):
        BrainMotion(trajectory_um=[(0.0, 0.0), (5.0, 0.0)]).build(
            acq, np.random.default_rng(0)
        )(scene)  # 5 px shift overruns the 1 px margin


def test_full_pipeline_with_motion_runs_end_to_end():
    acq = _acq(n_px=40, duration_s=1.5, bit_depth=8)  # 1 µm/px
    rng = np.random.default_rng(11)
    max_shift_um = 4.0
    margin = int(np.ceil(acq.um_to_px(max_shift_um))) + 1
    scene = Scene.zeros(acq, margin_px=margin)
    steps = [
        PlaceNeurons(
            density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 120.0)
        ),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        Bleaching(),
        CellOptics(),
        Composite(),
        Neuropil(amplitude=0.3),
        BrainMotion(model="walk", walk_step_um=0.4, max_shift_um=max_shift_um),
        Vignette(falloff=0.6),
        Leakage(profile="gaussian", level=0.1),
        Sensor(photons_per_unit=120.0),
    ]
    for sspec in steps:
        sspec.build(acq, rng)(scene)

    movie = scene.movie.values
    assert movie.shape == (acq.n_frames, 40, 40)  # cropped back to the sensor FOV
    np.testing.assert_array_equal(movie, np.round(movie))  # digitized counts
    assert movie.min() >= 0.0 and movie.max() <= 255.0
    assert movie.var() > 0.0
    assert scene.truth.shifts.shape == (acq.n_frames, 2)
    np.testing.assert_array_equal(scene.truth.shifts[0], [0.0, 0.0])

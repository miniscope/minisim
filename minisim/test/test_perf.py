"""Unit tests for the opt-in perf instrumentation.

Covers the :class:`~minisim.perf.PerfTracker` collector in isolation (span
accumulation, the optional-aware no-op, report formatting) and its two wiring
points: ``simulate(spec, perf=...)`` and the streaming ``_iter_count_frames``,
asserting both that the expected phases are recorded and that turning profiling
*on* does not change what the pipeline produces.
"""

import numpy as np

from minisim import (
    Acquisition,
    CellActivity,
    CellOptics,
    Composite,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    PerfTracker,
    PlaceNeurons,
    Sensor,
    Spec,
    Vignette,
    simulate,
)
from minisim.perf import measure
from minisim.steps import STEP_FOR_KIND
from minisim.video import _iter_count_frames


def _acq(n_px=48, fps=20.0, duration_s=1.0):
    return Acquisition(
        fps=fps,
        duration_s=duration_s,
        optics=Optics(magnification=8.0),
        image_sensor=ImageSensor(
            n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=8
        ),
    )


def _spec():
    """A spec exercising every instrumented phase: cells, neuropil, vignette,
    leakage, and the sensor (so digitize runs)."""
    return Spec(
        acquisition=_acq(),
        seed=3,
        steps=[
            PlaceNeurons(density_per_mm3=25000.0, soma_radius_um=4.0, depth_range_um=(0.0, 50.0)),
            CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
            CellOptics(),
            Composite(),
            Neuropil(n_components=2),
            Vignette(falloff=0.6),
            Leakage(profile="gaussian", level=0.1),
            Sensor(photons_per_unit=120.0),
        ],
    )


# --- the collector in isolation ------------------------------------------


def test_spans_accumulate_by_name():
    perf = PerfTracker()
    for _ in range(3):
        with perf.measure("phaseA"):
            pass
    with perf.measure("phaseB", domain="cell"):
        pass

    assert set(perf.spans) == {"phaseA", "phaseB"}
    assert perf.spans["phaseA"].calls == 3
    assert perf.spans["phaseB"].calls == 1
    assert perf.spans["phaseB"].domain == "cell"
    assert perf.total_seconds() >= 0.0


def test_measure_helper_is_noop_when_perf_is_none():
    # The optional-aware helper must accept None and simply run the block.
    ran = []
    with measure(None, "ignored"):
        ran.append(True)
    assert ran == [True]


def test_duration_recorded_even_on_exception():
    perf = PerfTracker()
    try:
        with perf.measure("boom"):
            raise ValueError("x")
    except ValueError:
        pass
    assert perf.spans["boom"].calls == 1


def test_report_handles_empty_and_populated():
    assert "no spans" in PerfTracker().report()

    perf = PerfTracker()
    with perf.measure("only"):
        pass
    text = perf.report()
    assert "only" in text
    assert "total" in text
    assert "100.0%" in text  # the lone span is 100% of tracked time


# --- wiring: simulate() ---------------------------------------------------


def test_simulate_records_each_step_and_finalize():
    perf = PerfTracker()
    spec = _spec()
    simulate(spec, perf=perf)

    # Spans are keyed by the executable step's name (a ClassVar), which is the
    # spec kind for most steps but not all (Composite's stage is "cells_only").
    expected = {STEP_FOR_KIND[s.kind].name for s in spec.steps} | {"finalize"}
    assert set(perf.spans) == expected
    assert all(stat.calls == 1 for stat in perf.spans.values())


def test_profiling_does_not_change_simulate_output():
    a = simulate(_spec())
    b = simulate(_spec(), perf=PerfTracker())
    np.testing.assert_array_equal(a.observed, b.observed)


# --- wiring: streaming render --------------------------------------------


def test_iter_count_frames_records_chunk_phases():
    perf = PerfTracker()
    spec = _spec()
    # Small chunk so several chunks run and the per-chunk phases accumulate calls.
    frames = list(_iter_count_frames(spec, chunk_frames=8, perf=perf))

    assert len(frames) == spec.acquisition.n_frames
    # Setup phases (once) plus the per-chunk render phases (once per chunk).
    assert "footprint_build" in perf.spans
    for phase in ("composite", "neuropil", "photon_field", "leakage", "digitize"):
        assert phase in perf.spans, phase
    n_chunks = -(-spec.acquisition.n_frames // 8)  # ceil
    assert perf.spans["digitize"].calls == n_chunks


def test_streaming_profiling_matches_unprofiled_frames():
    spec = _spec()
    plain = [f for _, f in _iter_count_frames(spec, chunk_frames=8)]
    timed = [f for _, f in _iter_count_frames(spec, chunk_frames=8, perf=PerfTracker())]
    for a, b in zip(plain, timed, strict=True):
        np.testing.assert_array_equal(a, b)

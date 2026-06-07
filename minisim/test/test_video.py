"""Streaming video export: bit-for-bit equivalence to simulate(), and file writing.

The core guarantee -- streamed frames equal ``simulate().observed`` exactly,
independent of chunk size -- is tested without any video dependency via the
``_iter_count_frames`` generator. The file-writing tests need ``mediapy`` + ffmpeg
and are skipped when ``mediapy`` is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from minisim import (
    Acquisition,
    BrainMotion,
    Bleaching,
    CellActivity,
    CellOptics,
    IlluminationProfile,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    PlaceNeurons,
    Render,
    Sensor,
    Spec,
    Vignette,
    simulate,
    simulate_video,
)
from minisim.video import _default_vmax, _iter_count_frames, _to_uint8


def _spec(duration_s=1.5, motion=True, sensor=True, n_px=48, seed=3):
    """A small but complete forward pipeline for streaming tests."""
    acq = Acquisition(
        fps=20.0,
        duration_s=duration_s,
        focal_depth_in_tissue_um="auto",
        optics=Optics(magnification=8.0, field_curvature_radius_um=600.0),
        image_sensor=ImageSensor(n_px_height=n_px, n_px_width=n_px, pixel_pitch_um=8.0, bit_depth=8),
    )
    steps = [
        PlaceNeurons(density_per_mm3=40000.0, soma_radius_um=5.0, depth_range_um=(0.0, 60.0)),
        CellActivity(active_rate_hz=5.0, tau_decay_s=0.4),
        Bleaching(),
        CellOptics(),
        Render(),
        Neuropil(n_components=2),
    ]
    if motion:
        steps.append(BrainMotion())
    steps += [IlluminationProfile(), Vignette(), Leakage()]
    if sensor:
        steps.append(Sensor(photons_per_unit=120.0))
    return Spec(acquisition=acq, seed=seed, steps=steps)


def _stream(spec, chunk):
    return np.stack([f for _, f in _iter_count_frames(spec, chunk)])


def test_streamed_frames_match_simulate_bit_for_bit():
    spec = _spec()
    observed = simulate(spec).observed
    streamed = _stream(spec, chunk=8)
    assert streamed.shape == observed.shape
    np.testing.assert_array_equal(streamed, observed)  # exact counts, not approximate


def test_stream_is_chunk_size_invariant():
    spec = _spec()
    # A chunk size that does not divide the frame count, and one that does.
    np.testing.assert_array_equal(_stream(spec, chunk=5), _stream(spec, chunk=8))


def test_stream_matches_simulate_without_motion():
    # No brain_motion -> canvas equals the sensor FOV (margin 0), so the chunk loop
    # passes the canvas straight through with no shift_and_crop. Still bit-for-bit.
    spec = _spec(motion=False)
    np.testing.assert_array_equal(_stream(spec, chunk=7), simulate(spec).observed)


def test_to_uint8_maps_range_to_full_gray():
    frame = np.array([[0.0, 128.0, 255.0]])
    out = _to_uint8(frame, 0.0, 255.0)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.array([[0, 128, 255]], dtype=np.uint8))
    # values past vmax clip to white, below vmin clip to black
    np.testing.assert_array_equal(_to_uint8(np.array([[-5.0, 999.0]]), 0.0, 255.0),
                                  np.array([[0, 255]], dtype=np.uint8))


def test_default_vmax_is_adc_range_with_sensor_else_errors():
    assert _default_vmax(_spec(sensor=True)) == 255.0  # 8-bit ADC full scale
    with pytest.raises(ValueError, match="vmax is required"):
        _default_vmax(_spec(sensor=False))


# --- file writing (needs mediapy + ffmpeg) ---------------------------------

mediapy = pytest.importorskip("mediapy")


def test_simulate_video_writes_decodable_file(tmp_path):
    spec = _spec()
    path = simulate_video(spec, tmp_path / "rec.avi", chunk_frames=8, progress=False)
    assert path.exists() and path.stat().st_size > 0
    decoded = np.asarray(mediapy.read_video(str(path)))
    # mjpeg decodes grayscale as RGB; frame count and frame size must match.
    assert decoded.shape[0] == simulate(spec).observed.shape[0]
    assert decoded.shape[1:3] == (48, 48)


def test_write_video_and_simulate_video_agree(tmp_path):
    # Recording.write_video (in-memory) and simulate_video (streamed) encode the same
    # frames, so the decoded videos are identical.
    spec = _spec()
    p_stream = simulate_video(spec, tmp_path / "stream.avi", chunk_frames=8, progress=False)
    p_mem = simulate(spec).write_video(tmp_path / "mem.avi", progress=False)
    a = np.asarray(mediapy.read_video(str(p_stream)))
    b = np.asarray(mediapy.read_video(str(p_mem)))
    np.testing.assert_array_equal(a, b)


def test_simulate_video_requires_vmax_without_sensor(tmp_path):
    with pytest.raises(ValueError, match="vmax is required"):
        simulate_video(_spec(sensor=False), tmp_path / "x.avi", progress=False)

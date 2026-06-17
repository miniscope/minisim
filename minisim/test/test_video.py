"""Streaming video export: bit-for-bit equivalence to simulate(), and file writing.

The core guarantee -- streamed frames equal ``simulate().observed`` exactly,
independent of chunk size -- is tested without any video dependency via the
``_iter_count_frames`` generator. The file-writing tests use ``cv2.VideoWriter``
(opencv is a core dependency that bundles ffmpeg), so they need no extra and no
system ffmpeg; frames are decoded back with ``cv2.VideoCapture``.
"""
from __future__ import annotations

import cv2
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
    PlaceNeurons,
    Sensor,
    Spec,
    Vasculature,
    VesselLayer,
    Vignette,
    simulate,
    simulate_video,
)
from minisim.video import _default_vmax, _iter_count_frames, _to_uint8


def _spec(
    duration_s=1.5, motion=True, sensor=True, neuropil=True, bleaching=True,
    vasculature=False, n_px=48, seed=3, exposure=120.0,
):
    """A small but complete forward pipeline for streaming tests.

    The optional effects (motion, sensor, neuropil, bleaching, vasculature) toggle so
    the bit-for-bit test can sweep a matrix of pipelines, not one fixed shape.
    """
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
    ]
    if bleaching:
        steps.append(Bleaching())
    steps += [CellOptics(), Composite()]
    if neuropil:
        steps.append(Neuropil(n_components=2))
    if vasculature:
        steps.append(Vasculature(enabled=True, layers=[VesselLayer(depth_um=20.0, n_roots=2)]))
    if motion:
        steps.append(BrainMotion())
    steps += [IlluminationProfile(), Vignette(), Leakage()]
    if sensor:
        steps.append(Sensor(photons_per_unit=exposure))
    return Spec(acquisition=acq, seed=seed, steps=steps)


def _stream(spec, chunk):
    return np.stack([f for _, f in _iter_count_frames(spec, chunk)])


@pytest.mark.parametrize(
    "motion,neuropil,bleaching,vasculature",
    [
        (True, True, True, False),  # the full pipeline
        (False, True, True, False),  # no motion -> canvas is the FOV, no shift_and_crop
        (True, False, True, False),  # no neuropil
        (True, True, False, False),  # no bleaching
        (False, False, False, False),  # the minimal count-producing chain
        (True, True, True, True),  # + vasculature (an RNG-consuming multiplicative mask)
        (False, False, False, True),  # vasculature alone over the minimal chain
    ],
)
def test_streamed_frames_match_simulate_bit_for_bit(motion, neuropil, bleaching, vasculature):
    # Sweep a matrix of pipelines (not one fixed spec): each RNG-consuming effect
    # toggled, so a draw-order desync in any of them is caught. Sensor stays on so
    # the counts are integer-valued (exact in float32) and the equality is meaningful.
    spec = _spec(motion=motion, neuropil=neuropil, bleaching=bleaching, vasculature=vasculature)
    observed = simulate(spec).observed
    streamed = _stream(spec, chunk=8)
    assert streamed.shape == observed.shape
    np.testing.assert_array_equal(streamed, observed)  # exact counts, not approximate


def test_streamed_auto_exposure_matches_simulate_bit_for_bit():
    # "auto" exposure is resolved analytically from the cells, so the streamer (which
    # never holds the whole movie) must resolve the SAME photons_per_unit as
    # simulate() and stay bit-for-bit. Sweep chunk sizes too, since the resolution
    # happens once before the chunk loop.
    spec = _spec(exposure="auto")
    observed = simulate(spec).observed
    assert observed.max() < 2 ** spec.acquisition.image_sensor.bit_depth  # not saturated
    np.testing.assert_array_equal(_stream(spec, chunk=8), observed)
    np.testing.assert_array_equal(_stream(spec, chunk=5), observed)


def test_iter_count_frames_raises_on_unreproduced_rng(monkeypatch):
    # The streamer reproduces simulate()'s RNG draws by hand; the consumes_rng guard
    # is what stops a newly RNG-consuming step from silently desyncing the stream.
    # Flip render's flag (render is a deterministic non-cell step, normally handled
    # in the chunk loop) to stand in for such a step: the walk must refuse loudly.
    from minisim.steps.tissue import CompositeStep

    monkeypatch.setattr(CompositeStep, "consumes_rng", True)
    with pytest.raises(NotImplementedError, match="consumes RNG"):
        _stream(_spec(), chunk=8)


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


# --- file writing (cv2.VideoWriter; opencv is a core dep, ffmpeg bundled) --


def _decode(path):
    """Decode a grayscale AVI back to a ``(frame, H, W)`` uint8 array via cv2.

    cv2.VideoCapture returns each frame as 3-channel BGR even for a grayscale file;
    the planes are identical for an uncompressed write, so take one.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr[..., 0] if fr.ndim == 3 else fr)
    cap.release()
    return np.array(frames)


def test_simulate_video_writes_decodable_file(tmp_path):
    spec = _spec()
    path = simulate_video(spec, tmp_path / "rec.avi", chunk_frames=8, progress=False)
    assert path.exists() and path.stat().st_size > 0
    decoded = _decode(path)
    assert decoded.shape[0] == simulate(spec).observed.shape[0]
    assert decoded.shape[1:3] == (48, 48)


def test_default_codec_roundtrips_counts_losslessly(tmp_path):
    # The default (Y800, uncompressed gray) must preserve the exact 8-bit counts: a
    # re-decoded frame equals simulate().observed bit-for-bit, with no lossy blocking.
    # This is the guard against silently regressing to a lossy default like MJPG.
    spec = _spec()
    observed = simulate(spec).observed  # counts are well within 8-bit (vmax=255)
    path = simulate_video(spec, tmp_path / "rec.avi", chunk_frames=8, progress=False)
    np.testing.assert_array_equal(_decode(path), observed.astype(np.uint8))


def test_write_video_and_simulate_video_agree(tmp_path):
    # Recording.write_video (in-memory) and simulate_video (streamed) encode the same
    # frames, so the decoded videos are identical.
    spec = _spec()
    p_stream = simulate_video(spec, tmp_path / "stream.avi", chunk_frames=8, progress=False)
    p_mem = simulate(spec).write_video(tmp_path / "mem.avi", progress=False)
    np.testing.assert_array_equal(_decode(p_stream), _decode(p_mem))


def test_simulate_video_requires_vmax_without_sensor(tmp_path):
    with pytest.raises(ValueError, match="vmax is required"):
        simulate_video(_spec(sensor=False), tmp_path / "x.avi", progress=False)

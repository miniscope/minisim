"""Stream a recording straight to a video file, one frame-chunk at a time.

:func:`~minisim.simulate.simulate` materializes the entire ``(frame, height, width)``
movie in memory — gigabytes for a long recording, growing linearly with duration.
:func:`simulate_video` instead computes the **duration-independent** and **small
per-frame** state once (footprints, the static optical fields, the motion
trajectory, the neuropil components, each cell's emission trace), then renders and
digitizes the movie in frame chunks, writing each chunk to disk and discarding it.
Peak memory is bounded by the chunk size, so arbitrarily long recordings can be
written.

The streamed counts are **bit-for-bit identical** to ``simulate().observed``: the
same RNG draws are consumed in the same order (the cell-domain steps run exactly as
``simulate`` runs them, and the neuropil/motion generators are the very functions
their steps call), and the sensor digitizes frame-by-frame so a chunk boundary can
never shift a draw. :func:`_iter_count_frames` is the shared generator; the public
:func:`simulate_video` only adds uint8 encoding, the video writer, and a progress bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from minisim.scene import Scene
from minisim.simulate import _motion_margin_px, inject_cell_deps, sensor_context
from minisim.spec import Spec
from minisim.steps.motion import brain_motion_shifts, shift_and_crop
from minisim.steps.sensor import leakage_field
from minisim.steps.tissue import neuropil_components

# Frames rendered+digitized per chunk. ~64 at 608x608 float64 is ~150 MB of working
# movie — small versus a full recording, large enough that per-chunk overhead is
# negligible. Bigger trades memory for slightly less Python overhead.
_DEFAULT_CHUNK_FRAMES = 64


def simulate_video(
    spec: Spec,
    path: str | Path,
    *,
    chunk_frames: int | None = None,
    fps: float | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    codec: str = "rawvideo",
    progress: bool = True,
) -> Path:
    """Simulate ``spec`` straight to a grayscale video at ``path``, streaming to disk.

    Renders and digitizes the recording in frame chunks and writes each to an
    incrementally-opened ``mediapy`` video, so the whole movie is never held in
    memory (unlike ``simulate(spec).observed``, which is). The produced counts match
    ``simulate(spec).observed`` exactly.

    ``vmax`` sets the count mapped to white (``vmin`` → black); it defaults to the
    sensor's full ADC range (``2**bit_depth - 1``) when the spec has a ``sensor``
    step, so the file faithfully shows the true ADC utilization (a dim, honest
    frame). For a sensorless (continuous-intensity) spec there is no natural scale,
    so ``vmax`` must be given. ``codec`` defaults to ``"rawvideo"``: uncompressed
    8-bit grayscale (fourcc ``Y800``), so the file carries the exact counts with no
    compression artifacts and opens directly in ImageJ/Fiji. It is therefore large
    (uncompressed: ~``n_frames * H * W`` bytes). For a small file pass ``"mjpeg"``
    (lossy, but Fiji-readable); ``"png"``/``"ffv1"`` are smaller and lossless but
    ffmpeg tags them ``MPNG``/``FFV1``, which Fiji's built-in AVI reader rejects.
    Requires the ``mediapy`` extra (``pip install 'minisim[notebook]'``). Returns
    ``path``.
    """
    acq = spec.acquisition
    fps = float(fps if fps is not None else acq.fps)
    fov = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
    chunk = int(chunk_frames or _DEFAULT_CHUNK_FRAMES)
    if vmax is None:
        vmax = _default_vmax(spec)
    media = _import_mediapy()
    path = Path(path)

    frames = (frame for _, frame in _iter_count_frames(spec, chunk))
    return _write_gray_video(
        media, frames, acq.n_frames, path, fov, fps, vmin, vmax, codec, progress
    )


def _write_gray_video(media, frames, total, path, fov, fps, vmin, vmax, codec, progress):
    """Encode an iterable of float ``(H, W)`` frames to a grayscale video at ``path``.

    The shared write tail behind both :func:`simulate_video` (streaming the
    chunked generator) and :meth:`minisim.recording.Recording.write_video`
    (replaying an in-memory movie): open the single-plane writer, map each frame
    to uint8 over ``[vmin, vmax]``, and drive a progress bar over ``total`` frames.
    """
    bar = _ProgressBar(total, progress, f"writing {path.name}")
    try:
        with _open_gray_writer(media, path, fov, fps, codec) as writer:
            for frame in frames:
                writer.add_image(_to_uint8(frame, vmin, vmax))
                bar.update()
    finally:
        bar.close()
    return path


def _iter_count_frames(spec: Spec, chunk_frames: int):
    """Yield ``(frame_index, frame)`` count arrays, frame-for-frame equal to
    ``simulate(spec).observed`` — without ever materializing the full movie.

    Runs the cell-domain steps exactly as ``simulate`` does (same RNG draws, same
    "auto"-focus injection so footprints match), pulls the neuropil/motion/field
    artifacts via the same generators their steps use, then renders + digitizes the
    movie in chunks of ``chunk_frames``. The shared core behind :func:`simulate_video`
    and the bit-for-bit test.
    """
    acq = spec.acquisition
    rng = np.random.default_rng(spec.seed)
    scene = Scene.zeros(acq, rng, margin_px=_motion_margin_px(spec, acq))
    n_frames = acq.n_frames
    canvas_shape = scene.canvas_shape
    fov = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
    sensor_hw = acq.image_sensor

    # Same up-front lookups as simulate(): the photon field feeds both "auto" focus
    # (so the resolved plane, and thus the footprints, match) and the later field
    # application; bleaching needs the illumination profile.
    illumination, sensor_spec, photon_field = sensor_context(spec, acq)

    neuropil = None  # (amplitude, spatial (k,Hc,Wc), temporal (k,frame))
    shifts = None  # (frame, 2) px, or None when there is no motion
    leak = None  # additive FOV field, or None

    # Walk the steps in order, consuming the RNG exactly as simulate() would: run the
    # cell-domain steps (they fill scene.cells), and for the RNG-consuming non-cell
    # steps call the same generators their steps call. Render/vasculature/illumination/
    # vignette/leakage/sensor draw no RNG, so building their (deterministic) artifacts
    # here -- or skipping them -- never desyncs the stream.
    for step_spec in spec.steps:
        kind = step_spec.kind
        if step_spec.domain == "cell":
            step = step_spec.build(acq, rng)
            inject_cell_deps(step, illumination, sensor_spec, photon_field)
            step(scene)
        elif kind == "neuropil":
            spatial, temporal, _ = neuropil_components(
                step_spec, acq, scene.cells, canvas_shape, n_frames, rng
            )
            neuropil = (step_spec.amplitude, spatial, temporal)
        elif kind == "brain_motion":
            shifts = brain_motion_shifts(step_spec, acq, n_frames, rng)
        elif kind == "leakage":
            leak = leakage_field(step_spec, acq, fov)
        # render / vasculature / illumination_profile / vignette / sensor: no RNG,
        # handled in the chunk loop (or already folded into photon_field).

    # The cells are now fully populated; stack the footprints and per-cell emission
    # (clean calcium dimmed by bleaching when present), exactly as RenderStep does.
    footprints, emissions = [], []
    for cell in scene.cells:
        fp = cell.footprint_observed if cell.footprint_observed is not None else cell.footprint_planted
        if fp is None or cell.trace is None:
            continue
        footprints.append(fp)
        emissions.append(cell.trace if cell.bleach is None else cell.trace * cell.bleach)
    A = np.stack(footprints) if footprints else None  # (unit, Hc, Wc)
    CB = np.stack(emissions) if emissions else None  # (unit, frame)
    ppu = sensor_spec.photons_per_unit if sensor_spec is not None else None

    for t0 in range(0, n_frames, chunk_frames):
        t1 = min(t0 + chunk_frames, n_frames)
        canvas = np.zeros((t1 - t0, canvas_shape[0], canvas_shape[1]))
        if A is not None:
            canvas += np.tensordot(CB[:, t0:t1], A, axes=([0], [0]))
        if neuropil is not None:
            amp, spatial, temporal = neuropil
            canvas += amp * np.tensordot(
                temporal[:, t0:t1], spatial, axes=([0], [0])
            ) / spatial.shape[0]
        # Motion crop (shift_and_crop) when there is motion; otherwise the canvas is
        # already the sensor FOV (no margin), so it passes through unchanged.
        frames = shift_and_crop(canvas, shifts[t0:t1], fov) if shifts is not None else canvas
        if photon_field is not None:
            frames = frames * photon_field
        if leak is not None:
            frames = frames + leak
        if ppu is not None:
            # Per-frame digitization on the shared rng -- identical to SensorStep, and
            # independent of the chunk boundaries (each frame draws its own noise).
            for i in range(frames.shape[0]):
                frames[i] = sensor_hw.photons_to_counts(
                    np.clip(frames[i] * ppu, 0.0, None), rng
                )
        for i in range(t1 - t0):
            yield t0 + i, frames[i]


def _open_gray_writer(media, path, shape, fps, codec):
    """A single-plane grayscale ``mediapy`` writer.

    Feeds the ``(H, W)`` uint8 frames as one luma plane and encodes a single plane
    (``encoded_format="gray"`` -- no RGB promotion, no chroma subsampling), so an
    8-bit count survives a lossless codec (``ffv1``) bit-for-bit. The earlier default
    (RGB roundtrip + ``mjpeg``) both quantized the data and added a small rounding
    error of its own.
    """
    return media.VideoWriter(
        str(path), shape=shape, fps=fps, codec=codec,
        input_format="gray", encoded_format="gray",
    )


def _default_vmax(spec: Spec) -> float:
    """White point for uint8 mapping: the sensor's full ADC range, or an error.

    A digitized recording maps the full ``[0, 2**bit_depth - 1]`` count range to
    ``[0, 255]`` (faithful, so saturation reads as white). A sensorless spec has no
    natural scale, so the caller must pass ``vmax`` explicitly.
    """
    if any(s.kind == "sensor" for s in spec.steps):
        return float(2 ** spec.acquisition.image_sensor.bit_depth - 1)
    raise ValueError(
        "vmax is required for a spec with no 'sensor' step (the movie is continuous "
        "intensity with no natural white point); pass an explicit vmax."
    )


def _to_uint8(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a single ``(H, W)`` frame to ``uint8`` grayscale over ``[vmin, vmax]``."""
    span = (vmax - vmin) or 1.0
    scaled = np.clip((frame - vmin) / span, 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def _import_mediapy():
    try:
        import mediapy

        return mediapy
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "writing video requires the 'mediapy' package (and ffmpeg on PATH); "
            "install the notebook extra: pip install 'minisim[notebook]'."
        ) from exc


class _ProgressBar:
    """Frame-write progress: a ``tqdm`` bar when available, else periodic stderr lines.

    Falls back to a no-op when ``enabled`` is False, and to coarse printed updates
    (≈5% steps) when ``tqdm`` is not installed, so the streaming writer shows
    progress on a minutes-long write without taking a hard dependency on ``tqdm``.
    """

    def __init__(self, total: int, enabled: bool, desc: str) -> None:
        self.total, self.desc, self.n = total, desc, 0
        self._tqdm = None
        self._step = max(total // 20, 1)
        if not enabled:
            return
        try:
            from tqdm.auto import tqdm

            self._tqdm = tqdm(total=total, desc=desc, unit="frame")
        except ImportError:
            print(f"{desc}: 0/{total} frames", file=sys.stderr, flush=True)

    def update(self) -> None:
        self.n += 1
        if self._tqdm is not None:
            self._tqdm.update(1)
        elif self.total and (self.n % self._step == 0 or self.n == self.total):
            pct = 100.0 * self.n / self.total
            print(f"{self.desc}: {self.n}/{self.total} frames ({pct:.0f}%)",
                  file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()

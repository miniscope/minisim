"""Stream a recording straight to a video file, one frame-chunk at a time.

:func:`~minisim.simulate.simulate` materializes the entire ``(frame, height, width)``
movie in memory - gigabytes for a long recording, growing linearly with duration.
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

The file is written with ``cv2.VideoWriter`` (opencv is a core dependency, bundling
its own ffmpeg), so writing a video needs neither the ``mediapy`` extra nor a system
ffmpeg. ``mediapy`` remains only for in-notebook *display* (the training notebooks),
not for writing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from minisim.footprint import RENDER_DTYPE, stack_dense
from minisim.perf import PerfTracker, measure
from minisim.scene import Scene
from minisim.simulate import _motion_margin_px, build_context
from minisim.spec import Spec
from minisim.steps import STEP_FOR_KIND
from minisim.steps.motion import brain_motion_shifts, shift_and_crop
from minisim.steps.sensor import leakage_field
from minisim.steps.tissue import (
    neuropil_components,
    vasculature_focal,
    vasculature_mask_field,
)

# Frames rendered+digitized per chunk. The dense footprint stack `A` is re-read by
# the composite contraction once per chunk, so fewer/larger chunks cut that memory
# traffic; 128 at 608x608 float64 is ~285 MB of working movie - still small versus a
# full recording, and it halves the A re-reads versus the old default of 64. Bigger
# keeps trading peak memory for less per-chunk overhead with diminishing returns.
_DEFAULT_CHUNK_FRAMES = 128


def simulate_video(
    spec: Spec,
    path: str | Path,
    *,
    chunk_frames: int | None = None,
    fps: float | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    codec: str = "Y800",
    progress: bool = True,
    perf: PerfTracker | None = None,
) -> Path:
    """Simulate ``spec`` straight to a grayscale video at ``path``, streaming to disk.

    Renders and digitizes the recording in frame chunks and writes each to an
    incrementally-opened ``cv2.VideoWriter``, so the whole movie is never held in
    memory (unlike ``simulate(spec).observed``, which is). The produced counts match
    ``simulate(spec).observed`` exactly.

    ``vmax`` sets the count mapped to white (``vmin`` → black); it defaults to the
    sensor's full ADC range (``2**bit_depth - 1``) when the spec has a ``sensor``
    step, so the file faithfully shows the true ADC utilization (a dim, honest
    frame). For a sensorless (continuous-intensity) spec there is no natural scale,
    so ``vmax`` must be given. ``codec`` is a 4-character opencv fourcc; it defaults
    to ``"Y800"``: uncompressed 8-bit grayscale, so the file carries the exact counts
    with no compression artifacts (large: ~``n_frames * H * W`` bytes). For a small
    lossy file pass ``"MJPG"``. Writing uses opencv's bundled ffmpeg, so no system
    ffmpeg or ``mediapy`` extra is needed. Returns ``path``.

    Pass a :class:`~minisim.perf.PerfTracker` as ``perf`` to record where the
    streamed write spends its time: the one-off ``setup`` (cell steps + footprint
    build), each per-chunk render sub-phase (``composite``, ``neuropil``,
    ``motion_crop``, ``photon_field``, ``leakage``, ``digitize``), and the
    ``encode+write`` tail. It is a no-op when ``None`` (the default).
    """
    acq = spec.acquisition
    fps = float(fps if fps is not None else acq.fps)
    fov = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
    chunk = int(chunk_frames or _DEFAULT_CHUNK_FRAMES)
    if vmax is None:
        vmax = _default_vmax(spec)
    path = Path(path)

    frames = (frame for _, frame in _iter_count_frames(spec, chunk, perf=perf))
    return _write_gray_video(
        frames, acq.n_frames, path, fov, fps, vmin, vmax, codec, progress, perf
    )


def _write_gray_video(frames, total, path, fov, fps, vmin, vmax, codec, progress, perf=None):
    """Encode an iterable of float ``(H, W)`` frames to a grayscale video at ``path``.

    The shared write tail behind both :func:`simulate_video` (streaming the
    chunked generator) and :meth:`minisim.recording.Recording.write_video`
    (replaying an in-memory movie): open a single-plane ``cv2.VideoWriter``
    (``isColor=False``), map each frame to uint8 over ``[vmin, vmax]``, and drive a
    progress bar over ``total`` frames.
    """
    writer = _open_gray_writer(path, fov, fps, codec)
    bar = _ProgressBar(total, progress, f"writing {path.name}")
    try:
        for frame in frames:
            with measure(perf, "encode+write"):
                writer.write(_to_uint8(frame, vmin, vmax))
            bar.update()
    finally:
        writer.release()
        bar.close()
    return path


def _iter_count_frames(spec: Spec, chunk_frames: int, *, perf: PerfTracker | None = None):
    """Yield ``(frame_index, frame)`` count arrays, frame-for-frame equal to
    ``simulate(spec).observed`` - without ever materializing the full movie.

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

    # Same resolved context as simulate(): the photon field feeds both "auto" focus
    # (so the resolved plane, and thus the footprints, match) via each cell step's
    # prepare(), and the later per-frame field application below.
    context = build_context(spec, acq)
    sensor_spec = context.sensor_spec
    photon_field = context.photon_field

    neuropil = None  # (amplitude, spatial (k,Hc,Wc), temporal (k,frame))
    shifts = None  # (frame, 2) px, or None when there is no motion
    leak = None  # additive FOV field, or None
    vasc_mask = None  # static (Hc,Wc) vessel transmission mask, or None when off

    # Walk the steps in order, consuming the RNG exactly as simulate() would: run the
    # cell-domain steps (they fill scene.cells), and for the RNG-consuming non-cell
    # steps call the same generators their steps call. Every RNG-consuming step must
    # be reproduced here in draw order (or, for the sensor, in the chunk loop below),
    # or the stream silently desyncs from simulate(). The else-branch enforces that
    # against Step.consumes_rng: a new RNG-consuming step added to the catalog without
    # being taught here raises loudly instead of corrupting the stream. Composite,
    # illumination_profile, and vignette draw no RNG (their fields are deterministic,
    # already folded into photon_field), so they are applied in the chunk loop and
    # ignored here. Vasculature *does* draw (vessel-tree growth), so it has its own
    # branch above that reproduces the draws and builds the mask.
    for step_spec in spec.steps:
        # Branch on step_spec.kind (not a local copy) so the discriminated union
        # narrows step_spec to the concrete spec type inside each branch.
        if step_spec.domain == "cell":
            step = step_spec.build(acq, rng)
            step.prepare(context)
            with measure(perf, step.name, domain="cell"):
                step(scene)
        elif step_spec.kind == "neuropil":
            with measure(perf, "neuropil_setup"):
                spatial, temporal, _ = neuropil_components(
                    step_spec, acq, scene.cells, canvas_shape, n_frames, rng
                )
            neuropil = (step_spec.amplitude, spatial, temporal)
        elif step_spec.kind == "vasculature":
            # Mirror VasculatureStep's guard exactly: when off it draws no RNG, so
            # the stream must not draw either. When on, build the same mask from the
            # same draws (cells + resolved focal are ready: the cell steps ran above).
            if step_spec.enabled and step_spec.layers:
                with measure(perf, "vasculature_setup"):
                    focal = vasculature_focal(scene, acq)
                    vasc_mask = vasculature_mask_field(
                        step_spec, acq, canvas_shape, focal, rng
                    )
        elif step_spec.kind == "brain_motion":
            with measure(perf, "motion_setup"):
                shifts = brain_motion_shifts(step_spec, acq, n_frames, rng)
        elif step_spec.kind == "leakage":
            leak = leakage_field(step_spec, acq, fov)
        elif step_spec.kind == "sensor":
            pass  # its per-frame RNG draws happen in the chunk loop below, in order
        elif STEP_FOR_KIND[step_spec.kind].consumes_rng:
            raise NotImplementedError(
                f"step {step_spec.kind!r} consumes RNG but the streaming video writer "
                "does not reproduce its draws, so simulate_video() would silently "
                "desync from simulate(). Teach minisim.video._iter_count_frames to "
                "replay this step's RNG draws in order (and extend the bit-for-bit "
                "test) before adding it to a streamed spec."
            )
        # else: a deterministic non-cell step (composite / illumination_profile /
        # vignette) -- no RNG, applied in the chunk loop.

    # The cells are now fully populated; rebuild the dense footprint stack (sparse
    # in storage) and per-cell emission (clean calcium dimmed by bleaching when
    # present), exactly as CompositeStep does, so the chunked render is bit-for-bit.
    with measure(perf, "footprint_build"):
        footprints, emissions = [], []
        for cell in scene.cells:
            fp = cell.observed_footprint()  # regenerated from planted (see CompositeStep)
            if fp is None or cell.trace is None:
                continue
            footprints.append(fp)
            emissions.append(cell.trace if cell.bleach is None else cell.trace * cell.bleach)
        # A and CB are float32 (RENDER_DTYPE), matching CompositeStep, so the
        # per-chunk contraction is single-precision and bit-identical to simulate().
        A = stack_dense(footprints, canvas_shape) if footprints else None  # (unit, Hc, Wc)
        CB = np.stack(emissions).astype(RENDER_DTYPE) if emissions else None  # (unit, frame)
    ppu = sensor_spec.photons_per_unit if sensor_spec is not None else None

    for t0 in range(0, n_frames, chunk_frames):
        t1 = min(t0 + chunk_frames, n_frames)
        with measure(perf, "composite"):
            canvas = np.zeros((t1 - t0, canvas_shape[0], canvas_shape[1]))
            if A is not None and CB is not None:
                canvas += np.tensordot(CB[:, t0:t1], A, axes=([0], [0]))
        if neuropil is not None:
            with measure(perf, "neuropil"):
                amp, spatial, temporal = neuropil
                canvas += amp * np.tensordot(
                    temporal[:, t0:t1], spatial, axes=([0], [0])
                ) / spatial.shape[0]
        if vasc_mask is not None:
            # Multiplicative vessel shadow, in the brain frame before the motion crop
            # (broadcasts the static (Hc,Wc) mask over the chunk's frames) - exactly
            # where VasculatureStep applies it, so it rides motion with the tissue.
            with measure(perf, "vasculature"):
                canvas *= vasc_mask
        # Motion crop (shift_and_crop) when there is motion; otherwise the canvas is
        # already the sensor FOV (no margin), so it passes through unchanged.
        if shifts is not None:
            with measure(perf, "motion_crop"):
                frames = shift_and_crop(canvas, shifts[t0:t1], fov)
        else:
            frames = canvas
        if photon_field is not None:
            with measure(perf, "photon_field"):
                frames = frames * photon_field
        if leak is not None:
            with measure(perf, "leakage"):
                frames = frames + leak
        if ppu is not None:
            # Per-frame digitization on the shared rng -- identical to SensorStep, and
            # independent of the chunk boundaries (each frame draws its own noise).
            with measure(perf, "digitize"):
                for i in range(frames.shape[0]):
                    frames[i] = sensor_hw.photons_to_counts(
                        np.clip(frames[i] * ppu, 0.0, None), rng
                    )
        for i in range(t1 - t0):
            yield t0 + i, frames[i]


def _open_gray_writer(path, fov, fps, codec):
    """A single-plane grayscale ``cv2.VideoWriter`` for the ``fov = (H, W)`` movie.

    Opened with ``isColor=False`` so each ``(H, W)`` uint8 frame is written as one
    grayscale plane (no RGB promotion / chroma subsampling), and with the ``codec``
    fourcc - the ``"Y800"`` default is uncompressed, so an 8-bit count survives
    bit-for-bit. ``VideoWriter`` reports failure by *not opening* rather than raising,
    so check ``isOpened`` and surface a clear error (a missing codec, an unwritable
    path) instead of silently producing an empty file.
    """
    h, w = fov
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*codec), fps, (w, h), isColor=False
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open {path!s} with fourcc {codec!r} at "
            f"{w}x{h}. The codec may be unavailable in this opencv build, or the "
            "path unwritable."
        )
    return writer


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

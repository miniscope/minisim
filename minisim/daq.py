"""Write a recording as a Miniscope-DAQ output folder, ground truth alongside.

The Miniscope DAQ software (Miniscope-DAQ-QT-Software) saves one folder per
recording::

    <base>/
        metaData.json            experiment level: start time, device list, ...
        <DeviceName>/
            metaData.json        device level: ROI, compression, frameRate, gain, ...
            timeStamps.csv       Frame Number,Time Stamp (ms),Buffer Index
            0.avi 1.avi ...      framesPerFile frames each, raw 8-bit GREY

:func:`write_miniscope_daq` produces the same layout from a minisim
:class:`~minisim.Spec` (streamed, never holding the movie) or an in-memory
:class:`~minisim.Recording`, so the folder can be handed straight to anything that
reads real DAQ output. The simulator's exact ground truth is saved in a clearly
separate ``ground_truth/`` directory *next to* the device folder (not inside it, so a
reader scanning the base folder or the device folder never trips over it)::

    <base>/ground_truth/
        spec.json                the full Spec that produced the recording
        ground_truth.zarr/       the GroundTruth (sparse footprints), reload with
                                 :func:`minisim.load_ground_truth`
        arrays.npz               dense arrays for readers without minisim
                                 (A_observed, A_planted, C, S, shifts, ...)

**What differs from a real DAQ folder.** The AVI header carries the true frame rate
(see :mod:`minisim.avi`); timestamps are the nominal frame times plus optional
Gaussian jitter (the DAQ's are wall-clock based, with a small negative first stamp
in the sample session; set ``start_offset_ms`` to mimic it) and ``Buffer Index`` is
always 0 (the software never fell behind); the frame counter has no gaps. Extra
``metaData.json`` keys beyond the DAQ's are not added except a ``"minisim"`` block
in the experiment-level file recording the generator version and spec hash.

**Streaming and memory.** The movie is rendered and digitized in frame chunks and
each frame written straight to the current AVI file, so peak memory is bounded by
``chunk_frames`` regardless of duration. The *ground truth* is not streamed: the
traces ``C``/``S`` are ``(unit, frame)`` float64 and are held in memory while the
movie renders, as are the neuropil components; footprints stay sparse. For
``n_units * n_frames * 8 B * ~2`` that is a few hundred MB for a thousand cells over
an hour at 20 fps. ``arrays.npz`` additionally materializes the dense footprints
``(unit, H, W)`` (float32) at the end; pass ``write_npz=False`` for very large
populations and read the zarr instead (footprints regenerate on demand).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from minisim.avi import AviWriter
from minisim.recording import (
    _GT_OPTIONAL,
    GroundTruth,
    Recording,
    save_ground_truth,
)
from minisim.spec import Spec
from minisim.video import (
    _DEFAULT_CHUNK_FRAMES,
    _default_vmax,
    _iter_count_frames,
    _ProgressBar,
    _to_uint8,
)

__all__ = ["DaqFolder", "daq_timestamps_ms", "write_miniscope_daq"]

GROUND_TRUTH_DIRNAME = "ground_truth"
_DEFAULT_START = datetime(2000, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class DaqFolder:
    """Where :func:`write_miniscope_daq` put things."""

    root: Path
    device_dir: Path
    avi_files: tuple[Path, ...]
    timestamps_csv: Path
    device_metadata: Path
    experiment_metadata: Path
    ground_truth_dir: Path | None
    n_frames: int


def daq_timestamps_ms(
    n_frames: int,
    fps: float,
    *,
    jitter_ms: float = 0.0,
    start_offset_ms: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Integer millisecond frame timestamps like a ``timeStamps.csv`` column.

    ``start_offset_ms + i * 1000 / fps`` plus iid Gaussian jitter of standard
    deviation ``jitter_ms`` (drawn from its own generator seeded by ``seed``, so it
    never perturbs the simulation's RNG), rounded to whole milliseconds and forced
    strictly increasing (a real DAQ never repeats or reverses a stamp).
    """
    if n_frames < 0:
        raise ValueError("n_frames must be >= 0.")
    t = start_offset_ms + np.arange(n_frames) * (1000.0 / fps)
    if jitter_ms > 0:
        t = t + np.random.default_rng([int(seed or 0), 0x7A5]).normal(
            0.0, jitter_ms, n_frames
        )
    ts = np.rint(t).astype(np.int64)
    for i in range(1, n_frames):  # strictly increasing, in place and O(n)
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1
    return ts


def write_miniscope_daq(
    source: Spec | Recording,
    out_dir: str | Path,
    *,
    frames_per_file: int = 1000,
    device_name: str = "Miniscope",
    device_type: str = "Miniscope_V4_BNO",
    jitter_ms: float = 0.0,
    start_offset_ms: float = 0.0,
    timestamp_seed: int | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    ground_truth: bool = True,
    write_npz: bool = True,
    device_metadata: dict | None = None,
    experiment_metadata: dict | None = None,
    start_time: datetime | None = None,
    chunk_frames: int | None = None,
    progress: bool = True,
    overwrite: bool = False,
) -> DaqFolder:
    """Write ``source`` as a Miniscope DAQ output folder under ``out_dir``.

    ``source`` is either a :class:`~minisim.Spec` (rendered and **streamed** to disk
    frame by frame, bounded memory; the counts equal ``simulate(spec).observed``) or
    an already-simulated :class:`~minisim.Recording` (its in-memory movie is written).

    Counts become 8-bit grey exactly as :func:`~minisim.simulate_video` does:
    ``vmax`` defaults to the sensor's full ADC range, so an 8-bit sensor's counts are
    written verbatim. A sensorless spec needs an explicit ``vmax``.

    Parameters
    ----------
    frames_per_file
        Frames per ``N.avi`` (the DAQ's ``framesPerFile``, default 1000); the last
        file holds the remainder.
    device_name, device_type
        The device folder name and ``deviceType`` field (the DAQ's defaults).
    jitter_ms, start_offset_ms, timestamp_seed
        Timestamp model, see :func:`daq_timestamps_ms`; ``timestamp_seed`` defaults
        to the spec's seed.
    ground_truth, write_npz
        Save the ground truth under ``ground_truth/`` (default) and, within it, the
        dense ``arrays.npz`` (default).
    device_metadata, experiment_metadata
        Extra / overriding keys merged into the device and experiment
        ``metaData.json``.
    start_time
        Recording start (naive, treated as UTC for ``msecSinceEpoch``); a fixed
        constant by default so the output is deterministic.
    chunk_frames
        Frames rendered per chunk when streaming a ``Spec``.
    overwrite
        Replace an existing device folder / ground-truth directory at ``out_dir``
        (otherwise ``FileExistsError``).
    """
    if frames_per_file < 1:
        raise ValueError(f"frames_per_file must be >= 1, got {frames_per_file}.")
    if isinstance(source, Recording):
        spec = source.spec
        # numpydantic's Shape[...] annotation is not statically an ndarray (see
        # [tool.pyright].exclude), so view it as one for the type checker.
        observed = np.asarray(source.observed)
        if observed.shape[0] == 0:
            raise ValueError("recording has no frames to write (observed is empty).")
    elif isinstance(source, Spec):
        spec = source
    else:
        raise TypeError(f"source must be a Spec or Recording, got {type(source)!r}.")

    acq = spec.acquisition
    fps = float(acq.fps)
    n_frames = acq.n_frames
    height, width = acq.image_sensor.n_px_height, acq.image_sensor.n_px_width
    if vmax is None:
        vmax = _default_vmax(spec)

    root = Path(out_dir).resolve()
    device_dir = root / device_name
    gt_dir = root / GROUND_TRUTH_DIRNAME
    for d in (device_dir, gt_dir):
        if d.exists():
            if not overwrite:
                raise FileExistsError(f"{d} already exists; pass overwrite=True.")
            shutil.rmtree(d)
    device_dir.mkdir(parents=True)

    sink: dict = {}
    if isinstance(source, Recording):
        frames = (observed[i] for i in range(n_frames))
        gt: GroundTruth | None = source.ground_truth
    else:
        frames = (
            f
            for _, f in _iter_count_frames(
                spec,
                int(chunk_frames or _DEFAULT_CHUNK_FRAMES),
                truth_sink=sink if ground_truth else None,
            )
        )
        gt = None

    bar = _ProgressBar(n_frames, progress, f"writing {root.name}")
    avi_files: list[Path] = []
    writer: AviWriter | None = None
    try:
        for i, frame in enumerate(frames):
            if i % frames_per_file == 0:
                if writer is not None:
                    writer.close()
                path = device_dir / f"{i // frames_per_file}.avi"
                avi_files.append(path)
                writer = AviWriter(path, width, height, fps)
            assert writer is not None
            writer.write(_to_uint8(np.asarray(frame, dtype=float), vmin, vmax))
            bar.update()
        if writer is not None:
            writer.close()
    finally:
        bar.close()
        if writer is not None:
            writer.abort()  # no-op once closed; releases the handle on error

    # --- timeStamps.csv -----------------------------------------------------
    ts = daq_timestamps_ms(
        n_frames,
        fps,
        jitter_ms=jitter_ms,
        start_offset_ms=start_offset_ms,
        seed=spec.seed if timestamp_seed is None else timestamp_seed,
    )
    csv_path = device_dir / "timeStamps.csv"
    with open(csv_path, "w", newline="") as fh:
        fh.write("Frame Number,Time Stamp (ms),Buffer Index\n")
        fh.writelines(f"{i},{t},0\n" for i, t in enumerate(ts.tolist()))

    # --- metaData.json (device + experiment) --------------------------------
    dev_meta = {
        "ROI": {"height": height, "leftEdge": 0, "topEdge": 0, "width": width},
        "compression": "GREY",
        "deviceDirectory": str(device_dir),
        "deviceID": 0,
        "deviceName": device_name,
        "deviceType": device_type,
        "ewl": 0,
        "frameRate": f"{fps:g}FPS",
        "framesPerFile": int(frames_per_file),
        "gain": 3.5,
        "led0": 6,
    }
    dev_meta.update(device_metadata or {})
    dev_meta_path = device_dir / "metaData.json"
    _dump_json(dev_meta, dev_meta_path)

    st = start_time or _DEFAULT_START
    st_ms = int(st.replace(tzinfo=timezone.utc).timestamp() * 1000)
    exp_meta = {
        "animalName": "minisim",
        "baseDirectory": str(root),
        "cameras": [],
        "experimentName": "minisim",
        "miniscopes": [device_name],
        "minisim": {
            "version": _minisim_version(),
            "spec_cache_key": spec.cache_key(),
            "seed": spec.seed,
            "ground_truth_dir": GROUND_TRUTH_DIRNAME if ground_truth else None,
        },
        "recordingStartTime": {
            "day": st.day,
            "hour": st.hour,
            "minute": st.minute,
            "month": st.month,
            "msec": st.microsecond // 1000,
            "msecSinceEpoch": st_ms,
            "second": st.second,
            "year": st.year,
        },
        "researcherName": "minisim",
    }
    exp_meta.update(experiment_metadata or {})
    exp_meta_path = root / "metaData.json"
    _dump_json(exp_meta, exp_meta_path)

    # --- ground truth -------------------------------------------------------
    gt_out: Path | None = None
    if ground_truth:
        gt = gt if gt is not None else sink["ground_truth"]
        gt_out = gt_dir
        gt_dir.mkdir(parents=True)
        (gt_dir / "spec.json").write_text(spec.model_dump_json(indent=2))
        save_ground_truth(gt, spec, gt_dir / "ground_truth.zarr")
        if write_npz:
            _write_npz(gt, spec, gt_dir / "arrays.npz")

    return DaqFolder(
        root=root,
        device_dir=device_dir,
        avi_files=tuple(avi_files),
        timestamps_csv=csv_path,
        device_metadata=dev_meta_path,
        experiment_metadata=exp_meta_path,
        ground_truth_dir=gt_out,
        n_frames=n_frames,
    )


def _minisim_version() -> str:
    try:
        return metadata.version("minisim")
    except metadata.PackageNotFoundError:
        return "unknown"


def _dump_json(obj: dict, path: Path) -> None:
    # sorted keys + 4-space indent: what Qt's QJsonDocument::toJson writes.
    path.write_text(json.dumps(obj, indent=4, sort_keys=True) + "\n")


def _write_npz(gt: GroundTruth, spec: Spec, path: Path) -> None:
    """Dense ground-truth arrays for readers that don't use minisim.

    Footprints are float32 ``(unit, H, W)`` over the sensor FOV; ``C``/``S`` are
    ``(unit, frame)``; ``shifts`` is ``(frame, 2)`` ``(dy, dx)`` in pixels (positive
    moves tissue down/right; present only with a ``brain_motion`` step) and, with
    ``BrainMotion(nonrigid=True)``, ``nonrigid_field`` ``(frame, gy, gx, 2)`` px at the
    FOV-pixel nodes ``nonrigid_nodes_y`` / ``nonrigid_nodes_x`` (cubic-spline
    interpolate between nodes). Optional
    per-effect fields are included when the recording has them.
    """
    arrays: dict[str, Any] = {
        "A_observed": gt.A_observed.astype(np.float32),
        "A_planted": gt.A_planted.astype(np.float32),
        "C": gt.C,
        "S": gt.S,
        "centers_um": gt.centers_um,
        "amplitude_per_cell": gt.amplitude_per_cell,
        "in_focus": gt.in_focus,
        "detectable": gt.detectable,
        "fps": np.asarray(spec.acquisition.fps),
        "pixel_size_um": np.asarray(spec.acquisition.pixel_size_um),
    }
    for name in _GT_OPTIONAL:  # every per-effect field the recording carries
        value = getattr(gt, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    np.savez_compressed(path, **arrays)

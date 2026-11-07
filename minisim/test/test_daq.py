"""Miniscope-DAQ folder writer: layout, chunking, metadata, timestamps, truth."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from minisim import (
    GroundTruth,
    daq_timestamps_ms,
    load_ground_truth,
    simulate,
    write_miniscope_daq,
)
from minisim.avi import read_avi_header
from minisim.cli import main as cli_main
from minisim.testing import make_recording

N_PX, DUR, FPS = 48, 5.0, 20.0  # 100 frames
FPF = 40  # -> files of 40, 40, 20


@pytest.fixture(scope="module")
def rec():
    return make_recording(
        n_cells=4, n_px=N_PX, duration_s=DUR, fps=FPS, seed=2, motion=True
    )


@pytest.fixture(scope="module")
def folder(rec, tmp_path_factory):
    out = tmp_path_factory.mktemp("daq")
    return write_miniscope_daq(rec, out, frames_per_file=FPF, progress=False)


def _decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img if img.ndim == 2 else img[:, :, 0])
    cap.release()
    return np.stack(frames)


def test_layout_and_names(folder):
    assert sorted(p.name for p in folder.root.iterdir()) == [
        "Miniscope",
        "ground_truth",
        "metaData.json",
    ]
    assert sorted(p.name for p in folder.device_dir.iterdir()) == [
        "0.avi",
        "1.avi",
        "2.avi",
        "metaData.json",
        "timeStamps.csv",
    ]
    assert sorted(p.name for p in folder.ground_truth_dir.iterdir()) == [
        "arrays.npz",
        "ground_truth.zarr",
        "spec.json",
    ]


def test_multi_file_split_and_frames_round_trip(folder, rec):
    counts = [read_avi_header(p)["avih"]["total_frames"] for p in folder.avi_files]
    assert counts == [40, 40, 20]
    movie = np.concatenate([_decode(p) for p in folder.avi_files])
    # An 8-bit sensor's counts are written verbatim.
    np.testing.assert_array_equal(movie, np.asarray(rec.observed).astype(np.uint8))
    h = read_avi_header(folder.avi_files[0])
    assert h["avih"]["usec_per_frame"] == 50_000
    assert h["strh"]["handler"] == b"GREY" and h["strf"]["compression"] == b"GREY"


def test_streaming_from_spec_matches_in_memory_recording(rec, folder, tmp_path):
    streamed = write_miniscope_daq(
        rec.spec, tmp_path / "s", frames_per_file=FPF, chunk_frames=7, progress=False
    )
    for a, b in zip(folder.avi_files, streamed.avi_files, strict=True):
        assert a.read_bytes() == b.read_bytes()
    assert streamed.timestamps_csv.read_text() == folder.timestamps_csv.read_text()


def test_streamed_ground_truth_equals_simulate(rec, tmp_path):
    streamed = write_miniscope_daq(rec.spec, tmp_path / "s", progress=False)
    gt, spec = load_ground_truth(streamed.ground_truth_dir / "ground_truth.zarr")
    ref = simulate(rec.spec).ground_truth
    assert spec == rec.spec
    for name in ("C", "S", "centers_um", "detectable", "shifts"):
        np.testing.assert_array_equal(getattr(gt, name), getattr(ref, name))
    np.testing.assert_allclose(gt.A_observed, ref.A_observed)
    np.testing.assert_array_equal(gt.A_planted, ref.A_planted)


def test_streamed_truth_has_effect_fields(tmp_path):
    from minisim import Leakage, Neuropil, Vignette

    r = make_recording(
        n_px=N_PX,
        duration_s=2.0,
        motion=True,
        extra_steps=[Neuropil(), Vignette(), Leakage(level=0.05)],
    )
    streamed = write_miniscope_daq(r.spec, tmp_path / "s", progress=False)
    gt, _ = load_ground_truth(streamed.ground_truth_dir / "ground_truth.zarr")
    for name in ("vignette", "leakage", "neuropil_spatial", "neuropil_population"):
        np.testing.assert_allclose(getattr(gt, name), getattr(r.ground_truth, name))


def test_npz_contents(folder, rec):
    z = np.load(folder.ground_truth_dir / "arrays.npz")
    gt = rec.ground_truth
    assert z["A_observed"].shape == (gt.n_units, N_PX, N_PX)
    np.testing.assert_allclose(z["A_planted"], gt.A_planted, rtol=1e-6)
    np.testing.assert_array_equal(z["C"], gt.C)
    np.testing.assert_array_equal(z["S"], gt.S)
    np.testing.assert_array_equal(z["shifts"], gt.shifts)
    assert float(z["fps"]) == FPS


def test_load_ground_truth_round_trips_type(folder, rec):
    gt, spec = load_ground_truth(folder.ground_truth_dir / "ground_truth.zarr")
    assert isinstance(gt, GroundTruth) and spec == rec.spec
    assert (
        folder.ground_truth_dir / "spec.json"
    ).read_text() == rec.spec.model_dump_json(indent=2)


def test_metadata_matches_daq_keys(folder):
    dev = json.loads(folder.device_metadata.read_text())
    assert dev["compression"] == "GREY"
    assert dev["frameRate"] == "20FPS"
    assert dev["framesPerFile"] == FPF
    assert dev["ROI"] == {"height": N_PX, "leftEdge": 0, "topEdge": 0, "width": N_PX}
    assert dev["deviceType"] == "Miniscope_V4_BNO"
    exp = json.loads(folder.experiment_metadata.read_text())
    assert exp["miniscopes"] == ["Miniscope"] and exp["cameras"] == []
    assert exp["recordingStartTime"]["msecSinceEpoch"] == 946684800000


def test_timestamps_csv(folder):
    with open(folder.timestamps_csv) as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["Frame Number", "Time Stamp (ms)", "Buffer Index"]
    body = np.array(rows[1:], dtype=int)
    assert body.shape == (100, 3)
    np.testing.assert_array_equal(body[:, 0], np.arange(100))  # continues across files
    np.testing.assert_array_equal(body[:, 1], np.arange(100) * 50)
    assert not body[:, 2].any()


def test_timestamp_jitter_is_seeded_and_monotonic():
    a = daq_timestamps_ms(500, 20.0, jitter_ms=3.0, start_offset_ms=-21, seed=1)
    b = daq_timestamps_ms(500, 20.0, jitter_ms=3.0, start_offset_ms=-21, seed=1)
    c = daq_timestamps_ms(500, 20.0, jitter_ms=3.0, start_offset_ms=-21, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a[0] in (-24, -23, -22, -21, -20, -19, -18) and (np.diff(a) > 0).all()
    assert abs(np.diff(a).mean() - 50) < 0.5


def test_deterministic_output_bytes(rec, tmp_path):
    a = write_miniscope_daq(
        rec, tmp_path / "a", frames_per_file=FPF, jitter_ms=2.0, progress=False
    )
    b = write_miniscope_daq(
        rec, tmp_path / "b", frames_per_file=FPF, jitter_ms=2.0, progress=False
    )
    for x, y in zip(a.avi_files, b.avi_files, strict=True):
        assert x.read_bytes() == y.read_bytes()
    assert a.timestamps_csv.read_bytes() == b.timestamps_csv.read_bytes()


def test_overwrite_guard(rec, tmp_path):
    write_miniscope_daq(rec, tmp_path, progress=False)
    with pytest.raises(FileExistsError):
        write_miniscope_daq(rec, tmp_path, progress=False)
    write_miniscope_daq(rec, tmp_path, progress=False, overwrite=True)


def test_no_ground_truth(rec, tmp_path):
    f = write_miniscope_daq(rec.spec, tmp_path, ground_truth=False, progress=False)
    assert f.ground_truth_dir is None and not (tmp_path / "ground_truth").exists()


def test_cli_generate_daq_from_spec(rec, tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(rec.spec.model_dump_json())
    out = tmp_path / "out"
    code = cli_main(
        [
            "generate",
            "--spec",
            str(spec_path),
            "--out",
            str(out),
            "--frames-per-file",
            "50",
            "--quiet",
        ]
    )
    assert code == 0
    assert [p.name for p in sorted((out / "Miniscope").glob("*.avi"))] == [
        "0.avi",
        "1.avi",
    ]
    # Refuses to clobber without --overwrite.
    assert (
        cli_main(["generate", "--spec", str(spec_path), "--out", str(out), "--quiet"])
        == 1
    )


def test_cli_preset_spec_construction():
    from minisim.cli import build_parser, spec_from_args

    args = build_parser().parse_args(
        [
            "generate",
            "--preset",
            "miniscope_v4",
            "--region",
            "ca1",
            "--duration",
            "2",
            "--seed",
            "5",
            "--motion",
            "physical",
            "--motion-um",
            "3",
            "--out",
            "x",
        ]
    )
    spec = spec_from_args(args)
    assert spec.seed == 5 and spec.acquisition.duration_s == 2.0
    assert spec.acquisition.image_sensor.n_px_height == 608
    motion = next(s for s in spec.steps if s.kind == "brain_motion")
    assert motion.motion_amplitude_um == 3.0

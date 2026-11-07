"""The dependency-free AVI writer: structure, index, and independent read-back.

The structure checks parse the file with a few ``struct`` calls written *here*
(not the library's own ``read_avi_header``), so a bug shared by writer and reader
cannot hide; ``cv2.VideoCapture`` (a core dependency) is the second, independent
decoder.
"""

from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest

from minisim.avi import AviWriter, read_avi_header


def _frames(n, h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w), dtype=np.uint8)


def _write(path, frames, fps=20.0):
    _, h, w = frames.shape
    with AviWriter(path, w, h, fps) as aw:
        for f in frames:
            aw.write(f)


def test_riff_structure_and_header_fields(tmp_path):
    frames = _frames(7, 12, 16)
    p = tmp_path / "a.avi"
    _write(p, frames, fps=20.0)
    b = p.read_bytes()

    assert b[:4] == b"RIFF" and b[8:12] == b"AVI "
    assert struct.unpack_from("<I", b, 4)[0] == len(b) - 8
    assert b[12:16] == b"LIST" and b[20:24] == b"hdrl"
    assert b[24:28] == b"avih" and struct.unpack_from("<I", b, 28)[0] == 56
    avih = struct.unpack_from("<14I", b, 32)
    assert avih[0] == 50_000  # usec/frame from the REAL 20 fps (not 60 fps)
    assert avih[3] & 0x10  # AVIF_HASINDEX
    assert avih[4] == 7  # total frames
    assert avih[6] == 1  # one stream
    assert (avih[8], avih[9]) == (16, 12)  # width, height

    # strl / strh
    assert b[88:92] == b"LIST" and b[96:100] == b"strl" and b[100:104] == b"strh"
    assert struct.unpack_from("<I", b, 104)[0] == 56
    assert b[108:112] == b"vids" and b[112:116] == b"GREY"  # fccHandler
    scale, rate = struct.unpack_from("<II", b, 108 + 20)
    assert rate / scale == 20.0
    assert struct.unpack_from("<I", b, 108 + 32)[0] == 7  # dwLength
    # strf: BITMAPINFOHEADER
    assert b[164:168] == b"strf" and struct.unpack_from("<I", b, 168)[0] == 40
    bih = struct.unpack_from("<IiiHH4sI", b, 172)
    assert bih[0] == 40 and (bih[1], bih[2]) == (16, 12)  # positive height
    assert (bih[3], bih[4]) == (1, 8)
    assert bih[5] == b"GREY"  # biCompression
    assert bih[6] == 16 * 12


def test_movi_chunks_and_idx1_offsets(tmp_path):
    frames = _frames(5, 8, 8)
    p = tmp_path / "a.avi"
    _write(p, frames)
    b = p.read_bytes()
    movi = b.find(b"movi")
    assert b[movi - 8 : movi - 4] == b"LIST"
    movi_size = struct.unpack_from("<I", b, movi - 8 + 4)[0]
    pos = movi + 4
    offsets = []
    for k in range(5):
        assert b[pos : pos + 4] == b"00dc"
        assert struct.unpack_from("<I", b, pos + 4)[0] == 64
        offsets.append(pos - movi)  # relative to the 'movi' fourcc
        data = np.frombuffer(b, np.uint8, 64, pos + 8).reshape(8, 8)
        np.testing.assert_array_equal(data, frames[k])  # rows top to bottom
        pos += 8 + 64
    assert movi_size == pos - movi  # LIST size covers 'movi' + chunks
    assert offsets[0] == 4
    assert b[pos : pos + 4] == b"idx1"
    n_idx = struct.unpack_from("<I", b, pos + 4)[0] // 16
    assert n_idx == 5
    for k in range(5):
        tag, flags, off, size = struct.unpack_from("<4sIII", b, pos + 8 + 16 * k)
        assert (tag, off, size) == (b"00dc", offsets[k], 64)
        assert flags & 0x10  # keyframe
    assert pos + 8 + 16 * 5 == len(b)  # idx1 is the last chunk


def test_cv2_round_trip_exact(tmp_path):
    frames = _frames(9, 20, 28, seed=4)
    p = tmp_path / "a.avi"
    _write(p, frames, fps=20.0)
    cap = cv2.VideoCapture(str(p))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 9
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
    for k in range(9):
        ok, img = cap.read()
        assert ok
        gray = img if img.ndim == 2 else img[:, :, 0]
        np.testing.assert_array_equal(gray, frames[k])
    assert not cap.read()[0]
    cap.release()


def test_row_padding_to_four_bytes(tmp_path):
    frames = _frames(3, 5, 6)  # width 6 -> stride 8
    p = tmp_path / "a.avi"
    _write(p, frames)
    h = read_avi_header(p)
    assert h["strf"]["size_image"] == 8 * 5
    off, size = h["frames"][1]
    assert size == 40
    raw = np.frombuffer(p.read_bytes(), np.uint8, 40, off).reshape(5, 8)
    np.testing.assert_array_equal(raw[:, :6], frames[1])
    assert not raw[:, 6:].any()


def test_deterministic_bytes(tmp_path):
    frames = _frames(4, 16, 16, seed=9)
    _write(tmp_path / "a.avi", frames)
    _write(tmp_path / "b.avi", frames)
    assert (tmp_path / "a.avi").read_bytes() == (tmp_path / "b.avi").read_bytes()


def test_fractional_fps_is_exact_in_rate_scale(tmp_path):
    _write(tmp_path / "a.avi", _frames(2, 8, 8), fps=29.97)
    h = read_avi_header(tmp_path / "a.avi")
    assert h["strh"]["rate"] / h["strh"]["scale"] == pytest.approx(29.97)
    assert h["avih"]["usec_per_frame"] == 33_367


def test_rejects_bad_frames(tmp_path):
    with AviWriter(tmp_path / "a.avi", 8, 8, 20.0) as aw:
        with pytest.raises(TypeError):
            aw.write(np.zeros((8, 8), dtype=np.float32))
        with pytest.raises(ValueError):
            aw.write(np.zeros((8, 9), dtype=np.uint8))
    with pytest.raises(ValueError):
        AviWriter(tmp_path / "b.avi", 8, 8, 0.0)


def test_zero_frames_is_a_valid_empty_file(tmp_path):
    with AviWriter(tmp_path / "a.avi", 8, 8, 20.0):
        pass
    h = read_avi_header(tmp_path / "a.avi")
    assert h["avih"]["total_frames"] == 0 and h["frames"] == [] and h["idx1"] == []

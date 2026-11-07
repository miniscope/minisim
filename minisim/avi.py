"""A dependency-free, streaming, uncompressed 8-bit grayscale AVI writer.

Real Miniscope-DAQ recordings are plain-RIFF AVI files carrying raw 8-bit
grayscale frames (fourcc ``GREY``). :class:`AviWriter` produces files with the same
structure from numpy + the standard library alone (no OpenCV, no ffmpeg), one frame
at a time, so a recording of any length can be written without ever holding the
movie in memory. The bytes are a pure function of the frames and the arguments
(nothing time- or path-dependent is embedded), so equal inputs give equal files.

File layout (all integers little-endian)::

    RIFF <size> 'AVI '
      LIST <size> 'hdrl'
        'avih' (56 B)   main header: usec/frame = round(1e6 / fps), total frames, W, H
        LIST <size> 'strl'
          'strh' (56 B) stream header: 'vids' / 'GREY', scale/rate = exact fps
          'strf' (40 B) BITMAPINFOHEADER: 8 bpp, biCompression = 'GREY', biHeight > 0
      LIST <size> 'movi'
        '00dc' <size> <frame bytes>   one chunk per frame, rows top to bottom
        ...
      'idx1' <size>                   one 16-byte entry per frame, offsets relative
                                      to the 'movi' fourcc (first frame = 4)

Two details worth knowing, both copied from what the Miniscope DAQ writes:

* **Row order.** Rows are stored top to bottom even though ``biHeight`` is positive
  (the DIB convention would be bottom-up). ffmpeg/OpenCV read such a ``GREY`` file
  back in file order, so a decoded frame equals the array that was written.
* **Row padding.** Each row is padded with zero bytes to a multiple of 4 (the AVI /
  DIB stride rule). For a width that is a multiple of 4 (the DAQ's 600 and 608) this
  is a no-op; ``strh`` / ``strf`` then describe the padded stride.

Unlike the DAQ's own files, the header carries the **true** frame rate (the DAQ
writes 60 fps into the AVI header regardless of the sensor rate; the rate in
``metaData.json`` / ``timeStamps.csv`` is authoritative there). There is no OpenDML
extension, so a single file is limited to ~4 GiB (:class:`AviWriter` refuses to
exceed it).
"""

from __future__ import annotations

import struct
from fractions import Fraction
from pathlib import Path
from types import TracebackType

import numpy as np

__all__ = ["AviWriter", "read_avi_header"]

_FOURCC_GREY = b"GREY"
_AVIF_HASINDEX = 0x10
_AVIF_ISINTERLEAVED = 0x100
_AVIF_TRUSTCKTYPE = 0x800
_AVIIF_KEYFRAME = 0x10
_MAX_RIFF_BYTES = 2**32 - 1


class AviWriter:
    """Write 8-bit grayscale frames to a plain-RIFF ``GREY`` AVI, streaming.

    Use as a context manager (or call :meth:`close`); each :meth:`write` appends one
    ``(height, width)`` ``uint8`` frame. The header fields that depend on the final
    frame count (``avih``/``strh`` lengths, the RIFF and ``movi`` sizes) are
    patched in on :meth:`close`, then the ``idx1`` index is appended.

    ``fps`` is stored exactly as ``scale/rate`` (the best rational with a
    denominator of at most 1000) and, rounded, as ``usec_per_frame``.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        if width < 1 or height < 1:
            raise ValueError(f"width and height must be >= 1, got {width}x{height}.")
        if not fps > 0:
            raise ValueError(f"fps must be > 0, got {fps}.")
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.stride = (self.width + 3) // 4 * 4
        self.frame_bytes = self.stride * self.height
        frac = Fraction(self.fps).limit_denominator(1000)
        self._rate, self._scale = frac.numerator, frac.denominator
        self.n_frames = 0
        self._closed = False
        self._pad = np.zeros((self.height, self.stride - self.width), dtype=np.uint8)
        self._f = open(self.path, "wb")
        try:
            self._write_header()
        except BaseException:
            self._f.close()
            raise

    # -- context manager -----------------------------------------------------
    def __enter__(self) -> AviWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:  # don't finalize a half-written file as if it were good
            self.abort()

    # -- header --------------------------------------------------------------
    def _write_header(self) -> None:
        f = self._f
        usec = int(round(1e6 / self.fps))
        w, h = self.width, self.height
        avih = struct.pack(
            "<14I",
            usec,
            int(round(self.frame_bytes * self.fps)),  # max bytes/sec
            0,  # padding granularity
            _AVIF_HASINDEX | _AVIF_ISINTERLEAVED | _AVIF_TRUSTCKTYPE,
            0,  # total frames (patched on close)
            0,  # initial frames
            1,  # streams
            self.frame_bytes,  # suggested buffer size
            w,
            h,
            0,
            0,
            0,
            0,
        )
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"vids",
            _FOURCC_GREY,
            0,  # flags
            0,  # priority
            0,  # language
            0,  # initial frames
            self._scale,
            self._rate,
            0,  # start
            0,  # length in frames (patched on close)
            self.frame_bytes,  # suggested buffer size
            0xFFFFFFFF,  # quality: default
            0,  # sample size (0 = variable; one frame per chunk)
            0,
            0,
            w,
            h,
        )
        strf = struct.pack(
            "<IiiHH4sIiiII",
            40,
            w,
            h,
            1,  # planes
            8,  # bit count
            _FOURCC_GREY,  # biCompression
            self.frame_bytes,  # biSizeImage
            0,
            0,
            0,  # colors used
            0,  # important colors
        )
        strl = b"strl" + _chunk(b"strh", strh) + _chunk(b"strf", strf)
        hdrl = b"hdrl" + _chunk(b"avih", avih) + _list(strl)
        f.write(b"RIFF" + struct.pack("<I", 0) + b"AVI ")  # size patched on close
        f.write(_list(hdrl))
        # LIST <size> 'movi': remember where the size lives, patched on close.
        f.write(b"LIST")
        self._movi_size_pos = f.tell()
        f.write(struct.pack("<I", 0) + b"movi")
        self._movi_fourcc_pos = f.tell() - 4
        # Offsets in idx1 are relative to the 'movi' fourcc; the first chunk is at +4.
        self._next_offset = 4
        self._index = bytearray()

    # -- frames --------------------------------------------------------------
    def write(self, frame: np.ndarray) -> None:
        """Append one ``(height, width)`` ``uint8`` frame."""
        if self._closed:
            raise ValueError("write() on a closed AviWriter.")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            raise TypeError(f"frames must be uint8, got {frame.dtype}.")
        if frame.shape != (self.height, self.width):
            raise ValueError(
                f"frame shape {frame.shape} != ({self.height}, {self.width})."
            )
        projected = (
            self._f.tell() + 8 + self.frame_bytes + 16 * (self.n_frames + 1) + 8 + 64
        )
        if projected > _MAX_RIFF_BYTES:
            raise ValueError(
                f"{self.path.name} would exceed the 4 GiB plain-RIFF limit at frame "
                f"{self.n_frames}; use a smaller frames_per_file."
            )
        data = frame if self.stride == self.width else np.hstack([frame, self._pad])
        self._f.write(b"00dc" + struct.pack("<I", self.frame_bytes))
        self._f.write(np.ascontiguousarray(data).data)
        self._index += struct.pack(
            "<4sIII", b"00dc", _AVIIF_KEYFRAME, self._next_offset, self.frame_bytes
        )
        self._next_offset += 8 + self.frame_bytes
        self.n_frames += 1

    def abort(self) -> None:
        """Close the file handle without finalizing (no-op once closed)."""
        if not self._closed:
            self._f.close()
            self._closed = True

    # -- finalize ------------------------------------------------------------
    def close(self) -> None:
        """Append ``idx1`` and patch the frame-count-dependent header fields."""
        if self._closed:
            return
        f = self._f
        movi_end = f.tell()
        f.write(_chunk(b"idx1", bytes(self._index)))
        end = f.tell()
        f.seek(self._movi_size_pos)
        f.write(struct.pack("<I", movi_end - self._movi_fourcc_pos))
        f.seek(4)
        f.write(struct.pack("<I", end - 8))
        # avih.dwTotalFrames (offset 12 + 16 within avih payload) and strh.dwLength.
        f.seek(_AVIH_TOTAL_FRAMES_POS)
        f.write(struct.pack("<I", self.n_frames))
        f.seek(_STRH_LENGTH_POS)
        f.write(struct.pack("<I", self.n_frames))
        f.close()
        self._closed = True


# Fixed byte positions of the patched fields: RIFF(12) + LIST hdrl header(12) +
# 'avih' chunk header(8) puts the avih payload at 32; strh payload follows the
# 56-byte avih payload, the 'LIST strl' header (12) and the 'strh' chunk header (8).
_AVIH_TOTAL_FRAMES_POS = 12 + 12 + 8 + 16
_STRH_LENGTH_POS = 12 + 12 + 8 + 56 + 12 + 8 + 32


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) % 2 else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + pad


def _list(body_with_type: bytes) -> bytes:
    """A ``LIST`` chunk whose payload is ``list type + children``."""
    return b"LIST" + struct.pack("<I", len(body_with_type)) + body_with_type


def read_avi_header(path: str | Path) -> dict:
    """Parse the header of a plain-RIFF AVI (a small, dependency-free inspector).

    Walks the RIFF chunks and returns the ``avih`` / ``strh`` / ``strf`` fields, the
    ``movi`` frame chunk offsets/sizes (from the chunks themselves) and the
    ``idx1`` entries. Intended for tests and quick validation of what
    :class:`AviWriter` (or the Miniscope DAQ) wrote - not a general AVI demuxer.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError(f"{path}: not a RIFF AVI file.")
    out: dict = {
        "riff_size": struct.unpack_from("<I", data, 4)[0],
        "file_size": len(data),
    }
    frames: list[tuple[int, int]] = []
    index: list[tuple[bytes, int, int, int]] = []

    def walk(start: int, end: int) -> None:
        pos = start
        while pos + 8 <= end:
            tag = data[pos : pos + 4]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            body = pos + 8
            if tag == b"LIST":
                kind = data[body : body + 4]
                if kind == b"movi":
                    out["movi_fourcc_pos"] = body
                    out["movi_size"] = size
                    walk_movi(body + 4, body + size)
                else:
                    walk(body + 4, body + size)
            elif tag == b"avih":
                names = (
                    "usec_per_frame max_bytes_per_sec padding flags total_frames "
                    "initial_frames streams suggested_buffer width height"
                ).split()
                out["avih"] = dict(
                    zip(names, struct.unpack_from("<10I", data, body), strict=True)
                )
            elif tag == b"strh":
                t, h = struct.unpack_from("<4s4s", data, body)
                rest = struct.unpack_from("<IHHIIIIIIII", data, body + 8)
                rc = struct.unpack_from("<hhhh", data, body + 48)
                out["strh"] = {
                    "type": t,
                    "handler": h,
                    "flags": rest[0],
                    "initial_frames": rest[3],
                    "scale": rest[4],
                    "rate": rest[5],
                    "start": rest[6],
                    "length": rest[7],
                    "suggested_buffer": rest[8],
                    "quality": rest[9],
                    "sample_size": rest[10],
                    "rc_frame": rc,
                }
            elif tag == b"strf":
                v = struct.unpack_from("<IiiHH4sIiiII", data, body)
                out["strf"] = dict(
                    zip(
                        "size width height planes bit_count compression size_image "
                        "xppm yppm clr_used clr_important".split(),
                        v,
                        strict=True,
                    )
                )
            elif tag == b"idx1":
                for k in range(size // 16):
                    index.append(struct.unpack_from("<4sIII", data, body + 16 * k))
            pos = body + size + (size & 1)

    def walk_movi(start: int, end: int) -> None:
        pos = start
        while pos + 8 <= end:
            tag = data[pos : pos + 4]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            if tag.endswith(b"dc"):
                frames.append((pos + 8, size))
            pos += 8 + size + (size & 1)

    walk(12, len(data))
    out["frames"] = frames
    out["idx1"] = index
    return out

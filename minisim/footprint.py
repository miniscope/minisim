"""Sparse, patch-based spatial footprints - a neuron's weight map without the zeros.

A neuron's spatial footprint (its fluorophore support, the ``A`` of ``A·C``) is
**local**: a soma plus proximal neurites occupy a small patch, while the canvas it
lives on is the whole sensor FOV (plus any motion margin). Stored as a dense
``(H, W)`` array per cell, ~98% of every footprint is zeros, and a recording with
``N`` cells carries ``N`` full frames twice over (planted *and* optically observed).

This module stores only what is non-zero. A :class:`Footprint` is a small dense
``patch`` plus the ``offset`` of its top-left corner on a known ``canvas_shape``;
the dense array is ``np.zeros(canvas_shape)`` with ``patch`` written at ``offset``,
but is never materialized in the pipeline - :class:`~minisim.steps.tissue.RenderStep`
composites each cell directly into its own canvas window, and metrics binarize each
patch in place. A :class:`FootprintStack` is the per-recording bundle of these,
the form ``GroundTruth.A_planted`` / ``A_observed`` take.

The patch is always the *tight* non-zero bounding box (see :meth:`Footprint.from_dense`),
so its size is set by the physics - soma radius, dendrite length, and the optical
PSF width - not by any guessed constant. Patches are held in ``float32``: a
peak-normalized weight in ``[0, 1]`` needs far less than ``float64``'s ~16 digits,
and the render accumulator (the movie) stays ``float64``, so nothing downstream
loses precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

# The dtype every footprint patch is stored in. Footprints are peak-normalized
# spatial weights in [0, 1]; float32's ~7 significant digits are ample, and this
# halves footprint memory and on-disk size versus float64. Kept named so a future
# change (e.g. to float16) is a one-line edit with a single place to validate.
PATCH_DTYPE = np.float32

# scipy gaussian_filter's default truncation: the kernel is exactly 0 beyond this
# many sigma, so growing a footprint's window by ceil(_GAUSS_TRUNCATE·sigma) before
# blurring yields a result bit-identical to filtering the whole canvas.
_GAUSS_TRUNCATE = 4.0


@dataclass(frozen=True)
class Footprint:
    """One neuron's spatial weight, stored as its non-zero patch on a canvas.

    Equivalent to the dense array ``D`` where ``D[offset[0]:offset[0]+ph,
    offset[1]:offset[1]+pw] = patch`` and ``D`` is otherwise zero, with ``D`` of
    shape :attr:`canvas_shape` and ``(ph, pw) = patch.shape``. Recover ``D`` with
    :meth:`to_dense`; build one from a dense array (trimming to the tight non-zero
    box) with :meth:`from_dense`.

    Attributes
    ----------
    offset
        ``(y0, x0)`` top-left corner of the patch in canvas pixel coordinates.
    patch
        The non-zero window, ``(ph, pw)`` in :data:`PATCH_DTYPE`. May be empty
        (shape ``(0, 0)``) for a footprint with no support (e.g. cropped entirely
        outside the FOV); :attr:`is_empty` flags this.
    canvas_shape
        ``(H, W)`` of the full frame the patch is placed on.
    """

    offset: tuple[int, int]
    patch: np.ndarray
    canvas_shape: tuple[int, int]

    @classmethod
    def from_dense(cls, dense: np.ndarray) -> Footprint:
        """Trim a dense ``(H, W)`` array to its tight non-zero box and store that.

        The patch is the smallest window containing every non-zero pixel, cast to
        :data:`PATCH_DTYPE`. An all-zero input yields an empty footprint (a
        ``(0, 0)`` patch at offset ``(0, 0)``) carrying the original canvas shape.
        """
        canvas_shape = (int(dense.shape[0]), int(dense.shape[1]))
        rows = np.any(dense != 0, axis=1)
        if not rows.any():
            empty = np.zeros((0, 0), dtype=PATCH_DTYPE)
            return cls(offset=(0, 0), patch=empty, canvas_shape=canvas_shape)
        cols = np.any(dense != 0, axis=0)
        y0 = int(np.argmax(rows))
        y1 = len(rows) - int(np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = len(cols) - int(np.argmax(cols[::-1]))
        patch = dense[y0:y1, x0:x1].astype(PATCH_DTYPE, copy=True)
        return cls(offset=(y0, x0), patch=patch, canvas_shape=canvas_shape)

    @property
    def is_empty(self) -> bool:
        """True when the patch holds no pixels - no support on the canvas."""
        return self.patch.size == 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """The patch extent as ``(y0, y1, x0, x1)`` half-open canvas indices."""
        y0, x0 = self.offset
        ph, pw = self.patch.shape
        return y0, y0 + ph, x0, x0 + pw

    def to_dense(self, dtype: np.dtype | type | None = None) -> np.ndarray:
        """Materialize the full ``canvas_shape`` array (the zeros made explicit).

        ``dtype`` defaults to the patch dtype (:data:`PATCH_DTYPE`); pass e.g.
        ``float`` when a downstream consumer wants ``float64``.
        """
        out = np.zeros(self.canvas_shape, dtype=dtype or self.patch.dtype)
        if not self.is_empty:
            y0, x0 = self.offset
            ph, pw = self.patch.shape
            out[y0 : y0 + ph, x0 : x0 + pw] = self.patch
        return out

    def crop(self, top: int, left: int, height: int, width: int) -> Footprint:
        """Crop to the window ``[top, top+height) × [left, left+width)``.

        Returns a footprint on the new ``(height, width)`` canvas whose patch is
        the intersection of this one with the window, offset rebased to the new
        origin. A footprint lying entirely outside the window yields an empty
        footprint on the new canvas - the signal a cell has left the FOV. This is
        how :func:`~minisim.recording.finalize` crops canvas footprints to the
        sensor FOV.
        """
        new_shape = (int(height), int(width))
        if self.is_empty:
            return Footprint(offset=(0, 0), patch=self.patch, canvas_shape=new_shape)
        y0, y1, x0, x1 = self.bbox
        iy0, iy1 = max(y0, top), min(y1, top + height)
        ix0, ix1 = max(x0, left), min(x1, left + width)
        if iy0 >= iy1 or ix0 >= ix1:
            empty = np.zeros((0, 0), dtype=self.patch.dtype)
            return Footprint(offset=(0, 0), patch=empty, canvas_shape=new_shape)
        sub = self.patch[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
        return Footprint(
            offset=(iy0 - top, ix0 - left),
            patch=np.ascontiguousarray(sub),
            canvas_shape=new_shape,
        )

    def add_into(self, canvas: np.ndarray, weights: np.ndarray | None = None) -> None:
        """Accumulate this footprint into ``canvas``, in place, touching only its window.

        With ``weights=None``, adds the 2-D patch into the matching ``(H, W)``
        window of ``canvas``. With ``weights`` of shape ``(T,)``, adds the outer
        product ``weights[:, None, None] · patch`` into the ``(T, H, W)`` window -
        the render primitive: a cell's patch scaled frame-by-frame by what it
        emits, written only where it has support. A no-op for an empty footprint.
        """
        if self.is_empty:
            return
        y0, x0 = self.offset
        ph, pw = self.patch.shape
        if weights is None:
            canvas[y0 : y0 + ph, x0 : x0 + pw] += self.patch
        else:
            canvas[:, y0 : y0 + ph, x0 : x0 + pw] += weights[:, None, None] * self.patch


def degrade_footprint(planted: Footprint, sigma_px: float, gain: float) -> Footprint:
    """Apply the optical PSF blur and the multiplicative light-loss to a footprint.

    ``observed = gain · (planted ⊛ Gaussian(sigma_px))``. The Gaussian convolution
    is the combined diffraction + defocus + scatter point-spread; it is
    sum-normalized, so it **conserves integrated intensity** -- that is what makes
    *defocus* intensity-conserving (it spreads light: the peak drops but the integral
    is unchanged). ``gain`` is the flat light-loss that actually removes signal:
    scatter ``attenuation(z)`` (depth) × ``collection_efficiency`` (``∝ NA²``). Both
    are focal-plane independent, so the observed footprint's integral is too.
    ``mode="constant"`` means light blurred past the canvas edge is lost -- physically
    honest for a cell near the boundary.

    Pure and deterministic: there is no RNG, so the observed footprint is a fixed
    function of ``(planted, sigma_px, gain)``. That is why the pipeline does not
    *store* observed footprints -- it regenerates them, bit-identically, at render
    time and for ``GroundTruth.A_observed`` (see :mod:`minisim.recording`).

    Works patch-to-patch: the planted footprint is already just the cell's non-zero
    window, so the blur grows that window by the PSF's truncation radius
    (``ceil(_GAUSS_TRUNCATE·sigma_px)``, clipped to the canvas) and convolves only
    there. Beyond ``_GAUSS_TRUNCATE·sigma_px`` the Gaussian is exactly zero, so the
    result is **bit-identical** to filtering the whole canvas, without ever
    materializing a canvas-sized array. An empty planted footprint passes through.
    """
    if planted.is_empty:
        return planted  # no support -> nothing to blur
    height, width = planted.canvas_shape
    y0, x0 = planted.offset
    ph, pw = planted.patch.shape
    pad = int(np.ceil(_GAUSS_TRUNCATE * sigma_px)) + 1
    oy0, ox0 = max(y0 - pad, 0), max(x0 - pad, 0)
    oy1, ox1 = min(y0 + ph + pad, height), min(x0 + pw + pad, width)
    # Place the planted patch into a zero window grown by the PSF reach, blur in
    # float64 (the working precision optics accumulates in), then store float32.
    window = np.zeros((oy1 - oy0, ox1 - ox0), dtype=float)
    window[y0 - oy0 : y0 - oy0 + ph, x0 - ox0 : x0 - ox0 + pw] = planted.patch
    blurred = gain * gaussian_filter(window, sigma=sigma_px, mode="constant")
    return Footprint(
        offset=(oy0, ox0),
        patch=blurred.astype(PATCH_DTYPE),
        canvas_shape=(height, width),
    )


def stack_dense(
    footprints: list[Footprint],
    canvas_shape: tuple[int, int],
    dtype: np.dtype | type = float,
) -> np.ndarray:
    """Materialize a list of footprints into a dense ``(n, H, W)`` stack.

    The render primitive. Footprints are stored sparse (one small patch per cell),
    but compositing is fastest as a single BLAS contraction ``tensordot(C, A)`` over
    a dense ``A`` - even though that multiplies the zeros, optimized matmul beats a
    Python per-cell loop, and deep cells' optically-scattered footprints are
    near-full-canvas anyway (little sparsity left to exploit). So the stack is
    rebuilt dense here, transiently, only for the contraction; persistent storage
    (``Cell`` footprints, ``GroundTruth.A_*``) stays sparse. Each patch is written
    into its own window via :meth:`Footprint.add_into`, so no per-cell full-canvas
    temporary is allocated.
    """
    a = np.zeros((len(footprints), int(canvas_shape[0]), int(canvas_shape[1])), dtype=dtype)
    for i, fp in enumerate(footprints):
        fp.add_into(a[i])
    return a


@dataclass(frozen=True)
class FootprintStack:
    """A recording's footprints - the sparse form of a dense ``(unit, H, W)`` stack.

    Holds one :class:`Footprint` per unit, all sharing :attr:`canvas_shape`. This
    is the type of ``GroundTruth.A_planted`` and ``A_observed``. It supports the
    access patterns the dense stack did - ``len``, integer and boolean-mask
    indexing (so ``A[detectable]`` still subsets units), iteration - plus
    :meth:`to_dense` to recover the ``(unit, H, W)`` array on demand, and the
    ``to_arrays`` / ``from_arrays`` pair the zarr (de)serializer uses.
    """

    footprints: tuple[Footprint, ...]
    canvas_shape: tuple[int, int]

    def __post_init__(self) -> None:
        for fp in self.footprints:
            if fp.canvas_shape != self.canvas_shape:
                raise ValueError(
                    f"Footprint canvas {fp.canvas_shape} != stack canvas "
                    f"{self.canvas_shape}; all footprints must share one canvas."
                )

    @classmethod
    def from_footprints(
        cls, footprints: list[Footprint], canvas_shape: tuple[int, int]
    ) -> FootprintStack:
        """Build from a list of footprints, defaulting an empty list to ``canvas_shape``.

        ``canvas_shape`` is required so an empty stack (no cells) still knows the
        frame it lives on, the way the old ``np.zeros((0, H, W))`` did.
        """
        return cls(footprints=tuple(footprints), canvas_shape=(int(canvas_shape[0]), int(canvas_shape[1])))

    @classmethod
    def from_dense(cls, dense: np.ndarray) -> FootprintStack:
        """Build from a dense ``(unit, H, W)`` stack, trimming each unit's patch."""
        canvas_shape = (int(dense.shape[1]), int(dense.shape[2]))
        return cls(
            footprints=tuple(Footprint.from_dense(d) for d in dense),
            canvas_shape=canvas_shape,
        )

    @property
    def n_units(self) -> int:
        """Number of footprints (units) in the stack."""
        return len(self.footprints)

    def __len__(self) -> int:
        return len(self.footprints)

    def __iter__(self):
        return iter(self.footprints)

    def __getitem__(self, index):
        """Index by ``int`` → :class:`Footprint`, or by slice / int array / bool
        mask → :class:`FootprintStack` (so ``A[mask]`` subsets units like the dense
        stack did)."""
        if isinstance(index, (int, np.integer)):
            return self.footprints[int(index)]
        if isinstance(index, slice):
            return FootprintStack(self.footprints[index], self.canvas_shape)
        idx = np.asarray(index)
        if idx.dtype == bool:
            if idx.shape != (len(self.footprints),):
                raise IndexError(
                    f"boolean mask {idx.shape} does not match {len(self.footprints)} units."
                )
            chosen = [fp for fp, keep in zip(self.footprints, idx, strict=True) if keep]
        else:
            chosen = [self.footprints[int(i)] for i in idx]
        return FootprintStack(tuple(chosen), self.canvas_shape)

    def crop(self, top: int, left: int, height: int, width: int) -> FootprintStack:
        """Crop every footprint to the window ``[top, top+height) × [left, left+width)``.

        Returns a new stack on the ``(height, width)`` canvas (see
        :meth:`Footprint.crop`). Used to take canvas-coordinate footprints down to
        the sensor FOV.
        """
        return FootprintStack(
            footprints=tuple(fp.crop(top, left, height, width) for fp in self.footprints),
            canvas_shape=(int(height), int(width)),
        )

    def to_dense(self, dtype: np.dtype | type | None = None) -> np.ndarray:
        """Materialize the full ``(unit, H, W)`` dense stack.

        The escape hatch for consumers that genuinely want a dense array (e.g.
        scoring against an external dense estimate). Returns ``(0, H, W)`` for an
        empty stack, matching the old ``_stack`` empty-shape behavior.
        """
        out_dtype = dtype or PATCH_DTYPE
        out = np.zeros((len(self.footprints), *self.canvas_shape), dtype=out_dtype)
        for i, fp in enumerate(self.footprints):
            if not fp.is_empty:
                y0, x0 = fp.offset
                ph, pw = fp.patch.shape
                out[i, y0 : y0 + ph, x0 : x0 + pw] = fp.patch
        return out

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten to three arrays for ragged storage (zarr): offsets, shapes, data.

        Returns ``(offsets (n, 2) int32, shapes (n, 2) int32, data (Σ pixels,)
        float32)`` - patches concatenated in row-major order, sliceable back via
        the per-unit pixel counts in ``shapes``. Round-trips through
        :meth:`from_arrays`.
        """
        n = len(self.footprints)
        offsets = np.zeros((n, 2), dtype=np.int32)
        shapes = np.zeros((n, 2), dtype=np.int32)
        flats = []
        for i, fp in enumerate(self.footprints):
            offsets[i] = fp.offset
            shapes[i] = fp.patch.shape
            flats.append(np.ravel(fp.patch))
        data = (
            np.concatenate(flats).astype(PATCH_DTYPE)
            if flats
            else np.zeros((0,), dtype=PATCH_DTYPE)
        )
        return offsets, shapes, data

    @classmethod
    def from_arrays(
        cls,
        offsets: np.ndarray,
        shapes: np.ndarray,
        data: np.ndarray,
        canvas_shape: tuple[int, int],
    ) -> FootprintStack:
        """Rebuild a stack from the :meth:`to_arrays` triple and a canvas shape."""
        offsets = np.asarray(offsets, dtype=np.int32)
        shapes = np.asarray(shapes, dtype=np.int32)
        data = np.asarray(data, dtype=PATCH_DTYPE)
        canvas_shape = (int(canvas_shape[0]), int(canvas_shape[1]))
        footprints = []
        pos = 0
        for (y0, x0), (ph, pw) in zip(offsets, shapes, strict=True):
            count = int(ph) * int(pw)
            patch = data[pos : pos + count].reshape(int(ph), int(pw))
            pos += count
            footprints.append(
                Footprint(offset=(int(y0), int(x0)), patch=patch, canvas_shape=canvas_shape)
            )
        return cls(footprints=tuple(footprints), canvas_shape=canvas_shape)

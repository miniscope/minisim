"""Unit tests for the sparse patch footprint types (:mod:`minisim.footprint`).

Covers the dense<->sparse round trips that the whole representation rests on:
``from_dense`` trims to the tight non-zero box, ``to_dense`` rebuilds the exact
canvas array, ``crop`` is FOV-window intersection (including the off-canvas
empty case), ``add_into`` is the render primitive, and ``FootprintStack`` indexing
/ ``to_arrays`` round-trip preserve units and values bit-for-bit.
"""

import numpy as np
import pytest

from minisim.footprint import PATCH_DTYPE, Footprint, FootprintStack


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(float)


# --- Footprint round trips -------------------------------------------------


def test_from_dense_trims_to_tight_bbox():
    dense = np.zeros((40, 50))
    dense[10:14, 20:27] = 1.0
    fp = Footprint.from_dense(dense)
    assert fp.offset == (10, 20)
    assert fp.patch.shape == (4, 7)
    assert fp.canvas_shape == (40, 50)
    assert fp.patch.dtype == PATCH_DTYPE


def test_to_dense_reconstructs_exactly():
    dense = _disk(64, 48, 30, 20, 6)
    fp = Footprint.from_dense(dense)
    np.testing.assert_array_equal(fp.to_dense(dtype=float), dense)


def test_all_zero_dense_is_empty_footprint():
    fp = Footprint.from_dense(np.zeros((16, 16)))
    assert fp.is_empty
    assert fp.patch.shape == (0, 0)
    assert not fp.to_dense().any()


def test_single_pixel_footprint():
    dense = np.zeros((10, 10))
    dense[3, 7] = 1.0
    fp = Footprint.from_dense(dense)
    assert fp.offset == (3, 7)
    assert fp.patch.shape == (1, 1)
    np.testing.assert_array_equal(fp.to_dense(dtype=float), dense)


# --- crop (FOV intersection) -----------------------------------------------


def test_crop_to_interior_window():
    dense = np.zeros((40, 40))
    dense[12:16, 18:22] = 2.0
    fp = Footprint.from_dense(dense)
    # Crop a centered 20x20 window starting at (10, 10): patch lands fully inside.
    cropped = fp.crop(top=10, left=10, height=20, width=20)
    assert cropped.canvas_shape == (20, 20)
    assert cropped.offset == (2, 8)
    expected = dense[10:30, 10:30]
    np.testing.assert_array_equal(cropped.to_dense(dtype=float), expected)


def test_crop_partial_overlap_clips_patch():
    dense = np.zeros((40, 40))
    dense[2:8, 2:8] = 1.0  # near the top-left corner
    fp = Footprint.from_dense(dense)
    cropped = fp.crop(top=5, left=5, height=20, width=20)
    # Only the [5:8, 5:8] sub-block survives, rebased to the new origin (0, 0).
    assert cropped.offset == (0, 0)
    assert cropped.patch.shape == (3, 3)
    np.testing.assert_array_equal(cropped.to_dense(dtype=float), dense[5:25, 5:25])


def test_crop_entirely_outside_is_empty():
    dense = np.zeros((40, 40))
    dense[2:6, 2:6] = 1.0
    fp = Footprint.from_dense(dense)
    cropped = fp.crop(top=20, left=20, height=10, width=10)
    assert cropped.is_empty
    assert cropped.canvas_shape == (10, 10)


# --- add_into (render primitive) -------------------------------------------


def test_add_into_2d():
    dense = _disk(32, 32, 16, 16, 5)
    fp = Footprint.from_dense(dense)
    canvas = np.zeros((32, 32))
    fp.add_into(canvas)
    np.testing.assert_allclose(canvas, dense)


def test_add_into_with_weights_matches_outer_product():
    dense = _disk(24, 24, 12, 10, 4)
    fp = Footprint.from_dense(dense)
    weights = np.array([0.0, 1.0, 2.5, -1.0])
    canvas = np.zeros((4, 24, 24))
    fp.add_into(canvas, weights)
    expected = weights[:, None, None] * dense[None]
    np.testing.assert_allclose(canvas, expected, rtol=1e-6)


def test_add_into_empty_is_noop():
    fp = Footprint.from_dense(np.zeros((8, 8)))
    canvas = np.ones((8, 8))
    fp.add_into(canvas)
    np.testing.assert_array_equal(canvas, np.ones((8, 8)))


# --- FootprintStack --------------------------------------------------------


def _stack_of(n, h=40, w=40):
    rng = np.random.default_rng(0)
    fps = []
    for _ in range(n):
        cy, cx = rng.integers(8, h - 8), rng.integers(8, w - 8)
        fps.append(Footprint.from_dense(_disk(h, w, cy, cx, 4)))
    return FootprintStack.from_footprints(fps, (h, w))


def test_stack_to_dense_matches_individual():
    stack = _stack_of(3)
    dense = stack.to_dense(dtype=float)
    assert dense.shape == (3, 40, 40)
    for i, fp in enumerate(stack):
        np.testing.assert_array_equal(dense[i], fp.to_dense(dtype=float))


def test_empty_stack_to_dense_keeps_canvas():
    stack = FootprintStack.from_footprints([], (40, 50))
    dense = stack.to_dense()
    assert dense.shape == (0, 40, 50)


def test_int_indexing_returns_footprint():
    stack = _stack_of(3)
    assert isinstance(stack[1], Footprint)


def test_bool_mask_indexing_subsets_units():
    stack = _stack_of(4)
    mask = np.array([True, False, True, False])
    sub = stack[mask]
    assert isinstance(sub, FootprintStack)
    assert len(sub) == 2
    np.testing.assert_array_equal(sub[0].to_dense(), stack[0].to_dense())
    np.testing.assert_array_equal(sub[1].to_dense(), stack[2].to_dense())


def test_bad_mask_length_raises():
    stack = _stack_of(3)
    with pytest.raises(IndexError):
        _ = stack[np.array([True, False])]


def test_canvas_mismatch_raises():
    fp = Footprint.from_dense(_disk(40, 40, 20, 20, 4))
    with pytest.raises(ValueError):
        FootprintStack(footprints=(fp,), canvas_shape=(32, 32))


def test_to_arrays_round_trip():
    stack = _stack_of(5)
    offsets, shapes, data = stack.to_arrays()
    assert offsets.shape == (5, 2)
    assert shapes.shape == (5, 2)
    rebuilt = FootprintStack.from_arrays(offsets, shapes, data, stack.canvas_shape)
    assert len(rebuilt) == 5
    np.testing.assert_array_equal(rebuilt.to_dense(), stack.to_dense())


def test_to_arrays_round_trip_empty():
    stack = FootprintStack.from_footprints([], (16, 16))
    offsets, shapes, data = stack.to_arrays()
    rebuilt = FootprintStack.from_arrays(offsets, shapes, data, (16, 16))
    assert len(rebuilt) == 0
    assert rebuilt.canvas_shape == (16, 16)

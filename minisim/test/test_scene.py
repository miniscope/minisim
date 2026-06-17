"""Unit tests for the ``minisim`` runtime substrate.

Covers the empty, correctly-shaped ``Scene`` built by ``Scene.zeros`` /
``Scene.ones`` (shape, dims, coords, fill, dtype, mutability), the RNG handling,
and the ``Cell`` / ``GroundTruthBuilder`` defaults. The step bodies that
*populate* the scene live in :mod:`minisim.steps`; this file only exercises the
substrate.
"""

import numpy as np
import xarray as xr

from minisim import Acquisition, ImageSensor

# Scene / Cell / GroundTruthBuilder are the mutable working-state internals: they
# live in minisim.scene, not the top-level public surface, so import them here from
# the submodule that owns them.
from minisim.scene import MOVIE_DIMS, Cell, GroundTruthBuilder, Scene


def _tiny_acquisition(**kw):
    """32×24 sensor, 25 s at 20 fps → a small, fast scene."""
    kw.setdefault("fps", 20.0)
    kw.setdefault("duration_s", 25.0)
    kw.setdefault("image_sensor", ImageSensor(n_px_height=32, n_px_width=24))
    return Acquisition(**kw)


# --- Scene.zeros / Scene.ones shape & content ------------------------------


def test_zeros_shape_dims_and_coords():
    acq = _tiny_acquisition()
    scene = Scene.zeros(acq)
    assert isinstance(scene.movie, xr.DataArray)
    assert scene.movie.dims == MOVIE_DIMS
    assert scene.movie.shape == (acq.n_frames, 32, 24)  # (500, 32, 24)
    np.testing.assert_array_equal(scene.movie["frame"].values, np.arange(acq.n_frames))
    np.testing.assert_array_equal(scene.movie["height"].values, np.arange(32))
    np.testing.assert_array_equal(scene.movie["width"].values, np.arange(24))


def test_zeros_is_all_zero_and_ones_is_all_one():
    acq = _tiny_acquisition()
    assert (Scene.zeros(acq).movie.values == 0.0).all()
    assert (Scene.ones(acq).movie.values == 1.0).all()


def test_working_movie_is_float64():
    # Honest-radiometry accumulation happens in float64; the downcast to
    # Output.store_dtype is a finalize() concern, not here.
    assert Scene.zeros(_tiny_acquisition()).movie.dtype == np.float64


def test_movie_is_mutable():
    # Unlike a frozen Spec, a Scene is mutated in place by steps.
    scene = Scene.zeros(_tiny_acquisition())
    scene.movie.loc[{"frame": 0, "height": 1, "width": 2}] = 5.0
    assert scene.movie.isel(frame=0, height=1, width=2) == 5.0


# --- lazy movie allocation --------------------------------------------------


def test_movie_is_lazy_until_accessed():
    # A fresh scene has no movie buffer; canvas_shape is known without forcing one
    # (so a cell-domain-only build never pays for the (n_frames, H, W) array).
    scene = Scene.zeros(_tiny_acquisition())
    assert scene.has_movie is False
    assert scene.canvas_shape == (32, 24)  # sensor FOV, no allocation
    assert scene.has_movie is False
    _ = scene.movie.values  # first access materializes it
    assert scene.has_movie is True


def test_canvas_shape_includes_margin_without_allocating():
    scene = Scene.zeros(_tiny_acquisition(), margin_px=4)
    assert scene.canvas_shape == (32 + 8, 24 + 8)
    assert scene.has_movie is False
    assert scene.movie.shape == (scene.acq.n_frames, 40, 32)  # margin honored on build


def test_canvas_shape_reads_a_hand_set_movie():
    # An explicitly assigned (oversized) movie defines the canvas the steps see.
    scene = Scene.zeros(_tiny_acquisition())
    scene.movie = xr.DataArray(
        np.zeros((scene.acq.n_frames, 50, 40)), dims=MOVIE_DIMS
    )
    assert scene.has_movie is True
    assert scene.canvas_shape == (50, 40)


# --- RNG handling ----------------------------------------------------------


def test_rng_defaults_to_a_generator():
    assert isinstance(Scene.zeros(_tiny_acquisition()).rng, np.random.Generator)


def test_provided_rng_is_stored_as_is():
    rng = np.random.default_rng(123)
    assert Scene.ones(_tiny_acquisition(), rng=rng).rng is rng


# --- fresh-scene defaults --------------------------------------------------


def test_fresh_scene_has_empty_collections():
    scene = Scene.zeros(_tiny_acquisition())
    assert scene.cells == []
    assert scene.snapshots == {}
    assert isinstance(scene.truth, GroundTruthBuilder)


def test_ground_truth_builder_starts_all_none():
    gt = GroundTruthBuilder()
    assert (
        gt.shifts
        is gt.vignette
        is gt.leakage
        is gt.neuropil_temporal
        is gt.neuropil_spatial
        is gt.neuropil_population
        is None
    )


# --- Cell record -----------------------------------------------------------


def test_cell_defaults_are_unpopulated():
    cell = Cell(center_um=(50.0, 10.0, 12.0))
    assert cell.center_um == (50.0, 10.0, 12.0)
    assert (
        cell.footprint_planted
        is cell.observed_sigma_px
        is cell.observed_gain
        is cell.trace
        is cell.spikes
        is cell.amplitude
        is cell.in_focus
        is cell.optical_brightness
        is cell.detectable
        is None
    )


def test_scene_internals_are_submodule_only():
    # The mutable working-state types are deliberately not on the top-level public
    # surface (they are an implementation detail of the pipeline, reachable via
    # minisim.scene for the step authors who build against them). Guard the
    # decision so the surface cannot silently regrow.
    import minisim

    for name in ("Scene", "Cell", "GroundTruthBuilder"):
        assert not hasattr(minisim, name), f"{name} leaked back onto the top-level surface"
        assert name not in minisim.__all__
        assert hasattr(__import__("minisim.scene", fromlist=[name]), name)

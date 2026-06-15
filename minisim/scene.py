"""Mutable runtime state for the ``minisim`` pipeline.

Where :mod:`minisim.spec` is the *immutable* description of a
recording, this module is its *mutable* counterpart - the working state the
executable steps read and write as they run. The split mirrors the rest of the
design: a ``Spec`` is a frozen pydantic tree you can serialize and hash; a
``Scene`` is a plain dataclass the steps mutate in place.

A step is a small callable ``step(scene) -> None`` that
either fills the per-cell records in :attr:`Scene.cells`, composites into
:attr:`Scene.movie`, or records its ground-truth contribution on
:attr:`Scene.truth`. Tests construct a ``Scene`` directly - usually with
:meth:`Scene.zeros` or :meth:`Scene.ones` - and run a single step against it;
that, not the full ``simulate()`` loop, is the primary unit-test substrate.

This file provides the empty, correctly-shaped substrate and its constructors
only. The step bodies that populate it live in :mod:`minisim.steps`; the
``finalize()`` that turns an exhausted ``Scene`` into a ``Recording`` (and its
numpydantic ``GroundTruth``) lives in :mod:`minisim.recording`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from minisim.footprint import Footprint, degrade_footprint
from minisim.spec import Acquisition

# Movie axis order, matching the convention used by 1p analysis pipelines (e.g.
# minian), so a recording's axes line up with what those pipelines expect.
MOVIE_DIMS = ("frame", "height", "width")


@dataclass
class Cell:
    """One simulated neuron - the per-cell record steps fill in, pre-composite.

    The fields mirror the per-cell structural columns of the eventual
    ``GroundTruth`` output one-for-one, so ``finalize()`` can
    stack them with no remapping. Each is populated by the step that owns it and
    is ``None`` until then:

    * ``center_um`` - set by ``place_neurons`` (the cell exists once it has a
      location). Placement is now purely spatial; brightness is not set here.
    * ``footprint_planted`` - the sharp, peak-normalized soma mask, also from
      ``place_neurons``; the ideal, optics-free target.
    * ``trace`` / ``spikes`` / ``amplitude`` - the noise-free calcium trace
      ``C``, spike train ``S``, and the per-cell brightness/expression gain that
      scales the whole trace, all from ``cell_activity``. The gain is biology
      (how much fluorescence this cell emits per spike); measurement noise, and
      hence any SNR, emerges later from ``optics`` and ``sensor``, not here.
    * ``bleach`` - the intact-fluorophore envelope ``B(t)`` from the optional
      ``bleaching`` step; ``composite`` emits ``trace · bleach``, leaving ``trace``
      the clean calcium. ``None`` until/unless bleaching runs.
    * ``observed_sigma_px`` / ``observed_gain`` / ``in_focus`` /
      ``optical_brightness`` - from the ``optics`` step. The observed
      (optically degraded) footprint is **not stored**: it is a deterministic
      function ``gain · (planted ⊛ Gaussian(sigma_px))`` of the planted footprint
      (see :func:`minisim.footprint.degrade_footprint`), so the optics step keeps
      only the two scalars and ``composite`` / ``GroundTruth.A_observed`` regenerate
      the footprint on demand. ``in_focus`` is the geometric in-focus flag and
      ``optical_brightness`` the depth-driven peak-brightness scalar.
    * ``detectable`` is *not* an optics-only property and so is **not** set by
      the optics step: it is a whole-pipeline flag (optical brightness ×
      illumination falloff, judged against the sensor noise floor) assembled in
      ``finalize()``.

    ``center_um`` is ``(z, y, x)`` in µm (depth first). ``z`` is depth below the
    tissue surface (0 = surface); ``y, x`` are lateral in the **optical-center
    frame** - the optical axis is ``(0, 0)``, ``+y`` down and ``+x`` right (image
    convention), so the frame is invariant to the motion margin. Pixel coordinates
    are a conversion away via ``acq.um_to_index`` and are not stored, to avoid drift.
    """

    center_um: tuple[float, float, float]
    footprint_planted: Footprint | None = None
    observed_sigma_px: float | None = None
    observed_gain: float | None = None
    trace: np.ndarray | None = None
    spikes: np.ndarray | None = None
    amplitude: float | None = None
    bleach: np.ndarray | None = None
    in_focus: bool | None = None
    optical_brightness: float | None = None
    detectable: bool | None = None

    def observed_footprint(self) -> Footprint | None:
        """The optically degraded footprint, regenerated from the planted one.

        Returns the planted footprint blurred + dimmed by the optics step's
        scalars (``observed_sigma_px``/``observed_gain``), or the planted footprint
        unchanged when the optics step has not run, or ``None`` if the cell has no
        footprint yet. Regenerated, not stored: it is a deterministic function of
        the planted footprint (see :func:`minisim.footprint.degrade_footprint`),
        and deep cells' observed footprints are near-full-canvas, so storing them
        would dominate memory. ``composite`` and ``GroundTruth.A_observed`` call this.
        """
        if (
            self.footprint_planted is None
            or self.observed_sigma_px is None
            or self.observed_gain is None
        ):
            return self.footprint_planted
        return degrade_footprint(
            self.footprint_planted, self.observed_sigma_px, self.observed_gain
        )


@dataclass
class GroundTruthBuilder:
    """Per-effect ground-truth side channel - each non-cell step writes its own.

    The per-*cell* truth lives on :attr:`Scene.cells`; this accumulator holds the
    per-*effect* fields that have no natural per-cell home. Each is
    ``None`` until the step that produces it runs, so a ``None`` value is the
    honest signal that the effect is absent from this recording:

    * ``shifts`` - rigid (dy, dx) per frame, from ``brain_motion``.
    * ``illumination`` / ``vignette`` / ``leakage`` - the static (height, width)
      optical fields (excitation falloff, collection falloff, additive glow).
    * ``neuropil_temporal`` / ``neuropil_spatial`` - the diffuse-background
      components; ``neuropil_population`` - the (frame,) population driver that
      modulates them (``None`` when no cells were active to drive it).
    * ``vasculature_mask`` - the static (height, width) vessel transmission mask
      from ``vasculature`` (``None`` when the step is off / has no layers).

    The steps that fill these slots live in :mod:`minisim.steps`, and
    ``finalize()`` reads them into the frozen ``GroundTruth``.
    """

    shifts: np.ndarray | None = None
    illumination: np.ndarray | None = None
    vignette: np.ndarray | None = None
    leakage: np.ndarray | None = None
    neuropil_temporal: np.ndarray | None = None
    neuropil_spatial: np.ndarray | None = None
    neuropil_population: np.ndarray | None = None
    vasculature_mask: np.ndarray | None = None
    # The concrete focal depth the optics step resolved (µm below the surface);
    # the one number "auto" focus turns into, recorded so it is observable.
    focal_depth_um: float | None = None


@dataclass
class Scene:
    """The mutable working state a pipeline of steps mutates in place.

    Holds the acquisition (which owns every µm↔px / s↔frame conversion), the RNG
    every stochastic step draws from, the movie being built, the per-cell
    records, the per-effect ground-truth side channel, and the optional
    per-stage snapshots. Unlike a ``Spec`` it is *not* frozen - mutation is the
    point.

    The working ``movie`` is held in **float64**: effects accumulate additively
    and multiplicatively across ~10 steps in honest radiometric units, and only
    the final ``sensor`` step quantizes to integer counts. The downcast to
    ``Output.store_dtype`` happens in ``finalize()``, not here.
    """

    acq: Acquisition
    rng: np.random.Generator
    cells: list[Cell] = field(default_factory=list)
    truth: GroundTruthBuilder = field(default_factory=GroundTruthBuilder)
    snapshots: dict[str, xr.DataArray] = field(default_factory=dict)
    # The working movie is allocated **lazily**: a partial build that runs only
    # cell-domain steps (e.g. ``until="optics"``) never writes a pixel, so it must
    # never pay for the ``(n_frames, H, W)`` buffer - at long durations that buffer
    # dominates both memory and time. ``_fill``/``_margin_px`` remember how to build
    # it on first access; ``_movie`` stays ``None`` until a movie-writing step (or
    # any ``.movie`` read) materializes it.
    _fill: float = 0.0
    _margin_px: int = 0
    _movie: xr.DataArray | None = field(default=None, repr=False)

    @property
    def movie(self) -> xr.DataArray:
        """The working movie, materialized on first access (lazy; see fields above)."""
        if self._movie is None:
            self._movie = self._materialize_movie()
        return self._movie

    @movie.setter
    def movie(self, value: xr.DataArray) -> None:
        self._movie = value

    @property
    def has_movie(self) -> bool:
        """Whether a movie buffer exists yet - ``True`` once a pixel step (or a
        ``.movie`` read) has materialized it. ``finalize`` uses this to skip the
        observed-movie cast for a cell-domain-only partial build."""
        return self._movie is not None

    @property
    def canvas_shape(self) -> tuple[int, int]:
        """The tissue-canvas ``(height, width)`` **without** materializing the movie.

        Equals the movie's spatial shape once one exists (so a hand-set oversized
        canvas is honored); otherwise the sensor FOV padded by the motion margin.
        Lets cell-domain steps (``place_neurons``, ``optics``) size their grid
        without forcing the buffer into existence.
        """
        if self._movie is not None:
            h, w = self._movie.values.shape[1:]
            return (int(h), int(w))
        sensor = self.acq.image_sensor
        return (
            sensor.n_px_height + 2 * self._margin_px,
            sensor.n_px_width + 2 * self._margin_px,
        )

    def _materialize_movie(self) -> xr.DataArray:
        """Build the ``(n_frames, H, W)`` working buffer filled to ``_fill``."""
        n_frames = self.acq.n_frames
        height, width = self.canvas_shape
        return xr.DataArray(
            np.full((n_frames, height, width), self._fill, dtype=np.float64),
            dims=list(MOVIE_DIMS),
            coords={
                "frame": np.arange(n_frames),
                "height": np.arange(height),
                "width": np.arange(width),
            },
            name="movie",
        )

    @classmethod
    def zeros(
        cls,
        acq: Acquisition,
        rng: np.random.Generator | None = None,
        margin_px: int = 0,
    ) -> Scene:
        """A blank scene whose movie is all zeros - the base for additive builds."""
        return cls._blank(acq, 0.0, rng, margin_px)

    @classmethod
    def ones(
        cls,
        acq: Acquisition,
        rng: np.random.Generator | None = None,
        margin_px: int = 0,
    ) -> Scene:
        """A scene whose movie is all ones - the substrate for multiplicative-field tests.

        A ``vignette`` or ``leakage`` step applied to an all-ones movie yields the
        bare field, which is exactly what a single-step test inspects.
        """
        return cls._blank(acq, 1.0, rng, margin_px)

    @classmethod
    def _blank(
        cls,
        acq: Acquisition,
        fill: float,
        rng: np.random.Generator | None,
        margin_px: int = 0,
    ) -> Scene:
        # ``margin_px`` pads the tissue canvas by that many pixels on every side
        # beyond the sensor FOV, so that under motion real, simulated tissue moves
        # into view at the edges (rather than a fabricated fill). The ``brain_motion``
        # step shifts this canvas and crops the centered sensor FOV back out; the
        # margin must be ≥ the maximum shift. ``simulate()`` sizes it from
        # the motion spec; tests pass it explicitly. ``margin_px=0`` is the plain
        # sensor-sized scene the non-motion steps use. The movie itself is not
        # allocated here - it is built lazily on first access (see the fields), so
        # ``fill``/``margin_px`` are just remembered.
        if rng is None:
            rng = np.random.default_rng()
        return cls(acq=acq, rng=rng, _fill=fill, _margin_px=margin_px)

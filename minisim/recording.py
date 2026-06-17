"""Typed simulator output - ``Recording`` / ``GroundTruth`` - and ``finalize()``.

Where :mod:`minisim.scene` is the mutable working state a pipeline of
steps fills in, this module is the *frozen, typed result* that state is distilled
into once the pipeline is exhausted. :func:`finalize` is the transform: it turns a
run-out ``Scene`` into a ``Recording`` carrying the observed movie, per-stage
snapshots, and the numpydantic-typed ``GroundTruth`` that tests, metrics, and the
training notebooks consume.

Two things that were deliberately deferred from earlier steps land here:

* **FOV cropping.** Under motion the tissue steps render on a canvas larger than
  the sensor (see :mod:`minisim.steps.motion`); cells carry canvas-sized
  footprints and canvas-frame positions. ``finalize`` crops them to the sensor
  FOV at the reference (zero-shift) frame - the template motion correction aligns
  back to - and drops cells whose reference footprint falls entirely in the
  margin (real tissue, but background that only flickers in transiently, not a
  recoverable unit).
* **Detectability.** ``detectable`` is not an optics-only property: a cell's peak
  signal (``optical_brightness``) is further dimmed by the illumination/vignette
  field at its position and then judged against a sensor-derived noise floor.
  ``finalize`` is the first point all three exist, so it is where the flag is set.
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from minisim.footprint import Footprint, FootprintStack, degrade_footprint
from minisim.scene import MOVIE_DIMS, Cell, Scene
from minisim.spec import Acquisition, Spec

# Minimum realized peak SNR (signal electrons over the sensor noise floor) for a
# cell to count as detectable. A provisional value: a future threshold
# calibration revisits it against observed metric distributions. Kept here, named,
# rather than buried as a literal so that calibration is a one-line change.
#
# STABILITY CONTRACT. This single constant sets the `detectable` flag, the
# `restrict_to_detectable` recall denominator in `minisim.testing.score`, and the
# "auto" focus/exposure yield objective - so its value is part of what a consumer's
# `assert report.recall > X` depends on. It is provisional and *not yet calibrated*
# against a real pipeline, so it may change before 1.0. Treat any change as a
# minor-version bump with a changelog entry, because it moves recall denominators
# under downstream tests (see the reproducibility & stability contract in the docs).
DETECT_SNR_THRESHOLD = 3.0

# On-disk layout for save()/load() (zarr group + sibling spec.json). The format
# version is the cross-version compatibility boundary: `load()` reads it back and
# refuses a layout it does not understand, rather than silently misreading. Any
# change to the on-disk layout (new/renamed datasets, a different sparse encoding)
# MUST bump this, so an older reader fails loudly with a migration message instead
# of mis-parsing. Within a single format version the spec hash is only an *advisory*
# staleness check (see `load`), so a benign spec-field addition that changes the
# hash does not brick already-saved recordings.
# v2: footprints stored sparse (planted patches; observed regenerated on load).
_FORMAT_VERSION = 2


class RecordingFormatWarning(UserWarning):
    """Advisory warning when a loaded recording's spec hash does not match.

    Raised by :meth:`Recording.load` when the stored ``spec_cache_key`` differs from
    the recomputed one *within a matching format version* - typically because this
    minisim serializes the spec slightly differently than the version that wrote the
    file (a benign field addition changes the hash), or because ``spec.json`` was
    hand-edited. The recording still loads (the arrays are independent of the hash);
    the warning flags that the spec it is paired with may be stale. A genuine layout
    incompatibility is a *hard* error instead, keyed on ``format_version``.
    """
# Plain GroundTruth array fields saved as datasets, split by whether they are always
# present or optional (None when their producing step is absent). The sparse planted
# footprints (and the fov crop) are handled separately; A_observed is not stored.
_GT_REQUIRED = (
    "C", "S", "centers_um", "amplitude_per_cell", "in_focus", "detectable",
)
_GT_OPTIONAL = (
    "observed_sigma_px", "observed_gain",
    "shifts", "illumination", "vignette", "leakage", "bleaching",
    "neuropil_temporal", "neuropil_spatial", "neuropil_population",
    "vasculature_mask", "vessel_overlap_fraction",
)

# Step kinds whose snapshot key differs from the kind, so Recording.stage() (and
# the until= argument of simulate()) accept either spelling. composite is the only
# step whose stage name (`cells_only`) is not its kind.
_STAGE_ALIASES = {"composite": "cells_only"}


class GroundTruth(BaseModel):
    """The per-recording truth: structural targets + per-cell and per-effect fields.

    The **planted vs observed footprint split is load-bearing**: ``A_observed`` is
    the optically degraded footprint - the recoverable target an analysis pipeline's
    estimate is matched against - while ``A_planted`` is the ideal, optics-free
    footprint that quantifies the irreducible limit. Both are exposed as dense
    ``(unit, height, width)`` arrays via properties, but neither is *stored* dense:

    * Footprints are stored sparse, as canvas-coordinate patches in :attr:`planted`
      (a :class:`~minisim.footprint.FootprintStack`); :attr:`fov_offset` /
      :attr:`fov_shape` crop them to the sensor FOV.
    * The observed footprint is **not stored at all**: it is the deterministic blur
      ``gain · (planted ⊛ Gaussian(sigma_px))`` of the planted one, so it is
      regenerated on demand from the per-unit :attr:`observed_sigma_px` /
      :attr:`observed_gain` scalars. Deep cells' observed footprints are
      near-full-canvas, so storing them dominated memory and disk; regenerating is
      bit-identical and far cheaper to keep.

    Per-effect fields are ``None`` when their step is absent from the recording.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # structural truth (sparse) --------------------------------------------
    # Canvas-coordinate planted footprints + the crop to the sensor FOV. Stored in
    # canvas coords (not pre-cropped) so the regenerated observed footprint exactly
    # matches blur-then-crop even for cells straddling the FOV edge.
    planted: FootprintStack
    fov_offset: tuple[int, int]  # (top, left) crop from canvas to sensor FOV
    fov_shape: tuple[int, int]  # (height, width) of the sensor FOV
    # Per-unit optics scalars defining the observed footprint; both None when the
    # optics step did not run (then A_observed falls back to A_planted).
    observed_sigma_px: NDArray[Shape["* unit"], float] | None = None
    observed_gain: NDArray[Shape["* unit"], float] | None = None

    C: NDArray[Shape["* unit, * frame"], float]
    S: NDArray[Shape["* unit, * frame"], float]

    # per-cell physical attributes -----------------------------------------
    # (z, y, x) µm: z is depth below the surface; y, x are lateral in the
    # optical-center frame (origin = optical axis / FOV center, +y down, +x right).
    centers_um: NDArray[Shape["* unit, 3"], float]
    # Per-cell brightness/expression gain (the clean input). NaN for a cell whose
    # ``cell_activity`` step has not run. SNR is deliberately absent: noise is a
    # downstream physical effect, so any SNR is computed later, not stored here.
    amplitude_per_cell: NDArray[Shape["* unit"], float]
    in_focus: NDArray[Shape["* unit"], bool]
    detectable: NDArray[Shape["* unit"], bool]

    # per-effect ground truth (None when that step is absent) ---------------
    shifts: NDArray[Shape["* frame, 2"], float] | None = None
    illumination: NDArray[Shape["* height, * width"], float] | None = None
    vignette: NDArray[Shape["* height, * width"], float] | None = None
    leakage: NDArray[Shape["* height, * width"], float] | None = None
    bleaching: NDArray[Shape["* unit, * frame"], float] | None = None
    neuropil_temporal: NDArray[Shape["* component, * frame"], float] | None = None
    neuropil_spatial: NDArray[Shape["* component, * height, * width"], float] | None = None
    neuropil_population: NDArray[Shape["* frame"], float] | None = None
    # The static vessel transmission mask (height, width) in (0, 1] from the
    # vasculature step (cropped to the FOV); None when the step is off / layer-less.
    vasculature_mask: NDArray[Shape["* height, * width"], float] | None = None
    # Per-cell fraction of footprint-integrated light absorbed by vessels, in [0, 1)
    # (0 = clear, ->1 = a vessel sits over the whole footprint). The scoreable
    # confound axis: stratify recall / footprint-correlation by vessel burden. None
    # when the vasculature step is off. The footprints themselves stay vessel-free
    # (A_observed is the single-cell optical truth); occlusion lives here, not in A.
    vessel_overlap_fraction: NDArray[Shape["* unit"], float] | None = None
    # The concrete focal depth (µm) the optics step resolved "auto" to - the plane
    # that maximized recoverable yield. A scalar, so persisted as a group attr (not
    # a dataset) by save/load; None when the optics step did not run.
    focal_depth_um: float | None = None
    # The concrete exposure (photons per intensity unit) the sensor step resolved
    # "auto" to - the level that lands the brightest cell near the top of the ADC
    # range. A scalar, persisted as a group attr; None when the sensor step did not
    # run. Equals Sensor.photons_per_unit when that was given numerically.
    exposure_photons_per_unit: float | None = None

    # Memoizes the regenerated dense A_observed (one blur pass over all units), so
    # repeated reads on the same object are free. A private attr, so it does not
    # affect equality, serialization, or the frozen field set.
    _observed_cache: np.ndarray | None = PrivateAttr(default=None)

    @property
    def n_units(self) -> int:
        """Number of ground-truth cells (units) in the recording."""
        return len(self.planted)

    @property
    def A_planted(self) -> np.ndarray:
        """The sharp, pre-optics footprints, dense ``(unit, height, width)`` over the FOV."""
        top, left = self.fov_offset
        h, w = self.fov_shape
        return self.planted.crop(top, left, h, w).to_dense(dtype=float)

    @property
    def A_observed(self) -> np.ndarray:
        """The optically degraded footprints - the recoverable target, dense ``(unit, H, W)``.

        Regenerated (and memoized) from the planted footprints and the per-unit
        ``observed_sigma_px`` / ``observed_gain`` scalars, then cropped to the FOV -
        bit-identical to what the optics step produced. Falls back to
        :attr:`A_planted` when the optics step did not run.
        """
        if self._observed_cache is None:
            self._observed_cache = self._regenerate_observed()
        return self._observed_cache

    def _regenerate_observed(self) -> np.ndarray:
        top, left = self.fov_offset
        h, w = self.fov_shape
        if self.observed_sigma_px is None:
            return self.A_planted  # no optics -> observed == planted
        observed = []
        for i, fp in enumerate(self.planted):
            sigma = float(self.observed_sigma_px[i])
            if np.isnan(sigma):  # a cell without optics params -> sharp footprint
                observed.append(fp)
            else:
                observed.append(degrade_footprint(fp, sigma, float(self.observed_gain[i])))
        return FootprintStack(tuple(observed), self.planted.canvas_shape).crop(
            top, left, h, w
        ).to_dense(dtype=float)

    @property
    def depth_um(self) -> np.ndarray:
        """Per-cell depth ``z`` (µm) - the first column of ``centers_um``.

        Exposed as a derived view rather than stored, to avoid drift. The lateral
        ``centers_um[:, 1:]`` are in the optical-center frame, so pixel coordinates
        are ``acq.um_to_index(y, x, fov_shape)`` (origin = FOV center), using the
        owning ``Recording.spec.acquisition``.
        """
        return self.centers_um[:, 0]

    def detectable_subset(self) -> GroundTruth:
        """Subset to detectable cells - the fair denominator for recall metrics.

        Slices the per-unit fields by the ``detectable`` mask (the planted stack,
        the optics scalars, and ``bleaching`` are all per-unit); the per-effect
        fields (shifts, vignette, neuropil, …) are not per-unit and are carried
        unchanged.
        """
        m = self.detectable
        return GroundTruth(
            planted=self.planted[m],
            fov_offset=self.fov_offset,
            fov_shape=self.fov_shape,
            observed_sigma_px=self.observed_sigma_px[m] if self.observed_sigma_px is not None else None,
            observed_gain=self.observed_gain[m] if self.observed_gain is not None else None,
            C=self.C[m],
            S=self.S[m],
            centers_um=self.centers_um[m],
            amplitude_per_cell=self.amplitude_per_cell[m],
            in_focus=self.in_focus[m],
            detectable=self.detectable[m],
            vessel_overlap_fraction=(
                self.vessel_overlap_fraction[m]
                if self.vessel_overlap_fraction is not None else None
            ),
            bleaching=self.bleaching[m] if self.bleaching is not None else None,
            shifts=self.shifts,
            illumination=self.illumination,
            vignette=self.vignette,
            leakage=self.leakage,
            neuropil_temporal=self.neuropil_temporal,
            neuropil_spatial=self.neuropil_spatial,
            neuropil_population=self.neuropil_population,
            vasculature_mask=self.vasculature_mask,
            focal_depth_um=self.focal_depth_um,
            exposure_photons_per_unit=self.exposure_photons_per_unit,
        )


class Recording(BaseModel):
    """A complete simulated recording: the spec, the observed movie, and the truth.

    ``observed`` holds the integer-valued sensor counts in a float container (per
    ``Output.store_dtype``). ``snapshots`` is populated only when
    ``Output.save_intermediates`` is set, keyed by each step's stage ``name``;
    ``stage()`` reads them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    spec: Spec
    observed: NDArray[Shape["* frame, * height, * width"], float]
    ground_truth: GroundTruth
    snapshots: dict[str, xr.DataArray] = Field(default_factory=dict)

    def stage(self, name: str) -> xr.DataArray:
        """Return the snapshot taken after the named stage.

        Accepts a stage name (the snapshot key) or the equivalent step ``kind``:
        ``stage("composite")`` resolves to the ``"cells_only"`` snapshot, matching
        the ``until=`` argument of :func:`~minisim.simulate`.
        """
        key = _STAGE_ALIASES.get(name, name)
        if key not in self.snapshots:
            raise KeyError(
                f"Stage '{name}' unavailable. Set Output.save_intermediates=True, "
                f"or pick from {sorted(self.snapshots)}."
            )
        return self.snapshots[key]

    def save(self, path: str | Path) -> None:
        """Persist this recording to a self-contained zarr directory at ``path``.

        Layout (one portable directory; ``path`` is conventionally
        ``{spec.cache_key()}.zarr`` but any path works)::

            {path}/
                spec.json            human-readable, diffable spec
                (group attrs)        format_version, spec_cache_key, gt_present,
                                     snapshot_names
                observed             (frame, height, width) in store_dtype
                ground_truth/        the GroundTruth arrays (optional ones only
                                     when not None; listed in the gt_present attr)
                    planted/         sparse footprints: offsets, shapes, data
                                     (+ canvas_shape/fov_offset/fov_shape attrs)
                snapshots/           per-stage movie values, only when non-empty

        Footprints are stored sparse (the ``planted`` subgroup holds the ragged
        patch arrays); the observed footprints are not stored -- they regenerate
        from the per-unit ``observed_sigma_px`` / ``observed_gain`` scalars.
        Snapshot coordinates are not stored - they are the trivial ``arange`` grid
        over ``MOVIE_DIMS`` and are rebuilt on :meth:`load`. The write is atomic: it
        builds a sibling ``{path}.tmp`` and renames it into place, so a crash never
        leaves a half-written directory that :meth:`load` would trust.
        """
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)

        gt = self.ground_truth
        root = zarr.open_group(str(tmp), mode="w")
        root.create_dataset("observed", data=np.asarray(self.observed))

        gt_group = root.create_group("ground_truth")
        for name in _GT_REQUIRED:
            gt_group.create_dataset(name, data=np.asarray(getattr(gt, name)))
        present = []
        for name in _GT_OPTIONAL:
            value = getattr(gt, name)
            if value is not None:
                gt_group.create_dataset(name, data=np.asarray(value))
                present.append(name)

        # Sparse planted footprints: ragged patches flattened to three datasets.
        offsets, shapes, data = gt.planted.to_arrays()
        planted_group = gt_group.create_group("planted")
        planted_group.create_dataset("offsets", data=offsets)
        planted_group.create_dataset("shapes", data=shapes)
        planted_group.create_dataset("data", data=data)
        planted_group.attrs["canvas_shape"] = list(gt.planted.canvas_shape)
        planted_group.attrs["fov_offset"] = list(gt.fov_offset)
        planted_group.attrs["fov_shape"] = list(gt.fov_shape)

        snapshot_names = sorted(self.snapshots)
        if snapshot_names:
            snap_group = root.create_group("snapshots")
            for name in snapshot_names:
                snap_group.create_dataset(name, data=np.asarray(self.snapshots[name].values))

        # focal_depth_um / exposure_photons_per_unit are scalars, not arrays: stash
        # them as group attrs (the resolved "auto" focus and exposure values).
        gt_group.attrs["focal_depth_um"] = gt.focal_depth_um
        gt_group.attrs["exposure_photons_per_unit"] = gt.exposure_photons_per_unit

        root.attrs["format_version"] = _FORMAT_VERSION
        root.attrs["spec_cache_key"] = self.spec.cache_key()
        root.attrs["gt_present"] = present
        root.attrs["snapshot_names"] = snapshot_names
        (tmp / "spec.json").write_text(self.spec.model_dump_json(indent=2))

        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Load a recording written by :meth:`save`.

        Two checks, with deliberately different severities:

        * **``format_version`` is a hard boundary.** If the on-disk layout version is
          not one this reader understands, ``load`` raises rather than risk
          mis-parsing a layout that changed incompatibly. Any layout change bumps the
          version, so an old file fails loudly with a re-simulate message.
        * **The spec hash is advisory.** Within a matching format version, a
          ``spec_cache_key`` mismatch only warns (:class:`RecordingFormatWarning`):
          the arrays load fine, and the most common cause is benign - this minisim
          serializes the spec slightly differently than the version that wrote the
          file, which changes the hash without changing the data. Hard-failing here
          would brick already-saved fixtures on a minisim upgrade. (The cache layer
          does not rely on this check - it keys recordings by filename, so a changed
          spec hash is a fresh filename and a clean miss.)

        Snapshots are rebuilt as ``DataArray``s over ``MOVIE_DIMS`` with ``arange``
        coordinates.
        """
        path = Path(path)
        root = zarr.open_group(str(path), mode="r")

        stored_format = root.attrs.get("format_version")
        if stored_format != _FORMAT_VERSION:
            raise ValueError(
                f"Cannot load {path}: on-disk format_version {stored_format!r} is not "
                f"readable by this minisim (which writes/reads format {_FORMAT_VERSION}). "
                f"The zarr layout changed incompatibly between versions; there is no "
                f"in-place migration - re-simulate the spec with this minisim version."
            )

        spec = Spec.model_validate_json((path / "spec.json").read_text())
        stored_key = root.attrs.get("spec_cache_key")
        if stored_key != spec.cache_key():
            warnings.warn(
                f"Spec hash mismatch loading {path}: stored {stored_key!r} != "
                f"recomputed {spec.cache_key()!r}. The recording still loads; this "
                f"usually means a newer minisim serializes the spec differently than "
                f"the one that wrote the file (a benign field addition changes the "
                f"hash), or spec.json was hand-edited. If the recording looks stale, "
                f"delete it and re-simulate.",
                RecordingFormatWarning,
                stacklevel=2,
            )

        gt_group = root["ground_truth"]
        fields = {name: np.asarray(gt_group[name]) for name in _GT_REQUIRED}
        for name in root.attrs.get("gt_present", []):
            fields[name] = np.asarray(gt_group[name])
        focal = gt_group.attrs.get("focal_depth_um")
        if focal is not None:
            fields["focal_depth_um"] = float(focal)
        exposure = gt_group.attrs.get("exposure_photons_per_unit")
        if exposure is not None:
            fields["exposure_photons_per_unit"] = float(exposure)

        # Rebuild the sparse planted footprints and the FOV crop.
        planted_group = gt_group["planted"]
        canvas_shape = tuple(int(v) for v in planted_group.attrs["canvas_shape"])
        fields["planted"] = FootprintStack.from_arrays(
            np.asarray(planted_group["offsets"]),
            np.asarray(planted_group["shapes"]),
            np.asarray(planted_group["data"]),
            canvas_shape,
        )
        fields["fov_offset"] = tuple(int(v) for v in planted_group.attrs["fov_offset"])
        fields["fov_shape"] = tuple(int(v) for v in planted_group.attrs["fov_shape"])
        ground_truth = GroundTruth(**fields)

        snapshots = {
            name: _movie_dataarray(np.asarray(root["snapshots"][name]))
            for name in root.attrs.get("snapshot_names", [])
        }
        return cls(
            spec=spec,
            observed=np.asarray(root["observed"]),
            ground_truth=ground_truth,
            snapshots=snapshots,
        )

    def write_video(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        vmin: float = 0.0,
        vmax: float | None = None,
        codec: str = "Y800",
        progress: bool = True,
    ) -> Path:
        """Write the in-memory ``observed`` movie to a grayscale video at ``path``.

        Maps counts to 8-bit gray over ``[vmin, vmax]`` (``vmax`` defaults to the
        sensor's full ADC range, ``2**bit_depth - 1``, so saturation reads as white)
        and encodes with ``cv2.VideoWriter`` + the ``codec`` fourcc (default
        ``"Y800"``: uncompressed 8-bit gray -- exact counts, no artifacts, but large;
        pass ``"MJPG"`` for a small lossy file). Uses opencv's bundled ffmpeg, so no
        system ffmpeg or ``mediapy`` extra is needed. Use this when you already hold a
        ``Recording``; to stream a long recording to disk **without** building the
        whole movie in memory, use :func:`minisim.video.simulate_video` instead.
        Returns ``path``.
        """
        from minisim.video import _default_vmax, _write_gray_video

        n_frames = self.observed.shape[0]
        if n_frames == 0:
            raise ValueError("recording has no frames to write (observed is empty).")
        fps = float(fps if fps is not None else self.spec.acquisition.fps)
        fov = (self.observed.shape[1], self.observed.shape[2])
        if vmax is None:
            vmax = _default_vmax(self.spec)
        frames = (self.observed[f] for f in range(n_frames))
        return _write_gray_video(
            frames, n_frames, Path(path), fov, fps, vmin, vmax, codec, progress
        )


# ---------------------------------------------------------------------------
# finalize: Scene -> Recording
# ---------------------------------------------------------------------------


def finalize(scene: Scene, spec: Spec) -> Recording:
    """Distill an exhausted ``Scene`` into a frozen, typed ``Recording``.

    Keeps each cell's canvas-coordinate planted footprint (sparse) plus its
    canvas-frame position rebased to the sensor FOV, drops cells left entirely in
    the motion margin, records the per-unit optics scalars (so ``A_observed`` can be
    regenerated rather than stored), assembles the per-cell structural truth, sets
    ``detectable`` from the realized optical × illumination peak versus the sensor
    noise floor (folding in vessel transmission, and recording each cell's
    vessel-occlusion burden), reads the per-effect fields off ``scene.truth``, and
    downcasts the working movie to ``Output.store_dtype`` for ``observed``.
    """
    acq = scene.acq
    fov_h = acq.image_sensor.n_px_height
    fov_w = acq.image_sensor.n_px_width
    n_frames = acq.n_frames
    # Footprints were stamped on the canvas (sensor FOV + any motion margin); read
    # the canvas size off them, since a brain_motion step crops scene.movie down to
    # the FOV after stamping. Falls back to the bare FOV when there are no cells.
    canvas_h, canvas_w = next(
        (c.footprint_planted.canvas_shape for c in scene.cells if c.footprint_planted is not None),
        (fov_h, fov_w),
    )
    # Centered crop from the canvas (sensor FOV + any motion margin) down to the FOV.
    margin_h = (canvas_h - fov_h) // 2
    margin_w = (canvas_w - fov_w) // 2

    sensor_spec = next((s for s in spec.steps if s.kind == "sensor"), None)
    # The exposure the sensor step resolved (the numeric value, or what "auto" chose);
    # None when no sensor ran, which is the signal detectability falls back on. Read
    # from the resolved value, never sensor_spec.photons_per_unit, which may be "auto".
    # Fallback: when finalize runs on a scene the SensorStep never touched (a hand-
    # built test scene, or a partial build that stops before the sensor) the resolved
    # value is absent - use the spec's exposure if it was given numerically. Only an
    # unrun "auto" stays unresolved, and then detectability rightly falls back to the
    # geometric in_focus flag.
    resolved_ppu = scene.truth.exposure_photons_per_unit
    if (
        resolved_ppu is None
        and sensor_spec is not None
        and sensor_spec.photons_per_unit != "auto"
    ):
        resolved_ppu = float(sensor_spec.photons_per_unit)
    # Both falloff fields are FOV-sized (built post-motion-crop), or None. Their
    # product is the per-pixel photon budget a cell's signal is dimmed by.
    illumination = scene.truth.illumination
    vignette = scene.truth.vignette
    photon_field = _combine_fields(illumination, vignette)
    # The vessel transmission mask, if the vasculature step ran. The canvas-sized
    # version aligns with the (canvas-coordinate) footprints for the per-cell
    # overlap integral; the FOV crop is sampled at each cell's position for
    # detectability, the same frame as photon_field.
    vasc_mask_canvas = scene.truth.vasculature_mask
    vasc_mask_fov = _crop_field(vasc_mask_canvas, fov_h, fov_w)

    planted_fps, traces, spikes, bleaches = [], [], [], []
    centers, amplitudes, in_focus, detectable, overlaps = [], [], [], [], []
    sigmas, gains = [], []
    for cell in scene.cells:
        if cell.footprint_planted is None:
            continue
        # Drop a cell whose planted footprint, cropped to the FOV, is empty: it sits
        # entirely in the motion margin -- real tissue, but background that only
        # flickers in transiently, not a recoverable unit.
        if cell.footprint_planted.crop(margin_h, margin_w, fov_h, fov_w).is_empty:
            continue
        # Optical-center frame: the origin is the optical axis for both the canvas
        # and the (centered-crop) FOV, so a cell's coordinates need no margin
        # adjustment - they are already FOV-relative, motion margin or not.
        z, y_um, x_um = cell.center_um

        trace = cell.trace if cell.trace is not None else np.zeros(n_frames)
        spike = cell.spikes if cell.spikes is not None else np.zeros(n_frames)
        ifocus = cell.in_focus if cell.in_focus is not None else True

        # Stored in canvas coords; A_planted/A_observed crop to the FOV on access.
        planted_fps.append(cell.footprint_planted)
        sigmas.append(cell.observed_sigma_px if cell.observed_sigma_px is not None else np.nan)
        gains.append(cell.observed_gain if cell.observed_gain is not None else np.nan)
        traces.append(trace)
        spikes.append(spike)
        bleaches.append(cell.bleach)
        centers.append((z, y_um, x_um))
        amplitudes.append(cell.amplitude if cell.amplitude is not None else float("nan"))
        in_focus.append(ifocus)
        # A vessel over the cell absorbs part of its light: dim the peak by the
        # transmission at the cell's position (detectability is a peak test) and
        # record the footprint-weighted occlusion as the scoreable confound axis.
        vessel_t = sample_field_at(vasc_mask_fov, y_um, x_um, acq.pixel_size_um)
        detectable.append(
            _is_detectable(cell, ifocus, y_um, x_um, photon_field, resolved_ppu, acq, vessel_t)
        )
        overlaps.append(_vessel_overlap(cell.observed_footprint(), vasc_mask_canvas))

    # Optics ran iff any surviving cell carries a sigma; then keep the per-unit
    # scalars so A_observed regenerates. Otherwise None -> A_observed == A_planted.
    optics_ran = any(not np.isnan(s) for s in sigmas)
    gt = GroundTruth(
        planted=FootprintStack.from_footprints(planted_fps, (canvas_h, canvas_w)),
        fov_offset=(margin_h, margin_w),
        fov_shape=(fov_h, fov_w),
        observed_sigma_px=np.array(sigmas, dtype=float) if optics_ran else None,
        observed_gain=np.array(gains, dtype=float) if optics_ran else None,
        C=_stack(traces, (0, n_frames)),
        S=_stack(spikes, (0, n_frames)),
        centers_um=np.array(centers, dtype=float).reshape(-1, 3),
        amplitude_per_cell=np.array(amplitudes, dtype=float),
        in_focus=np.array(in_focus, dtype=bool),
        detectable=np.array(detectable, dtype=bool),
        # Per-cell vessel occlusion, present only when the vasculature step ran.
        vessel_overlap_fraction=(
            np.array(overlaps, dtype=float) if vasc_mask_canvas is not None else None
        ),
        # Per-cell bleaching envelopes (unit, frame), present only if the bleaching
        # step ran; any cell without one (e.g. added afterward) gets a no-fade row.
        bleaching=(
            _stack([b if b is not None else np.ones(n_frames) for b in bleaches], (0, n_frames))
            if any(b is not None for b in bleaches)
            else None
        ),
        shifts=scene.truth.shifts,
        illumination=illumination,
        vignette=vignette,
        leakage=scene.truth.leakage,
        neuropil_temporal=scene.truth.neuropil_temporal,
        neuropil_spatial=_crop_components(scene.truth.neuropil_spatial, fov_h, fov_w),
        neuropil_population=scene.truth.neuropil_population,
        vasculature_mask=_crop_field(scene.truth.vasculature_mask, fov_h, fov_w),
        focal_depth_um=scene.truth.focal_depth_um,
        exposure_photons_per_unit=resolved_ppu,
    )
    # observed is always the sensor FOV: brain_motion already crops the movie,
    # but a partial build (until= before motion) can leave it canvas-sized, so
    # crop the centered FOV here too for consistency with the cropped footprints.
    # A build that ran only cell-domain steps (e.g. until="optics") never wrote a
    # pixel, so no movie was materialized: return an empty (0, H, W) stack rather
    # than allocating a full zero buffer only to discard it. The per-cell ground
    # truth (C, S, A, bleaching, ...) is fully populated either way.
    if scene.has_movie:
        movie = _center_crop_hw(scene.movie.values, fov_h, fov_w)
        observed_movie = movie.astype(spec.output.store_dtype)
    else:
        observed_movie = np.zeros((0, fov_h, fov_w), dtype=spec.output.store_dtype)
    return Recording(
        spec=spec, observed=observed_movie, ground_truth=gt, snapshots=scene.snapshots
    )


def _movie_dataarray(values: np.ndarray) -> xr.DataArray:
    """Rebuild a movie ``DataArray`` from stored values (the saved snapshot form).

    Snapshots are persisted as bare arrays; their coordinates are the trivial
    ``arange`` grid over ``MOVIE_DIMS``, reconstructed here so :meth:`Recording.load`
    returns the same ``(frame, height, width)`` labelling the pipeline produced.
    """
    return xr.DataArray(
        values,
        dims=list(MOVIE_DIMS),
        coords={dim: np.arange(size) for dim, size in zip(MOVIE_DIMS, values.shape, strict=True)},
        name="movie",
    )


def _center_crop_hw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Slice the trailing ``(H, W)`` axes of ``arr`` to a centered ``(h, w)`` window."""
    top = (arr.shape[-2] - h) // 2
    left = (arr.shape[-1] - w) // 2
    return arr[..., top : top + h, left : left + w]


def _crop_components(stack: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    """Crop each ``(component, H, W)`` field to the reference FOV (or pass None)."""
    return None if stack is None else _center_crop_hw(stack, h, w)


def _crop_field(field: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    """Crop a single ``(H, W)`` field to the centered reference FOV (or pass None)."""
    return None if field is None else _center_crop_hw(field, h, w)


def _stack(arrays: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    """Stack a per-unit list, or an empty array of ``empty_shape`` when there are none."""
    return np.stack(arrays) if arrays else np.zeros(empty_shape)


def _combine_fields(
    a: np.ndarray | None, b: np.ndarray | None
) -> np.ndarray | None:
    """Element-wise product of two optional FOV fields (each absent → identity)."""
    if a is None:
        return b
    if b is None:
        return a
    return a * b


def sample_field_at(
    field: np.ndarray | None, y_um: float, x_um: float, pixel_size_um: float
) -> float:
    """Value of a static FOV field at a µm position, with edge-clamped indexing.

    Converts an optical-center ``(y_um, x_um)`` position (origin = FOV center,
    ``+y`` down, ``+x`` right) to the nearest pixel (rounded and clipped to the
    field bounds) and returns ``field`` there. ``field`` is a sensor-FOV-sized
    array such as the combined illumination × vignette photon budget; an absent
    field (``None``) returns ``1.0``, the multiplicative identity, so callers can
    treat "no field" and "uniform field" alike. The one sampler shared by
    detectability (:func:`_is_detectable`) and the teaching notebook's per-cell SNR
    panels, so both read the field the same way.
    """
    if field is None:
        return 1.0
    h, w = field.shape
    iy = int(np.clip(round((h - 1) / 2.0 + y_um / pixel_size_um), 0, h - 1))
    ix = int(np.clip(round((w - 1) / 2.0 + x_um / pixel_size_um), 0, w - 1))
    return float(field[iy, ix])


def detection_snr(peak_dF, baseline, gain, read_e):
    """Transient SNR in detected electrons - the single detectability formula.

    ``gain`` converts a cell's ΔF units to detected electrons
    (``optical_brightness · illumination · photons_per_unit · QE``); ``peak_dF``
    and ``baseline`` are the transient height and the (non-negative) resting level
    in those same ΔF units. The noise floor is shot noise on the baseline
    electrons plus the sensor read noise::

        SNR = peak_dF·gain / sqrt(max(baseline,0)·gain + read_e²)

    Works on scalars or numpy arrays (so both ``finalize`` and the auto-focus
    yield scan share one definition). Where the noise floor is exactly zero the
    SNR is ``inf`` if there is any signal, else ``0``.
    """
    signal_e = peak_dF * gain
    noise_e = np.sqrt(np.maximum(baseline, 0.0) * gain + read_e * read_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = signal_e / noise_e
    return np.where(noise_e > 0, snr, np.where(signal_e > 0, np.inf, 0.0))


def _is_detectable(
    cell: Cell,
    in_focus: bool,
    y_um: float,
    x_um: float,
    photon_field: np.ndarray | None,
    photons_per_unit: float | None,
    acq: Acquisition,
    vessel_transmission: float = 1.0,
) -> bool:
    """Whether a cell's realized peak clears the sensor noise floor (and is in focus).

    The cell's peak ΔF is dimmed by its optical brightness (depth defocus +
    scatter), the illumination/vignette field at its position, and any vessel
    transmission there (``vessel_transmission`` in (0, 1], 1.0 = no vessel), scaled
    to detected electrons by the exposure and QE, then compared to the shot + read
    noise floor riding on its steady baseline (see :func:`detection_snr`). The
    vessel term is sampled at the cell's position because detectability is a peak
    test; the footprint-weighted occlusion is recorded separately as
    ``vessel_overlap_fraction``.

    ``photons_per_unit`` is the **resolved** exposure (the numeric value, or what
    ``"auto"`` chose at the sensor step), not the raw spec field. ``detectable``
    requires ``in_focus`` and ``SNR ≥ DETECT_SNR_THRESHOLD``. With no activity (no
    trace) a cell emits no transient and is not detectable; with no ``sensor`` step
    (``photons_per_unit is None``) there is no noise floor to test against, so
    detectability falls back to the geometric ``in_focus`` flag.
    """
    if cell.trace is None:
        return False
    if photons_per_unit is None:
        return in_focus
    if not in_focus:
        return False
    brightness = cell.optical_brightness if cell.optical_brightness is not None else 1.0
    illum = sample_field_at(photon_field, y_um, x_um, acq.pixel_size_um)
    gain = (
        brightness * illum * vessel_transmission
        * photons_per_unit * acq.image_sensor.quantum_efficiency
    )
    peak_dF = float(cell.trace.max() - cell.trace.min())
    baseline = float(cell.trace.min())
    snr = detection_snr(peak_dF, baseline, gain, acq.image_sensor.read_noise_e)
    return bool(snr >= DETECT_SNR_THRESHOLD)


def _vessel_overlap(footprint: Footprint | None, mask_canvas: np.ndarray | None) -> float:
    """Footprint-weighted fraction of a cell's light absorbed by vessels, in [0, 1).

    ``mask_canvas`` is the canvas-coordinate transmission mask M in (0, 1]; the
    occlusion is ``1 − Σ(A·M)/Σ(A)`` over the cell's (canvas-coordinate) footprint
    patch - 0 where no vessel touches the cell, approaching 1 as a vessel covers
    the whole footprint. Returns 0.0 when there is no mask or no footprint, so a
    vessel-free recording reports zero burden for every cell.
    """
    if mask_canvas is None or footprint is None or footprint.is_empty:
        return 0.0
    y0, x0 = footprint.offset
    ph, pw = footprint.patch.shape
    sub = mask_canvas[y0 : y0 + ph, x0 : x0 + pw]
    total = float(footprint.patch.sum())
    if total <= 0.0:
        return 0.0
    return float(1.0 - (footprint.patch * sub).sum() / total)

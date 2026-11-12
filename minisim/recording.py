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
* **Detectability.** ``detectable`` is a pure signal test, not an optics-only
  property: a cell is detectable when the matched-filter SNR of its realized
  signal - its degraded footprint (defocus, scatter, halo), dimmed by vessels,
  illumination and vignette, pooled over its pixels and over all its transients -
  against the per-pixel shot + read noise clears ``DETECT_SNR_THRESHOLD`` (see
  :func:`matched_filter_snr`). Defocus is not a hard gate - it lowers the SNR
  continuously, so a blurred but bright cell outside the depth of field can still be
  detectable, matching what a real pipeline recovers. ``finalize`` is the first point
  all the terms exist, so it is where the flag is set.
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
from minisim.scene import MOVIE_DIMS, Scene
from minisim.spec import Spec

# Minimum matched-filter SNR (see `matched_filter_snr`: the ideal detector's SNR
# for a cell's realized footprint and activity, pooled over its pixels and all its
# transients, against shot + read noise) for a cell to count as detectable. This is
# the *sole* detectability criterion: there is no separate in-focus/depth-of-field
# gate (defocus enters continuously through the SNR, the way a real pipeline
# experiences it). Calibrated in 1.1.0 against a strong pipeline's recovery: it is
# the SNR at which cala recovers half the cells on CaLab's simulated CA1 recordings
# with 6 px cells at the real recording's edge blur (fit 111, 95 % CI 104-117), and
# it holds without retuning on 10 px cells (50 % near 103). The ideal observer knows
# the footprint and the trace; a pipeline must estimate both, which is why the
# threshold is far above the few-sigma an ideal detector would need. Cells much
# larger than the pipeline's cell-size setting need 3-4x more (a size-prior effect,
# not noise), so the threshold is not raised for them. See the concepts docs.
#
# STABILITY CONTRACT. This single constant sets the `detectable` flag, the
# `restrict_to_detectable` recall denominator in `minisim.testing.score`, and the
# "auto" focus yield objective - so its value is part of what a consumer's
# `assert report.recall > X` depends on. Treat any change as a minor-version bump
# with a changelog entry, because it moves recall denominators under downstream
# tests (see the reproducibility & stability contract in the docs).
DETECT_SNR_THRESHOLD = 110.0

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
    "C",
    "S",
    "centers_um",
    "amplitude_per_cell",
    "in_focus",
    "detectable",
)
_GT_OPTIONAL = (
    "observed_sigma_px",
    "observed_gain",
    "observed_halo_sigma_px",
    "observed_halo_fraction",
    "shifts",
    "nonrigid_field",
    "nonrigid_nodes_y",
    "nonrigid_nodes_x",
    "illumination",
    "vignette",
    "leakage",
    "bleaching",
    "neuropil_temporal",
    "neuropil_spatial",
    "neuropil_population",
    "vasculature_mask",
    "vessel_overlap_fraction",
    "hot_pixel_mask",
    "hot_pixel_excess",
    "stuck_pixel_mask",
    "prnu_gain",
    "fpn_column",
    "fpn_row",
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
      :attr:`observed_gain` scalars (plus :attr:`observed_halo_sigma_px` /
      :attr:`observed_halo_fraction` when the tissue has a scatter halo). Deep cells' observed footprints are
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
    # Diffuse scatter wing of the PSF (Tissue.scatter_halo_fraction > 0): per-unit
    # wing width and weight, so A_observed reproduces (1-f)·G(sigma) + f·G(halo).
    # None when there is no halo (single-Gaussian PSF); NaN for a cell without optics.
    observed_halo_sigma_px: NDArray[Shape["* unit"], float] | None = None
    observed_halo_fraction: NDArray[Shape["* unit"], float] | None = None

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
    # Non-rigid deformation on top of ``shifts`` (None unless BrainMotion(nonrigid=True)):
    # control-node displacements (frame, gy, gx, 2) = (dy, dx) in px, same sign
    # convention as ``shifts`` (positive moves content down/right), evaluated at the
    # *observed* pixel. The nodes sit at FOV pixel coordinates ``nonrigid_nodes_y`` /
    # ``nonrigid_nodes_x`` (spanning the FOV edge to edge); between them the field is
    # natural-cubic-spline interpolated. ``shifts`` is the rigid component (the field's
    # spatial mean is ~0); use :meth:`displacement` for the full motion at any pixel.
    nonrigid_field: NDArray[Shape["* frame, * gy, * gx, 2"], float] | None = None
    nonrigid_nodes_y: NDArray[Shape["* gy"], float] | None = None
    nonrigid_nodes_x: NDArray[Shape["* gx"], float] | None = None
    illumination: NDArray[Shape["* height, * width"], float] | None = None
    vignette: NDArray[Shape["* height, * width"], float] | None = None
    leakage: NDArray[Shape["* height, * width"], float] | None = None
    bleaching: NDArray[Shape["* unit, * frame"], float] | None = None
    neuropil_temporal: NDArray[Shape["* component, * frame"], float] | None = None
    neuropil_spatial: NDArray[Shape["* component, * height, * width"], float] | None = (
        None
    )
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
    # Static sensor imperfections from the ``sensor_defects`` step (all None when the
    # step is absent): the boolean (height, width) hot-pixel mask, each hot pixel's
    # dark-current excess in ADU counts (0 elsewhere), the stuck-pixel mask, the
    # multiplicative per-pixel gain map (PRNU, 1 = nominal), and the additive
    # per-column / per-row fixed-pattern offsets in ADU. Sensor-frame fields: they do
    # not move with the brain.
    hot_pixel_mask: NDArray[Shape["* height, * width"], bool] | None = None
    hot_pixel_excess: NDArray[Shape["* height, * width"], float] | None = None
    stuck_pixel_mask: NDArray[Shape["* height, * width"], bool] | None = None
    prnu_gain: NDArray[Shape["* height, * width"], float] | None = None
    fpn_column: NDArray[Shape["* width"], float] | None = None
    fpn_row: NDArray[Shape["* height"], float] | None = None
    # The concrete focal depth (µm) the optics step resolved "auto" to - the plane
    # that maximized recoverable yield. A scalar, so persisted as a group attr (not
    # a dataset) by save/load; None when the optics step did not run.
    focal_depth_um: float | None = None

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
        if self.observed_sigma_px is None:
            return self.A_planted  # no optics -> observed == planted
        return self.observed_footprints().to_dense(dtype=float)

    def observed_footprints(self) -> FootprintStack:
        """The optically degraded footprints as a sparse, FOV-cropped stack.

        The same footprints as :attr:`A_observed` (regenerated from the planted
        patches and the per-unit optics scalars, bit-identical to the render), but
        kept sparse: one small patch per unit rather than a dense ``(unit, H, W)``
        array. Use it to loop over cells without paying for the dense stack (a
        600 px, 800-cell recording's dense ``A_observed`` is ~2 GB in float64).
        Falls back to the planted footprints when the optics step did not run.
        """
        top, left = self.fov_offset
        h, w = self.fov_shape
        if self.observed_sigma_px is None:
            return self.planted.crop(top, left, h, w)
        observed = []
        for i, fp in enumerate(self.planted):
            sigma = float(self.observed_sigma_px[i])
            if np.isnan(sigma):  # a cell without optics params -> sharp footprint
                observed.append(fp)
            else:
                halo = ()
                if self.observed_halo_fraction is not None:
                    f = float(self.observed_halo_fraction[i])
                    if f > 0.0:  # NaN (no optics) and 0 both mean "no wing"
                        halo = (float(self.observed_halo_sigma_px[i]), f)
                observed.append(
                    degrade_footprint(fp, sigma, float(self.observed_gain[i]), *halo)
                )
        return FootprintStack(tuple(observed), self.planted.canvas_shape).crop(
            top, left, h, w
        )

    @property
    def depth_um(self) -> np.ndarray:
        """Per-cell depth ``z`` (µm) - the first column of ``centers_um``.

        Exposed as a derived view rather than stored, to avoid drift. The lateral
        ``centers_um[:, 1:]`` are in the optical-center frame, so pixel coordinates
        are ``acq.um_to_index(y, x, fov_shape)`` (origin = FOV center), using the
        owning ``Recording.spec.acquisition``.
        """
        return self.centers_um[:, 0]

    def displacement(self, ys, xs, frames=None) -> np.ndarray:
        """Total ``(dy, dx)`` motion in px at the given pixels, ``(frame, point, 2)``.

        ``ys``/``xs`` are broadcast 1-D pixel coordinates of the query points (e.g. the
        centers of motion-correction patches), in FOV pixels. The result is the rigid
        ``shifts`` plus, when present, the interpolated ``nonrigid_field`` at each
        point, for every frame (or only the given ``frames`` index/slice/array).
        Positive values move content toward higher row/column indices, the
        convention of ``shifts``; a registration's correction is the negation.
        """
        if self.shifts is None:
            raise ValueError("this recording has no motion (no brain_motion step)")
        from minisim.steps.motion import grid_interp_matrix

        ys, xs = np.broadcast_arrays(
            np.atleast_1d(np.asarray(ys, dtype=float)),
            np.atleast_1d(np.asarray(xs, dtype=float)),
        )
        rigid = self.shifts if frames is None else self.shifts[frames]
        out = np.repeat(rigid[:, None, :], len(ys), axis=1).astype(float)
        if self.nonrigid_field is not None:
            field = (
                self.nonrigid_field if frames is None else self.nonrigid_field[frames]
            )
            my = grid_interp_matrix(self.nonrigid_nodes_y, ys)  # (P, gy)
            mx = grid_interp_matrix(self.nonrigid_nodes_x, xs)  # (P, gx)
            out += np.einsum("pi,fijc,pj->fpc", my, field, mx, optimize=True)
        return out

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
            observed_sigma_px=self.observed_sigma_px[m]
            if self.observed_sigma_px is not None
            else None,
            observed_gain=self.observed_gain[m]
            if self.observed_gain is not None
            else None,
            observed_halo_sigma_px=self.observed_halo_sigma_px[m]
            if self.observed_halo_sigma_px is not None
            else None,
            observed_halo_fraction=self.observed_halo_fraction[m]
            if self.observed_halo_fraction is not None
            else None,
            C=self.C[m],
            S=self.S[m],
            centers_um=self.centers_um[m],
            amplitude_per_cell=self.amplitude_per_cell[m],
            in_focus=self.in_focus[m],
            detectable=self.detectable[m],
            vessel_overlap_fraction=(
                self.vessel_overlap_fraction[m]
                if self.vessel_overlap_fraction is not None
                else None
            ),
            bleaching=self.bleaching[m] if self.bleaching is not None else None,
            shifts=self.shifts,
            nonrigid_field=self.nonrigid_field,
            nonrigid_nodes_y=self.nonrigid_nodes_y,
            nonrigid_nodes_x=self.nonrigid_nodes_x,
            illumination=self.illumination,
            vignette=self.vignette,
            leakage=self.leakage,
            neuropil_temporal=self.neuropil_temporal,
            neuropil_spatial=self.neuropil_spatial,
            neuropil_population=self.neuropil_population,
            vasculature_mask=self.vasculature_mask,
            focal_depth_um=self.focal_depth_um,
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

    @property
    def observed_movie(self) -> xr.DataArray:
        """The observed movie as a labeled ``(frame, height, width)`` DataArray.

        :attr:`observed` is stored as a bare array; this wraps it in the same
        :data:`~minisim.scene.MOVIE_DIMS` layout (with the trivial ``arange``
        coordinates) that :attr:`snapshots` and :meth:`stage` already return, so a
        recording can be fed straight into an analysis pipeline that expects
        labeled axes without re-attaching dim names. The array data is shared with
        :attr:`observed`, not copied.
        """
        return _movie_dataarray(np.asarray(self.observed))

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

        present = _write_ground_truth_group(root, gt)

        snapshot_names = sorted(self.snapshots)
        if snapshot_names:
            snap_group = root.create_group("snapshots")
            for name in snapshot_names:
                snap_group.create_dataset(
                    name, data=np.asarray(self.snapshots[name].values)
                )

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

        ground_truth = _read_ground_truth_group(root, root.attrs.get("gt_present", []))

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
        (
            c.footprint_planted.canvas_shape
            for c in scene.cells
            if c.footprint_planted is not None
        ),
        (fov_h, fov_w),
    )
    # Centered crop from the canvas (sensor FOV + any motion margin) down to the FOV.
    margin_h = (canvas_h - fov_h) // 2
    margin_w = (canvas_w - fov_w) // 2

    # Both falloff fields are FOV-sized (built post-motion-crop), or None.
    illumination = scene.truth.illumination
    vignette = scene.truth.vignette
    # The vessel transmission mask, if the vasculature step ran: canvas-sized, so
    # it aligns with the (canvas-coordinate) footprints for the per-cell overlap
    # integral.
    vasc_mask_canvas = scene.truth.vasculature_mask

    planted_fps, traces, spikes, bleaches = [], [], [], []
    centers, amplitudes, in_focus, has_trace, overlaps = [], [], [], [], []
    sigmas, gains, halo_sigmas, halo_fractions = [], [], [], []
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
        sigmas.append(
            cell.observed_sigma_px if cell.observed_sigma_px is not None else np.nan
        )
        gains.append(cell.observed_gain if cell.observed_gain is not None else np.nan)
        halo_sigmas.append(
            cell.observed_halo_sigma_px
            if cell.observed_halo_sigma_px is not None
            else np.nan
        )
        halo_fractions.append(
            cell.observed_halo_fraction
            if cell.observed_halo_fraction is not None
            else np.nan
        )
        traces.append(trace)
        spikes.append(spike)
        bleaches.append(cell.bleach)
        centers.append((z, y_um, x_um))
        amplitudes.append(
            cell.amplitude if cell.amplitude is not None else float("nan")
        )
        in_focus.append(ifocus)
        has_trace.append(cell.trace is not None)
        # Record the footprint-weighted vessel occlusion as the scoreable confound
        # axis (the occlusion itself enters detectability through the gain field).
        overlaps.append(_vessel_overlap(cell.observed_footprint(), vasc_mask_canvas))

    # Optics ran iff any surviving cell carries a sigma; then keep the per-unit
    # scalars so A_observed regenerates. Otherwise None -> A_observed == A_planted.
    optics_ran = any(not np.isnan(s) for s in sigmas)
    # The halo scalars exist only when some cell carries a wing (halo fraction > 0).
    halo_ran = any(not np.isnan(f) for f in halo_fractions)
    gt = GroundTruth(
        planted=FootprintStack.from_footprints(planted_fps, (canvas_h, canvas_w)),
        fov_offset=(margin_h, margin_w),
        fov_shape=(fov_h, fov_w),
        observed_sigma_px=np.array(sigmas, dtype=float) if optics_ran else None,
        observed_gain=np.array(gains, dtype=float) if optics_ran else None,
        observed_halo_sigma_px=(
            np.array(halo_sigmas, dtype=float) if halo_ran else None
        ),
        observed_halo_fraction=(
            np.array(halo_fractions, dtype=float) if halo_ran else None
        ),
        C=_stack(traces, (0, n_frames)),
        S=_stack(spikes, (0, n_frames)),
        centers_um=np.array(centers, dtype=float).reshape(-1, 3),
        amplitude_per_cell=np.array(amplitudes, dtype=float),
        in_focus=np.array(in_focus, dtype=bool),
        # Placeholder: set below from the matched-filter SNR, which needs the
        # assembled ground truth (footprints, traces and per-effect fields).
        detectable=np.zeros(len(in_focus), dtype=bool),
        # Per-cell vessel occlusion, present only when the vasculature step ran.
        vessel_overlap_fraction=(
            np.array(overlaps, dtype=float) if vasc_mask_canvas is not None else None
        ),
        # Per-cell bleaching envelopes (unit, frame), present only if the bleaching
        # step ran; any cell without one (e.g. added afterward) gets a no-fade row.
        bleaching=(
            _stack(
                [b if b is not None else np.ones(n_frames) for b in bleaches],
                (0, n_frames),
            )
            if any(b is not None for b in bleaches)
            else None
        ),
        shifts=scene.truth.shifts,
        nonrigid_field=scene.truth.nonrigid_field,
        nonrigid_nodes_y=scene.truth.nonrigid_nodes_y,
        nonrigid_nodes_x=scene.truth.nonrigid_nodes_x,
        illumination=illumination,
        vignette=vignette,
        leakage=scene.truth.leakage,
        neuropil_temporal=scene.truth.neuropil_temporal,
        neuropil_spatial=_crop_components(scene.truth.neuropil_spatial, fov_h, fov_w),
        neuropil_population=scene.truth.neuropil_population,
        vasculature_mask=_crop_field(scene.truth.vasculature_mask, fov_h, fov_w),
        focal_depth_um=scene.truth.focal_depth_um,
        **_defect_fields(scene.truth.sensor_defects),
    )
    snr = matched_filter_snr(gt, spec)
    gt = gt.model_copy(
        update={"detectable": detectable_from_snr(snr, gt.in_focus, has_trace)}
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


def _write_ground_truth_group(root, gt: GroundTruth) -> list[str]:
    """Write ``gt`` as the ``ground_truth`` subgroup of the open zarr ``root``.

    The one writer behind both :meth:`Recording.save` and :func:`save_ground_truth`,
    so the two layouts cannot drift. Returns the names of the optional datasets
    actually written (the ``gt_present`` attr a reader needs).
    """
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

    # focal_depth_um is a scalar, not an array: stash it as a group attr (the
    # resolved "auto" focus depth).
    gt_group.attrs["focal_depth_um"] = gt.focal_depth_um
    return present


def _read_ground_truth_group(root, present) -> GroundTruth:
    """Rebuild a :class:`GroundTruth` from the ``ground_truth`` subgroup of ``root``."""
    gt_group = root["ground_truth"]
    fields = {name: np.asarray(gt_group[name]) for name in _GT_REQUIRED}
    for name in present:
        fields[name] = np.asarray(gt_group[name])
    focal = gt_group.attrs.get("focal_depth_um")
    if focal is not None:
        fields["focal_depth_um"] = float(focal)

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
    return GroundTruth(**fields)


def save_ground_truth(gt: GroundTruth, spec: Spec, path: str | Path) -> None:
    """Persist just the ground truth (and its spec) to a zarr directory at ``path``.

    The movie-less sibling of :meth:`Recording.save`: the same ``ground_truth``
    layout and ``spec.json`` / ``format_version`` / ``spec_cache_key`` bookkeeping,
    with no ``observed`` dataset. Used by the Miniscope-DAQ writer, which streams
    the movie to AVI files and keeps the truth beside them. Read it back with
    :func:`load_ground_truth`. The write is atomic (sibling ``.tmp`` then rename).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    root = zarr.open_group(str(tmp), mode="w")
    present = _write_ground_truth_group(root, gt)
    root.attrs["format_version"] = _FORMAT_VERSION
    root.attrs["spec_cache_key"] = spec.cache_key()
    root.attrs["gt_present"] = present
    root.attrs["snapshot_names"] = []
    (tmp / "spec.json").write_text(spec.model_dump_json(indent=2))
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)


def load_ground_truth(path: str | Path) -> tuple[GroundTruth, Spec]:
    """Load ``(ground_truth, spec)`` written by :func:`save_ground_truth`.

    Also reads the ``ground_truth`` of a full :meth:`Recording.save` directory.
    Applies the same ``format_version`` hard check as :meth:`Recording.load`.
    """
    path = Path(path)
    root = zarr.open_group(str(path), mode="r")
    stored_format = root.attrs.get("format_version")
    if stored_format != _FORMAT_VERSION:
        raise ValueError(
            f"Cannot load {path}: on-disk format_version {stored_format!r} is not "
            f"readable by this minisim (which reads format {_FORMAT_VERSION})."
        )
    spec = Spec.model_validate_json((path / "spec.json").read_text())
    return _read_ground_truth_group(root, root.attrs.get("gt_present", [])), spec


def _defect_fields(defects) -> dict:
    """The ``GroundTruth`` kwargs for a realized ``SensorDefectMaps`` (empty if None)."""
    if defects is None:
        return {}
    return {
        "hot_pixel_mask": defects.hot_mask,
        "hot_pixel_excess": defects.hot_excess_counts,
        "stuck_pixel_mask": defects.stuck_mask,
        "prnu_gain": defects.prnu_gain,
        "fpn_column": defects.fpn_column,
        "fpn_row": defects.fpn_row,
    }


def _movie_dataarray(values: np.ndarray) -> xr.DataArray:
    """Rebuild a movie ``DataArray`` from stored values (the saved snapshot form).

    Snapshots are persisted as bare arrays; their coordinates are the trivial
    ``arange`` grid over ``MOVIE_DIMS``, reconstructed here so :meth:`Recording.load`
    returns the same ``(frame, height, width)`` labelling the pipeline produced.
    """
    return xr.DataArray(
        values,
        dims=list(MOVIE_DIMS),
        coords={
            dim: np.arange(size)
            for dim, size in zip(MOVIE_DIMS, values.shape, strict=True)
        },
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


def sample_field_at(
    field: np.ndarray | None, y_um: float, x_um: float, pixel_size_um: float
) -> float:
    """Value of a static FOV field at a µm position, with edge-clamped indexing.

    Converts an optical-center ``(y_um, x_um)`` position (origin = FOV center,
    ``+y`` down, ``+x`` right) to the nearest pixel (rounded and clipped to the
    field bounds) and returns ``field`` there. ``field`` is a sensor-FOV-sized
    array such as the combined illumination × vignette photon budget; an absent
    field (``None``) returns ``1.0``, the multiplicative identity, so callers can
    treat "no field" and "uniform field" alike. Used by the teaching notebook's
    per-cell SNR panels.
    """
    if field is None:
        return 1.0
    h, w = field.shape
    iy = int(np.clip(round((h - 1) / 2.0 + y_um / pixel_size_um), 0, h - 1))
    ix = int(np.clip(round((w - 1) / 2.0 + x_um / pixel_size_um), 0, w - 1))
    return float(field[iy, ix])


def peak_snr(peak_dF, baseline, gain, read_e):
    """Single-pixel, single-transient SNR in detected electrons (a diagnostic).

    ``gain`` converts a cell's ΔF units to detected electrons
    (``optical_brightness · illumination · photons_per_unit · QE``); ``peak_dF``
    and ``baseline`` are the transient height and the (non-negative) resting level
    in those same ΔF units. The noise floor is shot noise on the baseline
    electrons plus the sensor read noise::

        SNR = peak_dF·gain / sqrt(max(baseline,0)·gain + read_e²)

    This was minisim's detectability criterion up to 1.0.x. It looks at the
    brightest pixel of the best transient only, so it ignores how many pixels and
    how many transients a pipeline can pool, and with a sharp PSF it collapses
    under defocus (the peak drops as ``σ₀²/σ²`` while the integrated light stays).
    ``detectable`` now uses :func:`matched_filter_snr`; this function remains as a
    diagnostic (e.g. the teaching notebook's per-pixel view). Works on scalars or
    numpy arrays. Where the noise floor is exactly zero the SNR is ``inf`` if there
    is any signal, else ``0``.
    """
    signal_e = peak_dF * gain
    noise_e = np.sqrt(np.maximum(baseline, 0.0) * gain + read_e * read_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = signal_e / noise_e
    return np.where(noise_e > 0, snr, np.where(signal_e > 0, np.inf, 0.0))


#: Deprecated alias of :func:`peak_snr`, kept so 1.0.x code keeps importing. It is
#: no longer the detectability criterion (see :func:`matched_filter_snr`).
detection_snr = peak_snr


def transient_energy(C: np.ndarray, bleaching: np.ndarray | None = None) -> np.ndarray:
    """Per-unit temporal signal energy ``Σ_t Δ_t²`` of the emitted transients.

    ``Δ_t = (C_t − min_t C) · B_t``: the calcium trace above its resting level,
    faded by the bleaching envelope ``B`` when present (the light the cell
    actually emits above baseline). ``C`` is ``(unit, frame)``; returns
    ``(unit,)``. The time half of :func:`matched_filter_snr`.
    """
    C = np.asarray(C, dtype=float)
    if C.size == 0:
        return np.zeros(C.shape[0])
    delta = C - C.min(axis=1, keepdims=True)
    if bleaching is not None:
        delta = delta * bleaching
    return np.einsum("ut,ut->u", delta, delta)


def detection_noise_fields(
    ground_truth: GroundTruth, spec: Spec, footprints: FootprintStack | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-pixel signal gain and mean light, in detected electrons, over the FOV.

    Returns ``(gain_e, mean_e)``, both ``(H, W)``, or ``None`` when the spec has
    no ``sensor`` step (no noise floor). ``gain_e`` converts one unit of cell
    intensity at a pixel to detected electrons: the multiplicative fields applied
    after ``composite`` (vessel transmission, illumination, vignette) times
    ``photons_per_unit · quantum_efficiency``. ``mean_e`` is the expected
    electrons per pixel per frame of *all* the light the recording puts there,
    rebuilt from the ground truth by walking the spec's steps in order: every
    cell's footprint times its mean emitted trace (other cells' out-of-focus
    light included), the neuropil's mean, vessels, illumination, vignette and the
    additive leakage glow. Its square root is the shot noise.

    Approximations (documented, not modelled): the fields are evaluated at the
    reference (zero-motion) frame, the sensor-defect maps (hot pixels, PRNU, FPN)
    are left out, and the photon clip at 0 is ignored (all terms are
    non-negative). ``footprints`` is ``ground_truth.observed_footprints()``, passed
    in to skip regenerating it when the caller already has it.
    """
    sensor = next((s for s in spec.steps if s.kind == "sensor"), None)
    if sensor is None:
        return None
    acq = spec.acquisition
    h, w = ground_truth.fov_shape
    exposure = float(sensor.photons_per_unit) * acq.image_sensor.quantum_efficiency
    emitted = np.asarray(ground_truth.C, dtype=float)
    if ground_truth.bleaching is not None:
        emitted = emitted * ground_truth.bleaching
    mean_emit = emitted.mean(axis=1) if emitted.shape[1] else np.zeros(len(emitted))

    # Walk the steps in pipeline order (Spec keeps them canonical). A per-effect
    # field present in the ground truth without its step in the spec (a hand-built
    # scene) is still applied, after the spec's own steps, in canonical order.
    steps = [(s.kind, s) for s in spec.steps]
    kinds = {k for k, _ in steps}
    for kind, name in (
        ("vasculature", "vasculature_mask"),
        ("illumination_profile", "illumination"),
        ("vignette", "vignette"),
        ("leakage", "leakage"),
    ):
        if kind not in kinds and getattr(ground_truth, name) is not None:
            steps.append((kind, None))

    mean = np.zeros((h, w))
    gain = np.ones((h, w))
    cells_added = False
    for kind, step in steps:
        if kind == "composite":
            if footprints is None:
                footprints = ground_truth.observed_footprints()
            for k, fp in enumerate(footprints):
                if not fp.is_empty and mean_emit[k] != 0.0:
                    y0, x0 = fp.offset
                    ph, pw = fp.patch.shape
                    mean[y0 : y0 + ph, x0 : x0 + pw] += mean_emit[k] * fp.patch
            cells_added = True
        elif kind == "neuropil" and ground_truth.neuropil_spatial is not None:
            assert step is not None
            temporal = ground_truth.neuropil_temporal
            weights = temporal.mean(axis=1) if temporal is not None else 1.0
            spatial = np.asarray(ground_truth.neuropil_spatial, dtype=float)
            mean += (
                step.amplitude
                * np.tensordot(np.broadcast_to(weights, len(spatial)), spatial, 1)
                / step.n_components
            )
        elif kind in ("vasculature", "illumination_profile", "vignette"):
            field = {
                "vasculature": ground_truth.vasculature_mask,
                "illumination_profile": ground_truth.illumination,
                "vignette": ground_truth.vignette,
            }[kind]
            if field is not None:
                mean = mean * field
                if cells_added:
                    gain = gain * field
        elif kind == "leakage" and ground_truth.leakage is not None:
            mean = mean + ground_truth.leakage
    return gain * exposure, np.maximum(mean, 0.0) * exposure


def spatial_snr2(
    footprints, gain_e: np.ndarray, mean_e: np.ndarray, read_e: float
) -> np.ndarray:
    """Per-unit spatial matched-filter information ``Σ_i (g_i a_i)² / σ_i²``.

    ``footprints`` iterates sparse FOV :class:`~minisim.footprint.Footprint` s
    (``a``), ``gain_e``/``mean_e`` are from :func:`detection_noise_fields` and
    ``σ_i² = mean_e_i + read_e²`` (shot + read noise). The square root is the SNR
    of a footprint-weighted sum of one frame for one unit of ΔF. Returns
    ``(unit,)``.
    """
    inv_var = gain_e**2 / (mean_e + read_e * read_e)
    out = []
    for fp in footprints:
        if fp.is_empty:
            out.append(0.0)
            continue
        y0, x0 = fp.offset
        ph, pw = fp.patch.shape
        a = fp.patch.astype(float)
        out.append(float((a * a * inv_var[y0 : y0 + ph, x0 : x0 + pw]).sum()))
    return np.array(out, dtype=float)


def matched_filter_snr(ground_truth: GroundTruth, spec: Spec) -> np.ndarray | None:
    """Per-cell matched-filter detection SNR over space and time - ``detectable``'s test.

    The SNR of the ideal (known-template) detector for cell ``k``'s transients:
    a filter matched to its realized footprint *and* its realized activity,
    against the per-pixel shot + read noise of everything the recording puts on
    the sensor. Returns ``(unit,)``, or ``None`` with no ``sensor`` step.

    **Derivation.** In detected electrons, cell ``k`` adds ``s_it = g_i a_i Δ_t``
    to pixel ``i`` at frame ``t``: ``a`` is its observed footprint (the realized
    degraded footprint - defocus, scatter, halo - that the render uses), ``g_i``
    the per-pixel gain (vessel, illumination, vignette, exposure, QE) and
    ``Δ_t = (C_t − min C)·B_t`` its emitted transient above rest. The noise is
    independent over pixels and frames with variance ``σ_i² = λ_i + r²``: shot
    noise on the mean electrons ``λ_i`` of all light at that pixel plus the read
    noise ``r``. For a known signal in white Gaussian noise the matched filter
    ``Σ_it s_it y_it / σ_i²`` is the optimal detector and its SNR is::

        SNR² = Σ_t Σ_i s_it² / σ_i² = [Σ_i g_i² a_i² / σ_i²] · [Σ_t Δ_t²]

    The space-time product separates because the cell's signal is rank one.
    The spatial factor is what a footprint-weighted ROI gets from one frame; the
    temporal factor pools every transient (it equals the sum over isolated events
    of each event's squared SNR). Pipelines such as CNMF and cala pool exactly
    this way (they fit ``a`` and ``c`` jointly), which is why the old single-pixel
    peak test (:func:`peak_snr`) undercounted what they recover: with a sharp PSF,
    defocus drops a cell's peak as ``σ₀²/σ²`` but its spatial SNR only as
    ``σ₀/σ``, since the light is conserved and spread over more pixels.

    **What counts as noise.** Shot noise on the mean light (including other cells'
    out-of-focus light, the neuropil haze and leakage) and read noise. The
    *fluctuations* of the neuropil and of other cells are structured background
    that factorization pipelines model as components of their own (a background
    term, neighbouring cells), so they are not counted as noise here; their light
    still raises the shot noise. Bleaching fades ``Δ``. See
    :func:`detection_noise_fields` for the approximations in ``λ``.

    It is an *ideal-observer* SNR: a real pipeline must also estimate ``a`` and
    ``c``, so it needs more than the ideal detector; that gap is what
    :data:`DETECT_SNR_THRESHOLD` (calibrated against cala's recovery) absorbs.
    The temporal factor grows with the number of transients, so a longer
    recording makes the same cell more detectable, as it does for a pipeline.
    Cost: one pass over the sparse footprints plus one over ``C``.
    """
    if not any(s.kind == "sensor" for s in spec.steps):
        return None
    footprints = ground_truth.observed_footprints()
    fields = detection_noise_fields(ground_truth, spec, footprints)
    assert fields is not None
    gain_e, mean_e = fields
    read_e = spec.acquisition.image_sensor.read_noise_e
    spatial = spatial_snr2(footprints, gain_e, mean_e, read_e)
    energy = transient_energy(ground_truth.C, ground_truth.bleaching)
    return np.sqrt(spatial * energy)


def detectable_from_snr(
    snr: np.ndarray | None, in_focus: np.ndarray, has_trace: np.ndarray
) -> np.ndarray:
    """The ``detectable`` flag: ``snr ≥ DETECT_SNR_THRESHOLD`` for active cells.

    A cell without a calcium trace emits no transient and is never detectable.
    With no sensor (``snr is None``) there is no noise floor, so the flag falls
    back to the geometric ``in_focus`` flag.
    """
    has_trace = np.asarray(has_trace, dtype=bool)
    if snr is None:
        return np.asarray(in_focus, dtype=bool) & has_trace
    return (np.asarray(snr) >= DETECT_SNR_THRESHOLD) & has_trace


def _vessel_overlap(
    footprint: Footprint | None, mask_canvas: np.ndarray | None
) -> float:
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

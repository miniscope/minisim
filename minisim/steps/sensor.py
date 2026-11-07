"""Sensor-domain steps: the static optical fields, then digitization.

The sensor domain is the *static* frame - effects fixed to the optics and
detector that do **not** move with the brain (unlike the tissue-frame steps in
:mod:`minisim.steps.tissue`). They are applied after the motion
boundary:

* :class:`IlluminationProfileStep` (``illumination_profile``) - multiplicative
  radial excitation falloff (the LED lights the FOV unevenly).
* :class:`VignetteStep` (``vignette``) - multiplicative radial vignette on the
  emission / return path (collection light loss toward the edges).
* :class:`LeakageStep` (``leakage``) - additive static baseline; the *additive*
  contributor to the low-frequency background minian's glow-removal strips
  (which also bundles in the multiplicative illumination falloff + vignette above).
* :class:`SensorStep` (``sensor``) - the only step that turns honest radiometric
  intensity into integer ADC counts (shot + read noise, gain, quantization).

The three field steps each record their static ``(height, width)`` field to ground
truth. The illumination and vignette fields are load-bearing downstream:
``finalize()`` reads their *product* at each cell's lateral position to
finish the per-cell ``detectable`` flag, since both falloffs dim edge cells and so
shrink the usable FOV below the physical one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from minisim.scene import Scene
from minisim.steps.base import PipelineContext, Step

if TYPE_CHECKING:
    # Referenced only as string Generic bases (Step["Vignette"] etc.), which ruff's
    # F401 misses; pyright needs them in scope to resolve the forward references.
    from minisim.spec import (  # noqa: F401
        IlluminationProfile,
        Leakage,
        Sensor,
        SensorDefects,
        Vignette,
    )


def radius_grid(shape: tuple[int, int], center_px: tuple[float, float]) -> np.ndarray:
    """Per-pixel Euclidean distance (in pixels) from ``center_px``, shape ``shape``.

    The shared geometry behind both static fields - the radial falloff fields and
    the gaussian leakage glow are each a radial function of this distance.
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return np.hypot(yy - center_px[0], xx - center_px[1])


def falloff_center_px(
    shape: tuple[int, int], acq, center_offset_um: tuple[float, float]
) -> tuple[float, float]:
    """Bright-center pixel: the FOV center plus ``center_offset_um`` (µm → px)."""
    h, w = shape
    return (
        (h - 1) / 2.0 + acq.um_to_px(center_offset_um[0]),
        (w - 1) / 2.0 + acq.um_to_px(center_offset_um[1]),
    )


def radial_falloff(
    shape: tuple[int, int],
    center_px: tuple[float, float],
    falloff: float,
    exponent: float,
) -> np.ndarray:
    """Static radial falloff field ``1 − (1 − falloff)·(r / r_max)^exponent``.

    ``1`` at the bright ``center_px``, dropping to ``falloff`` at the farthest
    corner (``r_max`` is the distance to that corner, so the minimum is exactly
    ``falloff``); ``exponent`` shapes the rolloff (>1 keeps the center bright then
    dims sharply toward the rim). The single multiplicative-field shape shared by
    the **illumination profile** (excitation unevenness) and the **vignette**
    (emission/collection loss). A 1×1 FOV (``r_max == 0``) has no falloff.
    """
    h, w = shape
    r_max = max(
        np.hypot(center_px[0] - yc, center_px[1] - xc)
        for yc in (0, h - 1)
        for xc in (0, w - 1)
    )
    if r_max <= 0:
        return np.ones((h, w))
    return 1.0 - (1.0 - falloff) * (radius_grid(shape, center_px) / r_max) ** exponent


def combined_falloff_field(acq, illumination_spec, vignette_spec) -> np.ndarray | None:
    """Spec-only product of the illumination × vignette falloff fields, FOV-sized.

    A prediction of the per-pixel photon budget the sensor-domain
    :class:`IlluminationProfileStep` / :class:`VignetteStep` will apply, built
    straight from the sensor FOV shape and the spec parameters - so it is
    identical (by construction, same :func:`radial_falloff`) to what those steps
    record at run time, but is available *before* the pipeline reaches the sensor
    domain. ``None`` when neither field is present (no dimming).

    Auto-focus uses this to weight cells by the brightness their image will
    actually get, so the focal plane can be chosen for recoverable yield rather
    than raw defocus (see :func:`~minisim.steps.cell.resolve_focal_plane`).
    """
    shape = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
    field: np.ndarray | None = None
    for spec in (illumination_spec, vignette_spec):
        if spec is None:
            continue
        center = falloff_center_px(shape, acq, spec.center_offset_um)
        f = radial_falloff(shape, center, spec.falloff, spec.exponent)
        field = f if field is None else field * f
    return field


class IlluminationProfileStep(Step["IlluminationProfile"]):
    """Static excitation-illumination falloff: the LED lights the FOV unevenly.

    Multiplies every frame by a :func:`radial_falloff` field - brightest at the
    center (plus ``center_offset_um``), dimmer toward the edges - modelling the
    single excitation LED's uneven illumination of the tissue. Like the vignette it
    is a **sensor-frame** field (fixed to the scope, applied after the motion crop,
    so it does *not* move with the brain) and is recorded ``(height, width)`` to
    ground truth. Its excitation falloff also drives photobleaching faster at the
    bright center; that coupling is wired in ``BleachingStep`` (which evaluates this
    same field at each cell's rest position).
    """

    name = "illumination_profile"
    domain = "sensor"

    def __call__(self, scene: Scene) -> None:
        # Grid from the scene movie: a sensor-frame step runs after the motion crop,
        # so the movie is already the sensor FOV and the field must not extend into
        # the motion margin.
        shape = scene.movie.values.shape[1:]
        center = falloff_center_px(shape, self.acq, self.spec.center_offset_um)
        field = radial_falloff(shape, center, self.spec.falloff, self.spec.exponent)
        scene.movie.values[:] *= field
        scene.truth.illumination = field


class VignetteStep(Step["Vignette"]):
    """Multiplicative radial vignette: emission / return-path light loss (static).

    Multiplies every frame by a :func:`radial_falloff` field for the **collection**
    side - the physical return path trims light rays toward the field edges
    (aperture/relay clipping, plus poorer off-axis optical performance), so corners
    read dimmer regardless of how brightly the tissue was lit. Distinct from the
    illumination profile (excitation) but the same field shape; both are
    sensor-frame (applied after the motion crop, fixed to the detector, so they do
    *not* move with the brain). Recorded ``(height, width)`` to ground truth, where
    together with the illumination field it sets per-cell detectability in
    ``finalize()``.
    """

    name = "vignette"
    domain = "sensor"

    def __call__(self, scene: Scene) -> None:
        # Grid from the scene movie (the sensor FOV post-crop); see IlluminationProfileStep.
        shape = scene.movie.values.shape[1:]
        center = falloff_center_px(shape, self.acq, self.spec.center_offset_um)
        field = radial_falloff(shape, center, self.spec.falloff, self.spec.exponent)
        scene.movie.values[:] *= field
        scene.truth.vignette = field


def leakage_field(spec, acq, shape: tuple[int, int]) -> np.ndarray:
    """The static additive leakage field for a ``Leakage`` spec, FOV-sized.

    Factored out of :class:`LeakageStep` so the streaming video writer can add the
    same glow without running the step. ``uniform`` is a flat ``level``; ``gaussian``
    is a central glow ``level·exp(−r²/2σ²)`` (``sigma_um`` defaults to a quarter of
    the smaller FOV dimension). No RNG - deterministic from the spec.
    """
    h, w = shape
    if spec.profile == "uniform":
        return np.full((h, w), spec.level)
    sigma_um = spec.sigma_um if spec.sigma_um is not None else 0.25 * min(acq.fov_um)
    sigma_px = acq.um_to_px(sigma_um)
    r = radius_grid((h, w), ((h - 1) / 2.0, (w - 1) / 2.0))
    return spec.level * np.exp(-(r**2) / (2.0 * sigma_px**2))


class LeakageStep(Step["Leakage"]):
    """Additive static baseline - stray excitation light reaching the detector.

    Adds the same field to every frame: ``uniform`` is a flat ``level``
    everywhere; ``gaussian`` is a central glow ``level·exp(−r²/2σ²)`` peaking at
    the FOV center, with ``sigma_um`` defaulting to a quarter of the smaller FOV
    dimension when unset. Time-invariant and recorded ``(height, width)`` to
    ground truth. Being a sensor-frame step it is applied after motion and is not
    scaled by ``bleaching`` (stray excitation light reaching the detector neither
    moves with the brain nor bleaches).

    This is only the *additive* contributor to the smooth low-frequency
    background that minian's "glow removal" estimates and subtracts - that removal
    tackles the whole bundle, also including the *multiplicative* illumination
    falloff and vignette (see :class:`IlluminationProfileStep` /
    :class:`VignetteStep`). Leakage is one piece of what it strips, not its target.
    """

    name = "leakage"
    domain = "sensor"

    def __call__(self, scene: Scene) -> None:
        # Grid from the scene movie (the sensor FOV post-crop); see VignetteStep.
        field = leakage_field(self.spec, self.acq, scene.movie.values.shape[1:])
        scene.movie.values[:] += field
        scene.truth.leakage = field


@dataclass(frozen=True)
class SensorDefectMaps:
    """The realized static defect maps of a :class:`~minisim.spec.SensorDefects` step.

    All fields are FOV-sized (``(H, W)``) except the per-column / per-row offsets.
    ``hot_excess_counts`` is the dark-current excess in ADU at each hot pixel (zero
    elsewhere); ``extra_photons`` is that excess expressed as the equivalent expected
    photons the sensor model adds before noise. ``prnu_gain`` is the multiplicative
    gain map (1 = nominal) and ``offset_adu`` the combined column + row offset.
    """

    hot_excess_counts: np.ndarray
    extra_photons: np.ndarray
    stuck_mask: np.ndarray
    stuck_value: float
    prnu_gain: np.ndarray
    fpn_column: np.ndarray
    fpn_row: np.ndarray
    offset_adu: np.ndarray | None

    @property
    def hot_mask(self) -> np.ndarray:
        """Boolean ``(H, W)`` mask of the hot pixels."""
        return self.hot_excess_counts > 0


_DEFECT_SEED_SALT = (
    0x5E75_D3F  # separates the defect streams from any other use of a seed
)


def sensor_defect_maps(spec, acq, master_seed: int) -> SensorDefectMaps:
    """Draw the static defect maps for a ``SensorDefects`` spec.

    Generated from a ``SeedSequence([seed, salt])`` split into one independent child
    generator per effect (hot positions, hot magnitudes, stuck positions, PRNU,
    columns, rows), so changing one effect's parameters never reshuffles another's
    map, and the draws never touch the simulation RNG (adding the step leaves the
    cells, motion and noise of an otherwise identical spec unchanged). ``seed``
    defaults to ``master_seed`` (the spec seed).
    """
    sensor = acq.image_sensor
    h, w = sensor.n_px_height, sensor.n_px_width
    n_px = h * w
    seed = master_seed if spec.seed is None else spec.seed
    children = np.random.SeedSequence([int(seed), _DEFECT_SEED_SALT]).spawn(6)
    pos_rng, mag_rng, stuck_rng, prnu_rng, col_rng, row_rng = (
        np.random.default_rng(c) for c in children
    )

    n_hot = (
        spec.n_hot_pixels
        if spec.n_hot_pixels is not None
        else int(round(spec.hot_pixel_fraction * n_px))
    )
    n_hot = min(n_hot, n_px)
    hot = np.zeros(n_px)
    if n_hot > 0:
        idx = pos_rng.choice(n_px, size=n_hot, replace=False)
        # Pareto: x = x_min * (1-u)^(-1/shape), clipped (real max is ~12x the min).
        u = mag_rng.random(n_hot)
        excess = spec.hot_pixel_min_counts * (1.0 - u) ** (
            -1.0 / spec.hot_pixel_tail_shape
        )
        hot[idx] = np.minimum(excess, spec.hot_pixel_max_counts)
    hot = hot.reshape(h, w)

    stuck = np.zeros(n_px, dtype=bool)
    n_stuck = min(int(round(spec.stuck_pixel_fraction * n_px)), n_px)
    if n_stuck > 0:
        stuck[stuck_rng.choice(n_px, size=n_stuck, replace=False)] = True
    stuck = stuck.reshape(h, w)
    full_scale = float(2**sensor.bit_depth - 1)
    stuck_value = (
        full_scale if spec.stuck_pixel_value is None else float(spec.stuck_pixel_value)
    )

    prnu = np.ones((h, w))
    if spec.prnu_sd > 0:
        prnu = np.clip(1.0 + spec.prnu_sd * prnu_rng.standard_normal((h, w)), 0.0, None)
    col = col_rng.standard_normal(w) * spec.column_fpn_sd
    row = row_rng.standard_normal(h) * spec.row_fpn_sd
    offset = (
        (row[:, None] + col[None, :])
        if (spec.column_fpn_sd or spec.row_fpn_sd)
        else None
    )
    # ADU excess -> equivalent expected photons (inverse of QE and gain).
    extra_photons = hot / (sensor.quantum_efficiency * sensor.gain_adu_per_e)
    return SensorDefectMaps(
        hot_excess_counts=hot,
        extra_photons=extra_photons,
        stuck_mask=stuck,
        stuck_value=stuck_value,
        prnu_gain=prnu,
        fpn_column=col,
        fpn_row=row,
        offset_adu=offset,
    )


def digitize_frame(
    sensor, intensity: np.ndarray, photons_per_unit: float, rng, defects=None
) -> np.ndarray:
    """One frame of scene intensity -> integer-valued counts, with optional defects.

    The single digitization routine shared by :class:`SensorStep` and the streaming
    video writer, so both consume the RNG identically (one ``standard_normal`` per
    frame). With ``defects=None`` it is exactly ``photons_to_counts(clip(I * ppu))``.
    """
    photons = np.clip(intensity * photons_per_unit, 0.0, None)
    if defects is None:
        return sensor.photons_to_counts(photons, rng)
    photons = photons * defects.prnu_gain + defects.extra_photons
    counts = sensor.photons_to_counts(photons, rng, offset_adu=defects.offset_adu)
    if defects.stuck_mask.any():
        counts = np.where(defects.stuck_mask, defects.stuck_value, counts)
    return counts


class SensorDefectsStep(Step["SensorDefects"]):
    """Static sensor imperfections, recorded to ground truth (applied by ``sensor``).

    The defect maps are realized in :func:`~minisim.simulate.build_context` and consumed
    inside :class:`SensorStep`'s digitization (PRNU and dark current must act before
    the shot / read noise is drawn, so they cannot be a separate pass over counts).
    This step itself leaves the movie untouched and writes the maps to
    ``scene.truth`` so they reach ``GroundTruth``. No RNG is drawn from the
    simulation generator (the maps use their own seed).
    """

    name = "sensor_defects"
    domain = "sensor"

    defects: SensorDefectMaps | None = None

    def prepare(self, context: PipelineContext) -> None:
        self.defects = context.sensor_defects

    def __call__(self, scene: Scene) -> None:
        defects = self.defects
        if defects is None:  # run directly in a unit test, without prepare()
            defects = sensor_defect_maps(self.spec, self.acq, 0)
        scene.truth.sensor_defects = defects


class SensorStep(Step["Sensor"]):
    """Intensity → expected photons → digitized counts (the only count-producing step).

    Multiplies the working movie by ``photons_per_unit`` to get the per-pixel
    expected photon count (clipped at 0 - negative intensity from optional trace
    noise is unphysical light), then runs the image sensor's forward model to add
    shot + read noise and quantize to clipped integer counts. The result is
    written back into ``scene.movie`` as integer-valued counts in the float
    working container; the downcast to ``Output.store_dtype`` is a ``finalize()``
    concern.
    """

    name = "sensor"
    domain = "sensor"
    consumes_rng = True  # Poisson shot noise + Gaussian read noise per frame

    defects: SensorDefectMaps | None = None  # set by prepare() from the context

    def prepare(self, context: PipelineContext) -> None:
        self.defects = context.sensor_defects

    def __call__(self, scene: Scene) -> None:
        ppu = self.spec.photons_per_unit
        # Digitize FRAME BY FRAME, in order, on the shared rng: each frame draws its
        # combined shot+read noise (one standard_normal per frame, see
        # ImageSensor.photons_to_counts) before the next. The per-frame draw order is
        # what makes a chunked stream (minisim.video.simulate_video) reproduce these
        # counts bit-for-bit. The result is identical for any framing.
        sensor, rng = self.acq.image_sensor, self.rng
        movie = scene.movie.values
        for f in range(movie.shape[0]):
            movie[f] = digitize_frame(sensor, movie[f], ppu, rng, self.defects)

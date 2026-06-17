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

from typing import TYPE_CHECKING

import numpy as np

from minisim.recording import sample_field_at
from minisim.scene import Cell, Scene
from minisim.steps.base import PipelineContext, Step

if TYPE_CHECKING:
    # Referenced only as string Generic bases (Step["Vignette"] etc.), which ruff's
    # F401 misses; pyright needs them in scope to resolve the forward references.
    from minisim.spec import (  # noqa: F401
        Acquisition,
        IlluminationProfile,
        Leakage,
        Sensor,
        Vignette,
    )

# "auto" exposure aims the brightest cell's *mean* peak at this fraction of the ADC
# full scale: high enough to use the dynamic range, with enough headroom that the
# shot/read-noise tail on top of that mean does not clip (at 0.85 the realized peak
# lands ~0.95 of full scale with no saturated pixels; pushing to ~0.9 starts tipping
# the noise tail over the rail). The analog of the depth-of-field margin "auto"
# focus respects: bright, but deliberately short of saturation.
_EXPOSURE_TARGET_FRACTION = 0.85
# Fallback exposure when "auto" has no cell signal to scale against (no active
# cells, an empty scene): a nominal, well-behaved level so a blank recording is
# still digitized sanely. Matches the numeric Sensor default.
_EXPOSURE_FALLBACK_PHOTONS = 100.0


def resolve_exposure(
    cells: list[Cell],
    acq: Acquisition,
    sensor_spec: Sensor,
    photon_field: np.ndarray | None,
) -> float:
    """Resolve ``Sensor.photons_per_unit`` to a concrete exposure (photons/unit).

    A numeric value passes through unchanged. ``"auto"`` chooses the exposure that
    lands the **brightest cell's rendered peak** at :data:`_EXPOSURE_TARGET_FRACTION`
    of the sensor's ADC full scale - bright enough to use the dynamic range, with
    headroom so shot/read noise rarely saturates. This is the exposure analog of
    "auto" focus: a fixture gets a clear, well-exposed recording with no manual
    dialing of ``photons_per_unit``.

    The peak is estimated **analytically per cell** - ``observed_footprint_peak ×
    trace_peak × illumination`` at the cell's position - rather than from the
    realized movie, so ``simulate`` and the streaming writer
    (:func:`minisim.video.simulate_video`) resolve the *same* exposure from the same
    cell state and stay bit-for-bit identical (the streamer never holds the whole
    movie). ``photon_field`` is the illumination × vignette product, sampled at each
    cell's lateral position the same way detectability samples it.

    **v1 simplifications**, mirroring how "auto" focus documents its own: the
    estimate is per-cell, so it ignores where two footprints overlap (a hot spot can
    nudge slightly past the target) and the additive backgrounds (``neuropil`` /
    ``leakage``); for the clear-footprint fixtures this targets, cells dominate and
    are well separated, so the brightest pixel is a single soma peak. With no
    activity-bearing cell to scale against it falls back to
    :data:`_EXPOSURE_FALLBACK_PHOTONS`.
    """
    ppu = sensor_spec.photons_per_unit
    if ppu != "auto":
        return float(ppu)
    sensor = acq.image_sensor
    full_scale = float(2**sensor.bit_depth - 1)
    peak_intensity = 0.0
    for cell in cells:
        if cell.trace is None:
            continue
        footprint = cell.observed_footprint()
        if footprint is None or footprint.is_empty:
            continue
        illum = sample_field_at(
            photon_field, cell.center_um[1], cell.center_um[2], acq.pixel_size_um
        )
        cell_peak = float(footprint.patch.max()) * float(np.max(cell.trace)) * illum
        peak_intensity = max(peak_intensity, cell_peak)
    if peak_intensity <= 0.0:
        return _EXPOSURE_FALLBACK_PHOTONS
    # counts_peak ≈ intensity · ppu · QE · gain_adu_per_e; solve for the ppu that
    # puts the brightest peak at TARGET_FRACTION · full_scale.
    denom = peak_intensity * sensor.quantum_efficiency * sensor.gain_adu_per_e
    return _EXPOSURE_TARGET_FRACTION * full_scale / denom


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
    shape: tuple[int, int], center_px: tuple[float, float], falloff: float, exponent: float
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
    r_max = max(np.hypot(center_px[0] - yc, center_px[1] - xc) for yc in (0, h - 1) for xc in (0, w - 1))
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

    def __init__(self, spec, acq, rng) -> None:
        super().__init__(spec, acq, rng)
        # Pulled from the PipelineContext in prepare(): the illumination × vignette
        # product, sampled at each cell's position so "auto" exposure scales the
        # brightest cell by the light its image actually gets. None (no falloff
        # steps) means a uniform field. A step run directly in a unit test may skip
        # prepare and leave this None.
        self.photon_field: np.ndarray | None = None

    def prepare(self, context: PipelineContext) -> None:
        self.photon_field = context.photon_field

    def __call__(self, scene: Scene) -> None:
        # Resolve the exposure once (a no-op when photons_per_unit is numeric; the
        # brightest-cell scan when it is "auto"), record it so it is observable, and
        # digitize with it. The streaming writer resolves the same value the same
        # way, keeping the chunked counts bit-for-bit equal.
        ppu = resolve_exposure(scene.cells, self.acq, self.spec, self.photon_field)
        scene.truth.exposure_photons_per_unit = ppu
        # Digitize FRAME BY FRAME, in order, on the shared rng: each frame draws its
        # combined shot+read noise (one standard_normal per frame, see
        # ImageSensor.photons_to_counts) before the next. The per-frame draw order is
        # what makes a chunked stream (minisim.video.simulate_video) reproduce these
        # counts bit-for-bit. The result is identical for any framing.
        sensor, rng = self.acq.image_sensor, self.rng
        movie = scene.movie.values
        for f in range(movie.shape[0]):
            photons = np.clip(movie[f] * ppu, 0.0, None)
            movie[f] = sensor.photons_to_counts(photons, rng)

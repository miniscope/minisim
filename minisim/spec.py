"""Typed, serializable specification for the ``minisim`` pipeline.

This module defines the *contract* the simulator consumes: a tree of pydantic
v2 models describing the acquisition (a real, physical interface), an ordered
list of pipeline steps, and output formatting. It is the inverse of the minian
analysis pipeline expressed as data - the same ``Spec`` object a training
notebook walks through, a test parametrizes over, and a cache keys on.

Two layers, by design:

* **Layer 1 - what you read off a datasheet.** ``Optics.na``,
  ``Optics.magnification``, ``Tissue.scatter_mfp_emission_um`` and friends. Every knob a
  user touches here is a real, measurable property of a real scope or sample.
* **Layer 2 - what a step consumes.** Pixel size, PSF sigma, attenuation, noise
  variance - *derived* from Layer 1 by small, documented, individually-testable
  helpers.

This file defines the full Layer-1 surface, the unit conversions, the static
``AnyStep`` union, and the Layer-2 physics helpers -
``Optics.diffraction_sigma_um``/``defocus_sigma_um``, ``Tissue.attenuation``/
``scatter_sigma_um``, the combined ``Acquisition.cell_optics``, and the
``ImageSensor.photons_to_counts`` sensor model. ``StepSpec.build()``
turns a spec into its executable step by looking its ``kind`` up in the
declarative :data:`minisim.steps.STEP_FOR_KIND` table; the step bodies
themselves live in :mod:`minisim.steps`.

Units convention: **everything physical is in seconds and µm/mm - never frames
or pixels.** ``Acquisition`` owns every conversion to pixels/frames.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from minisim.steps.base import Step


class SpecWarning(UserWarning):
    """Advisory warning for unusual-but-legal spec configurations.

    The simulator distinguishes *invalid* configs (which raise) from *unusual*
    ones (which warn but still run) - e.g. a focal plane outside the cell depth
    range, or motion larger than the configured FOV margin.
    """


class _Base(BaseModel):
    """Common config for every spec model.

    ``extra="forbid"`` turns a mistyped field name into a construction-time
    error instead of a silently-ignored value. ``frozen=True`` makes specs
    immutable, which is what lets a recording be identified by its spec: the
    cache keys off the canonical JSON form (see ``Spec.cache_key``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Physical interface - Acquisition / Optics / Tissue (Layer 1 + unit conversions)
# ---------------------------------------------------------------------------


# Immersion/tissue refractive index used to derive the diffraction depth of field
# from NA (≈ n·λ/NA²); ~1.33 for the watery cortex a miniscope images into.
_DOF_IMMERSION_N = 1.33


class Optics(_Base):
    """Objective optics - the measurable lens properties of a 1-photon scope.

    Layer-2 phenomenological quantities (diffraction sigma, defocus blur) are
    *derived* from these fields by the helper methods below. Pixel
    size is a joint optics×sensor quantity (sensor pitch / magnification) and so
    lives on ``Acquisition``, not here.

    Typical 1-photon miniscope ranges: NA 0.3–0.6, magnification ~5–10×, GCaMP
    emission ~510–540 nm.
    """

    na: float = Field(
        gt=0, default=0.45, description="Numerical aperture of the GRIN objective."
    )
    magnification: float = Field(
        gt=0,
        default=8.0,
        description="Optical magnification (sensor side / object side).",
    )
    emission_nm: float = Field(
        gt=0,
        default=525.0,
        description="Fluorophore emission wavelength, nm (GCaMP ≈ 525).",
    )
    depth_of_field_um: float | Literal["auto"] = Field(
        default="auto",
        description="±in-focus half-depth around the focal plane, µm. 'auto' (default) "
        "derives it from NA as ≈ n·λ/NA² (the diffraction depth of field - the physical "
        "behavior, since DOF is set by the optics, not chosen); a number overrides it.",
    )
    field_curvature_radius_um: float | None = Field(
        default=None,
        description="Petzval field-curvature radius, µm (typical miniscope ≈ 2000–3000). "
        "Off-axis cells focus *shallower* by the spherical sagitta; None = ideal flat "
        "field. A miniscope has no room for a field flattener, so this is usually finite.",
    )

    @field_validator("field_curvature_radius_um")
    @classmethod
    def _check_curvature(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(
                f"field_curvature_radius_um ({v}) must be > 0, or None for a flat field."
            )
        return v

    @field_validator("depth_of_field_um")
    @classmethod
    def _check_dof(cls, v: float | Literal["auto"]) -> float | Literal["auto"]:
        if v != "auto" and v <= 0:
            raise ValueError(f"depth_of_field_um ({v}) must be > 0, or 'auto'.")
        return v

    @property
    def resolved_depth_of_field_um(self) -> float:
        """The in-focus half-depth, µm, resolving ``"auto"`` to ≈ n·λ/NA².

        A numeric ``depth_of_field_um`` is used as-is. ``"auto"`` derives the
        diffraction depth of field from the aperture: ``σ_z ≈ n·λ/NA²`` (immersion
        index ``n`` ≈ tissue), the same half-depth the in-focus check uses. Higher
        NA ⇒ shallower focus (DOF falls as 1/NA²), so a real scope's DOF is set by
        its optics rather than picked by hand. At NA 0.30, λ ≈ 525 nm this is
        ≈ 7.8 µm; at NA 0.45, ≈ 3.4 µm."""
        if self.depth_of_field_um != "auto":
            return float(self.depth_of_field_um)
        return _DOF_IMMERSION_N * (self.emission_nm / 1000.0) / self.na**2

    # ---- Layer-2 helpers: small, documented, individually-testable approximations ----

    @property
    def diffraction_sigma_um(self) -> float:
        """Diffraction-limited PSF width (Gaussian σ), µm.

        A Gaussian stand-in for the Airy disk: the diffraction FWHM is
        ``≈ 0.51·λ/NA`` and ``σ = FWHM / 2.355 ≈ 0.21·λ/NA``. Ignores
        aberrations and the finite Airy tails - adequate for showing how NA and
        emission wavelength set the resolution floor. Smaller NA ⇒ larger σ
        (blurrier). At NA 0.45, λ ≈ 525 nm this is σ ≈ 0.24 µm.

        Note - pixel-limited, not diffraction-limited. Across realistic 1-photon
        miniscope NAs (~0.1–0.6) and green emission this σ is only ~0.2–1.1 µm,
        i.e. at or below the *object-space pixel size* (sensor pitch ÷
        magnification, typically 1–2 µm). So lateral resolution is set by the
        pixel sampling, not by diffraction - the diffraction PSF is real but
        rarely the limiting blur (defocus and scatter usually dominate it too).
        A practical consequence: a cell's intrinsic shape can be generated on a
        fine, sub-pixel grid *independent of the sensor*, then resampled to
        whatever pixel size the sensor implies, because the optics never resolve
        anything finer than the pixel grid anyway. (The teaching notebook relies
        on exactly this so that changing magnification/pitch only rescales a cell
        rather than re-rasterizing - and re-randomizing - its shape.)
        """
        return 0.21 * (self.emission_nm / 1000.0) / self.na

    def defocus_sigma_um(self, z_um: float, focal_um: float) -> float:
        """Out-of-focus blur (Gaussian σ), µm, for a cell at depth ``z_um``.

        Geometric defocus broadens linearly with the distance from the focal
        plane: ``σ ≈ NA·|z − z_focal|``. Symmetric about the focal plane (zero
        at ``z == focal_um``) and larger for higher NA (shallower depth of
        field). INTENSITY-CONSERVING: defocus spreads light without losing it,
        so the matching peak drop is applied in
        :meth:`Acquisition.cell_optics`; that is what separates defocus
        (spreads) from scatter (attenuates). ``focal_um`` is passed explicitly
        because :attr:`Acquisition.focal_depth_in_tissue_um` may be ``"auto"``, resolved to a
        concrete depth by the optics step before this is called.
        """
        return self.na * abs(z_um - focal_um)

    @property
    def collection_efficiency(self) -> float:
        """Fraction of a cell's emitted light the objective collects - ``∝ NA²``.

        A lens gathers light over a collection cone whose solid angle grows with
        ``NA²`` (small-angle ``Ω ∝ sin²θ = NA²``), so a low-NA miniscope objective
        is *fundamentally* dimmer than a high-NA one - independently of focus or
        depth. This is a flat multiplicative light-loss applied alongside scatter
        :meth:`Tissue.attenuation`; the absolute proportionality constant
        (``1/4n²`` etc.) is absorbed into the ``sensor`` step's
        ``photons_per_unit`` exposure scale, so what matters here is the ``NA²``
        scaling. At NA 0.18 vs 0.45 this is a ~6× brightness difference."""
        return self.na**2

    def focal_curvature_shift_um(self, r_um: float) -> float:
        """Field-curvature focal shift at field radius ``r_um`` from the axis, µm.

        Without a field flattener - which a miniscope has no room for - off-axis
        points focus on a curved (≈spherical) surface, not a plane. A point at
        radius ``r`` from the optical axis comes into best focus *shallower*
        (nearer the objective) than the on-axis focal plane, by the spherical
        sagitta ``R − √(R² − r²) ≈ r²/(2R)``. Returns a **non-negative** shift to
        be *subtracted* from the central focal depth (the in-focus surface bows
        toward the objective at the edges, always, for miniscope/standard optics).
        Zero when :attr:`field_curvature_radius_um` is ``None`` (an ideal flat
        field). Typical radii are 2–3 mm - large vs a soma, so a cell can be
        evaluated at its center rather than warping the footprint across the field.
        """
        radius = self.field_curvature_radius_um
        if radius is None:
            return 0.0
        r = min(abs(r_um), radius)  # clamp keeps the sqrt real for pathological r > R
        return radius - math.sqrt(radius * radius - r * r)


class ImageSensor(_Base):
    """Physical and noise properties of the bare image sensor (the detector).

    Named *image sensor*, not *camera*, on purpose: a camera bundles optics on
    top of a sensor, whereas this is only the photosensitive array and its
    readout chain. The optics live separately on ``Optics``. Together with the
    exposure scale on the ``sensor`` step, the fields here fully specify the
    photons→counts conversion.

    Typical CMOS miniscope sensors: ~2–6 µm pixel pitch, QE 0.6–0.9, read noise
    1–5 e⁻ RMS, 8–12-bit ADC.
    """

    n_px_height: int = Field(gt=0, default=256, description="Sensor height, pixels.")
    n_px_width: int = Field(gt=0, default=256, description="Sensor width, pixels.")
    pixel_pitch_um: float = Field(
        gt=0, default=3.0, description="Physical sensor pixel pitch, µm."
    )
    quantum_efficiency: float = Field(
        gt=0, le=1, default=0.7, description="Photon → electron conversion efficiency."
    )
    read_noise_e: float = Field(
        ge=0, default=2.0, description="Read noise, electrons RMS."
    )
    gain_adu_per_e: float = Field(
        gt=0, default=1.0, description="Camera gain, ADU per electron."
    )
    bit_depth: int = Field(
        gt=0,
        default=8,
        description="ADC bit depth; counts clipped to [0, 2^bit_depth − 1].",
    )

    def photons_to_counts(
        self, photons: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Forward sensor model: incident photons → digitized ADC counts.

        The only place fluorescence becomes integer counts::

            λ      = photons · quantum_efficiency            # expected detected e⁻
            e⁻     = Normal(λ, √(λ + read_noise_e²))         # shot + read noise
            adu    = e⁻ · gain_adu_per_e
            counts = clip(floor(adu), 0, 2**bit_depth − 1)

        The two physical noise sources are combined into a **single Gaussian**. Shot
        noise is Poisson on the detected electrons (variance ``λ``); read noise is
        additive Gaussian in electrons (variance ``read_noise_e²``). Because the two
        are independent their variances add, and Poisson(λ) ≈ Normal(λ, λ) to high
        accuracy once ``λ`` is more than ~10 - which holds across essentially every
        pixel of a 1-photon recording, where a substantial background (out-of-focus
        fluorescence, neuropil, leakage glow) keeps ``λ`` well above that floor. So
        the electron count is drawn as ``Normal(λ, √(λ + read_noise_e²))``: one
        ``standard_normal`` draw that carries *both* the signal-dependent shot term
        (the ``√λ``, so brighter pixels are noisier) and the constant read term. This
        is the standard sensor-noise model, and it is far cheaper than sampling a
        large-λ Poisson; the only thing dropped is the discrete-Poisson shape at
        near-zero λ, the regime a 1p recording's background keeps it out of. Read
        noise dominates anyway in the dim, high-analog-gain corner.

        Quantization is ``floor`` (an ADC truncates), and counts are clipped to the
        converter's representable range. ``photons`` is the per-pixel expected photon
        count - the ``Sensor`` step produces it from scene intensity × its
        ``photons_per_unit`` exposure scale. Returns a float array holding
        integer-valued counts (the float container is set by ``Output.store_dtype``).
        """
        photons = np.asarray(photons, dtype=float)
        lam = photons * self.quantum_efficiency  # expected detected electrons
        # Shot (variance λ) + read (variance read_noise_e²) as one Gaussian: a single
        # standard_normal draw scaled by the combined std. One draw per call keeps the
        # streaming writer's per-frame replay in lock-step with simulate().
        std = np.sqrt(lam + self.read_noise_e**2)
        electrons = lam + std * rng.standard_normal(photons.shape)
        counts = np.floor(electrons * self.gain_adu_per_e)
        return np.clip(counts, 0.0, 2**self.bit_depth - 1)


class Tissue(_Base):
    """Light-scattering properties of the imaged tissue, as a function of depth.

    The fields parametrize Layer-2 helpers (``attenuation``, ``scatter_sigma``).
    Scattering has two separable consequences on a cell's image, modelled by two
    knobs: it *dims* the sharp signal (light scattered out of the collection cone
    is lost → :meth:`attenuation`, the ``scatter_mfp_*`` fields) and it *blurs*
    the footprint (forward-scattered light, ``g ≈ 0.88``, is recollected as a
    growing halo → :meth:`scatter_sigma_um`, ``scatter_blur_per_um``).

    Round-trip scattering, asymmetric. The signal makes two passes through
    tissue, but they attenuate very differently:

    * **Excitation in (≈470 nm)** is delivered by *widefield* illumination, so it
      reaches a cell as a *diffuse* fluence, not a ballistic beam. Diffuse light
      penetrates far (transport length ≈ 800 µm) and its fluence actually peaks a
      few hundred µm deep before falling (Ma et al. 2020, Neurophotonics
      7:031208), so over the depths a 1-photon scope images, excitation barely
      dims cells - a *long* effective MFP (``scatter_mfp_excitation_um``).
    * **Emission out (≈525 nm)** is the *image-forming* sharp signal, which
      decays at roughly the scattering MFP (``scatter_mfp_emission_um``). This leg
      dominates the depth-dimming.

    Modelling excitation as ballistic (a short, symmetric leg) double-counts the
    loss and makes deep cells unrealistically dim; the asymmetric split above is
    both more honest and what keeps the round trip from over-attenuating.

    Literature anchors (mouse cortex / gray matter). The *ballistic* scattering
    mean free path at blue/green is ≈ 40–50 µm (μ_s ≈ 200 cm⁻¹, g ≈ 0.86–0.89):
    ≈ 47 µm at 473 nm (Al-Juboori et al. 2013, PLoS ONE 8:e67626) and ≈ 38 µm at
    515 nm (Azimipour et al. 2014, Biomed. Opt. Express). That ballistic length is
    what sets the *blur* rate. The light an objective actually *collects* decays
    more slowly, because the strong forward scattering (g ≈ 0.88) is largely
    recollected - so the emission leg uses the *high end* of the scattering-MFP
    literature (~100 µm), and the diffuse excitation leg is longer still; their
    round trip gives an effective ≈ 85 µm (see :attr:`scatter_mfp_um`).
    """

    scatter_mfp_excitation_um: float = Field(
        gt=0,
        default=600.0,
        description="Effective attenuation MFP for the excitation leg (≈470 nm, in) - long, diffuse fluence, µm.",
    )
    scatter_mfp_emission_um: float = Field(
        gt=0,
        default=100.0,
        description="Effective attenuation MFP for the emission leg (≈525 nm, out) - the image-forming scattering MFP, µm.",
    )
    scatter_blur_per_um: float = Field(
        ge=0,
        default=0.05,
        description="Linear broadening of the footprint per µm of depth (µm sigma per µm depth).",
    )

    # ---- Layer-2 helpers: scattering as a function of absolute depth ----

    @property
    def scatter_mfp_um(self) -> float:
        """Effective round-trip (excitation × emission) attenuation MFP, µm.

        The two exponential legs multiply, ``exp(−z/mfp_ex)·exp(−z/mfp_em) =
        exp(−z/mfp_eff)``, so the combined length is the harmonic-style
        reciprocal sum ``1/mfp_eff = 1/mfp_ex + 1/mfp_em``. With the defaults
        (excitation 600 µm, emission 100 µm) this is ≈ 85.7 µm - dominated by the
        emission leg, since the diffuse excitation leg attenuates little over
        imaging depths (see the class docstring for why the legs are asymmetric).
        """
        return 1.0 / (
            1.0 / self.scatter_mfp_excitation_um + 1.0 / self.scatter_mfp_emission_um
        )

    def attenuation(self, z_um: float) -> float:
        """Fraction of light surviving the round-trip scatter from depth ``z_um`` - in (0, 1].

        Beer–Lambert decay applied on *both* legs the signal travels: excitation
        in (≈470 nm) then emission out (≈525 nm). The product collapses to a
        single exponential over the effective MFP, ``exp(−z / scatter_mfp_um)``
        (see :attr:`scatter_mfp_um`). Monotonically decreasing in depth and equal
        to 1 at the surface (``z = 0``). Genuinely *removes* light - unlike
        defocus - so a deep cell is irreversibly dimmer, the irreducible limit
        the module teaches.
        """
        return math.exp(-z_um / self.scatter_mfp_um)

    def scatter_sigma_um(self, z_um: float) -> float:
        """Scatter-induced footprint broadening (Gaussian σ), µm, at depth ``z_um``.

        Linear phenomenological model ``σ = scatter_blur_per_um · z``: deeper
        cells scatter more and so appear both larger and dimmer (see
        :meth:`attenuation`). Monotonically increasing in depth and zero at the
        surface. Unlike defocus this is not intensity-conserving - it co-occurs
        with attenuation. The rate is set by the ballistic scattering MFP
        (~40–50 µm at blue/green; Al-Juboori et al. 2013): with the default 0.05,
        a cell at 100 µm picks up σ ≈ 5 µm (FWHM ≈ 12 µm, about a soma diameter),
        so deep cells read as the blurry halos seen in real 1-photon data.
        """
        return self.scatter_blur_per_um * z_um


class Acquisition(_Base):
    """The physical acquisition: optics, image sensor, tissue, and sampling.

    Owns *all* unit conversions between the physical world (µm, seconds) and the
    sampled world (pixels, frames). Pixel size is the joint optics×sensor
    quantity ``image_sensor.pixel_pitch_um / optics.magnification``; FOV is then
    derived from the sensor's pixel count - any two of {FOV, pixel size, pixel
    count} fix the third.
    """

    optics: Optics = Field(default_factory=Optics)
    image_sensor: ImageSensor = Field(default_factory=ImageSensor)
    tissue: Tissue = Field(default_factory=Tissue)
    fps: float = Field(gt=0, default=20.0, description="Frame rate, frames per second.")
    duration_s: float = Field(
        gt=0, default=150.0, description="Recording duration, seconds."
    )
    focal_depth_in_tissue_um: float | Literal["auto"] = Field(
        default="auto",
        description="Depth of the focal plane below the tissue surface, µm (0 = surface), "
        "in the same coordinate as each cell's depth z. Cells above or below it defocus; "
        "'auto' resolves to the median realized cell depth at the optics step.",
    )
    front_working_distance_um: float | None = Field(
        default=None,
        description="Front working distance (lens front → focal point), µm - Miniscope V4 ≈ "
        "700. Informational only: it does NOT affect the simulation (the optics math uses "
        "focal_depth_in_tissue_um), but it's a physically relevant number for surgery/implant "
        "planning, so it's recorded here.",
    )

    @field_validator("focal_depth_in_tissue_um")
    @classmethod
    def _check_focal_depth(cls, v: float | Literal["auto"]) -> float | Literal["auto"]:
        if v != "auto" and v < 0:
            raise ValueError(f"focal_depth_in_tissue_um ({v}) must be ≥ 0, or 'auto'.")
        return v

    @field_validator("front_working_distance_um")
    @classmethod
    def _check_fwd(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"front_working_distance_um ({v}) must be > 0, or None.")
        return v

    @property
    def pixel_size_um(self) -> float:
        """Object-space size of one pixel, µm (sensor pitch / magnification)."""
        return self.image_sensor.pixel_pitch_um / self.optics.magnification

    @property
    def n_frames(self) -> int:
        """Number of frames in the recording (duration × fps, rounded)."""
        return round(self.duration_s * self.fps)

    @property
    def fov_um(self) -> tuple[float, float]:
        """Field of view (height, width) in µm - derived from pixels × pixel size."""
        return (
            self.image_sensor.n_px_height * self.pixel_size_um,
            self.image_sensor.n_px_width * self.pixel_size_um,
        )

    def um_to_px(self, um: float) -> float:
        """Convert a physical distance (µm) to pixels."""
        return um / self.pixel_size_um

    def um_to_index(
        self, y_um: float, x_um: float, shape: tuple[int, int]
    ) -> tuple[float, float]:
        """Array ``(row, col)`` for an optical-center µm position on a centered grid.

        Lateral positions live in the **optical-center frame**: the optical axis
        is ``(0, 0)`` µm, ``+y`` points down (increasing row) and ``+x`` right
        (increasing column), matching the image array. Both the canvas and the
        sensor FOV are centered on that axis, so this one map serves either by
        passing its own ``shape`` - ``(row, col) = grid_center + (y, x)/pixel_size``.
        Returns floats (sub-pixel); callers round/clip as needed. (Depth ``z`` is
        not lateral - it stays measured from the tissue surface, ``0`` = surface.)
        """
        inv = 1.0 / self.pixel_size_um
        return ((shape[0] - 1) / 2.0 + y_um * inv, (shape[1] - 1) / 2.0 + x_um * inv)

    def s_to_frame(self, s: float) -> float:
        """Convert a duration (seconds) to a (fractional) frame count."""
        return s * self.fps

    def cell_optics(self, z_um: float, focal_um: float) -> tuple[float, float]:
        """Combined per-cell optical degradation: ``(sigma_px, brightness)``.

        Folds the three Layer-2 effects into the two numbers the optics step
        applies to a footprint - a blur width (pixels) and a brightness scale.
        Blurs add in quadrature; brightness factors multiply::

            σ_0   = hypot(diffraction_sigma_um, scatter_sigma_um(z))   # all but defocus
            σ_tot = hypot(σ_0, defocus_sigma_um(z, focal))
            brightness = (σ_0² / σ_tot²) · attenuation(z) · collection_efficiency
            sigma_px   = σ_tot / pixel_size_um

        The ``σ_0²/σ_tot²`` factor is the peak drop that makes defocus
        intensity-conserving (a 2-D Gaussian's peak × area is constant); the two
        light-loss factors that actually remove signal are scatter
        ``attenuation(z)`` (depth) and ``collection_efficiency`` (``∝ NA²``, the
        objective's light-gathering power). Both are independent of the focal
        plane, so ``sigma_px² · brightness`` remains independent of the focal
        plane - the invariant the conservation test asserts. ``focal_um`` is the
        resolved (numeric) focal depth; ``diffraction_sigma_um > 0`` always, so
        ``σ_tot`` is never zero.

        Two distinct quantities come out, consumed separately by the optics
        step: ``sigma_px`` is the PSF width the footprint is *convolved* with,
        and the footprint's brightness then scales by ``attenuation(z)`` **alone**
        - the convolution itself produces the defocus peak drop, so applying
        ``brightness`` to the footprint too would double-count it. The returned
        ``brightness`` is instead the *point-source peak* scalar: how far a
        cell's peak signal sits above the noise, i.e. the per-cell effective
        brightness used (with the illumination field and sensor floor) to decide
        detectability at ``finalize()``.
        """
        sigma_0 = math.hypot(
            self.optics.diffraction_sigma_um, self.tissue.scatter_sigma_um(z_um)
        )
        sigma_total = math.hypot(sigma_0, self.optics.defocus_sigma_um(z_um, focal_um))
        brightness = (
            (sigma_0**2 / sigma_total**2)
            * self.tissue.attenuation(z_um)
            * self.optics.collection_efficiency
        )
        sigma_px = sigma_total / self.pixel_size_um
        return sigma_px, brightness


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class Output(_Base):
    """Final-array formatting - formatting only, never rescaling (honest radiometry)."""

    save_intermediates: bool = Field(
        default=False,
        description="Retain a snapshot after every step (test oracle + teaching visuals). "
        "Default False to keep memory flat for the programmatic and sweep paths; the "
        "two notebook presets opt in explicitly. When False, only `observed` + "
        "`ground_truth` are kept and stage() raises.",
    )
    store_dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="Float container for the integer-valued sensor counts.",
    )


# ---------------------------------------------------------------------------
# Step framework - StepSpec base + registry + the static AnyStep union
# ---------------------------------------------------------------------------


# The canonical execution order of the pipeline as a total order over step kinds.
# It follows the physical domains (cell -> tissue -> motion -> sensor) and fixes
# the within-domain order too. This is THE order steps run in, so the order a caller
# lists them in does not matter - :func:`order_steps` (used by both ``simulate``
# and :class:`Spec`) sorts any list into this one. It is a topological extension
# of every step's ``requires`` (a step always follows the kinds it consumes); the
# test suite asserts that, and that this is a permutation of the step catalog, so
# the order cannot silently drift as kinds are added.
_PIPELINE_ORDER: tuple[str, ...] = (
    "place_neurons",
    "cell_activity",
    "bleaching",
    "optics",
    "composite",
    "neuropil",
    "vasculature",
    "brain_motion",
    "illumination_profile",
    "vignette",
    "leakage",
    "sensor",
)
_KIND_RANK: dict[str, int] = {kind: i for i, kind in enumerate(_PIPELINE_ORDER)}

_StepT = TypeVar("_StepT", bound="StepSpec")


def order_steps(steps: Iterable[_StepT]) -> list[_StepT]:
    """Sort steps into the canonical pipeline order (:data:`_PIPELINE_ORDER`).

    :class:`Spec` runs every step list through this, so the order steps are listed
    in never matters. A *stable* sort: steps of equal rank keep their input order,
    and a kind absent from :data:`_PIPELINE_ORDER` (a future/unknown step) sorts to
    the end rather than raising, so no step is dropped. Returns a new list; the step
    specs themselves are not copied or mutated.
    """
    return sorted(steps, key=lambda s: _KIND_RANK.get(s.kind, len(_PIPELINE_ORDER)))


class StepSpec(_Base):
    """Base class for a single pipeline step's configuration.

    A concrete step spec carries its physical parameters and a literal ``kind``
    discriminator, and declares its ``domain`` (a class attribute). ``build()``
    turns the spec into the executable step that mutates a ``Scene``, resolving
    ``kind`` through the :data:`minisim.steps.STEP_FOR_KIND` table.

    ``requires`` declares the step kinds whose output this step consumes through
    the shared ``Scene`` (e.g. ``composite`` reads the footprints ``place_neurons``
    makes and the traces ``cell_activity`` makes). It is about *presence-order*, not
    completeness: a present requirement is placed before this step by the canonical
    ordering (:data:`_PIPELINE_ORDER`), but it may be absent entirely. Partial
    pipelines are first-class - a spec of ``[place_neurons, cell_activity,
    composite]`` with no sensor is valid, so targeted test data for a downstream
    calcium pipeline can exercise just a few stages.
    """

    domain: ClassVar[Literal["cell", "tissue", "motion", "sensor"]]
    requires: ClassVar[tuple[str, ...]] = ()
    kind: str

    def build(self, acq: Acquisition, rng) -> Step:
        """Return the executable step (a callable that mutates a Scene).

        Resolves this spec's ``kind`` through the declarative
        :data:`minisim.steps.STEP_FOR_KIND` table and constructs the matching
        step. The table is imported lazily because the step modules depend
        (through ``recording``) back on this module, so it cannot be imported at
        load time.
        """
        from minisim.steps import STEP_FOR_KIND

        return STEP_FOR_KIND[self.kind](self, acq, rng)


# ---------------------------------------------------------------------------
# Step catalog (cell → tissue → motion → sensor). Fields define the v1 surface;
# the executable `build()` bodies live in `minisim.steps`.
# ---------------------------------------------------------------------------


class NeuronPopulation(_Base):
    """One homogeneous group of neurons to place: a morphology + a 3-D distribution.

    A *population* is the unit ``place_neurons`` actually samples - one cell shape
    (soma-only or cytosolic) at one soma size, scattered at one volumetric density
    across one depth range. A single population is the common case; list several on
    :attr:`PlaceNeurons.populations` to build layered anatomy - e.g. a thin
    soma-targeted band over a deep cytosolic volume. The cell *count* is derived
    volumetrically from ``density_per_mm3`` and the depth thickness (see
    :func:`~minisim.steps.cell.sample_neurons`); brightness is biology and is drawn
    later in ``cell_activity``, never here.
    """

    density_per_mm3: float = Field(
        gt=0,
        default=25000.0,
        description="Cell volumetric density (cells/mm³); count = density × FOV area "
        "× depth thickness, the thickness floored at one soma diameter so a thin or "
        "planar layer still yields cells.",
    )
    soma_radius_um: float = Field(
        gt=0,
        default=7.0,
        description="Soma radius, µm (typical cortical neuron ≈ 5–10).",
    )
    irregularity: float = Field(
        ge=0,
        le=1,
        default=0.3,
        description="0 = smooth disk; higher = lumpier soma (low-pass-noise threshold).",
    )
    morphology: Literal["soma", "cytosolic"] = Field(
        default="soma",
        description="GCaMP targeting variant: 'soma' = soma-targeted (lumpy disk "
        "only); 'cytosolic' = standard GCaMP (soma + branched proximal dendrites).",
    )
    dendrite_length_um: float = Field(
        gt=0,
        default=24.0,
        description="Nominal proximal-dendrite reach, µm (cytosolic only); kept "
        "proximal so the arbor shapes the blurred footprint without blowing up the "
        "sparse-patch bounding box. The per-cell count is drawn randomly (not a "
        "spec input, so cells differ), and each dendrite's length jitters around "
        "this value and may branch.",
    )
    dendrite_width_um: float = Field(
        gt=0,
        default=3.0,
        description="Proximal-dendrite base width (diameter), µm; tapers to a "
        "~1 px thread at the tip (cytosolic only).",
    )
    depth_range_um: tuple[float, float] = Field(
        default=(0.0, 200.0), description="(min, max) depth into tissue, µm."
    )
    min_distance_um: float = Field(
        ge=0,
        default=0.0,
        description="3-D center-to-center minimum (Poisson-disk if > 0).",
    )
    positions_um: list[tuple[float, float, float]] | None = Field(
        default=None,
        description="Explicit soma centers as (z, y, x) µm tuples - depth first, "
        "matching Cell.center_um (not x,y,z). z is depth below the tissue surface "
        "(0 = surface); y, x are lateral in the optical-center frame: the optical "
        "axis is (0, 0), +y down and +x right (image convention), so a cell at "
        "(z, 0, 0) sits dead-center regardless of the motion margin, and these are "
        "the same coordinates GroundTruth.centers_um reports back. When given, these "
        "exact positions are placed instead of density-sampling, so the distribution "
        "fields (density_per_mm3, depth_range_um, min_distance_um) are ignored; the "
        "shape fields (soma_radius_um, irregularity, morphology, dendrites) still "
        "apply to each placed cell.",
    )

    @field_validator("depth_range_um")
    @classmethod
    def _check_depth_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo < 0:
            raise ValueError(f"depth_range_um min ({lo}) must be ≥ 0.")
        if hi < lo:
            raise ValueError(f"depth_range_um max ({hi}) must be ≥ min ({lo}).")
        return v


class PlaceNeurons(StepSpec, NeuronPopulation):
    """Place generic neurons in a 3-D µm volume, soma-only or with dendrites.

    'Place' is the verb - this *positions neurons in space* (anchored at the cell
    body); it is unrelated to hippocampal *place cells*. v1 models one generic
    excitable cell type (an irregular soma blob) with two GCaMP targeting variants
    via ``morphology``: ``"soma"`` (soma-targeted, body only) or ``"cytosolic"``
    (standard GCaMP, the soma plus a few tapering proximal dendrites). There is no
    further cell-type distinction and no spatial/behavioral tuning. Footprints are
    2-D masks carrying a scalar depth ``z``; out-of-focus neurons that become
    background emerge for free downstream from ``z`` + ``optics``.

    The step *is* a single :class:`NeuronPopulation`: its inherited fields
    (``morphology``, ``soma_radius_um``, ``depth_range_um``, …) describe that one
    group. To place several distinct groups together - a thin soma-targeted layer
    over a deep cytosolic volume, say - set :attr:`populations` to a list of
    ``NeuronPopulation`` instead; the step then samples each in turn (its own
    step-level population fields are ignored, and mixing the two raises).
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["place_neurons"] = "place_neurons"
    populations: list[NeuronPopulation] | None = Field(
        default=None,
        description="Distinct neuron populations to place together (e.g. a thin "
        "layer + a deep volume). None (default) = the step is a single population "
        "described by its own fields; a list = sample each entry in turn and ignore "
        "the step-level population fields.",
    )

    @property
    def resolved_populations(self) -> list[NeuronPopulation]:
        """The population(s) to place: explicit ``populations`` or the step-as-one.

        When ``populations`` is None the step is itself one population, so rebuild a
        plain :class:`NeuronPopulation` from the inherited fields - that way callers
        (the step, :func:`~minisim.steps.cell.sample_neurons`) iterate one uniform
        list either way.
        """
        if self.populations is not None:
            return self.populations
        return [
            NeuronPopulation(
                **{f: getattr(self, f) for f in NeuronPopulation.model_fields}
            )
        ]

    @model_validator(mode="after")
    def _check_populations(self) -> PlaceNeurons:
        if self.populations is None:
            return self
        if not self.populations:
            raise ValueError(
                "populations must list at least one NeuronPopulation, or be None."
            )
        # Flag step-level population fields left at a *non-default* value (they would
        # be silently ignored alongside `populations`). Comparing against the defaults
        # rather than `model_fields_set` keeps this stable across a serialization
        # round-trip: `model_dump()` marks every field as set, so a `model_fields_set`
        # check would spuriously fire whenever a populations-based spec is re-validated
        # (by `sweep()`, a cache/JSON reload, ...), making such a spec impossible to use.
        step_defaults = NeuronPopulation()
        clash = sorted(
            f
            for f in NeuronPopulation.model_fields
            if getattr(self, f) != getattr(step_defaults, f)
        )
        if clash:
            raise ValueError(
                f"place_neurons sets both populations and step-level population "
                f"field(s) {clash}; when using populations, set those on each "
                "NeuronPopulation entry instead (the step-level fields are ignored)."
            )
        return self


class CellActivity(StepSpec):
    """Calcium activity: 2-state Markov gate → Poisson spikes → double-exp kernel.

    Modeled on the CaLab web simulator: spikes are generated on a high-resolution
    grid (``spike_sim_hz``, ~300 Hz), convolved with the double-exponential kernel
    ``k(t) = exp(-t/τ_d) − exp(-t/τ_r)`` at that rate, then bin-averaged down to the
    camera frame rate (exposure integration). One spike per fine bin respects the
    ~3 ms refractory period. The ground-truth ``S`` is the per-frame spike *count*
    (the fine train is binned away - nothing recovers spikes faster than the frame
    rate). Indicator saturation and per-cell τ jitter are deferred to v1.1.

    Amplitude is biology and lives here as a single per-cell gain: ``brightness_cv``
    is the cell-to-cell spread of an overall expression/response gain that scales
    each cell's *whole* trace (baseline and transients together). The emitted trace
    is the **clean ground truth** ``C``; measurement noise is deliberately *not*
    added here. Photon shot noise and read noise enter at the ``sensor``, background
    fluctuations at ``neuropil`` - so any SNR is an emergent property of the physical
    chain, computable downstream, never an input.
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["cell_activity"] = "cell_activity"
    requires: ClassVar[tuple[str, ...]] = ("place_neurons",)  # needs cells to animate
    spike_sim_hz: float = Field(
        gt=0,
        default=300.0,
        description="High-res spike-simulation rate, Hz (~300 = a ~3 ms refractory); binned to the frame rate.",
    )
    # Defaults = CaLab's "moderate" SPIKE_ACTIVITY level; see spike_activity_params.
    p_quiescent_to_active: float = Field(
        gt=0, default=0.005, description="Per-frame quiescent→active transition prob."
    )
    p_active_to_quiescent: float = Field(
        gt=0, default=0.3, description="Per-frame active→quiescent transition prob."
    )
    active_rate_hz: float = Field(
        gt=0,
        default=150.0,
        description="Instantaneous firing rate while active, Hz (the in-burst rate).",
    )
    quiescent_rate_hz: float = Field(
        ge=0,
        default=0.6,
        description="Instantaneous firing rate while quiescent, Hz (the intrinsic background).",
    )
    tau_rise_s: float = Field(
        gt=0, default=0.05, description="Calcium rise time constant, s."
    )
    tau_decay_s: float = Field(
        gt=0, default=0.5, description="Calcium decay time constant, s."
    )
    brightness_cv: float = Field(
        ge=0,
        default=0.3,
        description="Cell-to-cell brightness spread: lognormal CV (mean 1) of the "
        "per-cell expression/response gain that scales the whole trace. 0 = every "
        "cell equally bright.",
    )
    f0: float = Field(ge=0, default=1.0, description="Baseline fluorescence.")
    trace_noise: float = Field(
        ge=0,
        default=0.0,
        description="Non-physical additive trace noise (default 0). An advanced "
        "override only; real noise enters at sensor/neuropil, not here.",
    )


class CellOptics(StepSpec):
    """Per-cell diffraction + defocus ``|z − z_f|`` + scatter(z) blur & attenuation.

    No tunable fields: blur and attenuation are fully determined by each cell's
    ``z`` plus the physical ``Optics``/``Tissue`` constants on ``Acquisition``.
    Writes the observed (degraded) footprint alongside the planted (sharp) one,
    sets the geometric ``in_focus`` flag, and stores the per-cell
    ``optical_brightness`` peak scalar. ``detectable`` is *not* set here - it is
    a whole-pipeline flag (optics × illumination vs the sensor noise floor)
    assembled in ``finalize()``.
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["optics"] = "optics"
    requires: ClassVar[tuple[str, ...]] = (
        "place_neurons",
    )  # degrades planted footprints


class Composite(StepSpec):
    """Composite ``Σ_i degraded_footprint_i × trace_i`` into the movie.

    The built step's snapshot name is ``"cells_only"``. The planted (sharp)
    ``A``/``C`` remain the ideal, optics-free target in ground truth.
    """

    domain: ClassVar[str] = "tissue"
    kind: Literal["composite"] = "composite"
    # Composites footprint × trace. optics is an optional enhancer (composite falls
    # back to the planted footprint), so it is not required.
    requires: ClassVar[tuple[str, ...]] = ("place_neurons", "cell_activity")


class Neuropil(StepSpec):
    """Additive diffuse background from the dendritic/axonal felt around the cells.

    A smooth spatial field (mesh-density variation on ``spatial_sigma_um``)
    modulated by a temporal envelope that is **biologically driven**: the haze is
    the aggregate calcium of the surrounding neural processes, so its time course
    is the local population activity, lagged and smoothed by the felt's
    integration (``population_tau_s``, short). Each component's envelope mixes
    that population driver with an independent slow drift (the unmodeled
    out-of-FOV/out-of-plane tissue, ``temporal_tau_s``, slow) at
    ``population_coupling``. This is the modeled diffuse mesh only - out-of-focus
    somata are a *separate* background that emerges for free from
    ``place_neurons`` + ``optics``.
    """

    domain: ClassVar[str] = "tissue"
    kind: Literal["neuropil"] = "neuropil"
    # Couples to the local population's calcium; falls back to independent drift if
    # absent, so cell_activity is order-only (must precede when present), not required.
    requires: ClassVar[tuple[str, ...]] = ("cell_activity",)
    spatial_sigma_um: float = Field(
        gt=0, default=40.0, description="Spatial smoothness of the mesh, µm."
    )
    temporal_tau_s: float = Field(
        gt=0,
        default=10.0,
        description="OU correlation time of the independent slow-drift leg, s (slow).",
    )
    population_tau_s: float = Field(
        gt=0,
        default=1.5,
        description="Low-pass time constant of the population-coupled leg, s: the felt's integration/lag, short relative to the drift.",
    )
    amplitude: float = Field(
        gt=0, default=0.5, description="Background amplitude relative to cell signal."
    )
    n_components: int = Field(
        ge=1, default=3, description="Number of independent diffuse components."
    )
    population_coupling: float = Field(
        ge=0,
        le=1,
        default=0.7,
        description="Fraction of the temporal envelope driven by local population activity vs independent slow drift (0=pure drift, 1=pure population).",
    )


class VesselLayer(_Base):
    """One depth's worth of randomly-grown blood vessels - a dark absorbing tree.

    A vessel absorbs both the excitation going in and the emission coming out
    (haemoglobin), so a layer is a *shadow* cast on the tissue behind it, not a
    light source. Each layer grows ``n_roots`` branching trees (see
    :func:`~minisim.steps.tissue.grow_vessel_tree`) entering from the field edges,
    tapering from ``root_radius_um`` trunks down to ``min_radius_um`` capillaries by
    Murray's law, then rasterizes them to a Beer-Lambert transmission mask and blurs
    it by the **defocus + scatter** at this layer's ``depth_um`` (reusing the optics
    model): a vessel near the focal plane is a crisp dark thread, one far from focus
    a broad soft shadow. List several layers on :class:`Vasculature` to stack scales
    and depths (e.g. a shallow large-caliber cortical layer above the cells, plus a
    deeper fine-capillary bed).

    All lengths are µm; ``depth_um`` shares the cell-``z`` / focal-plane coordinate
    (0 = surface), so a layer can sit above the cells (shallower) or, less commonly,
    below them.
    """

    depth_um: float = Field(
        ge=0,
        default=30.0,
        description="Depth of this vessel layer below the surface, µm (same coord as cell z / focal plane).",
    )
    n_roots: int = Field(
        ge=1,
        default=4,
        description="Number of vessel trees entering this layer from the field edges.",
    )
    root_radius_um: float = Field(
        gt=0, default=22.0, description="Trunk (thickest) vessel radius, µm."
    )
    min_radius_um: float = Field(
        gt=0,
        default=2.0,
        description="Capillary floor: a branch terminates once Murray-law tapering drops below this, µm.",
    )
    opacity: float = Field(
        gt=0,
        le=1,
        default=0.85,
        description="Peak absorption of a thick trunk in (0, 1]; transmission floors at 1 − opacity (real vessels are never fully black).",
    )
    absorption_per_um: float = Field(
        gt=0,
        default=0.04,
        description="Beer-Lambert absorption per µm of blood path length: sets the capillary-vs-trunk contrast.",
    )
    branch_prob: float = Field(
        ge=0,
        lt=1,
        default=0.2,
        description="Per-step bifurcation probability (a side branch peels off, the main vessel continues).",
    )
    tortuosity_deg: float = Field(
        ge=0,
        default=8.0,
        description="Std of the per-step heading perturbation, degrees (0 = ruler-straight vessels).",
    )
    branch_angle_deg: float = Field(
        ge=0,
        default=30.0,
        description="Base heading deviation at a bifurcation, degrees; the thinner child turns more.",
    )
    branch_area_main: float = Field(
        gt=0.5,
        lt=1.0,
        default=0.7,
        description="Main child's share of the conserved Murray cube-sum at a bifurcation (0.5 = symmetric, →1 = a thin twig off a barely-thinned trunk).",
    )
    step_per_radius: float = Field(
        gt=0,
        default=1.5,
        description="Growth step length as a multiple of the current radius.",
    )

    @model_validator(mode="after")
    def _check_radii(self) -> VesselLayer:
        if self.min_radius_um >= self.root_radius_um:
            raise ValueError(
                f"min_radius_um ({self.min_radius_um}) must be < root_radius_um "
                f"({self.root_radius_um}); the capillary floor is below the trunk."
            )
        return self


class Vasculature(StepSpec):
    """Dark, spatially-static absorbing vessels - a multiplicative shadow on the movie.

    Grows the ``layers`` of branching vessel trees, composites their Beer-Lambert
    transmission masks multiplicatively, and multiplies the result into the
    brain-frame movie (``movie *= M``). Because it is a *tissue*-domain effect
    applied before ``brain_motion``, the vessel pattern is fixed in the brain frame
    and rides the motion crop exactly with the cells - the high-contrast, temporally
    static landmark that motion-correction leans on (unlike the cells, whose
    brightness flickers with activity). It is also a tunable *confound*: a vessel
    crossing a soma corrupts its footprint and trace, so dialing vasculature up
    stress-tests footprint/dynamics extraction. That occlusion is scored, not
    hidden: each cell's footprint-weighted vessel burden is recorded as
    ``GroundTruth.vessel_overlap_fraction``, and a vessel over a soma dims its peak
    in the ``detectable`` test - but the footprints (``A_observed``) stay
    vessel-free, the single-cell optical truth the confound is measured against.

    **v1 simplification:** a single multiplicative attenuation of the already
    composited movie. Excitation and emission absorption are lumped into one static
    transmission, and light generated in front of the vessel is not exempted from
    the shadow (real out-of-plane / in-front fluorescence fills it in); the
    ``opacity`` floor stands in for that fill rather than modeling it. Good enough
    for a landmark and an occlusion confound; not a radiative-transfer model.

    Off by default: ``enabled=False`` and an empty ``layers`` both make the step a
    no-op, so it must be explicitly turned on with at least one
    :class:`VesselLayer`. The static absorbing mask is recorded to
    ``GroundTruth.vasculature_mask`` (cropped to the FOV) so the confound is
    scoreable. Temporal pulsation (cardiac / vasomotion) is deliberately deferred -
    the vessels are static here; only the brightness, not the position, would ever
    breathe, so the landmark property is unaffected either way.
    """

    domain: ClassVar[str] = "tissue"
    kind: Literal["vasculature"] = "vasculature"
    enabled: bool = Field(
        default=False,
        description="Master switch; with no layers the step is a no-op regardless.",
    )
    layers: list[VesselLayer] = Field(
        default_factory=list,
        description="Vessel layers to grow, each at its own depth/caliber. Empty = no vessels.",
    )


class Bleaching(StepSpec):
    """Per-cell, activity-driven photobleaching, opposed by protein turnover.

    Photobleaching is a per-photon hazard, so each cell loses intact fluorophore in
    proportion to how much it emits (its calcium activity × excitation intensity),
    while turnover replenishes it toward full expression. The realized envelope is a
    *cell-domain* effect computed before ``composite`` (see
    :class:`~minisim.steps.tissue.BleachingStep`), not a global movie multiply: busy
    or brightly-lit cells fade faster and to a lower floor, and with the light off
    the pool recovers, so the same model spans single recordings and repeated
    sessions. Defaults are calibrated to measured CA1 GCaMP6f bleaching curves
    across a wide range of excitation powers (bleaching linear in excitation;
    effective recovery ≈5.5 h, so darkness restores the pool within a couple of days).
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["bleaching"] = "bleaching"
    requires: ClassVar[tuple[str, ...]] = (
        "cell_activity",
    )  # bleaches each cell's emission
    bleach_susceptibility: float = Field(
        ge=0,
        default=6.3e-6,
        description="Bleach rate per second at unit excitation and baseline emission (the "
        "per-photon hazard); 0 disables bleaching. Calibrated to CA1 GCaMP6f.",
    )
    turnover_tau_s: float = Field(
        gt=0,
        default=20000.0,
        description="Effective fluorophore-recovery time constant, s (≈5.5 h, from the "
        "measured replenish rate). Restores the intact pool toward 1, opposing bleaching.",
    )
    excitation_intensity: float = Field(
        ge=0,
        default=1.0,
        description="Excitation level, dimensionless (1 = a typical continuous miniscope "
        "level). Deliberately unitless - absolute irradiance depends on the rig, depth, and "
        "optics. Scales the bleach rate linearly: the brighter-but-faster-fading trade-off.",
    )


class BrainMotion(StepSpec):
    """Rigid x,y translation of the whole tissue frame - the tissue→sensor boundary.

    The built step shifts the brain-frame canvas per frame and crops the sensor
    FOV from its center; it therefore requires a scene whose tissue canvas carries
    a margin ≥ the maximum shift (``Scene.zeros(acq, margin_px=…)``, sized
    automatically by ``simulate()``), so real off-FOV tissue moves into view
    instead of a fabricated fill. Ground truth records the per-frame ``(dy, dx)``
    displacement in **pixels**.

    Three sources of the trajectory, selected by ``model``:

    * ``"physical"`` (default): a 2-D damped harmonic oscillator. The brain is a
      damped mass elastically tethered to the (rigid) skull, driven on the dominant
      ``locomotion_axis`` by an always-on locomotion rhythm at ``locomotion_freq_hz``
      (mice/rats run at ~6-8 Hz) and on both axes by broadband sloshing noise. The
      restoring force bounds the motion physically; ``motion_amplitude_um`` sets the
      typical excursion and ``max_shift_um`` is the hard safety clamp (and the margin
      size). This is the realistic model the teaching notebook uses.
    * ``"walk"``: a bounded random walk (``walk_step_um`` per frame, clamped to the
      ``max_shift_um`` disk). Cheap and rhythm-free; kept for simple tests/fixtures.
    * an explicit ``trajectory_um`` overrides both, regardless of ``model``.

    Axial focus-drift motion is a deferred placeholder.
    """

    domain: ClassVar[str] = "motion"
    kind: Literal["brain_motion"] = "brain_motion"
    model: Literal["physical", "walk"] = Field(
        default="physical",
        description="Trajectory generator: 'physical' (driven damped oscillator) or 'walk' (bounded random walk). An explicit trajectory_um overrides both.",
    )
    trajectory_um: list[tuple[float, float]] | None = Field(
        default=None, description="Explicit per-frame (dy, dx) in µm; overrides model."
    )
    max_shift_um: float = Field(
        gt=0,
        default=15.0,
        description="Hard safety clamp on cumulative shift magnitude, µm (also sizes the tissue margin).",
    )
    # --- physical model ---
    locomotion_freq_hz: float = Field(
        gt=0,
        default=7.0,
        description="Locomotion (stride) drive frequency, Hz; mice/rats run at ~6-8 Hz.",
    )
    motion_amplitude_um: float = Field(
        gt=0,
        default=10.0,
        description="Extreme excursion (99th-percentile displacement radius), µm; most frames move less.",
    )
    locomotion_axis: Literal["y", "x"] = Field(
        default="y",
        description="Dominant motion axis the locomotion rhythm drives (y = height; the cross axis gets noise only).",
    )
    resonance_freq_hz: float = Field(
        gt=0,
        default=6.0,
        description="Natural frequency of the brain-on-skull oscillator, Hz.",
    )
    damping_ratio: float = Field(
        gt=0,
        default=0.5,
        description="Damping ratio ζ of the oscillator (<1 under-damped, sloshy; ≥1 over-damped).",
    )
    locomotion_fraction: float = Field(
        ge=0,
        le=1,
        default=0.25,
        description="Share of motion amplitude carried by the locomotion rhythm vs broadband sloshing noise (noise-dominated by default).",
    )
    # --- walk model ---
    walk_step_um: float = Field(
        ge=0, default=0.3, description="Random-walk step size, µm/frame (model='walk')."
    )

    @model_validator(mode="after")
    def _amplitude_within_clamp(self) -> BrainMotion:
        if self.model == "physical" and self.motion_amplitude_um > self.max_shift_um:
            raise ValueError(
                f"motion_amplitude_um ({self.motion_amplitude_um}) exceeds the max_shift_um "
                f"clamp ({self.max_shift_um}); raise max_shift_um so the calibrated motion is "
                "not crushed by the safety clamp."
            )
        return self


class IlluminationProfile(StepSpec):
    """Static excitation-illumination falloff - the LED lights the FOV unevenly.

    A single excitation LED illuminates the tissue brightest at the center and
    dimmer toward the edges, so peripheral cells fluoresce less to begin with.
    Modeled as a multiplicative radial falloff (``1`` at the bright center, dropping
    to ``falloff`` at the farthest corner, ``exponent`` shaping the rolloff) - fixed
    to the scope, so it does **not** move with the brain. Typically a gentle, broad
    rolloff (vs the sharper emission ``Vignette``). Being on the *excitation* side,
    this falloff also drives photobleaching faster at the bright center: that
    coupling is wired in ``Bleaching`` (which evaluates this field at each cell's
    rest position), the one way it differs from the collection-side vignette.
    """

    domain: ClassVar[str] = "sensor"
    kind: Literal["illumination_profile"] = "illumination_profile"
    falloff: float = Field(
        ge=0,
        le=1,
        default=0.7,
        description="Edge excitation relative to center (1 = uniform).",
    )
    exponent: float = Field(
        gt=0,
        default=2.0,
        description="Radial falloff exponent (gentle/broad by default).",
    )
    center_offset_um: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="(dy, dx) offset of the bright center from FOV center, µm.",
    )


class Vignette(StepSpec):
    """Static radial vignette on the emission / return path (collection light loss).

    The physical return path trims light rays toward the field edges (aperture and
    relay clipping, compounded by poorer off-axis optical performance), so corners
    read dimmer regardless of how brightly the tissue was lit. Same multiplicative
    radial-falloff shape as the ``IlluminationProfile`` but on the *collection* side
    - so it does not drive bleaching - and typically a sharper edge rolloff. Also
    fixed to the scope (does not move with the brain). Off-axis blur is a separate
    concern, deferred to a future optical-aberration step.
    """

    domain: ClassVar[str] = "sensor"
    kind: Literal["vignette"] = "vignette"
    falloff: float = Field(
        ge=0,
        le=1,
        default=0.5,
        description="Corner brightness relative to center (1 = none).",
    )
    exponent: float = Field(gt=0, default=2.0, description="Radial falloff exponent.")
    center_offset_um: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="(dy, dx) offset of the bright center from FOV center, µm.",
    )


class Leakage(StepSpec):
    """Static additive baseline (stray excitation light on the detector).

    One *additive* contributor to the smooth, low-frequency background that minian's
    'glow removal' estimates and subtracts - not its sole target: that removal also
    strips the *multiplicative* illumination falloff and vignette (see
    ``IlluminationProfile`` / ``Vignette``), since all three are smooth and static
    while the cells are sharp and moving."""

    domain: ClassVar[str] = "sensor"
    kind: Literal["leakage"] = "leakage"
    profile: Literal["uniform", "gaussian"] = Field(
        default="gaussian", description="Spatial baseline shape."
    )
    level: float = Field(ge=0, default=0.1, description="Additive baseline level.")
    sigma_um: float | None = Field(
        default=None,
        description="Spatial sigma for the gaussian profile, µm; None defaults to a "
        "quarter of the smaller FOV dimension. Ignored by the uniform profile.",
    )


class Sensor(StepSpec):
    """Photons → e⁻ → Poisson shot + read noise → ×gain → quantize → clip.

    The only step that produces integer-valued counts. The sensor *hardware*
    (QE, read noise, gain, bit depth, pixel pitch) lives on
    ``Acquisition.image_sensor`` and is read from there. The single field below
    is the exposure/flux scale - a scene property, not sensor hardware - which is
    why it stays on the step rather than the image-sensor spec.
    """

    domain: ClassVar[str] = "sensor"
    kind: Literal["sensor"] = "sensor"
    photons_per_unit: float = Field(
        gt=0,
        default=100.0,
        description="Photons per fluorescence intensity unit (exposure/flux scale); sets the "
        "shot-noise regime, and scales every light source in the frame (cells, neuropil, "
        "vasculature, leakage). A scene/illumination property, not sensor hardware (it lumps "
        "excitation power x integration time x collection efficiency), which is why it stays on "
        "the step rather than the image-sensor spec. Raise it (or the sensor gain) for a brighter "
        "recording; raising it also lifts the shot-noise-limited SNR.",
    )


# The v1 catalog is closed and known, so AnyStep is a hand-written static union:
# native pydantic, trivially debuggable, no import-order hazards. It is the
# single source of truth for the step catalog - adding a component means
# defining its StepSpec subclass and adding it here. Pydantic's Discriminator
# handles kind→class dispatch for deserialization, so no separate registry is
# needed; if later tooling wants an explicit map, derive it from this union.
AnyStep = Annotated[
    PlaceNeurons
    | CellActivity
    | CellOptics
    | Composite
    | Neuropil
    | Vasculature
    | Bleaching
    | BrainMotion
    | IlluminationProfile
    | Vignette
    | Leakage
    | Sensor,
    Discriminator("kind"),
]


# ---------------------------------------------------------------------------
# Top-level Spec + cross-field validation
# ---------------------------------------------------------------------------


class Spec(_Base):
    """A complete, reproducible recording specification.

    ``acquisition`` + ``steps`` (the ordered pipe) + ``seed`` + ``output`` fully
    determine a recording. The cross-field validators below catch genuinely
    invalid configs (raise) and flag unusual-but-legal ones (``SpecWarning``).
    """

    acquisition: Acquisition = Field(default_factory=Acquisition)
    seed: int = Field(default=42, description="RNG seed for full reproducibility.")
    steps: list[AnyStep]
    output: Output = Field(default_factory=Output)

    def cache_key(self) -> str:
        """SHA256 (first 16 hex chars) of the canonical JSON form. Stable across runs."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    @field_validator("steps")
    @classmethod
    def _canonicalize_order(cls, steps: list[AnyStep]) -> list[AnyStep]:
        """Store steps in canonical order (:func:`order_steps`), whatever order they
        were given, so every downstream consumer - ``simulate``, ``until=``, the
        snapshot keys, ``cache_key``, sweeps - sees the same sequence and two specs
        that differ only in listing order compare and cache as equal.
        """
        return order_steps(steps)

    @model_validator(mode="after")
    def _validate(self) -> Spec:
        self._check_unique_kinds()  # hard fail; everything below assumes unique kinds
        self._check_step_dependencies()
        by_kind = {s.kind: s for s in self.steps}
        self._check_footprint_vs_fov(by_kind)
        self._check_sampling_vs_kinetics(by_kind)
        self._warn_focal_plane(by_kind)
        self._warn_motion_magnitude(by_kind)
        return self

    # -- hard fails ---------------------------------------------------------

    def _check_unique_kinds(self) -> None:
        """Rule 4: each ``kind`` appears at most once - lets sweeps address a step
        by kind and keeps the snapshot dict (keyed by step name) collision-free."""
        dupes = sorted(
            k for k, n in Counter(s.kind for s in self.steps).items() if n > 1
        )
        if dupes:
            raise ValueError(
                f"Duplicate step kind(s) in spec: {dupes}. Each kind must be unique."
            )

    def _check_step_dependencies(self) -> None:
        """Rule 4b: a step's present ``requires`` kinds must precede it. Steps are
        already canonicalized and :data:`_PIPELINE_ORDER` is a topological extension
        of ``requires``, so this is a drift guard (an assertion that the two stay in
        sync) rather than a reachable error for known kinds."""
        present = {s.kind for s in self.steps}
        seen: set[str] = set()
        for s in self.steps:
            late = [r for r in s.requires if r in present and r not in seen]
            if late:
                raise ValueError(
                    f"Step {s.kind!r} resolves after {late}, which it consumes through "
                    "the shared Scene; the canonical pipeline order is inconsistent with "
                    f"its declared requires={s.requires!r}."
                )
            seen.add(s.kind)

    def _check_footprint_vs_fov(self, by_kind: Mapping[str, StepSpec]) -> None:
        """Rule 2: a soma larger than the entire FOV is a misconfiguration."""
        pc = by_kind.get("place_neurons")
        if not isinstance(pc, PlaceNeurons):
            return
        min_fov = min(self.acquisition.fov_um)
        max_radius = max(p.soma_radius_um for p in pc.resolved_populations)
        if 2 * max_radius > min_fov:
            raise ValueError(
                f"soma_radius_um={max_radius} µm gives a soma diameter larger than "
                f"the FOV ({min_fov:.3g} µm). Reduce the soma or enlarge the FOV."
            )

    def _check_sampling_vs_kinetics(self, by_kind: Mapping[str, StepSpec]) -> None:
        """Rule 3: ``tau_decay_s · fps`` must be ≳ 1, else the decay is unresolvable."""
        act = by_kind.get("cell_activity")
        if not isinstance(act, CellActivity):
            return
        samples_per_decay = act.tau_decay_s * self.acquisition.fps
        if samples_per_decay < 1.0:
            raise ValueError(
                f"tau_decay_s={act.tau_decay_s} s at fps={self.acquisition.fps} gives "
                f"{samples_per_decay:.3g} samples per decay (< 1); the calcium decay is "
                "unresolvable. Raise fps or tau_decay_s."
            )

    # -- advisory warnings --------------------------------------------------

    def _warn_focal_plane(self, by_kind: Mapping[str, StepSpec]) -> None:
        """Rule 6: a numeric focal depth outside the cell depth range is unusual."""
        focal = self.acquisition.focal_depth_in_tissue_um
        pc = by_kind.get("place_neurons")
        if focal == "auto" or not isinstance(pc, PlaceNeurons):
            return
        lo, hi = pc.depth_range_um
        if not (lo <= focal <= hi):
            warnings.warn(
                f"focal_depth_in_tissue_um={focal} µm is outside the cell depth range ({lo}, {hi}) µm.",
                SpecWarning,
                stacklevel=2,
            )

    def _warn_motion_magnitude(self, by_kind: Mapping[str, StepSpec]) -> None:
        """Rule 7: shifts beyond ~5% of the FOV likely indicate a misconfig."""
        mot = by_kind.get("brain_motion")
        if not isinstance(mot, BrainMotion):
            return
        min_fov = min(self.acquisition.fov_um)
        if mot.trajectory_um is not None:
            extent = max(
                (max(abs(dy), abs(dx)) for dy, dx in mot.trajectory_um), default=0.0
            )
        elif mot.model == "physical":
            extent = mot.motion_amplitude_um
        else:
            extent = mot.max_shift_um
        if extent > 0.05 * min_fov:
            warnings.warn(
                f"Motion extent {extent:.3g} µm exceeds 5% of the FOV ({min_fov:.3g} µm); "
                "likely a misconfiguration.",
                SpecWarning,
                stacklevel=2,
            )

"""Typed, serializable specification for the ``minisim`` pipeline.

This module defines the *contract* the simulator consumes: a tree of pydantic
v2 models describing the acquisition (a real, physical interface), an ordered
list of pipeline steps, and output formatting. It is the inverse of the minian
analysis pipeline expressed as data — the same ``Spec`` object a training
notebook walks through, a test parametrizes over, and a cache keys on.

Two layers, by design (see ``proposals/simulation-spec.md`` §2):

* **Layer 1 — what you read off a datasheet.** ``Optics.na``,
  ``Optics.magnification``, ``Tissue.scatter_mfp_emission_um`` and friends. Every knob a
  user touches here is a real, measurable property of a real scope or sample.
* **Layer 2 — what a step consumes.** Pixel size, PSF sigma, attenuation, noise
  variance — *derived* from Layer 1 by small, documented, individually-testable
  helpers.

This file defines the full Layer-1 surface, the unit conversions, the static
``AnyStep`` union (migration Step 2), and the Layer-2 physics helpers —
``Optics.diffraction_sigma_um``/``defocus_sigma_um``, ``Tissue.attenuation``/
``scatter_sigma_um``, the combined ``Acquisition.cell_optics``, and the
``ImageSensor.photons_to_counts`` sensor model (Step 3). The executable steps
that ``build()`` returns land in Step 5; until then ``StepSpec.build()`` is
intentionally unimplemented — these classes are the schema plus its physics,
not the execution engine.

Units convention: **everything physical is in seconds and µm/mm — never frames
or pixels.** ``Acquisition`` owns every conversion to pixels/frames.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

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
    ones (which warn but still run) — e.g. a focal plane outside the cell depth
    range, or steps listed out of the natural physical order.
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
# Physical interface — Acquisition / Optics / Tissue (Layer 1 + unit conversions)
# ---------------------------------------------------------------------------


# Immersion/tissue refractive index used to derive the diffraction depth of field
# from NA (≈ n·λ/NA²); ~1.33 for the watery cortex a miniscope images into.
_DOF_IMMERSION_N = 1.33


class Optics(_Base):
    """Objective optics — the measurable lens properties of a 1-photon scope.

    Layer-2 phenomenological quantities (diffraction sigma, defocus blur) are
    *derived* from these fields; their math arrives in migration Step 3. Pixel
    size is a joint optics×sensor quantity (sensor pitch / magnification) and so
    lives on ``Acquisition``, not here.

    Typical 1-photon miniscope ranges: NA 0.3–0.6, magnification ~5–10×, GCaMP
    emission ~510–540 nm.
    """

    na: float = Field(gt=0, default=0.45, description="Numerical aperture of the GRIN objective.")
    magnification: float = Field(gt=0, default=8.0, description="Optical magnification (sensor side / object side).")
    emission_nm: float = Field(gt=0, default=525.0, description="Fluorophore emission wavelength, nm (GCaMP ≈ 525).")
    depth_of_field_um: float | Literal["auto"] = Field(
        default="auto",
        description="±in-focus half-depth around the focal plane, µm. 'auto' (default) "
        "derives it from NA as ≈ n·λ/NA² (the diffraction depth of field — the physical "
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
    def _check_dof(cls, v: float | str) -> float | str:
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
        aberrations and the finite Airy tails — adequate for showing how NA and
        emission wavelength set the resolution floor. Smaller NA ⇒ larger σ
        (blurrier). At NA 0.45, λ ≈ 525 nm this is σ ≈ 0.24 µm.

        Note — pixel-limited, not diffraction-limited. Across realistic 1-photon
        miniscope NAs (~0.1–0.6) and green emission this σ is only ~0.2–1.1 µm,
        i.e. at or below the *object-space pixel size* (sensor pitch ÷
        magnification, typically 1–2 µm). So lateral resolution is set by the
        pixel sampling, not by diffraction — the diffraction PSF is real but
        rarely the limiting blur (defocus and scatter usually dominate it too).
        A practical consequence: a cell's intrinsic shape can be generated on a
        fine, sub-pixel grid *independent of the sensor*, then resampled to
        whatever pixel size the sensor implies, because the optics never resolve
        anything finer than the pixel grid anyway. (The teaching notebook relies
        on exactly this so that changing magnification/pitch only rescales a cell
        rather than re-rasterizing — and re-randomizing — its shape.)
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
        """Fraction of a cell's emitted light the objective collects — ``∝ NA²``.

        A lens gathers light over a collection cone whose solid angle grows with
        ``NA²`` (small-angle ``Ω ∝ sin²θ = NA²``), so a low-NA miniscope objective
        is *fundamentally* dimmer than a high-NA one — independently of focus or
        depth. This is a flat multiplicative light-loss applied alongside scatter
        :meth:`Tissue.attenuation`; the absolute proportionality constant
        (``1/4n²`` etc.) is absorbed into the ``sensor`` step's
        ``photons_per_unit`` exposure scale, so what matters here is the ``NA²``
        scaling. At NA 0.18 vs 0.45 this is a ~6× brightness difference."""
        return self.na**2

    def focal_curvature_shift_um(self, r_um: float) -> float:
        """Field-curvature focal shift at field radius ``r_um`` from the axis, µm.

        Without a field flattener — which a miniscope has no room for — off-axis
        points focus on a curved (≈spherical) surface, not a plane. A point at
        radius ``r`` from the optical axis comes into best focus *shallower*
        (nearer the objective) than the on-axis focal plane, by the spherical
        sagitta ``R − √(R² − r²) ≈ r²/(2R)``. Returns a **non-negative** shift to
        be *subtracted* from the central focal depth (the in-focus surface bows
        toward the objective at the edges, always, for miniscope/standard optics).
        Zero when :attr:`field_curvature_radius_um` is ``None`` (an ideal flat
        field). Typical radii are 2–3 mm — large vs a soma, so a cell can be
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
    pixel_pitch_um: float = Field(gt=0, default=3.0, description="Physical sensor pixel pitch, µm.")
    quantum_efficiency: float = Field(gt=0, le=1, default=0.7, description="Photon → electron conversion efficiency.")
    read_noise_e: float = Field(ge=0, default=2.0, description="Read noise, electrons RMS.")
    gain_adu_per_e: float = Field(gt=0, default=1.0, description="Camera gain, ADU per electron.")
    bit_depth: int = Field(gt=0, default=8, description="ADC bit depth; counts clipped to [0, 2^bit_depth − 1].")

    def photons_to_counts(self, photons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Forward sensor model: incident photons → digitized ADC counts.

        The only place fluorescence becomes integer counts (spec §6)::

            e⁻     = Poisson(photons · quantum_efficiency)   # shot noise
            e⁻    += Normal(0, read_noise_e)                 # read noise, electrons
            adu    = e⁻ · gain_adu_per_e
            counts = clip(floor(adu), 0, 2**bit_depth − 1)

        Shot noise is Poisson on the *detected* electrons: photon arrival is
        Poisson and detection thins it by QE, which stays Poisson. Read noise is
        additive Gaussian in electrons. Quantization is ``floor`` (an ADC
        truncates), and counts are clipped to the converter's representable
        range. ``photons`` is the per-pixel expected photon count — the
        ``Sensor`` step produces it from scene intensity × its
        ``photons_per_unit`` exposure scale. Returns a float array holding
        integer-valued counts (the float container is set by
        ``Output.store_dtype``).
        """
        photons = np.asarray(photons, dtype=float)
        electrons = rng.poisson(photons * self.quantum_efficiency).astype(float)
        electrons += rng.normal(0.0, self.read_noise_e, size=electrons.shape)
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

    Round-trip scattering. The signal makes two scattering-attenuated passes
    through tissue: excitation light travels *in* (GCaMP ≈ 470 nm) and emission
    travels *out* (≈ 525 nm), so a cell at depth ``z`` is attenuated on both legs
    (see :meth:`attenuation`). Shorter wavelengths scatter more, so the blue
    excitation leg has the shorter MFP.

    Literature anchors (mouse cortex / gray matter). The *ballistic* scattering
    mean free path at blue/green is ≈ 40–50 µm (μ_s ≈ 200 cm⁻¹, g ≈ 0.86–0.89):
    ≈ 47 µm at 473 nm (Al-Juboori et al. 2013, PLoS ONE 8:e67626) and ≈ 38 µm at
    515 nm (Azimipour et al. 2014, Biomed. Opt. Express). That ballistic length
    is what sets the *blur* rate. The light an objective actually *collects*
    decays more slowly than ballistic, because the strong forward scattering is
    largely recollected and widefield excitation penetrates diffusely (transport
    length ≈ 800 µm; Ma et al. 2020, Neurophotonics 7:031208) — so the per-leg
    *effective attenuation* MFPs below are deliberately milder (~90–110 µm) than
    the bare ballistic numbers; their round trip gives an effective ≈ 50 µm.
    """

    scatter_mfp_excitation_um: float = Field(
        gt=0,
        default=90.0,
        description="Effective attenuation MFP for the excitation leg (≈470 nm, in), µm.",
    )
    scatter_mfp_emission_um: float = Field(
        gt=0,
        default=110.0,
        description="Effective attenuation MFP for the emission leg (≈525 nm, out), µm.",
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
        (90 / 110 µm) this is ≈ 49.5 µm — i.e. ~2× steeper depth-dimming than a
        single 100 µm pass, reflecting that the light is attenuated both going in
        and coming out.
        """
        return 1.0 / (1.0 / self.scatter_mfp_excitation_um + 1.0 / self.scatter_mfp_emission_um)

    def attenuation(self, z_um: float) -> float:
        """Fraction of light surviving the round-trip scatter from depth ``z_um`` — in (0, 1].

        Beer–Lambert decay applied on *both* legs the signal travels: excitation
        in (≈470 nm) then emission out (≈525 nm). The product collapses to a
        single exponential over the effective MFP, ``exp(−z / scatter_mfp_um)``
        (see :attr:`scatter_mfp_um`). Monotonically decreasing in depth and equal
        to 1 at the surface (``z = 0``). Genuinely *removes* light — unlike
        defocus — so a deep cell is irreversibly dimmer, the irreducible limit
        the module teaches.
        """
        return math.exp(-z_um / self.scatter_mfp_um)

    def scatter_sigma_um(self, z_um: float) -> float:
        """Scatter-induced footprint broadening (Gaussian σ), µm, at depth ``z_um``.

        Linear phenomenological model ``σ = scatter_blur_per_um · z``: deeper
        cells scatter more and so appear both larger and dimmer (see
        :meth:`attenuation`). Monotonically increasing in depth and zero at the
        surface. Unlike defocus this is not intensity-conserving — it co-occurs
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
    derived from the sensor's pixel count — any two of {FOV, pixel size, pixel
    count} fix the third.
    """

    optics: Optics = Field(default_factory=Optics)
    image_sensor: ImageSensor = Field(default_factory=ImageSensor)
    tissue: Tissue = Field(default_factory=Tissue)
    fps: float = Field(gt=0, default=20.0, description="Frame rate, frames per second.")
    duration_s: float = Field(gt=0, default=150.0, description="Recording duration, seconds.")
    focal_depth_in_tissue_um: float | Literal["auto"] = Field(
        default="auto",
        description="Depth of the focal plane below the tissue surface, µm (0 = surface), "
        "in the same coordinate as each cell's depth z. Cells above or below it defocus; "
        "'auto' resolves to the median realized cell depth at the optics step.",
    )
    front_working_distance_um: float | None = Field(
        default=None,
        description="Front working distance (lens front → focal point), µm — Miniscope V4 ≈ "
        "700. Informational only: it does NOT affect the simulation (the optics math uses "
        "focal_depth_in_tissue_um), but it's a physically relevant number for surgery/implant "
        "planning, so it's recorded here.",
    )

    @field_validator("focal_depth_in_tissue_um")
    @classmethod
    def _check_focal_depth(cls, v: float | str) -> float | str:
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
        """Field of view (height, width) in µm — derived from pixels × pixel size."""
        return (
            self.image_sensor.n_px_height * self.pixel_size_um,
            self.image_sensor.n_px_width * self.pixel_size_um,
        )

    def um_to_px(self, um: float) -> float:
        """Convert a physical distance (µm) to pixels."""
        return um / self.pixel_size_um

    def s_to_frame(self, s: float) -> float:
        """Convert a duration (seconds) to a (fractional) frame count."""
        return s * self.fps

    def cell_optics(self, z_um: float, focal_um: float) -> tuple[float, float]:
        """Combined per-cell optical degradation: ``(sigma_px, brightness)``.

        Folds the three Layer-2 effects into the two numbers the optics step
        applies to a footprint — a blur width (pixels) and a brightness scale.
        Blurs add in quadrature; brightness factors multiply (spec §2)::

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
        plane — the invariant the conservation test asserts. ``focal_um`` is the
        resolved (numeric) focal depth; ``diffraction_sigma_um > 0`` always, so
        ``σ_tot`` is never zero.

        Two distinct quantities come out, consumed separately by the optics step
        (5b): ``sigma_px`` is the PSF width the footprint is *convolved* with,
        and the footprint's brightness then scales by ``attenuation(z)`` **alone**
        — the convolution itself produces the defocus peak drop, so applying
        ``brightness`` to the footprint too would double-count it. The returned
        ``brightness`` is instead the *point-source peak* scalar: how far a
        cell's peak signal sits above the noise, i.e. the per-cell effective
        brightness used (with the illumination field and sensor floor) to decide
        detectability at ``finalize()``.
        """
        sigma_0 = math.hypot(self.optics.diffraction_sigma_um, self.tissue.scatter_sigma_um(z_um))
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
    """Final-array formatting — formatting only, never rescaling (honest radiometry)."""

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
# Step framework — StepSpec base + registry + the static AnyStep union
# ---------------------------------------------------------------------------


# Natural physical order of the pipeline. A step's domain is a class-level
# attribute (not a serialized field); the Spec validator warns if the list
# departs from this order.
_DOMAIN_RANK: dict[str, int] = {"cell": 0, "tissue": 1, "motion": 2, "sensor": 3}


class StepSpec(_Base):
    """Base class for a single pipeline step's configuration.

    A concrete step spec carries its physical parameters and a literal ``kind``
    discriminator, and declares its ``domain`` (a class attribute used for
    ordering checks). ``build()`` turns the spec into the executable step that
    mutates a ``Scene``; those bodies arrive in migration Step 5.
    """

    domain: ClassVar[Literal["cell", "tissue", "motion", "sensor"]]
    kind: str

    def build(self, acq: Acquisition, rng) -> Step:
        """Return the executable step (a callable that mutates a Scene).

        Unimplemented until migration Step 5 — at this stage these classes are
        the typed schema/contract, not the execution engine.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.build() is implemented in migration Step 5; "
            "Step 2 defines the spec surface only."
        )


# ---------------------------------------------------------------------------
# Step catalog (cell → tissue → motion → sensor). Fields define the v1 surface;
# `build()` bodies and the no-op placeholder steps (vasculature, fancier motion)
# arrive in Step 5.
# ---------------------------------------------------------------------------


class PlaceNeurons(StepSpec):
    """Place generic neurons in a 3-D µm volume, soma-only or with dendrites.

    'Place' is the verb — this *positions neurons in space* (anchored at the cell
    body); it is unrelated to hippocampal *place cells*. v1 models one generic
    excitable cell type (an irregular soma blob) with two GCaMP targeting variants
    via ``morphology``: ``"soma"`` (soma-targeted, body only) or ``"cytosolic"``
    (standard GCaMP, the soma plus a few tapering proximal dendrites). There is no
    further cell-type distinction and no spatial/behavioral tuning. Footprints are
    2-D masks carrying a scalar depth ``z``; out-of-focus neurons that become
    background emerge for free downstream from ``z`` + ``optics``.
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["place_neurons"] = "place_neurons"
    density_per_mm3: float = Field(
        gt=0,
        default=25000.0,
        description="Cell volumetric density (cells/mm³); count = density × FOV area "
        "× depth thickness, the thickness floored at one soma diameter so a thin or "
        "planar layer still yields cells.",
    )
    soma_radius_um: float = Field(gt=0, default=7.0, description="Soma radius, µm (typical cortical neuron ≈ 5–10).")
    irregularity: float = Field(
        ge=0,
        le=1,
        default=0.3,
        description="0 = smooth disk; higher = lumpier soma (low-pass-noise threshold).",
    )
    morphology: Literal["soma", "cytosolic"] = Field(
        default="soma",
        description="GCaMP targeting variant: 'soma' = soma-targeted (lumpy disk "
        "only); 'cytosolic' = standard GCaMP (soma + tapering proximal dendrites).",
    )
    n_dendrites: int = Field(
        ge=0,
        default=4,
        description="Proximal dendrites grown per cell when morphology='cytosolic' "
        "(ignored for 'soma').",
    )
    dendrite_length_um: float = Field(
        gt=0,
        default=45.0,
        description="Proximal-dendrite length, µm (cytosolic only).",
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
        ge=0, default=0.0, description="3-D center-to-center minimum (Poisson-disk if > 0)."
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

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.cell import PlaceNeuronsStep

        return PlaceNeuronsStep(self, acq, rng)


class CellActivity(StepSpec):
    """Calcium activity: 2-state Markov gate → Poisson spikes → double-exp kernel.

    Modeled on the CaLab web simulator: spikes are generated on a high-resolution
    grid (``spike_sim_hz``, ~300 Hz), convolved with the double-exponential kernel
    ``k(t) = exp(-t/τ_d) − exp(-t/τ_r)`` at that rate, then bin-averaged down to the
    camera frame rate (exposure integration). One spike per fine bin respects the
    ~3 ms refractory period. The ground-truth ``S`` is the per-frame spike *count*
    (the fine train is binned away — nothing recovers spikes faster than the frame
    rate). Indicator saturation and per-cell τ jitter are deferred to v1.1.

    Amplitude is biology and lives here as a single per-cell gain: ``brightness_cv``
    is the cell-to-cell spread of an overall expression/response gain that scales
    each cell's *whole* trace (baseline and transients together). The emitted trace
    is the **clean ground truth** ``C``; measurement noise is deliberately *not*
    added here. Photon shot noise and read noise enter at the ``sensor``, background
    fluctuations at ``neuropil`` — so any SNR is an emergent property of the physical
    chain, computable downstream, never an input.
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["cell_activity"] = "cell_activity"
    spike_sim_hz: float = Field(gt=0, default=300.0, description="High-res spike-simulation rate, Hz (~300 = a ~3 ms refractory); binned to the frame rate.")
    # Defaults = CaLab's "moderate" SPIKE_ACTIVITY level; see spike_activity_params.
    p_quiescent_to_active: float = Field(gt=0, default=0.005, description="Per-frame quiescent→active transition prob.")
    p_active_to_quiescent: float = Field(gt=0, default=0.3, description="Per-frame active→quiescent transition prob.")
    active_rate_hz: float = Field(gt=0, default=150.0, description="Instantaneous firing rate while active, Hz (the in-burst rate).")
    quiescent_rate_hz: float = Field(ge=0, default=0.6, description="Instantaneous firing rate while quiescent, Hz (the intrinsic background).")
    tau_rise_s: float = Field(gt=0, default=0.05, description="Calcium rise time constant, s.")
    tau_decay_s: float = Field(gt=0, default=0.5, description="Calcium decay time constant, s.")
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

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.cell import CellActivityStep

        return CellActivityStep(self, acq, rng)


class CellOptics(StepSpec):
    """Per-cell diffraction + defocus(|z − z_f|) + scatter(z) blur & attenuation.

    No tunable fields: blur and attenuation are fully determined by each cell's
    ``z`` plus the physical ``Optics``/``Tissue`` constants on ``Acquisition``.
    Writes the observed (degraded) footprint alongside the planted (sharp) one,
    sets the geometric ``in_focus`` flag, and stores the per-cell
    ``optical_brightness`` peak scalar. ``detectable`` is *not* set here — it is
    a whole-pipeline flag (optics × illumination vs the sensor noise floor)
    assembled in ``finalize()`` (Step 6).
    """

    domain: ClassVar[str] = "cell"
    kind: Literal["optics"] = "optics"

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.cell import CellOpticsStep

        return CellOpticsStep(self, acq, rng)


class Render(StepSpec):
    """Composite ``Σ_i degraded_footprint_i × trace_i`` into the movie.

    The built step's snapshot name is ``"cells_only"``. The planted (sharp)
    ``A``/``C`` remain the ideal CNMF target in ground truth.
    """

    domain: ClassVar[str] = "tissue"
    kind: Literal["render"] = "render"

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.tissue import RenderStep

        return RenderStep(self, acq, rng)


class Neuropil(StepSpec):
    """Additive diffuse background: smooth spatial field × slow OU temporal."""

    domain: ClassVar[str] = "tissue"
    kind: Literal["neuropil"] = "neuropil"
    spatial_sigma_um: float = Field(gt=0, default=40.0, description="Spatial smoothness of the mesh, µm.")
    temporal_tau_s: float = Field(gt=0, default=10.0, description="OU temporal correlation time, s (slow).")
    amplitude: float = Field(gt=0, default=0.3, description="Background amplitude relative to cell signal.")
    n_components: int = Field(ge=1, default=3, description="Number of independent diffuse components.")

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.tissue import NeuropilStep

        return NeuropilStep(self, acq, rng)


class Vasculature(StepSpec):
    """Dark absorbing mask × (slow dilation + cardiac). Placeholder no-op for v1."""

    domain: ClassVar[str] = "tissue"
    kind: Literal["vasculature"] = "vasculature"
    enabled: bool = Field(default=False, description="Placeholder; multiplicative absorption lands in v1.1.")

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.tissue import VasculatureStep

        return VasculatureStep(self, acq, rng)


class Bleaching(StepSpec):
    """Global temporal decay of fluorophores (not the additive sensor leakage)."""

    domain: ClassVar[str] = "tissue"
    kind: Literal["bleaching"] = "bleaching"
    model: Literal["mono_exp", "bi_exp"] = Field(
        default="mono_exp",
        description="Decay curve family. Only 'mono_exp' is implemented in v1; 'bi_exp' "
        "is rejected at construction (final_fraction alone underdetermines it).",
    )
    final_fraction: float = Field(
        gt=0, le=1, default=0.65, description="Brightness at the last frame relative to the first."
    )

    @field_validator("model")
    @classmethod
    def _model_implemented(cls, v: str) -> str:
        # Fail fast at construction rather than mid-run: a single final_fraction
        # does not pin a two-component curve, so faking one would be dishonest.
        if v == "bi_exp":
            raise ValueError(
                "model='bi_exp' is not implemented in v1 (mono_exp only); "
                "a single final_fraction does not determine a bi-exponential curve."
            )
        return v

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.tissue import BleachingStep

        return BleachingStep(self, acq, rng)


class BrainMotion(StepSpec):
    """Rigid x,y translation of the whole tissue frame — the tissue→sensor boundary.

    The built step shifts the brain-frame canvas per frame and crops the sensor
    FOV from its center; it therefore requires a scene whose tissue canvas carries
    a margin ≥ the maximum shift (``Scene.zeros(acq, margin_px=…)``, sized
    automatically by ``simulate()``), so real off-FOV tissue moves into view
    instead of a fabricated fill. Ground truth records the per-frame ``(dy, dx)``
    displacement in **pixels**. OU/jump and axial focus-drift motion are deferred
    placeholders.
    """

    domain: ClassVar[str] = "motion"
    kind: Literal["brain_motion"] = "brain_motion"
    trajectory_um: list[tuple[float, float]] | None = Field(
        default=None, description="Explicit per-frame (dy, dx) in µm; else a bounded random walk."
    )
    walk_step_um: float = Field(ge=0, default=0.3, description="Random-walk step size, µm/frame.")
    max_shift_um: float = Field(gt=0, default=5.0, description="Bound on cumulative shift magnitude, µm.")

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.motion import BrainMotionStep

        return BrainMotionStep(self, acq, rng)


class Vignette(StepSpec):
    """Static radial illumination falloff (lumped excitation × collection)."""

    domain: ClassVar[str] = "sensor"
    kind: Literal["vignette"] = "vignette"
    falloff: float = Field(
        ge=0, le=1, default=0.6, description="Corner brightness relative to center (1 = none)."
    )
    exponent: float = Field(gt=0, default=2.0, description="Radial falloff exponent.")
    center_offset_um: tuple[float, float] = Field(
        default=(0.0, 0.0), description="(dy, dx) offset of the bright center from FOV center, µm."
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.sensor import VignetteStep

        return VignetteStep(self, acq, rng)


class Leakage(StepSpec):
    """Static additive baseline — what minian's 'glow removal' subtracts."""

    domain: ClassVar[str] = "sensor"
    kind: Literal["leakage"] = "leakage"
    profile: Literal["uniform", "gaussian"] = Field(default="gaussian", description="Spatial baseline shape.")
    level: float = Field(ge=0, default=0.1, description="Additive baseline level.")
    sigma_um: float | None = Field(
        default=None,
        description="Spatial sigma for the gaussian profile, µm; None defaults to a "
        "quarter of the smaller FOV dimension. Ignored by the uniform profile.",
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.sensor import LeakageStep

        return LeakageStep(self, acq, rng)


class Sensor(StepSpec):
    """Photons → e⁻ → Poisson shot + read noise → ×gain → quantize → clip.

    The only step that produces integer-valued counts. The sensor *hardware*
    (QE, read noise, gain, bit depth, pixel pitch) lives on
    ``Acquisition.image_sensor`` and is read from there. The single field below
    is the exposure/flux scale — a scene property, not sensor hardware — which is
    why it stays on the step rather than the image-sensor spec.
    """

    domain: ClassVar[str] = "sensor"
    kind: Literal["sensor"] = "sensor"
    photons_per_unit: float = Field(
        gt=0,
        default=100.0,
        description="Photons per fluorescence intensity unit (exposure/flux scale); sets the "
        "shot-noise regime. A scene/illumination property, not sensor hardware.",
    )

    def build(self, acq: Acquisition, rng) -> Step:
        from minisim.steps.sensor import SensorStep

        return SensorStep(self, acq, rng)


# The v1 catalog is closed and known, so AnyStep is a hand-written static union:
# native pydantic, trivially debuggable, no import-order hazards. It is the
# single source of truth for the step catalog — adding a component means
# defining its StepSpec subclass and adding it here. Pydantic's Discriminator
# handles kind→class dispatch for deserialization, so no separate registry is
# needed; if later tooling wants an explicit map, derive it from this union.
AnyStep = Annotated[
    PlaceNeurons
    | CellActivity
    | CellOptics
    | Render
    | Neuropil
    | Vasculature
    | Bleaching
    | BrainMotion
    | Vignette
    | Leakage
    | Sensor,
    Discriminator("kind"),
]


# ---------------------------------------------------------------------------
# Top-level Spec + cross-field validation (spec §11)
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

    @model_validator(mode="after")
    def _validate(self) -> Spec:
        self._check_unique_kinds()  # hard fail; everything below assumes unique kinds
        by_kind = {s.kind: s for s in self.steps}
        self._warn_domain_order()
        self._check_footprint_vs_fov(by_kind)
        self._check_sampling_vs_kinetics(by_kind)
        self._warn_focal_plane(by_kind)
        self._warn_motion_magnitude(by_kind)
        return self

    # -- hard fails ---------------------------------------------------------

    def _check_unique_kinds(self) -> None:
        """Rule 4: each ``kind`` appears at most once — lets sweeps address a step
        by kind and keeps the snapshot dict (keyed by step name) collision-free."""
        dupes = sorted(k for k, n in Counter(s.kind for s in self.steps).items() if n > 1)
        if dupes:
            raise ValueError(f"Duplicate step kind(s) in spec: {dupes}. Each kind must be unique.")

    def _check_footprint_vs_fov(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 2: a soma larger than the entire FOV is a misconfiguration."""
        pc = by_kind.get("place_neurons")
        if pc is None:
            return
        min_fov = min(self.acquisition.fov_um)
        if 2 * pc.soma_radius_um > min_fov:
            raise ValueError(
                f"soma_radius_um={pc.soma_radius_um} µm gives a soma diameter larger than "
                f"the FOV ({min_fov:.3g} µm). Reduce the soma or enlarge the FOV."
            )

    def _check_sampling_vs_kinetics(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 3: ``tau_decay_s · fps`` must be ≳ 1, else the decay is unresolvable."""
        act = by_kind.get("cell_activity")
        if act is None:
            return
        samples_per_decay = act.tau_decay_s * self.acquisition.fps
        if samples_per_decay < 1.0:
            raise ValueError(
                f"tau_decay_s={act.tau_decay_s} s at fps={self.acquisition.fps} gives "
                f"{samples_per_decay:.3g} samples per decay (< 1); the calcium decay is "
                "unresolvable. Raise fps or tau_decay_s."
            )

    # -- advisory warnings --------------------------------------------------

    def _warn_domain_order(self) -> None:
        """Rule 5: steps out of cell→tissue→motion→sensor order are legal but unusual."""
        ranks = [_DOMAIN_RANK[type(s).domain] for s in self.steps]
        if any(b < a for a, b in zip(ranks, ranks[1:])):
            order = " → ".join(f"{s.kind}({type(s).domain})" for s in self.steps)
            warnings.warn(
                f"Steps depart from the natural cell→tissue→motion→sensor order: {order}.",
                SpecWarning,
                stacklevel=2,
            )

    def _warn_focal_plane(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 6: a numeric focal depth outside the cell depth range is unusual."""
        focal = self.acquisition.focal_depth_in_tissue_um
        pc = by_kind.get("place_neurons")
        if focal == "auto" or pc is None:
            return
        lo, hi = pc.depth_range_um
        if not (lo <= focal <= hi):
            warnings.warn(
                f"focal_depth_in_tissue_um={focal} µm is outside the cell depth range ({lo}, {hi}) µm.",
                SpecWarning,
                stacklevel=2,
            )

    def _warn_motion_magnitude(self, by_kind: dict[str, StepSpec]) -> None:
        """Rule 7: shifts beyond ~5% of the FOV likely indicate a misconfig."""
        mot = by_kind.get("brain_motion")
        if mot is None:
            return
        min_fov = min(self.acquisition.fov_um)
        if mot.trajectory_um is not None:
            extent = max((max(abs(dy), abs(dx)) for dy, dx in mot.trajectory_um), default=0.0)
        else:
            extent = mot.max_shift_um
        if extent > 0.05 * min_fov:
            warnings.warn(
                f"Motion extent {extent:.3g} µm exceeds 5% of the FOV ({min_fov:.3g} µm); "
                "likely a misconfiguration.",
                SpecWarning,
                stacklevel=2,
            )

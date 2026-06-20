"""Named, realistic starting points: standard scopes and brain regions.

A *scope* and a *region* are the two orthogonal halves of a recording's physical
setup, and this module names the common ones so a test or a notebook can grab a
real, validated starting point in one line instead of hand-tuning a dozen
fields:

* a :class:`Scope` is the measurable hardware - objective optics + image sensor -
  everything that does not depend on what tissue you point it at
  (:func:`miniscope_v4`, :func:`generic_1p`);
* a :class:`Region` is the biology you point it at - the cell population (depth,
  density, morphology), the tissue scatter, the diffuse neuropil haze, and the
  region's characteristic vessel confound (:func:`ca1`, :func:`cortex_l23`).

The two compose: :func:`build_spec` assembles any scope × any region into a
validated :class:`~minisim.Spec`, so you can swap the scope or the region
independently and override the rest with :func:`~minisim.sweep`. This module is
the source of truth for the V4 optics/sensor *values* and the standard-region
anatomy: the ``build_recording`` studio reads those numbers from here, and a
parity test (``test_studio_presets_match_library_presets``) fails if the studio
and the library ever drift apart.

The Miniscope V4 optics/sensor numbers are confirmed by D. Aharoni; the
region anatomy (CA1 pyramidal band, neocortex L2/3) follows the same values the
anatomy notebook teaches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from minisim.spec import (
    Acquisition,
    AnyStep,
    CellActivity,
    CellOptics,
    Composite,
    IlluminationProfile,
    ImageSensor,
    Leakage,
    NeuronPopulation,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Sensor,
    Spec,
    Tissue,
    Vasculature,
    VesselLayer,
    Vignette,
    _Base,
)

# A focal plane is either a concrete depth into tissue (µm) or "auto" (resolved to
# the median realized cell depth at the optics step).
FocalDepth = float | Literal["auto"]


class Scope(_Base):
    """A miniscope's measurable hardware: objective optics + image sensor.

    The reusable physical front-end - everything independent of the tissue you
    image. Besides the objective ``optics`` and the bare ``image_sensor``, a scope
    carries its own static field signature: the excitation-side ``illumination``
    falloff, the collection-side ``vignette``, and the stray-light ``leakage``
    glow - all fixed to the instrument, not the tissue (``None`` to omit any).
    ``photons_per_unit`` is the rig's typical exposure/flux scale (excitation
    power x integration time), the brightness :func:`build_spec` gives the sensor
    by default. ``focal_depth_in_tissue_um`` and ``front_working_distance_um`` are
    the two acquisition-level fields that travel with the scope rather than the
    region. Compose with a :class:`Region` via :func:`build_spec`.
    """

    optics: Optics = Field(
        description="Objective optics (NA, magnification, emission)."
    )
    image_sensor: ImageSensor = Field(
        description="The bare image sensor (pixel count/pitch, QE, noise, bit depth)."
    )
    illumination: IlluminationProfile | None = Field(
        default=None,
        description="Excitation-side illumination falloff (center-bright LED), or None.",
    )
    vignette: Vignette | None = Field(
        default=None, description="Collection-side emission vignette, or None."
    )
    leakage: Leakage | None = Field(
        default=None, description="Additive stray-light leakage glow, or None."
    )
    photons_per_unit: float = Field(
        gt=0,
        default=100.0,
        description="The rig's typical exposure/flux scale; the default brightness "
        "build_spec gives the sensor.",
    )
    focal_depth_in_tissue_um: FocalDepth = Field(
        default="auto",
        description="Focal-plane depth into tissue, µm, or 'auto' (median cell depth).",
    )
    front_working_distance_um: float | None = Field(
        default=None,
        description="Front working distance (lens front → focal point), µm; informational.",
    )

    @property
    def pixel_size_um(self) -> float:
        """Object-space size of one pixel, µm (sensor pitch / magnification)."""
        return self.image_sensor.pixel_pitch_um / self.optics.magnification

    @property
    def fov_um(self) -> tuple[float, float]:
        """Field of view ``(height, width)`` in µm at this scope's settings."""
        px = self.pixel_size_um
        return (self.image_sensor.n_px_height * px, self.image_sensor.n_px_width * px)


class Region(_Base):
    """A standard imaging target: cell population + tissue scatter + vessels.

    The biology half of a recording - what a :class:`Scope` is pointed at.
    ``population`` carries the cell distribution (depth range, density,
    morphology); ``tissue`` the depth-dependent scatter; ``neuropil`` the diffuse
    background haze from the surrounding dendritic/axonal felt, or ``None`` for a
    clean background; ``vasculature`` the region's characteristic dark-vessel
    confound, or ``None`` for no vessels. Compose with a scope via
    :func:`build_spec`.
    """

    population: NeuronPopulation = Field(
        description="The cell distribution to place (depth, density, morphology)."
    )
    tissue: Tissue = Field(
        description="Depth-dependent light scatter of the imaged tissue."
    )
    neuropil: Neuropil | None = Field(
        default=None,
        description="Diffuse background haze from the surrounding felt, or None.",
    )
    vasculature: Vasculature | None = Field(
        default=None, description="The region's dark-vessel confound, or None."
    )


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

# UCLA Miniscope V4 optics/sensor (confirmed by D. Aharoni). The magnification is
# set so the 608 px × 4.8 µm sensor yields the V4's ~1.0 mm field of view
# (FOV = n_px · pitch / mag); an NA 0.3 GRIN objective; 525 nm GCaMP emission; a
# ~2500 µm Petzval field-curvature radius (a miniscope has no field flattener);
# 700 µm front working distance (informational, for implant planning).
_V4_NA = 0.3
_V4_MAGNIFICATION = 2.9  # -> FOV = 608 · 4.8 / 2.9 ≈ 1.0 mm
_V4_EMISSION_NM = 525.0
_V4_FIELD_CURVATURE_UM = 2500.0
_V4_N_PX = 608
_V4_PIXEL_PITCH_UM = 4.8
_V4_FWD_UM = 700.0
_V4_BIT_DEPTH = 8  # V4 digitizes to 8-bit raw counts (confirmed); the rest of the
# sensor noise model (QE, read noise, gain) stays at the library defaults for now.
# The V4's characteristic static field signature (excitation glow, emission vignette,
# stray-light leakage) and its bright deep-tissue exposure - the "V4 look".
_V4_PHOTONS_PER_UNIT = 600.0  # bright enough that a deep field clears the noise floor
_V4_ILLUMINATION_FALLOFF = 0.7  # gentle center-bright excitation rolloff
_V4_VIGNETTE_FALLOFF = 0.6  # moderate emission vignette (corner ~60% of center)
_V4_LEAKAGE_LEVEL = 0.08  # gentle center glow; higher buries cells under the bloom


def miniscope_v4() -> Scope:
    """UCLA Miniscope V4: NA 0.3 GRIN, 608×608 px at 4.8 µm pitch, ~1.0 mm FOV.

    Confirmed V4 numbers (D. Aharoni): magnification 2.9 so the sensor sees a
    ~1.0 mm field of view, 525 nm GCaMP emission, a ~2500 µm field-curvature
    radius, an 8-bit sensor ADC, and a 700 µm front working distance. The scope
    also carries the V4's characteristic static field signature - a gentle
    center-bright excitation glow, a moderate emission vignette, and a soft
    stray-light leakage - plus a bright deep-tissue exposure, so a
    :func:`build_spec` recording reproduces the V4 look without hand-adding it.
    The focal plane defaults to ``"auto"`` (tracks the placed layer), as the
    anatomy notebook does. The remaining sensor-noise fields (QE, read noise,
    gain) are left at the library defaults until measured V4 values land.
    """
    return Scope(
        optics=Optics(
            na=_V4_NA,
            magnification=_V4_MAGNIFICATION,
            emission_nm=_V4_EMISSION_NM,
            field_curvature_radius_um=_V4_FIELD_CURVATURE_UM,
        ),
        image_sensor=ImageSensor(
            n_px_height=_V4_N_PX,
            n_px_width=_V4_N_PX,
            pixel_pitch_um=_V4_PIXEL_PITCH_UM,
            bit_depth=_V4_BIT_DEPTH,
        ),
        illumination=IlluminationProfile(falloff=_V4_ILLUMINATION_FALLOFF),
        vignette=Vignette(falloff=_V4_VIGNETTE_FALLOFF),
        leakage=Leakage(profile="gaussian", level=_V4_LEAKAGE_LEVEL),
        photons_per_unit=_V4_PHOTONS_PER_UNIT,
        focal_depth_in_tissue_um="auto",
        front_working_distance_um=_V4_FWD_UM,
    )


def generic_1p() -> Scope:
    """A neutral generic 1-photon scope - the library default optics and sensor.

    NA 0.45, magnification 8×, 256×256 px at 3.0 µm pitch (a small ~96 µm FOV).
    A convenient, fast neutral baseline when the specific instrument does not
    matter; use :func:`miniscope_v4` for a realistic V4 setup.
    """
    return Scope(
        optics=Optics(), image_sensor=ImageSensor(), focal_depth_in_tissue_um="auto"
    )


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def ca1() -> Region:
    """Hippocampal CA1 imaged through an implanted GRIN lens.

    CA1 reads as a thin pyramidal band: cytosolic GCaMP somata (radius ~5 µm) at
    ~45000 cells/mm³ over a ~140-160 µm slab (the anatomy notebook's CA1 preset).
    Neuropil haze is *moderate*: the imaged band is the densely-packed pyramidal
    soma layer, with most of the dendritic/axonal felt in the adjacent strata
    (radiatum/oriens) outside the thin imaged slab. Vasculature is on but *less
    pronounced* than cortex - thinner, lower-contrast vessels just above the
    pyramidal band (D. Aharoni).
    """
    return Region(
        population=NeuronPopulation(
            density_per_mm3=45000.0,
            soma_radius_um=5.0,
            morphology="cytosolic",
            depth_range_um=(140.0, 160.0),
        ),
        tissue=Tissue(),
        neuropil=Neuropil(amplitude=0.4),
        vasculature=Vasculature(
            enabled=True,
            layers=[
                VesselLayer(
                    depth_um=120.0,  # just above the 140-160 µm pyramidal band
                    n_roots=3,
                    root_radius_um=14.0,  # thinner than cortex
                    opacity=0.65,  # less pronounced
                )
            ],
        ),
    )


def cortex_l23() -> Region:
    """Neocortex layer 2/3 (standard cytosolic GCaMP).

    L2/3 excitatory cells: cytosolic GCaMP somata (radius ~6 µm), sparsely
    labeled (~8000 cells/mm³) and spread through a deeper, thicker 100-200 µm
    band than CA1, so depth-dependent scatter and defocus matter more. Neuropil
    haze is *prominent*: the imaged band is dense with dendrites and axons woven
    among the sparse somata, so the diffuse background reads strongly relative to
    the cells. Vasculature is *thick and on top of the cells* (D. Aharoni): a
    shallow layer of large-caliber, high-contrast vessels above the imaged band.
    """
    return Region(
        population=NeuronPopulation(
            density_per_mm3=8000.0,
            soma_radius_um=6.0,
            morphology="cytosolic",
            depth_range_um=(100.0, 200.0),
        ),
        tissue=Tissue(),
        neuropil=Neuropil(amplitude=0.6),
        vasculature=Vasculature(
            enabled=True,
            layers=[
                VesselLayer(
                    depth_um=80.0,  # sits above the 100-200 µm cell band
                    n_roots=2,
                    root_radius_um=25.0,  # thick trunks
                    opacity=0.9,  # pronounced, high-contrast
                    branch_prob=0.15,
                    tortuosity_deg=5.0,
                )
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def build_spec(
    scope: Scope,
    region: Region,
    *,
    duration_s: float = 150.0,
    fps: float = 20.0,
    seed: int = 0,
    populations: Sequence[NeuronPopulation] | None = None,
    activity: CellActivity | None = None,
    sensor: Sensor | None = None,
    extra_steps: Sequence[AnyStep] = (),
    include_neuropil: bool = True,
    include_vasculature: bool = True,
    include_scope_fields: bool = True,
    save_intermediates: bool = False,
) -> Spec:
    """Assemble a validated :class:`~minisim.Spec` from a scope × a region.

    Builds the :class:`~minisim.Acquisition` from the scope's optics/sensor and
    the region's tissue, then the minimal forward chain
    (``place_neurons → cell_activity → optics → composite → sensor``) plus the
    region's neuropil and vasculature and the scope's static fields
    (illumination, vignette, leakage) when present. The result is a real frozen
    ``Spec``: drop it into :func:`~minisim.simulate`, or pass it to
    :func:`~minisim.sweep` as the base for a parameter grid.

    Parameters
    ----------
    scope, region
        The two halves to compose; see :func:`miniscope_v4` / :func:`ca1` etc.
    duration_s, fps, seed
        Sampling and the master RNG seed.
    populations
        Override the region's cell distribution - e.g. a hand-placed pair of
        overlapping cells (a list of :class:`~minisim.NeuronPopulation` with
        explicit ``positions_um``). ``None`` uses the region's own population.
    activity
        The :class:`~minisim.CellActivity` model; ``None`` uses its defaults.
    sensor
        The :class:`~minisim.Sensor` exposure step; ``None`` uses the scope's own
        ``photons_per_unit`` exposure (V4 ≈ 600 for a bright deep-tissue field,
        the generic scope 100). Pass an explicit ``Sensor(...)`` to override.
    extra_steps
        Steps the scope/region do not already supply - typically ``BrainMotion``
        or ``Bleaching`` (the region's neuropil/vasculature and the scope's
        illumination/vignette/leakage come in automatically; see the
        ``include_*`` toggles). The ``Spec`` re-sorts into canonical pipeline
        order, so order here is free, and a duplicate ``kind`` raises.
    include_neuropil
        When ``False``, drop the region's neuropil haze (a clean, background-free
        movie). Ignored when the region has no neuropil.
    include_vasculature
        When ``False``, drop the region's vessel layer (a clean, vessel-free
        movie). Ignored when the region has no vasculature.
    include_scope_fields
        When ``False``, drop the scope's static fields (illumination, vignette,
        leakage) for a flat-field, glow-free movie. Ignored for a scope that sets
        none of them.
    save_intermediates
        Persist per-step snapshots (see :class:`~minisim.Output`).

    Returns
    -------
    Spec
        Validated end to end - an impossible combination (a soma larger than the
        FOV, say) raises here, at build time.
    """
    acquisition = Acquisition(
        optics=scope.optics,
        image_sensor=scope.image_sensor,
        tissue=region.tissue,
        fps=fps,
        duration_s=duration_s,
        focal_depth_in_tissue_um=scope.focal_depth_in_tissue_um,
        front_working_distance_um=scope.front_working_distance_um,
    )
    cells = list(populations) if populations is not None else [region.population]
    steps: list[AnyStep] = [
        PlaceNeurons(populations=cells),
        activity or CellActivity(),
        CellOptics(),
        Composite(),
    ]
    if include_neuropil and region.neuropil is not None:
        steps.append(region.neuropil)
    if include_vasculature and region.vasculature is not None:
        steps.append(region.vasculature)
    if include_scope_fields:
        steps.extend(
            f
            for f in (scope.illumination, scope.vignette, scope.leakage)
            if f is not None
        )
    steps.append(sensor or Sensor(photons_per_unit=scope.photons_per_unit))
    steps.extend(extra_steps)
    return Spec(
        acquisition=acquisition,
        seed=seed,
        steps=steps,
        output=Output(save_intermediates=save_intermediates),
    )

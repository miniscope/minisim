"""Named, realistic starting points: standard scopes and brain regions.

A *scope* and a *region* are the two orthogonal halves of a recording's physical
setup, and this module names the common ones so a test or a notebook can grab a
real, validated starting point in one line instead of hand-tuning a dozen
fields:

* a :class:`Scope` is the measurable hardware - objective optics + image sensor -
  everything that does not depend on what tissue you point it at
  (:func:`miniscope_v4`, :func:`generic_1p`);
* a :class:`Region` is the biology you point it at - the cell population (depth,
  density, morphology), the tissue scatter, and the region's characteristic
  vessel confound (:func:`ca1`, :func:`cortex_l23`).

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
from dataclasses import dataclass
from typing import Literal

from minisim.spec import (
    Acquisition,
    AnyStep,
    CellActivity,
    CellOptics,
    Composite,
    ImageSensor,
    NeuronPopulation,
    Optics,
    Output,
    PlaceNeurons,
    Sensor,
    Spec,
    Tissue,
    Vasculature,
    VesselLayer,
)

# A focal plane is either a concrete depth into tissue (µm) or "auto" (resolved to
# the median realized cell depth at the optics step).
FocalDepth = float | Literal["auto"]


@dataclass(frozen=True)
class Scope:
    """A miniscope's measurable hardware: objective optics + image sensor.

    The reusable physical front-end - everything independent of the tissue you
    image. ``focal_depth_in_tissue_um`` and ``front_working_distance_um`` are the
    two acquisition-level fields that travel with the scope rather than the
    region. Compose with a :class:`Region` via :func:`build_spec`.
    """

    optics: Optics
    image_sensor: ImageSensor
    focal_depth_in_tissue_um: FocalDepth = "auto"
    front_working_distance_um: float | None = None

    @property
    def pixel_size_um(self) -> float:
        """Object-space size of one pixel, µm (sensor pitch / magnification)."""
        return self.image_sensor.pixel_pitch_um / self.optics.magnification

    @property
    def fov_um(self) -> tuple[float, float]:
        """Field of view ``(height, width)`` in µm at this scope's settings."""
        px = self.pixel_size_um
        return (self.image_sensor.n_px_height * px, self.image_sensor.n_px_width * px)


@dataclass(frozen=True)
class Region:
    """A standard imaging target: cell population + tissue scatter + vessels.

    The biology half of a recording - what a :class:`Scope` is pointed at.
    ``population`` carries the cell distribution (depth range, density,
    morphology); ``tissue`` the depth-dependent scatter; ``vasculature`` the
    region's characteristic dark-vessel confound, or ``None`` for no vessels.
    Compose with a scope via :func:`build_spec`.
    """

    population: NeuronPopulation
    tissue: Tissue
    vasculature: Vasculature | None = None


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


def miniscope_v4() -> Scope:
    """UCLA Miniscope V4: NA 0.3 GRIN, 608×608 px at 4.8 µm pitch, ~1.0 mm FOV.

    Confirmed V4 numbers (D. Aharoni): magnification 2.9 so the sensor sees a
    ~1.0 mm field of view, 525 nm GCaMP emission, a ~2500 µm field-curvature
    radius, and a 700 µm front working distance. The focal plane defaults to
    ``"auto"`` (tracks the placed layer), as the anatomy notebook does.
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
        ),
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
    Vasculature is on but *less pronounced* than cortex - thinner, lower-contrast
    vessels just above the pyramidal band (D. Aharoni).
    """
    return Region(
        population=NeuronPopulation(
            density_per_mm3=45000.0,
            soma_radius_um=5.0,
            morphology="cytosolic",
            depth_range_um=(140.0, 160.0),
        ),
        tissue=Tissue(),
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
    band than CA1, so depth-dependent scatter and defocus matter more.
    Vasculature is *thick and on top of the cells* (D. Aharoni): a shallow layer
    of large-caliber, high-contrast vessels above the imaged band.
    """
    return Region(
        population=NeuronPopulation(
            density_per_mm3=8000.0,
            soma_radius_um=6.0,
            morphology="cytosolic",
            depth_range_um=(100.0, 200.0),
        ),
        tissue=Tissue(),
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
    include_vasculature: bool = True,
    save_intermediates: bool = False,
) -> Spec:
    """Assemble a validated :class:`~minisim.Spec` from a scope × a region.

    Builds the :class:`~minisim.Acquisition` from the scope's optics/sensor and
    the region's tissue, then the minimal forward chain
    (``place_neurons → cell_activity → optics → composite → sensor``) plus the
    region's vasculature when present. The result is a real frozen ``Spec``: drop
    it into :func:`~minisim.simulate`, or pass it to :func:`~minisim.sweep` as the
    base for a parameter grid.

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
        The :class:`~minisim.Sensor` exposure step; ``None`` uses its defaults
        (``photons_per_unit=100``). For a brightly-exposed deep-tissue recording
        pass e.g. ``Sensor(photons_per_unit=600)``.
    extra_steps
        Additional steps to append - ``BrainMotion``, ``Neuropil``,
        ``Vignette``, ``IlluminationProfile``, ``Leakage``, ``Bleaching``. The
        ``Spec`` re-sorts into canonical pipeline order, so order here is free.
    include_vasculature
        When ``False``, drop the region's vessel layer (a clean, vessel-free
        movie). Ignored when the region has no vasculature.
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
    if include_vasculature and region.vasculature is not None:
        steps.append(region.vasculature)
    steps.append(sensor or Sensor())
    steps.extend(extra_steps)
    return Spec(
        acquisition=acquisition,
        seed=seed,
        steps=steps,
        output=Output(save_intermediates=save_intermediates),
    )

"""The executable-step contract for the ``minisim`` pipeline.

Where a ``StepSpec`` (:mod:`minisim.spec`) is the *immutable,
serializable description* of a step, a ``Step`` is its *executable counterpart*:
a small callable that mutates a :class:`~minisim.scene.Scene` in place.
``StepSpec.build()`` turns the former into the latter, binding it to the
``Acquisition`` (which owns every µm↔px / s↔frame conversion) and the RNG it
draws from.

A step does exactly one thing to the scene — fill per-cell records, composite
into the movie, or record a ground-truth contribution — so a single step can be
constructed and run against a hand-built ``Scene`` in a unit test. That, not the
full ``simulate()`` loop (migration Step 6), is the primary test substrate.

Each step captures the RNG handed to ``build()`` and draws from *that* generator
(not ``scene.rng``), so a step is reproducible from its construction regardless
of which scene it is later run against — matching the ``build(acq, rng)``
signature the orchestrator and the unit tests both use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar

if TYPE_CHECKING:
    import numpy as np

    from minisim.scene import Scene
    from minisim.spec import Acquisition, IlluminationProfile, Sensor, StepSpec

#: The concrete :class:`~minisim.spec.StepSpec` subtype a step is built from.
#: Parametrizing ``Step`` on it (``class RenderStep(Step[Render])``) makes every
#: ``self.spec`` access resolve to the concrete spec's typed fields instead of the
#: untyped base.
SpecT = TypeVar("SpecT", bound="StepSpec")


@dataclass(frozen=True)
class PipelineContext:
    """Cross-step context resolved once before a run, offered to each step's
    :meth:`Step.prepare`.

    A few cell-domain steps depend on parameters defined by *later* sensor-domain
    steps: ``bleaching`` bleaches faster under brighter excitation
    (``illumination``), and ``optics`` "auto" focus weights cells by the photon
    budget their image gets and the sensor noise floor (``photon_field`` /
    ``sensor_spec``). Rather than the orchestrator reaching into each step by name,
    the context is assembled up front (:func:`minisim.simulate.build_context`) and
    handed to every step via ``prepare``; each step pulls only the fields it needs.

    Every field is optional: a step absent from the spec leaves its slot ``None``
    and dependents fall back to a uniform / no-op default.
    """

    illumination: IlluminationProfile | None = None
    sensor_spec: Sensor | None = None
    photon_field: np.ndarray | None = None


class Step(Generic[SpecT]):
    """Base class for an executable pipeline step.

    Subclasses set the two class attributes and implement :meth:`__call__`:

    * ``name`` — the snapshot/stage name (usually the spec ``kind``; ``render``
      is the exception, whose stage is ``"cells_only"``).
    * ``domain`` — the pipeline domain (``cell`` → ``tissue`` → ``motion`` →
      ``sensor``), the same ordering axis the spec validator checks.

    The constructor stores the originating spec, the acquisition, and the RNG;
    subclasses read parameters off ``self.spec`` and physical conversions off
    ``self.acq``.
    """

    name: ClassVar[str]
    domain: ClassVar[Literal["cell", "tissue", "motion", "sensor"]]

    def __init__(
        self, spec: SpecT, acq: Acquisition, rng: np.random.Generator
    ) -> None:
        self.spec: SpecT = spec
        self.acq = acq
        self.rng = rng

    def prepare(self, context: PipelineContext) -> None:
        """Pull any cross-step dependencies off the resolved ``context``.

        Called by the orchestrator after :meth:`StepSpec.build` and before
        :meth:`__call__`. The base implementation is a no-op; a step that needs a
        sibling step's resolved spec or field (e.g. ``bleaching``'s illumination,
        ``optics``' photon budget) overrides this to read it from ``context``.
        Keeping it a *pull* (the step takes what it needs) rather than a *push*
        (the orchestrator sets attributes by name) keeps each cross-step
        dependency declared on the step that actually has it. A step run directly
        in a unit test may skip ``prepare`` and set the dependency itself.
        """

    def __call__(self, scene: Scene) -> None:
        """Mutate ``scene`` in place. Implemented by each concrete step."""
        raise NotImplementedError(f"{type(self).__name__} does not implement __call__.")

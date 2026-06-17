"""Executable pipeline steps for ``minisim``.

Each step is the runtime counterpart of a ``StepSpec`` (see
:mod:`minisim.spec`): a small callable that mutates a ``Scene`` in
place, returned by the spec's ``build()`` method. They are organized by pipeline
domain - ``cell`` → ``tissue`` → ``motion`` → ``sensor`` - mirroring the forward
order biology → optics → motion → sensor.

The minimal runnable chain is ``place_neurons`` → ``cell_activity`` → ``composite``
→ ``sensor``; ``optics`` degrades the footprints, the field effects
(``neuropil``/``bleaching``/``vignette``/``leakage``, plus the absorbing
``vasculature`` mask) layer on top, and ``brain_motion`` is the brain→sensor frame
boundary - together the full forward pipeline. The :class:`Step` base and the
physics helpers
(:func:`calcium_kernel`, :func:`neuron_footprint`, :func:`bleaching_pool`,
:func:`ou_process`, :func:`bounded_random_walk`, …) are exposed here for direct
unit testing and teaching.
"""

from minisim.footprint import degrade_footprint
from minisim.steps.base import Step
from minisim.steps.cell import (
    CellActivityStep,
    CellOpticsStep,
    PlaceNeuronsStep,
    calcium_kernel,
    kernel_timing,
    neuron_footprint,
    resolve_focal_plane,
    sample_neurons,
    spike_activity_params,
    tau_from_kernel_timing,
)
from minisim.steps.motion import (
    BrainMotionStep,
    bounded_random_walk,
    brain_motion_shifts,
    physical_brain_motion,
    shift_and_crop,
)
from minisim.steps.sensor import (
    IlluminationProfileStep,
    LeakageStep,
    SensorStep,
    VignetteStep,
    combined_falloff_field,
    falloff_center_px,
    leakage_field,
    radial_falloff,
    radius_grid,
    resolve_exposure,
)
from minisim.steps.tissue import (
    BleachingStep,
    CompositeStep,
    NeuropilStep,
    VasculatureStep,
    VesselGrowth,
    bleaching_floor,
    bleaching_pool,
    dark_recovery,
    grow_vessel_tree,
    murray_children,
    neuropil_components,
    neuropil_envelope,
    ou_process,
    population_envelope,
    rasterize_vessels,
    smooth_spatial_field,
    vasculature_focal,
    vasculature_mask_field,
    vessels_to_mask,
)

#: The declarative spec→step mapping, keyed by ``StepSpec.kind``. This single
#: table *is* the relationship between a spec and its executable step;
#: ``StepSpec.build`` looks the class up here rather than each spec hardcoding an
#: import. It lives in the steps package (not ``spec``) because the step modules
#: depend, through ``recording``, back on ``spec`` - so ``spec`` cannot import the
#: classes at module load. ``test_spec`` asserts it stays in 1:1 sync with the
#: spec ``kind`` catalog.
STEP_FOR_KIND: dict[str, type[Step]] = {
    "place_neurons": PlaceNeuronsStep,
    "cell_activity": CellActivityStep,
    "optics": CellOpticsStep,
    "composite": CompositeStep,
    "neuropil": NeuropilStep,
    "vasculature": VasculatureStep,
    "bleaching": BleachingStep,
    "brain_motion": BrainMotionStep,
    "illumination_profile": IlluminationProfileStep,
    "vignette": VignetteStep,
    "leakage": LeakageStep,
    "sensor": SensorStep,
}

__all__ = [
    "STEP_FOR_KIND",
    "BleachingStep",
    "BrainMotionStep",
    "CellActivityStep",
    "CellOpticsStep",
    "CompositeStep",
    "IlluminationProfileStep",
    "LeakageStep",
    "NeuropilStep",
    "PlaceNeuronsStep",
    "SensorStep",
    "Step",
    "VasculatureStep",
    "VesselGrowth",
    "VignetteStep",
    "bleaching_floor",
    "bleaching_pool",
    "bounded_random_walk",
    "brain_motion_shifts",
    "calcium_kernel",
    "combined_falloff_field",
    "dark_recovery",
    "degrade_footprint",
    "falloff_center_px",
    "grow_vessel_tree",
    "kernel_timing",
    "leakage_field",
    "murray_children",
    "neuron_footprint",
    "neuropil_components",
    "neuropil_envelope",
    "ou_process",
    "physical_brain_motion",
    "population_envelope",
    "radial_falloff",
    "radius_grid",
    "rasterize_vessels",
    "resolve_exposure",
    "resolve_focal_plane",
    "sample_neurons",
    "shift_and_crop",
    "smooth_spatial_field",
    "spike_activity_params",
    "tau_from_kernel_timing",
    "vasculature_focal",
    "vasculature_mask_field",
    "vessels_to_mask",
]

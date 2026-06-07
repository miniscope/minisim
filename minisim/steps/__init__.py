"""Executable pipeline steps for ``minisim``.

Each step is the runtime counterpart of a ``StepSpec`` (see
:mod:`minisim.spec`): a small callable that mutates a ``Scene`` in
place, returned by the spec's ``build()`` method. They are organized by pipeline
domain — ``cell`` → ``tissue`` → ``motion`` → ``sensor`` — mirroring the forward
order biology → optics → motion → sensor.

Migration Step 5a lands the minimal runnable chain (``place_neurons`` →
``cell_activity`` → ``render`` → ``sensor``); optics is 5b, the field effects
(``neuropil``/``bleaching``/``vignette``/``leakage``, plus the ``vasculature``
no-op placeholder) are 5c, and ``brain_motion`` is 5d — the full forward
pipeline. The :class:`Step` base and the physics helpers
(:func:`calcium_kernel`, :func:`neuron_footprint`, :func:`bleaching_pool`,
:func:`ou_process`, :func:`bounded_random_walk`, …) are exposed here for direct
unit testing and teaching.
"""

from minisim.steps.base import Step
from minisim.steps.cell import (
    CellActivityStep,
    CellOpticsStep,
    PlaceNeuronsStep,
    calcium_kernel,
    degrade_footprint,
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
)
from minisim.steps.tissue import (
    BleachingStep,
    NeuropilStep,
    RenderStep,
    VasculatureStep,
    bleaching_pool,
    neuropil_components,
    neuropil_envelope,
    ou_process,
    population_envelope,
    smooth_spatial_field,
)

__all__ = [
    "BleachingStep",
    "BrainMotionStep",
    "CellActivityStep",
    "CellOpticsStep",
    "IlluminationProfileStep",
    "LeakageStep",
    "NeuropilStep",
    "PlaceNeuronsStep",
    "RenderStep",
    "SensorStep",
    "Step",
    "VasculatureStep",
    "VignetteStep",
    "bleaching_pool",
    "bounded_random_walk",
    "brain_motion_shifts",
    "calcium_kernel",
    "combined_falloff_field",
    "falloff_center_px",
    "degrade_footprint",
    "kernel_timing",
    "leakage_field",
    "neuron_footprint",
    "neuropil_components",
    "neuropil_envelope",
    "ou_process",
    "physical_brain_motion",
    "population_envelope",
    "radial_falloff",
    "radius_grid",
    "resolve_focal_plane",
    "sample_neurons",
    "shift_and_crop",
    "spike_activity_params",
    "tau_from_kernel_timing",
    "smooth_spatial_field",
]

"""Physically-driven synthetic 1-photon miniscope data: generator + teaching tool.

minisim builds a recording forward from its physical components (biology ->
optics -> motion -> sensor), the inverse of an analysis pipeline like minian.
Each recording ships with the exact ground truth that generated it, so it is
suitable for benchmarking pipelines, testing analysis code, and teaching the
anatomy of miniscope data.

The top-level surface is the typed :class:`Spec` plus :func:`simulate` (and the
cached :func:`simulate_cached`), the recovery :mod:`~minisim.metrics`, and the
:func:`sweep` helper for parameter studies.
"""

from minisim.cache import cache_dir, cache_path, simulate_cached
from minisim.metrics import (
    Match,
    SpikeScore,
    field_pearson,
    hungarian_match,
    shift_rmse,
    spike_precision_recall,
    trace_pearson,
)
from minisim.recording import GroundTruth, Recording, finalize
from minisim.scene import Cell, GroundTruthBuilder, Scene
from minisim.simulate import simulate
from minisim.spec import (
    Acquisition,
    AnyStep,
    Bleaching,
    BrainMotion,
    CellActivity,
    CellOptics,
    IlluminationProfile,
    ImageSensor,
    Leakage,
    Neuropil,
    Optics,
    Output,
    PlaceNeurons,
    Render,
    Sensor,
    Spec,
    SpecWarning,
    StepSpec,
    Tissue,
    Vasculature,
    Vignette,
)
from minisim.sweep import SweptSpec, sweep
from minisim.video import simulate_video

__all__ = [
    "Acquisition",
    "AnyStep",
    "Bleaching",
    "BrainMotion",
    "Cell",
    "CellActivity",
    "CellOptics",
    "GroundTruth",
    "GroundTruthBuilder",
    "IlluminationProfile",
    "ImageSensor",
    "Leakage",
    "Match",
    "Neuropil",
    "Optics",
    "Output",
    "PlaceNeurons",
    "Recording",
    "Render",
    "Scene",
    "Sensor",
    "Spec",
    "SpecWarning",
    "SpikeScore",
    "StepSpec",
    "SweptSpec",
    "Tissue",
    "Vasculature",
    "Vignette",
    "cache_dir",
    "cache_path",
    "field_pearson",
    "finalize",
    "hungarian_match",
    "shift_rmse",
    "simulate",
    "simulate_cached",
    "simulate_video",
    "spike_precision_recall",
    "sweep",
    "trace_pearson",
]

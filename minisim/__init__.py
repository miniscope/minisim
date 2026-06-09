"""Physically-driven synthetic 1-photon miniscope data: generator + teaching tool.

Minisim builds a recording forward from its physical components (biology ->
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
    footprint_mask,
    footprint_roi_trace,
    hungarian_match,
    shift_rmse,
    spike_precision_recall,
    trace_pearson,
)
from minisim.perf import PerfTracker
from minisim.recording import (
    DETECT_SNR_THRESHOLD,
    GroundTruth,
    Recording,
    detection_snr,
    finalize,
    sample_field_at,
)
from minisim.scene import Cell, GroundTruthBuilder, Scene
from minisim.simulate import simulate
from minisim.spec import (
    Acquisition,
    AnyStep,
    Bleaching,
    BrainMotion,
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
    SpecWarning,
    StepSpec,
    Tissue,
    Vasculature,
    Vignette,
    order_steps,
)
from minisim.sweep import SweptSpec, sweep
from minisim.video import simulate_video

__all__ = [
    "DETECT_SNR_THRESHOLD",
    "Acquisition",
    "AnyStep",
    "Bleaching",
    "BrainMotion",
    "Cell",
    "CellActivity",
    "CellOptics",
    "Composite",
    "GroundTruth",
    "GroundTruthBuilder",
    "IlluminationProfile",
    "ImageSensor",
    "Leakage",
    "Match",
    "NeuronPopulation",
    "Neuropil",
    "Optics",
    "Output",
    "PerfTracker",
    "PlaceNeurons",
    "Recording",
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
    "detection_snr",
    "field_pearson",
    "finalize",
    "footprint_mask",
    "footprint_roi_trace",
    "hungarian_match",
    "order_steps",
    "sample_field_at",
    "shift_rmse",
    "simulate",
    "simulate_cached",
    "simulate_video",
    "spike_precision_recall",
    "sweep",
    "trace_pearson",
]

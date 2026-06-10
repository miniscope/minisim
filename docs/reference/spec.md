# Spec and steps

The {py:class}`~minisim.Spec` is the one object you build to describe a
recording: an acquisition, a seed, an ordered list of steps, and output options.
It validates the whole pipeline on construction.

## Spec

```{eval-rst}
.. autopydantic_model:: minisim.Spec
```

## Acquisition and physical models

The acquisition owns all unit conversions between the physical world (µm,
seconds) and the sampled world (pixels, frames), and composes the three physical
models below.

```{eval-rst}
.. autopydantic_model:: minisim.Acquisition
```

```{eval-rst}
.. autopydantic_model:: minisim.Optics
```

```{eval-rst}
.. autopydantic_model:: minisim.ImageSensor
```

```{eval-rst}
.. autopydantic_model:: minisim.Tissue
```

## Output

```{eval-rst}
.. autopydantic_model:: minisim.Output
```

## Steps

`Spec.steps` is the forward chain. Each step below is a
{py:class}`~minisim.StepSpec`; `AnyStep` is the discriminated union of them.
Each `kind` may appear at most once in a spec.

```{eval-rst}
.. autopydantic_model:: minisim.StepSpec
```

### The forward chain

```{eval-rst}
.. autopydantic_model:: minisim.PlaceNeurons
```

A `PlaceNeurons` step is itself a single {py:class}`~minisim.NeuronPopulation`
(its fields describe one group of cells). To place several distinct groups at
once - a thin soma-targeted layer over a deep cytosolic volume, say - set
`populations` to a list of `NeuronPopulation` instead. A population can be
density-sampled (`density_per_mm3` over a `depth_range_um`) or placed at exact
`positions_um` centers `(z, y, x)`; the two kinds can be mixed in one spec.

```{eval-rst}
.. autopydantic_model:: minisim.NeuronPopulation
```

```{eval-rst}
.. autopydantic_model:: minisim.CellActivity
```

```{eval-rst}
.. autopydantic_model:: minisim.CellOptics
```

```{eval-rst}
.. autopydantic_model:: minisim.Composite
```

```{eval-rst}
.. autopydantic_model:: minisim.Neuropil
```

```{eval-rst}
.. autopydantic_model:: minisim.Vasculature
```

```{eval-rst}
.. autopydantic_model:: minisim.VesselLayer
```

```{eval-rst}
.. autopydantic_model:: minisim.Bleaching
```

```{eval-rst}
.. autopydantic_model:: minisim.BrainMotion
```

```{eval-rst}
.. autopydantic_model:: minisim.IlluminationProfile
```

```{eval-rst}
.. autopydantic_model:: minisim.Vignette
```

```{eval-rst}
.. autopydantic_model:: minisim.Leakage
```

```{eval-rst}
.. autopydantic_model:: minisim.Sensor
```

## Warnings

```{eval-rst}
.. autoclass:: minisim.SpecWarning
```

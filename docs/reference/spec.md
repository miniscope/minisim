# Spec and steps

The {py:class}`~minisim.Spec` is the one object you build to describe a
recording: an acquisition, a seed, an ordered list of steps, and output options.
It validates the whole pipeline on construction.

## Spec

```{autopydantic_model} minisim.Spec
```

## Acquisition and physical models

The acquisition owns all unit conversions between the physical world (µm,
seconds) and the sampled world (pixels, frames), and composes the three physical
models below.

```{autopydantic_model} minisim.Acquisition
```

```{autopydantic_model} minisim.Optics
```

```{autopydantic_model} minisim.ImageSensor
```

```{autopydantic_model} minisim.Tissue
```

## Output

```{autopydantic_model} minisim.Output
```

## Steps

`Spec.steps` is the forward chain. Each step below is a
{py:class}`~minisim.StepSpec`; `AnyStep` is the discriminated union of them.
Each `kind` may appear at most once in a spec.

```{autopydantic_model} minisim.StepSpec
```

### The forward chain

```{autopydantic_model} minisim.PlaceNeurons
```

A `PlaceNeurons` step is itself a single {py:class}`~minisim.NeuronPopulation`
(its fields describe one group of cells). To place several distinct groups at
once - a thin soma-targeted layer over a deep cytosolic volume, say - set
`populations` to a list of `NeuronPopulation` instead. A population can be
density-sampled (`density_per_mm3` over a `depth_range_um`) or placed at exact
`positions_um` centers `(z, y, x)`; the two kinds can be mixed in one spec.

```{autopydantic_model} minisim.NeuronPopulation
```

```{autopydantic_model} minisim.CellActivity
```

```{autopydantic_model} minisim.CellOptics
```

```{autopydantic_model} minisim.Composite
```

```{autopydantic_model} minisim.Neuropil
```

```{autopydantic_model} minisim.Vasculature
```

```{autopydantic_model} minisim.Bleaching
```

```{autopydantic_model} minisim.BrainMotion
```

```{autopydantic_model} minisim.IlluminationProfile
```

```{autopydantic_model} minisim.Vignette
```

```{autopydantic_model} minisim.Leakage
```

```{autopydantic_model} minisim.Sensor
```

## Warnings

```{autoclass} minisim.SpecWarning
```

# Recovery metrics

Functions to score an analysis pipeline's output against the ground truth. See
the {doc}`benchmarking guide <../howto/benchmark>` for how they fit together.

## Spatial matching

```{autofunction} minisim.hungarian_match
```

```{autoclass} minisim.Match
```

## Temporal scores

```{autofunction} minisim.trace_pearson
```

```{autofunction} minisim.spike_precision_recall
```

```{autoclass} minisim.SpikeScore
```

## Field and motion

```{autofunction} minisim.field_pearson
```

```{autofunction} minisim.shift_rmse
```

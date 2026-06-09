# Recovery metrics

Functions to score an analysis pipeline's output against the ground truth. See
the {doc}`benchmarking guide <../howto/benchmark>` for how they fit together.

## Spatial matching

```{eval-rst}
.. autofunction:: minisim.hungarian_match
```

```{eval-rst}
.. autoclass:: minisim.Match
```

## Temporal scores

```{eval-rst}
.. autofunction:: minisim.trace_pearson
```

```{eval-rst}
.. autofunction:: minisim.spike_precision_recall
```

```{eval-rst}
.. autoclass:: minisim.SpikeScore
```

## Field and motion

```{eval-rst}
.. autofunction:: minisim.field_pearson
```

```{eval-rst}
.. autofunction:: minisim.shift_rmse
```

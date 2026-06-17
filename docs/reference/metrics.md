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
.. autofunction:: minisim.activity_similarity
```

```{eval-rst}
.. autoclass:: minisim.ActivityScore
```

## Field and motion

```{eval-rst}
.. autofunction:: minisim.field_pearson
```

```{eval-rst}
.. autofunction:: minisim.shift_rmse
```

```{eval-rst}
.. autofunction:: minisim.global_shift_from_trajectories
```

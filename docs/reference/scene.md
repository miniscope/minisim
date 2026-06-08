# Scene construction

The scene layer is the mutable working state that pipeline steps read and write
(canvas, movie, per-cell footprints). Most users never touch it directly; it is
public so step authors and notebooks can. {py:func}`~minisim.finalize` turns a
built {py:class}`~minisim.Scene` into a {py:class}`~minisim.Recording`.

## Scene

```{autoclass} minisim.Scene
```

## Cell

```{autoclass} minisim.Cell
```

## GroundTruthBuilder

```{autoclass} minisim.GroundTruthBuilder
```

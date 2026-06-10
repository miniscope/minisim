# Scene construction

The scene layer is the mutable working state that pipeline steps read and write
(canvas, movie, per-cell footprints). Most users never touch it directly; it is
public so step authors and notebooks can. {py:func}`~minisim.finalize` turns a
built {py:class}`~minisim.Scene` into a {py:class}`~minisim.Recording`.

## Scene

```{eval-rst}
.. autoclass:: minisim.Scene
```

## Cell

```{eval-rst}
.. autoclass:: minisim.Cell
```

## GroundTruthBuilder

```{eval-rst}
.. autoclass:: minisim.GroundTruthBuilder
```

# Scene construction

The scene layer is the mutable working state that pipeline steps read and write
(canvas, movie, per-cell footprints). Most users never touch it directly, so it is
not part of the top-level surface; it lives in the {mod}`minisim.scene` submodule
for the step authors and notebooks that build against it.
{py:func}`~minisim.finalize` turns a built {py:class}`~minisim.scene.Scene` into a
{py:class}`~minisim.Recording`.

## Scene

```{eval-rst}
.. autoclass:: minisim.scene.Scene
```

## Cell

```{eval-rst}
.. autoclass:: minisim.scene.Cell
```

## GroundTruthBuilder

```{eval-rst}
.. autoclass:: minisim.scene.GroundTruthBuilder
```

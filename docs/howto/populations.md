# Place layered populations

A {py:class}`~minisim.PlaceNeurons` step is itself a single
{py:class}`~minisim.NeuronPopulation`: its fields (`morphology`,
`soma_radius_um`, `depth_range_um`, `density_per_mm3`, ...) describe one group of
cells. The common case is exactly that - one population:

```python
from minisim import PlaceNeurons

PlaceNeurons(morphology="cytosolic", soma_radius_um=7.0, depth_range_um=(0.0, 200.0))
```

## Combine several populations

To build layered anatomy - say a thin soma-targeted band sitting over a deeper
cytosolic volume - set `populations` to a list of
{py:class}`~minisim.NeuronPopulation`. The step samples each in turn and
concatenates the cells; the step's own population fields are then unused (setting
both raises, so there is no ambiguity about which wins).

```python
from minisim import PlaceNeurons, NeuronPopulation

PlaceNeurons(populations=[
    NeuronPopulation(                       # thin soma-targeted layer
        morphology="soma",
        depth_range_um=(50.0, 60.0),
        density_per_mm3=80000.0,
    ),
    NeuronPopulation(                       # deep cytosolic volume
        morphology="cytosolic",
        depth_range_um=(100.0, 300.0),
        density_per_mm3=25000.0,
    ),
])
```

Each population's count is volumetric (`density_per_mm3 × FOV area × depth
thickness`), and any `min_distance_um` spacing is enforced *within* a population,
not across them - so adjacent layers may interpenetrate at their shared boundary,
the physical case.

## Place cells at exact positions

A population can also be placed at explicit centers instead of being
density-sampled: set `positions_um` to a list of `(z, y, x)` µm tuples - depth,
row, column, in the tissue frame (origin = canvas top-left, the same coordinates
the ground truth reports back). Note the **depth-first** order, matching
`Cell.center_um` rather than `x, y, z`.

```python
PlaceNeurons(
    morphology="cytosolic",
    soma_radius_um=8.0,
    positions_um=[
        (120.0, 128.0, 128.0),   # (z=depth, y=row, x=col)
        (120.0, 60.0, 60.0),
    ],
)
```

When `positions_um` is given, the distribution fields (`density_per_mm3`,
`depth_range_um`, `min_distance_um`) are ignored; the shape fields
(`soma_radius_um`, `irregularity`, `morphology`, dendrites) still apply to each
placed cell. Because positions live on the *population*, you can mix an
explicit-position population and a density-sampled one in the same `populations`
list - handy for dropping a few cells at known spots inside an otherwise random
field.

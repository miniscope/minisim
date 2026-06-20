# Presets and `build_spec`

Named, realistic starting points for a recording, and the composer that turns
them into a {py:class}`~minisim.Spec`. A **scope** and a **region** are the two
orthogonal halves of a recording's physical setup; this module names the common
ones so a test or a notebook can grab a validated starting point in one line
instead of hand-tuning a dozen fields.

`Scope`, `Region`, and `build_spec` are importable from the top level
(`from minisim import Scope, Region, build_spec`); the named factory functions
live in the `minisim.presets` submodule.

## Scope

A scope is the measurable hardware - objective optics plus image sensor - and the
instrument-fixed effects that ride with it: the excitation `illumination`
falloff, the collection-side `vignette`, the stray-light `leakage` glow, and the
rig's typical `photons_per_unit` exposure. Everything here is independent of the
tissue you point it at.

```{eval-rst}
.. autopydantic_model:: minisim.Scope
```

```{eval-rst}
.. autofunction:: minisim.presets.miniscope_v4
```

```{eval-rst}
.. autofunction:: minisim.presets.generic_1p
```

## Region

A region is the biology a scope is pointed at: the cell `population` (depth,
density, morphology), the depth-dependent `tissue` scatter, the diffuse
`neuropil` haze from the surrounding dendritic/axonal felt, and the region's
characteristic dark-vessel `vasculature` confound.

```{eval-rst}
.. autopydantic_model:: minisim.Region
```

```{eval-rst}
.. autofunction:: minisim.presets.ca1
```

```{eval-rst}
.. autofunction:: minisim.presets.cortex_l23
```

## Compose with `build_spec`

`build_spec` assembles any scope × any region into a validated
{py:class}`~minisim.Spec`. It builds the {py:class}`~minisim.Acquisition` from
the scope's optics/sensor and the region's tissue, then the forward chain
`place_neurons → cell_activity → optics → composite`, and appends the region's
neuropil and vasculature, the scope's static fields (illumination, vignette,
leakage), and a sensor exposed at the scope's `photons_per_unit` - each gated by
an `include_*` toggle so you can drop any of them for a clean baseline. Swap the
scope or region independently, override the rest with
{py:func}`~minisim.sweep`, or hand-place cells via the `populations` argument.

```{eval-rst}
.. autofunction:: minisim.build_spec
```

## Example

```python
from minisim import build_spec, simulate
from minisim.presets import miniscope_v4, ca1

# a Miniscope V4 imaging hippocampal CA1, full V4 look (illumination, vignette,
# leakage, neuropil, vasculature) included by default
spec = build_spec(miniscope_v4(), ca1(), duration_s=30.0, seed=0)
rec = simulate(spec)

# a clean, confound-free baseline of the same anatomy
clean = build_spec(
    miniscope_v4(), ca1(),
    include_neuropil=False, include_vasculature=False, include_scope_fields=False,
)
```

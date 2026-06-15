# Testing helpers

Convenience helpers for using minisim as test fixtures for an analysis pipeline.
Unlike the rest of the reference, these live in the `minisim.testing` submodule
(`from minisim.testing import make_recording, score`). See the how-to guide
{doc}`../howto/use_in_test_suite` for the recommended `pytest` wiring.

## make_recording

```{eval-rst}
.. autofunction:: minisim.testing.make_recording
```

## score

```{eval-rst}
.. autofunction:: minisim.testing.score
```

## Estimate

```{eval-rst}
.. autoclass:: minisim.testing.Estimate
```

## Report

```{eval-rst}
.. autoclass:: minisim.testing.Report
   :members:
```

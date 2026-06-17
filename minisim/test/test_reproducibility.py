"""Golden-master reproducibility lock for the public fixture surface.

``minisim.testing.make_recording`` is a *contract*: downstream test suites
(minian, CaImAn, suite2p) assert against the recordings it produces, so a silent
change to its output - new defaults, a tweaked activity model, a recalibrated
detection threshold, an RNG-order change - would flip those assertions red across
every consumer at once, for a reason invisible in their own diff.

These tests pin the output of a fixed-seed ``make_recording`` so that *minisim's
own* CI fails the moment that output moves. A failure here is not a bug to paper
over: it means the fixture contract changed. When the change is intentional, re-pin
the values in the same commit and call it out in the changelog (a minor-version
bump), so the break is deliberate and announced rather than silent. See the
reproducibility & stability contract in the docs.

Two kinds of pin, by what is portable across platforms. The digitized ``observed``
movie (integer counts) and the cell ``centers_um`` (pure RNG draws) are byte-exact
on every platform, so they are locked by SHA256. The calcium traces ``gt.C`` are
continuous and built from transcendental decay math, whose last bit differs between
the Windows and Linux math libraries; hashing them is not portable, so ``C`` is
pinned within a tight tolerance instead. The tolerance (rtol=1e-4) sits far above
the ~1e-12 platform noise yet far below any real change to the activity model.
"""

import hashlib

import numpy as np
import pytest

from minisim.testing import make_recording


def _sha256(array) -> str:
    """SHA256 of an array's raw bytes, after a contiguous + dtype-stable view."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# Golden hashes for make_recording(seed=0) at its documented defaults. Re-pin
# these (deliberately, with a changelog note) whenever the fixture output changes.
_GOLDEN_SEED = 0
_GOLDEN_OBSERVED_SHA = "e59c184bf0acf2c9e1c4021bdf1eaeae6fdeb46137956a852e785e35d372eddc"
_GOLDEN_CENTERS_SHA = "1789804afad9ecffee2fc27e8e1321d708274618aea5066fa7d62a73f6247dc5"
_GOLDEN_SHAPE = (40, 128, 128)
_GOLDEN_N_UNITS = 6
_GOLDEN_N_DETECTABLE = 6
# gt.C is continuous (non-portable to hash); pinned by shape + reduction stats.
_GOLDEN_C_SHAPE = (6, 40)
_GOLDEN_C_SUM = 6160.93338586201
_GOLDEN_C_MAX = 72.45677929083958


def test_make_recording_observed_is_byte_stable():
    rec = make_recording(seed=_GOLDEN_SEED)
    assert rec.observed.shape == _GOLDEN_SHAPE
    assert rec.observed.dtype == np.float32
    assert _sha256(rec.observed) == _GOLDEN_OBSERVED_SHA, (
        "make_recording(seed=0).observed changed. If this is intentional, re-pin "
        "the golden hash here and note the fixture change in the changelog."
    )


def test_make_recording_ground_truth_is_stable():
    gt = make_recording(seed=_GOLDEN_SEED).ground_truth
    assert gt.n_units == _GOLDEN_N_UNITS
    assert int(gt.detectable.sum()) == _GOLDEN_N_DETECTABLE
    # centers_um are pure RNG draws -> byte-exact on every platform.
    assert _sha256(gt.centers_um) == _GOLDEN_CENTERS_SHA
    # C is continuous transcendental math -> pin within tolerance, not by hash.
    assert gt.C.shape == _GOLDEN_C_SHAPE
    assert float(gt.C.sum()) == pytest.approx(_GOLDEN_C_SUM, rel=1e-4)
    assert float(gt.C.max()) == pytest.approx(_GOLDEN_C_MAX, rel=1e-4)


def test_make_recording_is_repeatable_within_a_run():
    # Same seed, same process: identical down to the byte, independent of the
    # pinned golden values above (guards reproducibility even if a future re-pin
    # is in flight).
    a = make_recording(seed=3, n_cells=4, duration_s=1.0)
    b = make_recording(seed=3, n_cells=4, duration_s=1.0)
    assert _sha256(a.observed) == _sha256(b.observed)
    assert _sha256(a.ground_truth.C) == _sha256(b.ground_truth.C)

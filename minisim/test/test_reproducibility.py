"""Golden-master reproducibility lock for the public fixture surface.

``minisim.testing.make_recording`` is a *contract*: downstream test suites
(minian, CaImAn, suite2p) assert against the recordings it produces, so a silent
change to its output - new defaults, a tweaked activity model, a recalibrated
detection threshold, an RNG-order change - would flip those assertions red across
every consumer at once, for a reason invisible in their own diff.

These tests pin the byte-exact output of a fixed-seed ``make_recording`` so that
*minisim's own* CI fails the moment that output moves. A failure here is not a bug
to paper over: it means the fixture contract changed. When the change is
intentional, re-pin the hashes in the same commit and call it out in the changelog
(a minor-version bump), so the break is deliberate and announced rather than
silent. See the reproducibility & stability contract in the docs.
"""

import hashlib

import numpy as np

from minisim.testing import make_recording


def _sha256(array) -> str:
    """SHA256 of an array's raw bytes, after a contiguous + dtype-stable view."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# Golden hashes for make_recording(seed=0) at its documented defaults. Re-pin
# these (deliberately, with a changelog note) whenever the fixture output changes.
_GOLDEN_SEED = 0
_GOLDEN_OBSERVED_SHA = "e59c184bf0acf2c9e1c4021bdf1eaeae6fdeb46137956a852e785e35d372eddc"
_GOLDEN_C_SHA = "b765db9c5189653ee845efd08ba8a391cd643b6b5f82d649999de1045d8f8777"
_GOLDEN_CENTERS_SHA = "1789804afad9ecffee2fc27e8e1321d708274618aea5066fa7d62a73f6247dc5"
_GOLDEN_SHAPE = (40, 128, 128)
_GOLDEN_N_UNITS = 6
_GOLDEN_N_DETECTABLE = 6


def test_make_recording_observed_is_byte_stable():
    rec = make_recording(seed=_GOLDEN_SEED)
    assert rec.observed.shape == _GOLDEN_SHAPE
    assert rec.observed.dtype == np.float32
    assert _sha256(rec.observed) == _GOLDEN_OBSERVED_SHA, (
        "make_recording(seed=0).observed changed. If this is intentional, re-pin "
        "the golden hash here and note the fixture change in the changelog."
    )


def test_make_recording_ground_truth_is_byte_stable():
    gt = make_recording(seed=_GOLDEN_SEED).ground_truth
    assert gt.n_units == _GOLDEN_N_UNITS
    assert int(gt.detectable.sum()) == _GOLDEN_N_DETECTABLE
    assert _sha256(gt.C) == _GOLDEN_C_SHA
    assert _sha256(gt.centers_um) == _GOLDEN_CENTERS_SHA


def test_make_recording_is_repeatable_within_a_run():
    # Same seed, same process: identical down to the byte, independent of the
    # pinned golden values above (guards reproducibility even if a future re-pin
    # is in flight).
    a = make_recording(seed=3, n_cells=4, duration_s=1.0)
    b = make_recording(seed=3, n_cells=4, duration_s=1.0)
    assert _sha256(a.observed) == _sha256(b.observed)
    assert _sha256(a.ground_truth.C) == _sha256(b.ground_truth.C)

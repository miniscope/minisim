"""Cell-domain steps: place neurons, then give them calcium activity.

These are the first two steps of the forward pipeline - pure biology, before any
optics or sensor effect:

* :class:`PlaceNeuronsStep` positions neurons in a 3-D µm volume and stamps
  a sharp, pre-optics footprint for each. The *distribution* half (sampling
  centers only, no footprints) is factored into :func:`sample_neurons` so it can
  be reused cheaply for teaching/visualization at full FOV.
* :class:`CellActivityStep` gives every soma a calcium trace built from a
  2-state Markov spike model convolved with a double-exponential indicator
  kernel, plus a per-cell brightness gain that scales the trace.

Both only fill per-cell records on the scene (``scene.cells``); nothing is drawn
into the movie until ``composite`` (:mod:`minisim.steps.tissue`). The
optical degradation that turns the *planted* (sharp) footprint into the
*observed* (blurred, attenuated) one is the ``optics`` step; until it has run,
``composite`` composites the planted footprint directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import oaconvolve

from minisim.footprint import Footprint
from minisim.recording import DETECT_SNR_THRESHOLD, detection_snr
from minisim.scene import Cell, Scene
from minisim.steps.base import PipelineContext, Step
from minisim.steps.tissue import murray_children

if TYPE_CHECKING:
    # CellActivity / CellOptics are referenced only as string Generic bases
    # (Step["CellActivity"]), which ruff's F401 does not count as a use; pyright
    # needs them in scope to resolve those forward references.
    from minisim.spec import (
        Acquisition,
        CellActivity,  # noqa: F401
        CellOptics,  # noqa: F401
        NeuronPopulation,
        Optics,
        PlaceNeurons,
    )

# Guards the noise normalization for a degenerate (flat) low-pass field; far
# below any physically meaningful intensity.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# place_neurons
# ---------------------------------------------------------------------------


# Proximal-dendrite rendering constants (cytosolic morphology only). Dendrites
# are graded dimmer than the soma and taper to a thread, so blur, defocus, and
# the sensor noise floor erase the thin distal structure first - the "we lose thin
# features fast" lesson falls out of the physics for free. Their reach is kept
# *proximal* (bounded by the spec's dendrite length): a far-reaching arbor would
# blow up the sparse-footprint bounding box and merely re-create, in the per-cell
# ground-truth A, the diffuse felt the ``neuropil`` step already models. What the
# proximal arbor *does* add - that ``neuropil`` cannot, being cell-unattached - is a
# lobed, asymmetric *shape* to each cell's blurred footprint.
_DENDRITE_BASE_INTENSITY = 0.55  # planted weight where a dendrite leaves the soma
_DENDRITE_TIP_INTENSITY = 0.05  # ...tapering to this at the distal tip (faint)
_DENDRITE_TIP_WIDTH_PX = 0.5  # minimum stamp radius, keeps the thread continuous
_DENDRITE_WANDER_RAD = 0.16  # per-step heading random walk, radians (gently wavy, not curly)
_DENDRITE_ANGLE_JITTER_RAD = 0.4  # jitter on the evenly spaced launch angles
# The number of primary dendrites is drawn *per cell* (not a spec input), so cells
# differ from one another: a clamped Poisson around this mean.
_DENDRITE_MEAN_COUNT = 5.0
_DENDRITE_COUNT_RANGE = (2, 9)  # clamp the Poisson draw to a plausible range
_DENDRITE_LENGTH_JITTER = (0.6, 1.25)  # per-dendrite multiple of the nominal length
# Bounded bifurcation: a dendrite may branch a few times, a side branch peeling off
# (Murray's-law width split) while the main continues thinner - the realistic arbor
# shape, kept cheap by capping the branch count and the reach.
_DENDRITE_BRANCH_PROB = 0.06  # per-step bifurcation probability
_DENDRITE_MAX_BRANCHES = 2  # cap per primary dendrite (keeps the patch small)
_DENDRITE_BRANCH_AREA_MAIN = 0.62  # main child's share of the Murray cube-sum
_DENDRITE_BRANCH_ANGLE_RAD = 0.7  # heading split at a bifurcation


def neuron_footprint(
    shape: tuple[int, int],
    center_px: tuple[float, float],
    radius_px: float,
    irregularity: float,
    rng: np.random.Generator,
    *,
    morphology: str = "soma",
    dendrite_length_px: float = 0.0,
    dendrite_width_px: float = 0.0,
) -> np.ndarray:
    """A sharp, peak-normalized neuron footprint - the *planted* spatial weight A.

    Models the cell's true (pre-optics) fluorophore support, with **no optical
    blur** - diffraction/defocus/scatter are applied later by the ``optics`` step.
    It is peak-normalized (``max == 1`` at the soma) so a cell's brightness is
    carried entirely by its calcium trace, not baked into the footprint.

    Two GCaMP targeting variants are supported via ``morphology``:

    * ``"soma"`` - soma-targeted GCaMP (e.g. SomaGCaMP / riboGCaMP): a single
      filled, possibly lumpy disk, the soma body only.
    * ``"cytosolic"`` - standard cytosolic GCaMP (GCaMP6/7/8…): the same soma
      disk plus a *random* number of **branched, bounded proximal dendrites**
      (drawn per cell, so cells differ - see :func:`_stamp_dendrites`). The
      dendrites are *graded* (dimmer than the soma), *thin*, and *taper* to a
      thread, so their fine distal structure is exactly what diffraction, defocus,
      scatter, and the sensor noise floor erase first - a faithful demonstration of
      how quickly fine neurites become unresolvable - while the surviving proximal
      arbor gives each cell's blurred footprint a lobed, asymmetric shape.

    The soma is **identical** in both variants; ``"cytosolic"`` only *adds*
    dendrites after the soma is drawn, so ``"soma"`` (the default) reproduces the
    soma-only footprint bit-for-bit.

    ``irregularity`` ∈ [0, 1] warps the soma boundary: at ``0`` it is a clean
    disk; above ``0`` the per-pixel radius is modulated by a low-pass-filtered
    noise field (smoothed on the soma's own length scale), giving a lumpy outline
    that is more soma-like than a perfect circle while staying coarser than the
    optics will later blur away. Typical cortical somata are ~5–10 µm radius.

    Note - the shape is *physical*, the grid is *sampling*. This routine
    rasterizes a continuous µm-space shape onto whatever pixel grid the caller
    passes (via ``shape`` and the ``*_px`` arguments). The cell's true geometry is
    intrinsic and independent of pixel size; only how finely it is sampled depends
    on the sensor. In the normal pipeline the planted footprint is rasterized once
    at the sensor's own scale, which is fine because the result is then blurred by
    the (coarser) optics. But a caller that needs the *same* cell across multiple
    pixel sizes should generate it once on a fixed fine grid and resample, rather
    than re-rasterizing per grid - re-rasterizing re-draws the ``rng`` noise field
    at the new size and so changes the lumpy outline. This costs nothing in
    fidelity: a 1-photon miniscope is pixel-limited, never diffraction-limited
    (see :meth:`Optics.diffraction_sigma_um`), so a sub-pixel reference grid holds
    more detail than the optics can ever resolve.
    """
    h, w = shape
    cy, cx = center_px
    footprint = np.zeros((h, w), dtype=float)
    # The soma is local: it reaches at most radius_px·(1 + irregularity) from the
    # center (the +irregularity headroom covers the lumpy-boundary wobble). Compute
    # the hypot/threshold/noise only inside that bounding box and write it into the
    # full canvas - bit-for-bit the same as filling the whole grid wherever the soma
    # is, just far cheaper than touching every pixel for a cell that occupies a tiny
    # patch. Dendrites are stamped afterwards and self-window (see :func:`_stamp_disk`).
    reach = radius_px * (1.0 + max(irregularity, 0.0)) + 1.0
    y0 = max(int(math.floor(cy - reach)), 0)
    y1 = min(int(math.ceil(cy + reach)) + 1, h)
    x0 = max(int(math.floor(cx - reach)), 0)
    x1 = min(int(math.ceil(cx + reach)) + 1, w)
    if y0 < y1 and x0 < x1:
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(yy - cy, xx - cx)
        if irregularity > 0:
            # Low-pass noise on the soma's own scale → a smoothly lumpy boundary,
            # normalized to ~[-1, 1] so `irregularity` is the fractional radius wobble.
            noise = gaussian_filter(
                rng.standard_normal((y1 - y0, x1 - x0)),
                sigma=max(radius_px / 2.0, 1.0),
            )
            noise /= max(noise.max(), -noise.min()) + _EPS  # scale to ~[-1, 1]
            r_eff = radius_px * (1.0 + irregularity * noise)
        else:
            r_eff = radius_px
        # A 0/1 membership mask is already peak-normalized (max == 1) by construction.
        footprint[y0:y1, x0:x1] = (dist <= r_eff).astype(float)
    # Cytosolic GCaMP fills the proximal dendrites too. Stamp them *after* the
    # soma so the soma's RNG draw above is untouched - "soma" stays bit-identical.
    if morphology == "cytosolic" and dendrite_length_px > 0:
        _stamp_dendrites(footprint, cy, cx, radius_px, dendrite_length_px, dendrite_width_px, rng)
    if not footprint.any():
        # Sub-pixel soma: keep at least the nearest pixel lit so the cell is
        # never silently empty.
        iy = int(np.clip(round(cy), 0, h - 1))
        ix = int(np.clip(round(cx), 0, w - 1))
        footprint[iy, ix] = 1.0
    return footprint


def _stamp_disk(
    footprint: np.ndarray, y: float, x: float, radius_px: float, intensity: float
) -> None:
    """Paint one filled disk into ``footprint`` via ``maximum`` (overlaps never sum).

    Works only on the disk's local bounding box, so laying down a dendrite is
    cheap regardless of canvas size. ``maximum`` keeps the soma's peak at 1 where
    a dendrite overlaps it, preserving peak-normalization.
    """
    h, w = footprint.shape
    y0 = max(int(np.floor(y - radius_px)), 0)
    y1 = min(int(np.ceil(y + radius_px)) + 1, h)
    x0 = max(int(np.floor(x - radius_px)), 0)
    x1 = min(int(np.ceil(x + radius_px)) + 1, w)
    if y0 >= y1 or x0 >= x1:
        return  # disk fell entirely off the canvas
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = ((yy - y) ** 2 + (xx - x) ** 2 <= radius_px**2) * intensity
    sub = footprint[y0:y1, x0:x1]
    np.maximum(sub, disk, out=sub)


def _stamp_dendrites(
    footprint: np.ndarray,
    cy: float,
    cx: float,
    radius_px: float,
    length_px: float,
    width_px: float,
    rng: np.random.Generator,
) -> None:
    """Grow a random number of branched, bounded proximal dendrites out of the soma.

    The **count is drawn per cell** (a clamped Poisson around
    :data:`_DENDRITE_MEAN_COUNT`), not a spec input, so no two cells look alike. Each
    primary dendrite launches from just inside the soma edge at a roughly evenly
    spaced (then jittered) angle and walks outward in ~1 px steps with a small
    per-step heading wobble, so it curves gently rather than spiking out straight.
    Its width and intensity **taper** base→tip, and it may **bifurcate** up to
    :data:`_DENDRITE_MAX_BRANCHES` times - a side branch peels off with a Murray's-law
    width split (:func:`~minisim.steps.tissue.murray_children`) while the main branch
    continues thinner, the realistic arbor shape. ``length_px`` **bounds the reach**
    (with a per-dendrite jitter only), keeping the arbor *proximal*: that is what
    shapes the blurred footprint without exploding the sparse-patch bounding box (a
    far-reaching arbor would just re-create the ``neuropil`` felt in the per-cell A).
    Laid down as a chain of overlapping disks (:func:`_stamp_disk`), so it stays
    continuous and peak-normalized (the soma keeps the peak at 1).
    """
    n = int(np.clip(rng.poisson(_DENDRITE_MEAN_COUNT), *_DENDRITE_COUNT_RANGE))
    # Roughly even angular spread, then jittered, so dendrites don't all clump.
    base = rng.uniform(0.0, 2.0 * np.pi)
    angles = base + np.arange(n) * (2.0 * np.pi / n)
    angles = angles + rng.normal(0.0, _DENDRITE_ANGLE_JITTER_RAD, size=n)
    half_w = 0.5 * width_px  # base stamp radius (width_px is a diameter)
    for theta in angles:
        length = length_px * rng.uniform(*_DENDRITE_LENGTH_JITTER)
        # Start just inside the soma so the dendrite connects without a gap.
        y0 = cy + 0.8 * radius_px * np.sin(theta)
        x0 = cx + 0.8 * radius_px * np.cos(theta)
        # DFS over branches: each entry is one branch's start state
        # (y, x, heading, remaining_length_px, base_width, branches_left).
        stack = [(y0, x0, theta, length, width_px, _DENDRITE_MAX_BRANCHES)]
        while stack:
            y, x, heading, rem, w, branches = stack.pop()
            n_steps = max(int(round(rem)), 2)
            for i in range(n_steps):
                frac = i / (n_steps - 1)  # 0 at this branch's base .. 1 at its tip
                rad = max(0.5 * w * (1.0 - frac), _DENDRITE_TIP_WIDTH_PX)
                # Intensity tracks width (thick→bright, thin→faint), so child branches
                # and distal tips fade out - exactly what optics erase first.
                intensity = _DENDRITE_TIP_INTENSITY + min(rad / half_w, 1.0) * (
                    _DENDRITE_BASE_INTENSITY - _DENDRITE_TIP_INTENSITY
                )
                _stamp_disk(footprint, y, x, rad, intensity)
                heading += rng.normal(0.0, _DENDRITE_WANDER_RAD)
                y += np.sin(heading)
                x += np.cos(heading)
                if branches > 0 and i < n_steps - 2 and rng.random() < _DENDRITE_BRANCH_PROB:
                    w_main, w_side = murray_children(w, _DENDRITE_BRANCH_AREA_MAIN)
                    side = -1.0 if rng.random() < 0.5 else 1.0
                    # Side branch peels off and grows the rest of the way (a little
                    # shorter); the main continues, thinner and bending slightly back.
                    stack.append((
                        y, x, heading + side * _DENDRITE_BRANCH_ANGLE_RAD,
                        (n_steps - 1 - i) * 0.8, w_side, branches - 1,
                    ))
                    heading -= side * 0.5 * _DENDRITE_BRANCH_ANGLE_RAD
                    w = w_main
                    branches -= 1


def sample_neurons(
    spec: PlaceNeurons,
    fov_h_um: float,
    fov_w_um: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Sample the soma *distribution* for a ``PlaceNeurons`` spec over a FOV.

    This is the half of ``place_neurons`` that decides **where cells go** - with no
    footprint stamping, so it is cheap even at a full sensor FOV (the per-cell
    :func:`neuron_footprint` paints are the expensive part). :class:`PlaceNeuronsStep`
    samples the same centers (every population, in order) and then stamps a footprint
    per center; teaching code (the anatomy notebook's placement widget) can call this
    directly to show the population layout without paying for footprints.

    Each of the spec's :attr:`~minisim.spec.PlaceNeurons.resolved_populations` is
    sampled in turn and the centers concatenated - so a layered spec (a thin band
    plus a deep volume) returns every layer's cells. Spacing (``min_distance_um``) is
    enforced *within* a population, not across them, so distinct layers may
    interpenetrate at their depth boundary (the physical case for adjacent layers).

    Placement is **purely spatial**: brightness is no longer drawn here. Per-cell
    response gain (how bright a cell is) is biology that belongs to the calcium
    response, so it is drawn in :class:`CellActivityStep` (``brightness_cv``); SNR is
    an emergent measurement property and is not an input anywhere.

    Returns ``centers``: a list of ``(z, y, x)`` µm tuples.
    """
    centers: list[tuple[float, float, float]] = []
    for pop in spec.resolved_populations:
        centers.extend(_sample_population(pop, fov_h_um, fov_w_um, rng))
    return centers


def _sample_population(
    pop: NeuronPopulation,
    fov_h_um: float,
    fov_w_um: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Sample one population's soma centers over the FOV: a list of ``(z, y, x)`` µm.

    When the population gives explicit ``positions_um`` those exact centers are
    returned verbatim (consuming no ``rng``), and the distribution fields are
    ignored. Otherwise the count is **volumetric**:
    ``round(density_per_mm3 · area_mm2 · thickness)``, where the slab thickness is
    ``depth_range_um`` width **floored at one soma diameter** (``2 · soma_radius_um``)
    so a thin - or strictly planar (``lo == hi``) - layer still yields cells rather
    than zero. A thicker slab therefore holds proportionally more cells, the physical
    behavior. Centers are drawn uniformly in ``(y, x)`` across the FOV and in ``z``
    across ``depth_range_um``; if ``min_distance_um > 0`` they are rejection-sampled
    (Poisson-disk style) to a 3-D center-to-center minimum within this population.
    """
    if pop.positions_um is not None:
        return list(pop.positions_um)  # exact centers; no sampling, no rng draw
    area_mm2 = (fov_h_um / 1000.0) * (fov_w_um / 1000.0)
    lo, hi = pop.depth_range_um
    thickness_um = max(hi - lo, 2.0 * pop.soma_radius_um)  # floor: one soma diameter
    count = round(pop.density_per_mm3 * area_mm2 * (thickness_um / 1000.0))
    return _sample_centers(
        count, fov_h_um, fov_w_um, pop.depth_range_um, pop.min_distance_um, rng
    )


def _sample_centers(
    count: int,
    fov_h_um: float,
    fov_w_um: float,
    depth_range_um: tuple[float, float],
    min_distance_um: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    z_lo, z_hi = depth_range_um

    def draw() -> tuple[float, float, float]:
        return (
            rng.uniform(z_lo, z_hi),
            rng.uniform(-fov_h_um / 2.0, fov_h_um / 2.0),
            rng.uniform(-fov_w_um / 2.0, fov_w_um / 2.0),
        )

    if min_distance_um <= 0:
        return [draw() for _ in range(count)]

    # Poisson-disk-style rejection sampling. Capped so an over-dense request
    # ends with fewer cells rather than looping forever (an honest outcome:
    # you cannot pack more than the minimum spacing allows).
    centers: list[tuple[float, float, float]] = []
    attempts, max_attempts = 0, max(1000, 100 * count)
    while len(centers) < count and attempts < max_attempts:
        attempts += 1
        cand = draw()
        if all(math.dist(cand, c) >= min_distance_um for c in centers):
            centers.append(cand)
    return centers


def _sample_brightness(cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-cell expression/response gain: lognormal, **mean 1**, given CV.

    ``cv == 0`` makes every cell equally bright (gain 1). Otherwise the gain is
    lognormal with mean exactly 1 (so changing the spread does not change the
    population's total light budget) and a right tail, the familiar "a few bright
    cells over a dimmer majority".
    """
    if n == 0:
        return np.empty(0)
    if cv <= 0:
        return np.ones(n)
    sigma = np.sqrt(np.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma  # makes E[gain] == 1
    return np.exp(rng.normal(mu, sigma, size=n))


class PlaceNeuronsStep(Step["PlaceNeurons"]):
    """Position neurons in a 3-D µm volume and stamp a planted footprint each.

    Placement (*where cells go*) is delegated to :func:`sample_neurons`; this step
    adds the part that function omits - stamping a peak-normalized planted footprint
    (:func:`neuron_footprint`, soma-only or soma + proximal dendrites per
    ``spec.morphology``) at each sampled center. Placement is purely spatial now:
    per-cell brightness is drawn later in ``cell_activity``, not here.

    The cell count is volumetric - ``round(density_per_mm3 · canvas_area_mm2 ·
    thickness)`` over the **canvas** area (the scene movie's grid, which a motion
    margin may enlarge beyond the sensor FOV) and the ``depth_range_um`` thickness
    (floored at one soma diameter); see :func:`sample_neurons`. The per-cell depth
    is consumed later by the ``optics`` step for the ``in_focus`` /
    ``detectable`` flags.
    """

    name = "place_neurons"
    domain = "cell"
    consumes_rng = True  # samples cell positions and the irregular footprint noise

    def __call__(self, scene: Scene) -> None:
        spec = self.spec
        acq, rng = self.acq, self.rng
        # Fill whatever canvas the scene movie defines, not the bare sensor: a
        # motion margin enlarges the canvas beyond the sensor FOV so
        # that real, simulated tissue moves into view at the edges. At margin 0
        # the canvas equals the sensor FOV. Cell positions are in the
        # optical-center frame (origin = optical axis, +y down, +x right), which is
        # invariant to the motion margin; um_to_index maps them onto the canvas grid.
        shape = scene.canvas_shape  # (height, width) of the canvas, no movie alloc
        fov_h_um = shape[0] * acq.pixel_size_um
        fov_w_um = shape[1] * acq.pixel_size_um

        # Sample every population's centers *first* (tagging each with its source
        # population), then stamp - so sample_neurons() reproduces this exact
        # placement (stamping consumes rng, so interleaving sampling with it would
        # desync the two), and a single-population spec stays identical to the flat
        # form.
        planned: list[tuple[tuple[float, float, float], NeuronPopulation]] = [
            (center, pop)
            for pop in spec.resolved_populations
            for center in _sample_population(pop, fov_h_um, fov_w_um, rng)
        ]
        for (z, y, x), pop in planned:
            # Rasterize the cell onto the canvas grid, then keep only its non-zero
            # patch: a soma + neurites cover a tiny window of a frame that may be a
            # full sensor FOV, so storing the dense canvas array would be ~98%
            # zeros. The dense grid here is transient (one cell, freed at once); the
            # Cell holds the trimmed Footprint. See :mod:`minisim.footprint`.
            footprint = Footprint.from_dense(
                neuron_footprint(
                    shape,
                    acq.um_to_index(y, x, shape),
                    acq.um_to_px(pop.soma_radius_um),
                    pop.irregularity,
                    rng,
                    morphology=pop.morphology,
                    dendrite_length_px=acq.um_to_px(pop.dendrite_length_um),
                    dendrite_width_px=acq.um_to_px(pop.dendrite_width_um),
                )
            )
            scene.cells.append(
                Cell(center_um=(z, y, x), footprint_planted=footprint)
            )


# ---------------------------------------------------------------------------
# cell_activity
# ---------------------------------------------------------------------------


def calcium_kernel(tau_rise_s: float, tau_decay_s: float, fps: float) -> np.ndarray:
    """Double-exponential calcium-indicator kernel, sampled at the frame rate.

    ``k(t) = exp(-t/τ_decay) − exp(-t/τ_rise)`` - the canonical CaLab-style
    impulse response of a fluorescent calcium indicator: a fast rise (``τ_rise``)
    onto a slow decay (``τ_decay``). Sampled at ``1/fps`` intervals out to
    ``5·τ_decay`` (where the response has decayed to <1%) and peak-normalized, so
    convolving it with an amplitude-weighted spike train yields a ΔF trace whose
    per-spike height is the spike amplitude. Requires ``τ_rise < τ_decay`` (a
    rise slower than the decay is not a physical indicator response). Typical
    GCaMP: ``τ_rise`` ~0.05 s, ``τ_decay`` ~0.3–0.7 s.
    """
    if tau_rise_s >= tau_decay_s:
        raise ValueError(
            f"tau_rise_s ({tau_rise_s}) must be < tau_decay_s ({tau_decay_s}) "
            "for a double-exponential indicator kernel."
        )
    length = max(int(np.ceil(tau_decay_s * 5.0 * fps)), 2)
    t = np.arange(length) / fps
    k = np.exp(-t / tau_decay_s) - np.exp(-t / tau_rise_s)
    return k / k.max()


# --- kernel timing: (time-to-peak, FWHM) <-> (tau_rise, tau_decay) ----------
#
# The double-exp kernel g(t) = e^{-t/τd} - e^{-t/τr} has a useful scale property:
# write r = τr/τd and t = τd·u, and g(τd·u) = e^{-u} - e^{-u/r} depends on r alone.
# So both the time-to-peak and the FWHM are τd times a function of r only, and their
# *ratio* depends only on r. That turns the two-feature inversion into a single
# robust 1-D root-find on r, then τd = t_peak / peak_u(r) and τr = r·τd. Users think
# in observable kernel features (how fast it rises, how wide it is); the simulator
# keeps physical time constants. These two helpers bridge the two.


def _peak_u(r: float) -> float:
    """Location (in u = t/τd units) of the kernel peak, for r = τr/τd ∈ (0, 1)."""
    return r * math.log(r) / (r - 1.0)


def _shape_u(u: float, r: float) -> float:
    return math.exp(-u) - math.exp(-u / r)


def _half_max_u(r: float) -> tuple[float, float]:
    """The two u where the kernel is half its peak (u_lo < u_peak < u_hi)."""
    from scipy.optimize import brentq

    up = _peak_u(r)
    half = _shape_u(up, r) / 2.0
    # brentq returns a float here (full_output defaults to False); scipy types it
    # as a float|tuple union, so cast to keep the arithmetic below typed.
    u_lo = cast("float", brentq(lambda u: _shape_u(u, r) - half, 1e-12, up))
    hi = up
    while _shape_u(hi, r) > half:  # grow until the decay falls below half max
        hi *= 2.0
    u_hi = cast("float", brentq(lambda u: _shape_u(u, r) - half, up, hi))
    return u_lo, u_hi


def kernel_timing(tau_rise_s: float, tau_decay_s: float) -> tuple[float, float]:
    """``(time_to_peak_s, fwhm_s)`` of the double-exp kernel for given time constants.

    The forward direction of :func:`tau_from_kernel_timing`. Requires
    ``tau_rise_s < tau_decay_s``.
    """
    if tau_rise_s >= tau_decay_s:
        raise ValueError(
            f"tau_rise_s ({tau_rise_s}) must be < tau_decay_s ({tau_decay_s})."
        )
    r = tau_rise_s / tau_decay_s
    u_lo, u_hi = _half_max_u(r)
    return _peak_u(r) * tau_decay_s, (u_hi - u_lo) * tau_decay_s


# Achievable ratio range t_peak/FWHM, over r in (0, 1): -> 0 as r -> 0 (instant rise,
# FWHM -> ln2·τd), -> ~0.409 as r -> 1 (the alpha-function limit). Targets outside
# this are physically impossible for this kernel family, so we clamp into it.
_R_LO, _R_HI = 1e-4, 1.0 - 1e-6


def tau_from_kernel_timing(t_peak_s: float, fwhm_s: float) -> tuple[float, float]:
    """Invert observable kernel features to ``(tau_rise_s, tau_decay_s)``.

    ``t_peak_s`` is the time from a spike to the kernel's peak; ``fwhm_s`` is the
    kernel's full width at half maximum. Because the peak/FWHM ratio depends only on
    ``r = tau_rise/tau_decay``, this solves a single 1-D root-find for ``r`` and then
    recovers ``tau_decay = t_peak / peak_u(r)`` and ``tau_rise = r·tau_decay``. A
    ratio outside the kernel's achievable range (roughly ``0 < t_peak/fwhm < 0.41``)
    is clamped to the nearest feasible shape.
    """
    from scipy.optimize import brentq

    def ratio(r: float) -> float:
        u_lo, u_hi = _half_max_u(r)
        return _peak_u(r) / (u_hi - u_lo)

    rho = t_peak_s / fwhm_s
    lo, hi = ratio(_R_LO), ratio(_R_HI)
    if rho <= lo:
        r = _R_LO
    elif rho >= hi:
        r = _R_HI
    else:
        r = cast("float", brentq(lambda rr: ratio(rr) - rho, _R_LO, _R_HI))
    tau_decay = t_peak_s / _peak_u(r)
    return r * tau_decay, tau_decay


# CaLab's SPIKE_ACTIVITY_LEVELS (sparse, moderate, dense), expressed in our units:
# (p_quiescent_to_active /frame, p_active_to_quiescent /frame, active_rate_hz,
# quiescent_rate_hz). The Hz values are CaLab's per-bin spike probs x its 300 Hz sim
# rate (0.3/0.5/0.7 -> 90/150/210; 0.001/0.002/0.005 -> 0.3/0.6/1.5). Going denser
# couples three things: bursts start more often (p_q2a up), last longer (p_a2q down),
# and fire harder (active_rate up) -- the path that keeps the look realistic.
# Attribution: CaLab web simulator (simulation-quality-presets.ts).
_ACTIVITY_SPARSE = (0.002, 0.4, 90.0, 0.3)
_ACTIVITY_MODERATE = (0.005, 0.3, 150.0, 0.6)
_ACTIVITY_DENSE = (0.01, 0.2, 210.0, 1.5)


def spike_activity_params(activity: float) -> tuple[float, float, float, float]:
    """Map a single ``activity`` ∈ [0, 1] onto CaLab's sparse→moderate→dense path.

    Returns ``(p_quiescent_to_active, p_active_to_quiescent, active_rate_hz,
    quiescent_rate_hz)`` for :class:`~minisim.spec.CellActivity`. ``0`` is
    sparse, ``0.5`` is moderate (CaLab's default, the screenshot regime), ``1`` is
    dense; values interpolate piecewise-linearly through CaLab's three
    ``SPIKE_ACTIVITY_LEVELS`` and are clamped to ``[0, 1]``.

    This is the whole spike-activity control: CaLab moves all four Markov parameters
    together along this single density axis (it has no separate rate/burstiness
    knobs), and that coupling is what keeps firing realistic, dense bursts plus a
    little background, with no dead stretches, across the entire range.
    """
    a = min(max(activity, 0.0), 1.0)
    if a <= 0.5:
        t, lo, hi = a / 0.5, _ACTIVITY_SPARSE, _ACTIVITY_MODERATE
    else:
        t, lo, hi = (a - 0.5) / 0.5, _ACTIVITY_MODERATE, _ACTIVITY_DENSE
    p_q2a, p_a2q, active_rate_hz, quiescent_rate_hz = (
        lo_i + t * (hi_i - lo_i) for lo_i, hi_i in zip(lo, hi, strict=True)
    )
    return p_q2a, p_a2q, active_rate_hz, quiescent_rate_hz


class CellActivityStep(Step["CellActivity"]):
    """Give each soma a calcium trace, the CaLab way: 300 Hz spikes → kernel → bin.

    Spikes are generated on a **high-resolution grid** (``spike_sim_hz``, ~300 Hz)
    rather than at the camera frame rate, then convolved with the calcium kernel at
    that fine rate and **bin-averaged down** to the imaging rate (which is what the
    camera's exposure integration physically does). Two payoffs: one spike per fine
    bin is at most ~3 ms apart, so the **refractory period** is respected and bursts
    cannot pack unphysically tight; and sub-frame spike timing survives the kernel,
    which matters for fast indicators. (Attribution: this matches the CaLab web
    simulator's spike model.)

    Per cell, a 2-state Markov gate (quiescent ↔ active) modulates the per-bin spike
    probability between ``quiescent_rate_hz`` and ``active_rate_hz`` (each ÷
    ``spike_sim_hz``). Bursting comes from a **high in-active rate** concentrated into
    short active bouts, not from a high mean rate; :func:`spike_activity_params` maps a
    single ``activity`` density axis onto the gate's four Markov parameters. The gate itself is stepped at the
    frame rate (bouts are ~seconds, so sub-frame onset timing is irrelevant once we
    bin); the spikes it gates are the part that runs at ``spike_sim_hz``.

    Amplitude is biology and enters as a single **per-cell** expression/response gain
    (lognormal spread ``brightness_cv``, mean 1) that scales each cell's *whole* trace
    - baseline and transients together, so a bright cell is brighter everywhere. No
    measurement noise is added here: the trace is the clean ground-truth ``C``;
    shot/read noise enter at the ``sensor`` and background at ``neuropil``, so SNR is
    emergent, never set here.

    Writes ``cell.trace`` (the calcium trace ``C``), ``cell.spikes`` (the per-frame
    spike *count* train ``S`` - the fine 300 Hz train is binned away, since nothing
    downstream recovers spikes faster than the frame rate), and ``cell.amplitude``
    (the per-cell gain); these are the ideal deconvolution targets in ground truth.
    """

    name = "cell_activity"
    domain = "cell"
    consumes_rng = True  # Markov gate, spike draws, per-cell brightness

    def __call__(self, scene: Scene) -> None:
        spec = self.spec
        n_frames, fps = self.acq.n_frames, self.acq.fps
        bins = max(int(round(spec.spike_sim_hz / fps)), 1)  # fine bins per frame
        hr_fps = bins * fps  # realized high-res rate (integer multiple of fps)
        kernel = calcium_kernel(spec.tau_rise_s, spec.tau_decay_s, hr_fps)
        # Burn-in: generate a lead-in of ~6 decay constants and trim it away. A raw
        # convolution opens cold (no spikes before frame 0), so the trace would ramp
        # from baseline up to its stationary level over a few tau_decay; the lead-in
        # gives frame 0 a realistic history of already-decaying tails instead. Six
        # tau_decay leaves <0.3% of a spike's tail unrepresented.
        pad = int(np.ceil(6.0 * spec.tau_decay_s * fps))
        n_total = n_frames + pad
        gains = _sample_brightness(spec.brightness_cv, len(scene.cells), self.rng)
        # The 2-state gate is the same Markov chain for every cell, so step them all
        # together: one O(n_total) time-loop over a per-cell state vector, instead of
        # a Python loop per cell. The fine spikes the gate modulates, and the calcium
        # convolution, stay per-cell -- each is a long (n_total*bins) array, so
        # batching them would trade back the memory we just stopped wasting on the
        # movie for little speed.
        states = self._gate_states(spec, len(scene.cells), n_total, self.rng)
        for cell, gain, state in zip(scene.cells, gains, states, strict=True):
            fine = self._fine_spikes(state, bins, hr_fps, spec, self.rng)
            # Short kernel vs a long fine train -> overlap-add convolution (O(n log k))
            # beats a full-length FFT (fftconvolve, O(n log n)) and allocates less.
            calcium = oaconvolve(fine, kernel)[: fine.size]
            # Bin-average to the imaging rate (camera exposure integration), then drop
            # the burn-in lead-in. The per-cell gain scales the whole trace (baseline +
            # transients), so a bright cell emits more light everywhere -- a higher
            # emergent SNR later.
            clean = calcium.reshape(n_total, bins).mean(axis=1)[pad:]
            trace = gain * (spec.f0 + clean)
            if spec.trace_noise > 0:
                trace = trace + self.rng.normal(0.0, spec.trace_noise, size=n_frames)
            cell.trace = trace
            cell.spikes = fine.reshape(n_total, bins).sum(axis=1)[pad:]  # per-frame counts
            cell.amplitude = float(gain)

    @staticmethod
    def _gate_states(
        spec, n_cells: int, n_frames: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Per-cell quiescent↔active gate for all cells at once: ``(n_cells, n_frames)``.

        The 2-state Markov chain is identical across cells, so one time-loop over a
        ``(n_cells,)`` boolean ``active`` vector replaces the per-cell Python loop.
        Each frame draws **one** uniform per cell and flips it with the same rule the
        scalar chain used (an active cell stays active unless it fires the a→q
        transition; a quiescent cell flips on q→a), so the draw count per cell is
        unchanged. Gates start in the **stationary** active fraction, so a recording
        does not always open with a quiet stretch.
        """
        p_q2a, p_a2q = spec.p_quiescent_to_active, spec.p_active_to_quiescent
        f_stationary = p_q2a / (p_q2a + p_a2q)
        active = rng.random(n_cells) < f_stationary
        states = np.empty((n_cells, n_frames), dtype=bool)
        for f in range(n_frames):
            r = rng.random(n_cells)
            active = np.where(active, r >= p_a2q, r < p_q2a)
            states[:, f] = active
        return states

    @staticmethod
    def _fine_spikes(
        state: np.ndarray, bins: int, hr_fps: float, spec, rng: np.random.Generator
    ) -> np.ndarray:
        """Binary spike train on the high-res grid for one cell (length ``state.size * bins``).

        Expands the cell's per-frame gate ``state`` to the fine grid and draws a
        Bernoulli per ~3 ms bin at ``rate / hr_fps`` (``active`` vs ``quiescent``
        rate). One spike per fine bin enforces the refractory period for free, and
        the sub-frame spike timing survives into the calcium kernel.
        """
        rate = np.where(np.repeat(state, bins), spec.active_rate_hz, spec.quiescent_rate_hz)
        p_spike = np.minimum(rate / hr_fps, 1.0)
        return (rng.random(state.size * bins) < p_spike).astype(float)


# ---------------------------------------------------------------------------
# optics
# ---------------------------------------------------------------------------


# Candidate focal planes scanned by the yield-maximizing "auto" focus. Over a
# realistic ~200 µm imaging slab this is ~2 µm spacing - well under the depth of
# field (≈ 8 µm at NA 0.3), so the optimum is sampled finely enough.
_FOCUS_SCAN_N = 96


def resolve_focal_plane(
    cells: list[Cell],
    focal_depth_in_tissue_um: float | str,
    optics: Optics | None = None,
    *,
    acq: Acquisition | None = None,
    sensor_spec=None,
    photon_field: np.ndarray | None = None,
) -> float:
    """Resolve ``Acquisition.focal_depth_in_tissue_um`` to a concrete focal depth, µm.

    A numeric value is the focal depth below the surface as-is. ``"auto"`` chooses
    the plane that maximizes **recoverable-cell yield** - the count of cells whose
    realized transient clears the sensor detection floor - which is what an
    experimenter actually focuses for, not the sharpest *average* cell.

    Each cell at field radius ``r`` focuses at ``focal_eff = focal − shift(r)``
    (off-axis cells focus shallower under field curvature), so defocus is
    ``σ = NA·|z − focal_eff| = NA·|e − focal|`` in the cell's **effective depth**
    ``e = z + shift(r)``. The radius is ``r = hypot(y, x)`` straight off the cell's
    optical-center coordinates (the optical axis is the frame origin). For a
    candidate focal the per-cell detection SNR folds in the defocus peak-drop,
    scatter attenuation, the illumination × vignette photon budget at the cell's
    lateral position, and the shot + read noise floor (the same
    :func:`~minisim.recording.detection_snr` ``finalize`` uses). The scan picks the
    focal with the most detectable cells, ties broken by total in-focus signal.
    Because scatter dims deep cells and the falloff fields dim edge cells, this
    lands shallower / re-centered versus a naive blur optimum - and shifts once you
    account for vignetting, which is the point.

    The yield scan needs both ``acq`` (for the optics/sensor physics) and a
    ``sensor_spec`` (the noise floor that defines "recoverable"); ``photon_field``
    is the optional pre-built illumination × vignette product (FOV-sized, sampled
    at each cell's optical-center position). **Without a sensor**
    there is no floor, so ``"auto"`` falls back to the geometric optimum: the focal
    that minimizes total defocus ``Σ NA·|e − focal|`` is the **median effective
    depth** (median ``z`` when no curvature info is supplied). An empty scene falls
    back to the surface (``0.0``). This is the one place ``"auto"`` becomes
    concrete; every downstream read sees a number.
    """
    if focal_depth_in_tissue_um != "auto":
        return float(focal_depth_in_tissue_um)
    if not cells:
        return 0.0
    depths = np.array([cell.center_um[0] for cell in cells], dtype=float)
    if optics is not None and optics.field_curvature_radius_um is not None:
        # Field radius is the distance from the optical axis = the frame origin.
        shifts = np.array(
            [
                optics.focal_curvature_shift_um(math.hypot(cell.center_um[1], cell.center_um[2]))
                for cell in cells
            ],
            dtype=float,
        )
    else:
        shifts = np.zeros_like(depths)
    effective = depths + shifts

    # No sensor -> no noise floor to define "recoverable": fall back to the focal
    # that minimizes total defocus blur, i.e. the median effective depth.
    if acq is None or sensor_spec is None:
        return float(np.median(effective))
    scored = [i for i, cell in enumerate(cells) if cell.trace is not None]
    if not scored:
        return float(np.median(effective))  # nothing to detect -> geometric optimum
    return _max_yield_focus(cells, scored, effective, acq, sensor_spec, photon_field)


def _max_yield_focus(
    cells: list[Cell],
    scored: list[int],
    effective: np.ndarray,
    acq: Acquisition,
    sensor_spec,
    photon_field: np.ndarray | None,
) -> float:
    """Scan candidate focal planes; return the one maximizing detectable-cell yield.

    Only cells with a calcium trace (indices ``scored``) can be detected. For each
    candidate focal ``F`` the per-cell detection SNR is computed fully vectorized
    over (candidates × cells): the defocus peak-drop ``σ₀²/(σ₀² + (NA·(e−F))²)``
    times the focal-independent gain (scatter attenuation × NA² collection ×
    illumination × exposure × QE), against the shot + read floor. A cell counts when
    it clears ``DETECT_SNR_THRESHOLD`` - there is no separate depth-of-field gate, so
    the objective matches the realized ``detectable`` flag (see
    :func:`~minisim.recording._is_detectable`): defocus dims a cell through the SNR
    rather than excluding it at a hard cutoff. Ties in count are broken by total
    detected signal so the plane sits where the detectable cells are brightest.

    Vessel occlusion is **not** folded in here: vasculature is a later tissue-domain
    effect, not yet grown when the focus is chosen. So a vessel over a soma can drop
    that cell from the realized ``detectable`` set (see ``_is_detectable``) without
    shifting the chosen plane - a deliberate v1 simplification.
    """
    optics, tissue = acq.optics, acq.tissue
    eff = effective[scored]
    z = np.array([cells[i].center_um[0] for i in scored], dtype=float)
    # Focal-independent per-cell physics (σ₀² and the gain that is not the defocus
    # peak-drop). σ₀ is diffraction + scatter blur; gain folds in everything that
    # dims a cell regardless of focus.
    sigma0_sq = optics.diffraction_sigma_um**2 + np.array(
        [tissue.scatter_sigma_um(zi) ** 2 for zi in z]
    )
    atten = np.array([tissue.attenuation(zi) for zi in z])
    illum = _photon_budget_at(cells, scored, photon_field, acq)
    qe = acq.image_sensor.quantum_efficiency
    read_e = acq.image_sensor.read_noise_e
    ppu = float(sensor_spec.photons_per_unit)
    gain_const = atten * optics.collection_efficiency * illum * ppu * qe
    peak_dF_list, baseline_list = [], []
    for i in scored:
        trace = cells[i].trace
        assert trace is not None  # scored cells are activity-bearing (have a trace)
        peak_dF_list.append(float(trace.max() - trace.min()))
        baseline_list.append(float(trace.min()))
    peak_dF = np.array(peak_dF_list)
    baseline = np.array(baseline_list)

    candidates = np.linspace(eff.min(), eff.max(), _FOCUS_SCAN_N)
    d = eff[None, :] - candidates[:, None]  # (n_candidates, n_cells)
    peak_drop = sigma0_sq / (sigma0_sq + (optics.na * d) ** 2)
    gain = gain_const * peak_drop
    snr = detection_snr(peak_dF, baseline, gain, read_e)
    detectable = snr >= DETECT_SNR_THRESHOLD
    yields = detectable.sum(axis=1)
    signals = np.where(detectable, peak_dF * gain, 0.0).sum(axis=1)

    best_yield = yields.max()
    tied = np.flatnonzero(yields == best_yield)
    best = tied[np.argmax(signals[tied])]
    return float(candidates[best])


def _photon_budget_at(
    cells: list[Cell],
    scored: list[int],
    photon_field: np.ndarray | None,
    acq: Acquisition,
) -> np.ndarray:
    """Illumination × vignette factor at each scored cell's sensor-FOV position.

    All ones when no field was supplied. The field is the sensor FOV; cells live in
    the optical-center frame whose origin is the FOV center, so each cell maps to a
    FOV pixel via :meth:`~minisim.Acquisition.um_to_index` (no margin bookkeeping).
    """
    if photon_field is None:
        return np.ones(len(scored))
    h, w = photon_field.shape
    rows, cols = [], []
    for i in scored:
        r, c = acq.um_to_index(cells[i].center_um[1], cells[i].center_um[2], (h, w))
        rows.append(r)
        cols.append(c)
    iy = np.clip(np.round(rows), 0, h - 1).astype(int)
    ix = np.clip(np.round(cols), 0, w - 1).astype(int)
    return photon_field[iy, ix]


class CellOpticsStep(Step["CellOptics"]):
    """Degrade each planted footprint by diffraction + defocus(|z−focal|) + scatter(z).

    Reads each cell's depth ``z`` and the physical ``Optics``/``Tissue``
    constants (via :meth:`Acquisition.cell_optics`) - there are no tunable
    fields. For every cell it:

    * stores the two scalars that define the observed footprint -- ``sigma_px``
      (the total PSF width) and ``gain = attenuation(z) · collection_efficiency``
      (the flat light-loss) -- as ``observed_sigma_px`` / ``observed_gain``. The
      observed footprint ``gain · (planted ⊛ Gaussian(σ_total))`` -- the blurred,
      dimmed footprint CNMF could recover -- is **not stored**: it is a pure
      function of those scalars and the planted footprint, so ``composite`` and
      ``GroundTruth.A_observed`` regenerate it on demand (deep cells' observed
      footprints are near-full-canvas, so storing them dominated memory and disk);
    * sets ``in_focus`` geometrically (``|z − focal_eff| ≤`` the NA-derived depth
      of field), where ``focal_eff`` includes the field-curvature shift;
    * stores ``optical_brightness`` - the per-cell *peak* scalar from
      ``cell_optics`` (defocus drops the peak as ``σ₀²/σ_total²``; scatter
      ``attenuation(z)`` and ``collection_efficiency ∝ NA²`` dim it). Footprint
      *integral* scales with that same ``gain``, but a cell's *detectability*
      turns on its peak, which defocus also lowers - hence two distinct
      quantities. ``detectable`` itself is left for ``finalize()``, where this
      peak combines with the illumination field and the sensor noise floor.

    The *central* focal plane is resolved once for the whole scene from
    ``Acquisition.focal_depth_in_tissue_um`` (``"auto"`` → the focus that minimizes
    total defocus, i.e. the median curvature-corrected effective depth; see
    :func:`resolve_focal_plane`). When ``Optics.field_curvature_radius_um`` is set,
    each cell's effective focal depth is that plane minus the field-curvature
    sagitta at its distance from the optical axis (the frame origin), so off-axis cells
    focus shallower and blur out toward the edges - the sharp-center/soft-edge look
    of an un-flattened miniscope. Cells without a planted footprint are skipped.
    """

    name = "optics"
    domain = "cell"

    def __init__(self, spec, acq, rng) -> None:
        super().__init__(spec, acq, rng)
        # Pulled from the PipelineContext in prepare() so "auto" focus can choose
        # the plane that maximizes detectable yield: the sensor spec is the noise
        # floor, the photon field is the (FOV-sized) illumination × vignette
        # product. Absent (no sensor step), focus falls back to the geometric
        # defocus optimum.
        self.sensor_spec = None
        self.photon_field = None

    def prepare(self, context: PipelineContext) -> None:
        self.sensor_spec = context.sensor_spec
        self.photon_field = context.photon_field

    def __call__(self, scene: Scene) -> None:
        acq = self.acq
        # Optical axis = the optical-center frame origin (0, 0), so a cell's field
        # radius is just hypot(y, x). Off-axis cells focus shallower by the
        # field-curvature sagitta, so each cell sees its own focal depth (no
        # footprint warping: the curvature over one soma is negligible vs the ~mm
        # curvature radius). "auto" focus maximizes recoverable yield: pass the
        # optics (so it folds in field curvature) and the sensor/photon-field
        # context (so it weights cells by the brightness their image actually gets).
        focal = resolve_focal_plane(
            scene.cells,
            acq.focal_depth_in_tissue_um,
            acq.optics,
            acq=acq,
            sensor_spec=self.sensor_spec,
            photon_field=self.photon_field,
        )
        scene.truth.focal_depth_um = focal  # the resolved "auto" plane, made observable
        dof = acq.optics.resolved_depth_of_field_um
        for cell in scene.cells:
            if cell.footprint_planted is None:
                continue
            z = cell.center_um[0]
            r = math.hypot(cell.center_um[1], cell.center_um[2])
            focal_eff = focal - acq.optics.focal_curvature_shift_um(r)
            sigma_px, brightness = acq.cell_optics(z, focal_eff)
            # Keep only the two scalars that define the observed footprint; render
            # and GroundTruth.A_observed regenerate it via degrade_footprint.
            cell.observed_sigma_px = sigma_px
            cell.observed_gain = acq.tissue.attenuation(z) * acq.optics.collection_efficiency
            cell.in_focus = abs(z - focal_eff) <= dof
            cell.optical_brightness = brightness

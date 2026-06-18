"""Structural metrics - the shared oracle comparing estimates to ground truth.

Standalone and dependency-light: the unit tests, the parameter-matrix tests, and
both training notebooks all score CNMF output against a :class:`GroundTruth`
through this one module. It knows nothing about tests or thresholds - callers
supply those (and the fair recall denominator, ``GroundTruth.detectable_subset``).

Every function takes unit-first arrays whose dim order matches *both* sides of the
comparison: CNMF emits ``A`` as ``(unit_id, height, width)`` and ``C``/``S`` as
``(unit_id, frame)``, exactly the layout of ``GroundTruth.A_observed``/``C``/``S``.
Inputs pass through :func:`numpy.asarray`, so ``xr.DataArray`` (what minian's CNMF
returns) and plain ``ndarray`` are both accepted.

The pipeline is: :func:`hungarian_match` pairs estimated footprints to true ones
by spatial overlap, then :func:`trace_pearson` / :func:`activity_similarity` score
the temporal recovery of those pairs. :func:`shift_rmse` and :func:`field_pearson`
score the per-effect ground truth (motion, vignette/leakage).

**Two pitfalls this module is built to absorb.**

*Pixel weights matter.* A footprint is not a flat blob: the bright core carries
most of a cell's identity. :func:`hungarian_match` therefore offers weighted
similarities (``"cosine"``, ``"weighted_jaccard"``) that compare the *intensity
profile*, not only a binary mask (``"iou"``), so a footprint that overlaps the
right pixels but smears its weight is not scored the same as a tight match.

*A global frame offset is not a recovery error.* After motion correction the
estimated footprints live in the pipeline's template frame, which can sit a few
pixels off minisim's zero-shift reference - and the offset differs between
pipelines (rigid vs non-rigid, different templates). :func:`hungarian_match`
accepts a known global ``shift`` (derive it from the motion trajectories) or
estimates one (``shift="auto"``); :func:`shift_rmse` can ``align`` away the same
constant offset. Both keep a uniform translation from masquerading as a miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Default fraction of each footprint's energy retained when binarizing it for IoU
# (CaImAn-style): the mask is the *smallest* set of highest-value pixels whose
# summed intensity reaches this fraction of the footprint's total. 0.9 keeps the
# bright core and discards the low-intensity skirt that blur/scatter spread out.
DEFAULT_ENERGY_FRAC = 0.9

# Default intensity-relative threshold for a *naive analyst* footprint ROI
# (:func:`footprint_mask`): the visible bright extent, pixels above this fraction
# of the footprint peak. This is deliberately distinct from DEFAULT_ENERGY_FRAC -
# energy masks are for IoU *scoring* (the smallest core holding 90% of the
# energy), whereas this is the rough region of interest a person would draw by eye
# to read out a trace, the un-demixed baseline the demixing comparison is against.
DEFAULT_ROI_REL_THRESHOLD = 0.18

# The footprint similarities :func:`hungarian_match` can pair and score with.
# ``"iou"`` is the binary energy-mask Jaccard (the v1 default, kept). The other two
# weigh pixels by intensity: ``"cosine"`` is the angle between the flattened
# footprints (scale-free by construction); ``"weighted_jaccard"`` (Ruzicka) is
# sum(min)/sum(max) on sum-normalized footprints, the graded analogue of IoU.
SIMILARITY_METRICS = ("iou", "weighted_jaccard", "cosine")


@dataclass(frozen=True)
class Match:
    """The result of pairing estimated footprints to true ones by spatial overlap.

    ``pairing`` is the optimal one-to-one assignment (maximizing total similarity)
    with pure non-overlapping pairs dropped, so it is safe to feed straight into the
    temporal metrics. The threshold-dependent quality summaries (:meth:`recall`,
    :meth:`precision`) count only pairs whose similarity clears ``threshold``.

    ``metric`` records which similarity was used (see :data:`SIMILARITY_METRICS`)
    and ``shift`` the global ``(dy, dx)`` translation applied to the estimate before
    scoring (``(0, 0)`` when none). ``iou_matrix`` / ``mean_iou`` remain as aliases
    for the generic ``similarity_matrix`` / ``mean_similarity`` so existing callers
    keep working; they hold the chosen metric, not necessarily IoU.

    Empty denominators (no estimated or no true cells, no matched pairs) report
    ``0.0`` rather than ``nan`` - convenient for ``assert metric >= bound`` tests.
    """

    similarity_matrix: np.ndarray  # (n_est, n_true) pairwise footprint similarity
    pairing: tuple[tuple[int, int], ...]  # optimal (est_idx, true_idx) pairs, sim > 0
    metric: str = "iou"  # which similarity was used (see SIMILARITY_METRICS)
    shift: tuple[int, int] = (0, 0)  # global (dy, dx) applied to A_est before scoring

    @property
    def iou_matrix(self) -> np.ndarray:
        """Alias for :attr:`similarity_matrix` (holds the chosen ``metric``)."""
        return self.similarity_matrix

    @property
    def n_est(self) -> int:
        return int(self.similarity_matrix.shape[0])

    @property
    def n_true(self) -> int:
        return int(self.similarity_matrix.shape[1])

    def matched_pairs(self, threshold: float = 0.5) -> list[tuple[int, int]]:
        """The assigned pairs whose similarity is at least ``threshold`` (true positives)."""
        return [(i, j) for i, j in self.pairing if self.similarity_matrix[i, j] >= threshold]

    def recall(self, threshold: float = 0.5) -> float:
        """True positives over the number of true cells (``0.0`` if there are none)."""
        if self.n_true == 0:
            return 0.0
        return len(self.matched_pairs(threshold)) / self.n_true

    def precision(self, threshold: float = 0.5) -> float:
        """True positives over the number of estimated cells (``0.0`` if there are none)."""
        if self.n_est == 0:
            return 0.0
        return len(self.matched_pairs(threshold)) / self.n_est

    @property
    def mean_similarity(self) -> float:
        """Mean similarity over the matched (positive-overlap) pairs (``0.0`` if none)."""
        if not self.pairing:
            return 0.0
        return float(np.mean([self.similarity_matrix[i, j] for i, j in self.pairing]))

    @property
    def mean_iou(self) -> float:
        """Alias for :attr:`mean_similarity` (the chosen ``metric``, not always IoU)."""
        return self.mean_similarity


class ActivityScore(NamedTuple):
    """Per-matched-pair recovery of the deconvolved activity (see :func:`activity_similarity`).

    The deconvolved estimate (CNMF/minian ``S``) is not a spike train: it is a
    non-negative estimate of neural activity rate, scaled by an unknown factor that
    maps activity to calcium-kernel amplitude. So each pair is scored without
    binarizing and without assuming a common scale:

    * ``correlation`` - Pearson r between estimated and true activity (scale- and
      offset-invariant; the *shape* match). ``nan`` for a constant trace.
    * ``scale`` - the recovered non-negative amplitude factor alpha that best maps
      the estimate onto the true activity (the unknown gain, made explicit).
    * ``variance_explained`` - proportion of the true activity's variance the
      non-negatively scaled estimate accounts for, in ``(-inf, 1]``. ``nan`` when
      the true activity is constant.
    """

    correlation: np.ndarray
    scale: np.ndarray
    variance_explained: np.ndarray


def hungarian_match(
    A_est,
    A_true,
    *,
    metric: str = "iou",
    energy_frac: float = DEFAULT_ENERGY_FRAC,
    shift: tuple[float, float] | str | None = None,
) -> Match:
    """Optimally pair estimated spatial footprints to true ones by overlap.

    Forms the pairwise footprint-similarity matrix under ``metric`` and runs
    :func:`scipy.optimize.linear_sum_assignment` to find the assignment maximizing
    total similarity. Pairs with zero similarity are dropped from the returned
    :attr:`Match.pairing`.

    Parameters
    ----------
    A_est, A_true
        Footprint stacks ``(n, height, width)``, non-negative. Negative values
        (if any) are clipped to zero before scoring.
    metric
        Footprint similarity (see :data:`SIMILARITY_METRICS`): ``"iou"`` (binary
        energy-mask Jaccard, the default), ``"cosine"``, or ``"weighted_jaccard"``.
        The two weighted metrics compare the intensity profile, so pixel weights -
        not just which pixels are lit - drive the match.
    energy_frac
        For ``metric="iou"`` only: fraction of each footprint's energy its binary
        mask retains, in ``(0, 1]``.
    shift
        Global translation applied to ``A_est`` before scoring, to absorb a uniform
        offset between the estimate's (motion-corrected) frame and the true frame.
        ``None`` applies nothing; a ``(dy, dx)`` tuple applies that known shift
        (rounded to whole pixels - derive it from the motion trajectories);
        ``"auto"`` estimates one by aligning the intensity-weighted centroids of the
        two footprint stacks. The applied integer shift is recorded on
        :attr:`Match.shift`.
    """
    if metric not in SIMILARITY_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}; choose from {SIMILARITY_METRICS}."
        )
    A_est = np.clip(np.asarray(A_est, dtype=float), 0.0, None)
    A_true = np.clip(np.asarray(A_true, dtype=float), 0.0, None)

    applied = _resolve_shift(shift, A_est, A_true)
    sim = _similarity_matrix(A_est, A_true, metric, energy_frac)
    if applied != (0, 0):
        sim_shifted = _similarity_matrix(_shift_stack(A_est, *applied), A_true, metric, energy_frac)
        # An *estimated* shift is only adopted if it genuinely improves the matched
        # overlap; a known/explicit shift is trusted as given. This keeps a misfired
        # auto-estimate (a flat overlap surface, a dominant false positive) from
        # making an otherwise-good recovery look like a total miss.
        if shift != "auto" or _assigned_total(sim_shifted) > _assigned_total(sim):
            sim = sim_shifted
        else:
            applied = (0, 0)

    pairing = _assign(sim)
    return Match(similarity_matrix=sim, pairing=pairing, metric=metric, shift=applied)


def trace_pearson(C_est, C_true, pairing) -> np.ndarray:
    """Per-matched-pair Pearson correlation between estimated and true traces.

    Returns one correlation per ``(est_idx, true_idx)`` in ``pairing`` (a constant
    trace has undefined correlation and yields ``nan``). ``C_est``/``C_true`` are
    ``(unit, frame)``.
    """
    C_est = np.asarray(C_est, dtype=float)
    C_true = np.asarray(C_true, dtype=float)
    out = []
    for i, j in pairing:
        a, b = C_est[i], C_true[j]
        if a.std() == 0 or b.std() == 0:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out, dtype=float)


def activity_similarity(S_est, S_true, pairing) -> ActivityScore:
    """Scale-aware recovery of the deconvolved activity over the matched units.

    For each matched pair the estimated activity ``S_est[i]`` is compared to the
    true activity ``S_true[j]`` (``(unit, frame)`` arrays) **without binarizing**:
    the deconvolved estimate is a continuous activity rate, related to the truth by
    an unknown non-negative amplitude factor, so the comparison must be invariant to
    that scale. Per pair this returns (see :class:`ActivityScore`):

    * the Pearson ``correlation`` (scale- and offset-invariant shape match);
    * the recovered non-negative ``scale`` alpha minimizing
      ``|| S_true - (alpha * S_est + beta) ||`` (the best amplitude map, ``alpha >= 0``);
    * the ``variance_explained`` by that fit, ``1 - SS_res / SS_tot``.

    A constant estimate yields ``scale = 0`` and ``variance_explained = 0`` (it
    explains nothing); a constant truth yields ``nan`` for both correlation and
    variance explained (there is nothing to explain). This is the deconvolution
    scorecard CaLab uses (per-cell amplitude alpha plus proportion of variance
    explained), not a spike-timing precision/recall.
    """
    S_est = np.asarray(S_est, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    corr, scale, pve = [], [], []
    for i, j in pairing:
        a, b = S_est[i], S_true[j]
        corr.append(
            np.nan if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
        )
        alpha, var_exp = _nonneg_affine_fit(a, b)
        scale.append(alpha)
        pve.append(var_exp)
    return ActivityScore(
        correlation=np.array(corr, dtype=float),
        scale=np.array(scale, dtype=float),
        variance_explained=np.array(pve, dtype=float),
    )


def shift_rmse(shifts_est, shifts_true, *, correction: bool = False, align: bool = False) -> float:
    """Root-mean-square error (pixels) between two ``(frame, 2)`` shift trajectories.

    RMSE over all frames and both axes. The two arrays must share a **sign
    convention**: a motion-*correction* estimate is the negation of the applied
    ``GroundTruth.shifts``, so pass ``correction=True`` to negate ``shifts_est``
    first (the common case, since a motion-correction pipeline emits the shift it
    would apply to undo the motion); leave it ``False`` to compare two trajectories
    already in the same convention.

    They must also share an **origin**, and that is the catch across pipelines: a
    correction trajectory is relative to whatever template the pipeline registered
    to, so it can carry an arbitrary constant per-axis offset versus minisim's
    zero-shift reference. ``align=True`` removes the best constant offset (the
    per-axis mean residual) before the RMSE, scoring how well the *motion* was
    tracked rather than which frame was called zero. Leave it ``False`` to hold the
    pipeline to the absolute origin too.
    """
    e = np.asarray(shifts_est, dtype=float)
    t = np.asarray(shifts_true, dtype=float)
    if correction:
        e = -e
    diff = e - t
    if align:
        diff = diff - diff.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(diff**2)))


def global_shift_from_trajectories(shifts_est, shifts_true, *, correction: bool = True) -> tuple[int, int]:
    """The constant ``(dy, dx)`` offset between two motion trajectories, in pixels.

    The estimated correction and the true applied motion track the same brain
    movement, so their difference is (up to noise) a constant: the offset between
    the pipeline's registration template and minisim's zero-shift reference. That
    constant is exactly the global translation between the estimated and true
    footprint frames, so it can be read off the trajectories rather than searched
    for. Returns the per-axis mean difference rounded to whole pixels, ready to pass
    as :func:`hungarian_match`'s ``shift``. ``correction=True`` negates the estimate
    first (the usual convention, matching :func:`shift_rmse`).
    """
    e = np.asarray(shifts_est, dtype=float)
    t = np.asarray(shifts_true, dtype=float)
    if correction:
        e = -e
    offset = (e - t).mean(axis=0)
    return int(round(float(offset[0]))), int(round(float(offset[1])))


def field_pearson(est, true) -> float:
    """Pearson correlation between two 2-D fields (vignette, leakage), flattened.

    Scale- and offset-invariant, so it scores the *shape* of the recovered field
    rather than its absolute level. Returns ``nan`` if either field is constant.
    """
    a = np.asarray(est, dtype=float).ravel()
    b = np.asarray(true, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# naive footprint ROI - the un-demixed baseline the demixing comparison beats
# ---------------------------------------------------------------------------


def footprint_mask(a, rel: float = DEFAULT_ROI_REL_THRESHOLD) -> np.ndarray:
    """Boolean mask of a footprint's bright extent: pixels above ``rel × peak``.

    The rough region of interest a person would draw around a cell by eye - an
    intensity-relative threshold on a single footprint ``(height, width)``, not the
    energy-fraction core :func:`hungarian_match` uses for scoring (see
    :data:`DEFAULT_ROI_REL_THRESHOLD` for why the two differ). An all-zero (or
    all-negative) footprint yields an all-False mask.
    """
    a = np.asarray(a, dtype=float)
    peak = float(a.max()) if a.size else 0.0
    if peak <= 0:
        return np.zeros(a.shape, dtype=bool)
    return a > rel * peak


def footprint_roi_trace(movie, a, rel: float = DEFAULT_ROI_REL_THRESHOLD) -> np.ndarray:
    """Naive footprint-ROI trace: the movie averaged over a cell's footprint mask.

    Mean of ``movie`` ``(frame, height, width)`` over the pixels in
    :func:`footprint_mask` of ``a``, frame by frame - exactly what reading out a
    hand-drawn ROI gives, with **no unmixing**. It is *not* the true calcium ``C``:
    the mask also collects neighbour light the optics blur and the tissue scatter
    in, plus any additive background (neuropil, leakage), so it is the contaminated
    baseline that motivates demixing. An empty mask yields all zeros.
    """
    movie = np.asarray(movie, dtype=float)
    mask = footprint_mask(a, rel)
    if not mask.any():
        return np.zeros(movie.shape[0])
    return movie[:, mask].mean(axis=1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nonneg_affine_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Best non-negative ``alpha`` and the variance of ``b`` it explains.

    Fits ``b ≈ alpha·a + beta`` by least squares with ``alpha >= 0`` (the amplitude
    map from estimated activity ``a`` to true activity ``b`` cannot be negative),
    and returns ``(alpha, variance_explained)`` where ``variance_explained =
    1 - SS_res/SS_tot``. A constant ``a`` cannot explain anything, so it yields
    ``(0.0, 0.0)``; a constant ``b`` has no variance to explain, so it yields
    ``(0.0, nan)``.
    """
    var_a = float(a.var())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0, float("nan")
    if var_a == 0.0:
        return 0.0, 0.0
    alpha = float(np.cov(a, b, bias=True)[0, 1] / var_a)
    alpha = max(0.0, alpha)
    beta = float(b.mean() - alpha * a.mean())
    ss_res = float(((b - (alpha * a + beta)) ** 2).sum())
    return alpha, 1.0 - ss_res / ss_tot


def _resolve_shift(
    shift: tuple[float, float] | str | None, A_est: np.ndarray, A_true: np.ndarray
) -> tuple[int, int]:
    """Turn the ``shift`` argument into a concrete integer ``(dy, dx)`` to apply."""
    if shift is None:
        return (0, 0)
    if shift == "auto":
        return _overlap_shift(A_est, A_true)
    if isinstance(shift, str):
        raise ValueError(f"Unknown shift mode {shift!r}; use 'auto', a (dy, dx) tuple, or None.")
    dy, dx = shift
    return int(round(float(dy))), int(round(float(dx)))


# Cap on the magnitude of an *estimated* global shift, as a fraction of the FOV per
# axis. The overlap search is a fallback (the trajectory-derived offset is exact and
# unbounded); bounding it keeps a pathological cross-correlation peak - e.g. a lone
# bright false positive landing on a true cell - from inventing a large translation.
_MAX_AUTO_SHIFT_FRAC = 0.25

# Fraction of the peak below which a summed-footprint pixel is treated as background
# when forming the binary support for registration (see _overlap_shift). Drops the
# near-zero blur tail (a degraded footprint spans many orders of magnitude) without
# eating into the real footprint body.
_SUPPORT_REL_THRESHOLD = 1e-2


def _overlap_shift(A_est: np.ndarray, A_true: np.ndarray) -> tuple[int, int]:
    """Whole-pixel ``(dy, dx)`` for ``A_est`` that best aligns it onto ``A_true``.

    Uses **phase correlation on the binary supports** of the summed footprint images:
    the normalized cross-power spectrum has a sharp peak at the translation between
    the two, robust to the broad low-frequency envelope that makes a raw cross-
    correlation drift to large lags. Registering the *binary support* (rather than the
    raw intensity sum) is what makes it robust on real footprints: a summed footprint
    image spans many orders of magnitude - bright cores over a near-zero blur tail -
    and the phase-only whitening would otherwise amplify that near-zero tail to equal
    weight with the real signal, drowning the peak. Thresholding to a support at
    ``_SUPPORT_REL_THRESHOLD`` of the peak gives clean, well-conditioned edges to
    align, and caps any single footprint - real or false-positive - at weight 1. The
    peak is searched within ``_MAX_AUTO_SHIFT_FRAC`` of the FOV per axis. The candidate
    is only a *proposal* - :func:`hungarian_match` adopts it only if it improves the
    matched overlap. Returns ``(0, 0)`` when either stack carries no mass.
    """
    est_img = A_est.sum(axis=0)
    true_img = A_true.sum(axis=0)
    if est_img.max() <= 0.0 or true_img.max() <= 0.0:
        return (0, 0)
    h, w = est_img.shape
    est_sup = (est_img > _SUPPORT_REL_THRESHOLD * est_img.max()).astype(float)
    true_sup = (true_img > _SUPPORT_REL_THRESHOLD * true_img.max()).astype(float)
    cross = np.fft.rfft2(true_sup) * np.conj(np.fft.rfft2(est_sup))
    cross /= np.abs(cross) + 1e-12  # phase only: a delta at the translation, no envelope drift
    cc = np.fft.fftshift(np.fft.irfft2(cross, s=(h, w)))
    cy, cx = h // 2, w // 2
    my = min(int(_MAX_AUTO_SHIFT_FRAC * h), cy)
    mx = min(int(_MAX_AUTO_SHIFT_FRAC * w), cx)
    window = cc[cy - my : cy + my + 1, cx - mx : cx + mx + 1]
    py, px = np.unravel_index(int(np.argmax(window)), window.shape)
    return int(py - my), int(px - mx)


def _assign(sim: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Optimal one-to-one ``(est, true)`` pairing maximizing total similarity, sim > 0 pairs."""
    rows, cols = linear_sum_assignment(sim, maximize=True)
    return tuple((int(i), int(j)) for i, j in zip(rows, cols, strict=True) if sim[i, j] > 0)


def _assigned_total(sim: np.ndarray) -> float:
    """Total similarity of the optimal assignment - the score a global shift is judged by."""
    return float(sum(sim[i, j] for i, j in _assign(sim)))


def _shift_stack(A: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate the ``(height, width)`` axes of a footprint stack by ``(dy, dx)``, zero-filled."""
    out = np.zeros_like(A)
    h, w = A.shape[-2:]
    sy, dy_dst = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    sx, dx_dst = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., dy_dst, dx_dst] = A[..., sy, sx]
    return out


def _similarity_matrix(
    A_est: np.ndarray, A_true: np.ndarray, metric: str, energy_frac: float
) -> np.ndarray:
    """Pairwise footprint similarity ``(n_est, n_true)`` under the chosen ``metric``."""
    if metric == "iou":
        masks_est = _energy_masks(A_est, energy_frac)
        masks_true = _energy_masks(A_true, energy_frac)
        return _iou_matrix(masks_est, masks_true)
    # Explicit pixel count: a zero-unit stack cannot infer a -1 reshape dim.
    n_pix = int(np.prod(A_est.shape[1:]))
    e = A_est.reshape(A_est.shape[0], n_pix)
    t = A_true.reshape(A_true.shape[0], n_pix)
    if metric == "cosine":
        return _cosine_matrix(e, t)
    return _weighted_jaccard_matrix(e, t)


def _cosine_matrix(e: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two stacks of flattened footprints."""
    en = np.linalg.norm(e, axis=1)
    tn = np.linalg.norm(t, axis=1)
    denom = en[:, None] * tn[None, :]
    sim = e @ t.T
    return np.where(denom > 0, sim / np.where(denom > 0, denom, 1.0), 0.0)


def _weighted_jaccard_matrix(e: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pairwise weighted Jaccard (Ruzicka) between sum-normalized flattened footprints.

    Each footprint is normalized to unit sum (so absolute brightness drops out, only
    the relative weight profile matters), then the similarity is
    ``Σ min(e, t) / Σ max(e, t)`` - the graded analogue of binary IoU. Computed a
    row at a time to keep the ``(n_est, n_true, n_pixels)`` comparison out of memory.
    """
    e = _row_normalize_sum(e)
    t = _row_normalize_sum(t)
    out = np.zeros((e.shape[0], t.shape[0]), dtype=float)
    for i, row in enumerate(e):
        inter = np.minimum(row[None, :], t).sum(axis=1)
        union = np.maximum(row[None, :], t).sum(axis=1)
        out[i] = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return out


def _row_normalize_sum(A: np.ndarray) -> np.ndarray:
    """Scale each row to unit sum (rows that sum to zero are left at zero)."""
    totals = A.sum(axis=1, keepdims=True)
    return np.divide(A, totals, out=np.zeros_like(A), where=totals > 0)


def _energy_masks(A: np.ndarray, energy_frac: float) -> np.ndarray:
    """Binarize each footprint to the smallest pixel set holding ``energy_frac`` energy.

    For each footprint, pixels are ranked by intensity (high to low) and the top
    ones are kept until their cumulative intensity reaches ``energy_frac`` of the
    footprint's total. An all-zero footprint yields an all-False mask.
    """
    if not 0.0 < energy_frac <= 1.0:
        raise ValueError(f"energy_frac must be in (0, 1], got {energy_frac}.")
    # Flatten each footprint to its pixel count, keeping a 0-row (recovered-nothing)
    # stack valid.
    n_pix = int(np.prod(A.shape[1:]))
    flat = np.clip(A, 0.0, None).reshape(A.shape[0], n_pix)
    masks = np.zeros(flat.shape, dtype=bool)
    if energy_frac >= 1.0:
        # The whole support. Going through the cumulative-energy search here is not
        # just wasteful but numerically unstable: for a footprint spanning many orders
        # of magnitude (e.g. a blurred footprint with a near-zero tail), the descending
        # cumulative sum goes flat across the tail, and `searchsorted` for the total -
        # which is summed in a different order and so differs by rounding - can land in
        # that flat region and collapse the mask to the bright core. Keep every
        # positive pixel directly instead.
        return (flat > 0).reshape(A.shape)
    for i, f in enumerate(flat):
        total = f.sum()
        if total <= 0:
            continue
        order = np.argsort(f)[::-1]  # descending intensity
        csum = np.cumsum(f[order])
        k = int(np.searchsorted(csum, energy_frac * total, side="left")) + 1
        masks[i, order[: min(k, f.size)]] = True
    return masks.reshape(A.shape)


def _iou_matrix(masks_est: np.ndarray, masks_true: np.ndarray) -> np.ndarray:
    """Pairwise IoU (Jaccard) between two stacks of boolean masks → ``(n_est, n_true)``.

    Intersections come from a single mask-vs-mask matmul (footprints flattened to
    rows); unions are ``area_est + area_true − intersection``.
    """
    # Flatten each mask to its pixel count, keeping a 0-row stack valid.
    e = masks_est.reshape(masks_est.shape[0], int(np.prod(masks_est.shape[1:]))).astype(np.float32)
    t = masks_true.reshape(masks_true.shape[0], int(np.prod(masks_true.shape[1:]))).astype(np.float32)
    intersection = e @ t.T  # (n_est, n_true)
    area_e = e.sum(axis=1)[:, None]
    area_t = t.sum(axis=1)[None, :]
    union = area_e + area_t - intersection
    return np.where(union > 0, intersection / np.where(union > 0, union, 1.0), 0.0)

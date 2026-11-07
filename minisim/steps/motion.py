"""Motion-domain step: rigid x,y brain motion - the tissue→sensor boundary.

``brain_motion`` is the single step in the motion domain and the hinge of the
whole reference-frame design. Everything before it (cells, neuropil, bleaching)
is **brain-frame** content that moves with the tissue; everything after it
(vignette, leakage, sensor) is **sensor-frame** and static. This step is where
the brain frame is translated relative to the static sensor.

To keep the motion honest, the upstream tissue steps render on a canvas **larger
than the sensor** (``Scene.zeros(acq, margin_px=…)``): real, simulated tissue
sits just off the sensor FOV. This step shifts that canvas per frame and crops
the centered sensor FOV back out, so the content that moves in at the edges is
genuine tissue, never a fabricated fill - and because the FOV crop never reaches
the canvas edge (the margin is ≥ the maximum shift), no edge fill ever enters the
result. The per-frame displacement is recorded to ground truth as the motion a
correction stage must estimate.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import cv2
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import lfilter

from minisim.scene import MOVIE_DIMS, Scene
from minisim.steps.base import Step

if TYPE_CHECKING:
    # Referenced only as the string Generic base Step["BrainMotion"], which ruff's
    # F401 misses; pyright needs it in scope to resolve the forward reference.
    from minisim.spec import BrainMotion  # noqa: F401

# High-res integration rate for the physical oscillator, Hz. The trajectory is
# integrated this finely then bin-averaged down to the frame rate (the same
# fine-then-downsample idiom as cell_activity), so the exposure averaging and the
# aliasing of stride-rate motion at the camera rate emerge honestly. ~300 Hz keeps
# tens of steps per cycle for the 6-8 Hz band the symplectic integrator sees.
_INTEGRATION_HZ = 300.0
# Random drift of the locomotion frequency: real gait is near-periodic over a few
# strides but far from metronomic (the animal speeds up and slows down), so the stride
# frequency wanders by this fraction (a low-pass-filtered process), broadening the
# spectral line into a band rather than a razor-sharp tone.
_LOCOMOTION_FREQ_CV = 0.15
_LOCOMOTION_FREQ_TAU_S = 0.3  # time constant of that wander
# Correlation time of the broadband behavioral forcing. Real brain motion is dominated
# by LOW frequencies (postural shifts, walking, head movement, slow drift), so the
# acceleration noise is red (low-pass white) with this time constant rather than white
# -- most of the motion's power then sits below the stride rhythm, not at it.
_BEHAVIORAL_TAU_S = 0.12
# Percentile of the displacement radius calibrated to motion_amplitude_um: a high
# percentile, so amplitude is the *extreme* excursion while the bulk of frames sit
# well inside it (most of a recording moves less than the worst moments). Robust to
# the single largest swing, unlike the raw peak.
_EXCURSION_PCTILE = 99.0
# Guards divisions by a (near-)zero normalization; far below any real motion scale.
_EPS = 1e-12


def bounded_random_walk(
    n_frames: int, step_px: float, max_px: float, rng: np.random.Generator
) -> np.ndarray:
    """A 2-D random walk of per-frame ``(dy, dx)`` positions, bounded to a disk.

    Starts at ``(0, 0)`` on frame 0 (the reference view) and adds an independent
    ``Normal(0, step_px)`` increment per axis each frame; whenever the cumulative
    displacement would leave the radius-``max_px`` disk it is rescaled back onto
    the boundary. Returns an ``(n_frames, 2)`` array in pixels. The bound keeps
    motion within the tissue margin so the FOV crop always lands on real content.
    Sequential by construction (each position depends on the previous), so an
    explicit loop - cheap at the recording lengths the simulator targets.
    """
    pos = np.zeros((n_frames, 2))
    for f in range(1, n_frames):
        cand = pos[f - 1] + rng.normal(0.0, step_px, size=2)
        magnitude = float(np.hypot(cand[0], cand[1]))
        if magnitude > max_px:
            cand *= max_px / magnitude
        pos[f] = cand
    return pos


def _lowpass(white: np.ndarray, tau_s: float, dt: float) -> np.ndarray:
    """One-pole low-pass of a white series → a slow, unit-variance process."""
    alpha = dt / (tau_s + dt)
    slow = np.asarray(lfilter([alpha], [1.0, -(1.0 - alpha)], white))
    std = float(slow.std())
    return slow / std if std > _EPS else slow


def _integrate_dho(accel: np.ndarray, dt: float, w0: float, zeta: float) -> np.ndarray:
    """Position response of a damped harmonic oscillator to an acceleration drive.

    Solves ``x'' + 2ζω₀ x' + ω₀² x = a(t)`` per column of ``accel`` with the
    semi-implicit (symplectic) Euler scheme, which is a 2nd-order linear recurrence
    and so runs as an IIR filter (``lfilter``): vectorized in C, no Python loop,
    fast even for long recordings. Starts from rest (x = v = 0).

    The scheme ``v_i = v_{i-1} + dt(a_i − 2ζω₀v_{i-1} − ω₀²x_{i-1})``, ``x_i =
    x_{i-1} + dt·v_i`` eliminates to ``x_i = (2 − c₁dt − c₂dt²)x_{i-1} − (1 −
    c₁dt)x_{i-2} + dt²a_i`` with ``c₁ = 2ζω₀``, ``c₂ = ω₀²``.
    """
    c1, c2 = 2.0 * zeta * w0, w0 * w0
    b = [dt * dt]
    a = [1.0, -(2.0 - c1 * dt - c2 * dt * dt), (1.0 - c1 * dt)]
    return np.asarray(lfilter(b, a, accel, axis=0))


def physical_brain_motion(
    n_frames: int,
    fps: float,
    *,
    locomotion_freq_hz: float,
    resonance_freq_hz: float,
    damping_ratio: float,
    locomotion_fraction: float,
    locomotion_axis: int,
    amplitude_px: float,
    max_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """2-D brain trajectory from a driven damped harmonic oscillator, per-frame px.

    The tissue is a damped mass elastically tethered to the rigid skull:
    ``x'' + 2ζω₀ x' + ω₀² x = a_drive(t)``. The drive is an always-on locomotion
    rhythm (a stride-frequency sinusoidal acceleration whose frequency slowly
    wanders, so the spectral line is a narrow band not a pure tone) on the dominant
    ``locomotion_axis``, plus broadband acceleration noise on both axes. The two
    drives are integrated separately, each normalized to unit RMS displacement, then
    mixed by ``locomotion_fraction`` so that knob is the rhythm's share of the motion.

    Integrated at :data:`_INTEGRATION_HZ` then bin-averaged to ``fps`` (exposure
    integration), so stride-rate motion is honestly aliased at the camera rate. The
    oscillator is linear in the drive, so the realized trajectory is scaled in one
    shot to ``amplitude_px`` (the :data:`_EXCURSION_PCTILE` displacement radius),
    then clamped onto the ``max_px`` safety disk. Frame 0 is pinned to the origin
    (the reference view). Returns ``(n_frames, 2)`` in pixels.
    """
    bins = max(int(round(_INTEGRATION_HZ / fps)), 1)
    dt = 1.0 / (bins * fps)
    n_hr = n_frames * bins
    w0, zeta = 2.0 * np.pi * resonance_freq_hz, damping_ratio

    # Locomotion acceleration: a sinusoid whose instantaneous frequency drifts slowly
    # about locomotion_freq_hz (near-periodic gait, not a metronome).
    freq = locomotion_freq_hz * (
        1.0
        + _LOCOMOTION_FREQ_CV
        * _lowpass(rng.standard_normal(n_hr), _LOCOMOTION_FREQ_TAU_S, dt)
    )
    a_loco = np.zeros((n_hr, 2))
    a_loco[:, locomotion_axis] = np.sin(2.0 * np.pi * np.cumsum(freq) * dt)
    # Broadband behavioral acceleration on both axes, RED (low-pass white) so its
    # power sits at low frequencies -- the slow drift, walking, and head movement that
    # dominate real brain motion, with the stride rhythm riding above it.
    a_noise = np.stack(
        [_lowpass(rng.standard_normal(n_hr), _BEHAVIORAL_TAU_S, dt) for _ in range(2)],
        axis=1,
    )

    # Decimate each component's position to the frame rate (exposure-window mean),
    # pin frame 0 to the reference, normalize to unit RMS radius, then mix.
    def _frames(accel: np.ndarray) -> np.ndarray:
        pos = (
            _integrate_dho(accel, dt, w0, zeta).reshape(n_frames, bins, 2).mean(axis=1)
        )
        pos -= pos[0]
        rms = float(np.sqrt(np.mean(pos[:, 0] ** 2 + pos[:, 1] ** 2)))
        return pos / rms if rms > _EPS else pos

    traj = locomotion_fraction * _frames(a_loco) + (
        1.0 - locomotion_fraction
    ) * _frames(a_noise)

    # One-shot amplitude calibration (linear system), then the hard safety clamp.
    radius = np.hypot(traj[:, 0], traj[:, 1])
    pctile = float(np.percentile(radius, _EXCURSION_PCTILE))
    if pctile > _EPS:
        traj *= amplitude_px / pctile
    mag = np.hypot(traj[:, 0], traj[:, 1])
    over = mag > max_px
    if over.any():
        traj[over] *= (max_px / mag[over])[:, None]
    return traj


# --- "miniscope" model: empirically calibrated rigid motion -------------------------
# Target spectrum of real Miniscope V4 CA1 brain motion (20-min, 20 fps): the share of
# the total motion variance in each band. Cumulative power below 0.1/0.5/1/2/5 Hz is
# 39/63/76/86/96 %, so the bands (0, 0.1], (0.1, 0.5], (0.5, 2], (2, 5], (5, Nyquist]
# carry 39/24/23/10/4 %. A broadband, 1/f-like spectrum - most of the power is slow,
# but a long tail reaches the Nyquist rate.
_MINISCOPE_BAND_EDGES_HZ = (0.0, 0.1, 0.5, 2.0, 5.0)
_MINISCOPE_BAND_FRACTIONS = (0.39, 0.24, 0.23, 0.10, 0.04)
# Direction jitter of a burst about the dominant axis (animals move along a preferred
# axis, but not exactly), degrees.
_BURST_AXIS_JITTER_DEG = 25.0
# Fraction of the burst's duration spent rising (fast onset, slower relaxation).
_BURST_RISE_FRACTION = 0.3
# Bursts may already be under way at frame 0: onsets are drawn from this far before it.
_BURST_LEAD_S = 3.0
# The broadband residual never falls below this share of a band's target power, so a
# burst-heavy draw cannot starve a band (the realized total then overshoots slightly).
_BASE_FLOOR_FRACTION = 0.2


def _burst_shape(u: np.ndarray, rise: float = _BURST_RISE_FRACTION) -> np.ndarray:
    """Unit-peak burst envelope on normalized time ``u`` in [0, 1); zero outside.

    ``sin²`` rise over the first ``rise`` of the episode, ``cos²`` relaxation over the
    rest: smooth (zero slope at both ends and at the peak), asymmetric like a startle
    or postural shift.
    """
    out = np.zeros_like(u)
    up = (u >= 0) & (u < rise)
    down = (u >= rise) & (u < 1.0)
    out[up] = np.sin(0.5 * np.pi * u[up] / rise) ** 2
    out[down] = np.cos(0.5 * np.pi * (u[down] - rise) / (1.0 - rise)) ** 2
    return out


def _band_layout(freqs: np.ndarray, fps: float) -> list[np.ndarray]:
    """Per-band index arrays over the positive FFT bins (k >= 1), one per target band.

    Bands above the Nyquist rate are dropped. The last band extends to Nyquist.
    """
    nyq = fps / 2.0
    edges = [e for e in _MINISCOPE_BAND_EDGES_HZ if e < nyq] + [nyq]
    bands = []
    for lo, hi in itertools.pairwise(edges):
        k = np.nonzero((freqs > lo) & (freqs <= hi) & (freqs > 0))[0]
        bands.append(k)
    return bands


def draw_bursts(
    t: np.ndarray,
    rate_hz: float,
    duration_s: tuple[float, float],
    peak_px: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sum of Poisson-onset burst episodes on the time grid ``t``.

    Returns ``(burst, n_bursts)``: the ``(len(t), 2)`` displacement in the principal
    frame (columns: major, minor axis) and the number of episodes drawn (onsets run
    from :data:`_BURST_LEAD_S` before ``t[0]`` so one may already be under way).
    """
    burst = np.zeros((len(t), 2))
    n_bursts = int(rng.poisson(rate_hz * (t[-1] + _BURST_LEAD_S)))
    onsets = rng.uniform(-_BURST_LEAD_S, t[-1], size=n_bursts)
    durs = rng.uniform(duration_s[0], duration_s[1], size=n_bursts)
    peaks = rng.uniform(peak_px[0], peak_px[1], size=n_bursts)
    signs = rng.choice([-1.0, 1.0], size=n_bursts)
    phis = np.deg2rad(rng.normal(0.0, _BURST_AXIS_JITTER_DEG, size=n_bursts))
    for t0, dur, amp, sgn, phi in zip(onsets, durs, peaks, signs, phis, strict=True):
        env = _burst_shape((t - t0) / dur)
        burst[:, 0] += sgn * amp * np.cos(phi) * env
        burst[:, 1] += sgn * amp * np.sin(phi) * env
    return burst, n_bursts


def miniscope_brain_motion(
    n_frames: int,
    fps: float,
    *,
    rms_major_px: float,
    axis_ratio: float,
    axis_deg: float,
    burst_rate_hz: float,
    burst_duration_s: tuple[float, float],
    burst_peak_px: tuple[float, float],
    drift_px_per_s: float,
    max_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rigid ``(dy, dx)`` trajectory calibrated to real Miniscope brain motion, in px.

    Three additive parts, all in a principal frame whose major axis points
    ``axis_deg`` degrees from +y toward +x (so ``axis_deg=60.6`` leans mostly along x):

    * **Bursts** - Poisson-onset movement episodes (``burst_rate_hz``), each lasting
      ``burst_duration_s`` (uniform) and peaking at ``burst_peak_px`` (uniform) with a
      random sign, mostly along the major axis (:data:`_BURST_AXIS_JITTER_DEG`).
    * **Broadband noise** - Gaussian noise with a piecewise-flat spectrum whose
      per-band power is the target (:data:`_MINISCOPE_BAND_FRACTIONS` of the axis
      variance) *minus the power the bursts already put in that band*, so the
      total follows the 1/f-like target with the bursts counted. Axis variances are
      ``rms_major_px²`` and ``(axis_ratio·rms_major_px)²``.
    * **Drift** - a linear ramp of ``drift_px_per_s`` in a random direction.

    Frame 0 is pinned to the origin (the reference view) and the result is clamped to
    the ``max_px`` disk. Needs at least 2 frames (otherwise zeros).
    """
    n = n_frames
    if n < 2:
        return np.zeros((n, 2))
    t = np.arange(n) / fps
    theta = np.deg2rad(axis_deg)
    major = np.array([np.cos(theta), np.sin(theta)])  # (dy, dx)
    minor = np.array([-np.sin(theta), np.cos(theta)])

    burst, _ = draw_bursts(t, burst_rate_hz, burst_duration_s, burst_peak_px, rng)

    # --- broadband noise, filling each band's target power net of the bursts ---
    big_n = 1 << int(np.ceil(np.log2(2 * n)))  # power of two >= 2n: fast, no wrap
    freqs = np.fft.rfftfreq(big_n, d=1.0 / fps)
    bands = _band_layout(freqs, fps)
    fractions = np.array(_MINISCOPE_BAND_FRACTIONS[: len(bands)], dtype=float)
    fractions /= fractions.sum()
    sigma2 = np.array([rms_major_px**2, (axis_ratio * rms_major_px) ** 2])
    base = np.zeros((n, 2))
    for ax in range(2):
        spectrum = np.fft.rfft(burst[:, ax], big_n)
        burst_power = np.array(
            [2.0 * np.sum(np.abs(spectrum[k]) ** 2) / (big_n * n) for k in bands]
        )
        target = sigma2[ax] * fractions
        power = np.maximum(target - burst_power, _BASE_FLOOR_FRACTION * target)
        # A band with no FFT bins (a recording shorter than the band's period) hands
        # its power to the next populated band (else the previous one).
        counts = np.array([len(k) for k in bands])
        for b in np.nonzero(counts == 0)[0]:
            later = [i for i in range(b + 1, len(bands)) if counts[i] > 0]
            earlier = [i for i in range(b) if counts[i] > 0]
            dest = later[0] if later else (earlier[-1] if earlier else None)
            if dest is not None:
                power[dest] += power[b]
            power[b] = 0.0
        gain = np.zeros(len(freqs))
        for p_b, k in zip(power, bands, strict=True):
            if len(k):
                gain[k] = np.sqrt(p_b * big_n / (2.0 * len(k)))
        white = rng.standard_normal(big_n)
        base[:, ax] = np.fft.irfft(np.fft.rfft(white) * gain, big_n)[:n]

    pc = base + burst
    traj = pc[:, :1] * major + pc[:, 1:] * minor
    drift_dir = rng.uniform(0.0, 2.0 * np.pi)
    traj += (
        t[:, None] * drift_px_per_s * np.array([np.cos(drift_dir), np.sin(drift_dir)])
    )
    traj -= traj[0]
    mag = np.hypot(traj[:, 0], traj[:, 1])
    over = mag > max_px
    if over.any():
        traj[over] *= (max_px / mag[over])[:, None]
    return traj


# --- non-rigid deformation -------------------------------------------------------
# Short smoothing of the rigid-motion magnitude that drives the deformation amplitude.
_NONRIGID_COUPLING_SMOOTH_S = 0.25
# Gaussian smoothing of the control-node noise, in grid cells: makes neighbouring
# nodes move together (a coherent local deformation, not independent jitter).
_NONRIGID_SPATIAL_SIGMA = 0.7
# Deformation is larger toward the FOV edges: node weight 1 + boost·ρ² (ρ = 0 at the
# center, 1 at a corner), then renormalized to unit mean-square.
_NONRIGID_EDGE_BOOST = 1.5


def nonrigid_nodes(
    fov_shape: tuple[int, int], grid: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Control-node coordinates ``(ys, xs)`` in FOV pixels, spanning the FOV edge to edge."""
    out = []
    for size, g in zip(fov_shape, grid, strict=True):
        out.append(
            np.array([(size - 1) / 2.0]) if g == 1 else np.linspace(0, size - 1, g)
        )
    return out[0], out[1]


def grid_interp_matrix(nodes: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """``(len(coords), len(nodes))`` matrix interpolating node values to ``coords``.

    Natural cubic-spline interpolation (exact at the nodes, C² between them), linear
    in the node values so a whole field is ``My @ node_values @ Mx.T``. Coordinates are
    clamped to the node span.
    """
    coords = np.clip(np.asarray(coords, dtype=float), nodes[0], nodes[-1])
    if len(nodes) == 1:
        return np.ones((len(coords), 1))
    return CubicSpline(nodes, np.eye(len(nodes)), bc_type="natural")(coords)


def nonrigid_node_field(
    rigid_px: np.ndarray,
    fps: float,
    grid: tuple[int, ...],
    *,
    quiet_px: float,
    coupling: float,
    tau_s: float,
    max_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Control-node displacements ``(frame, gy, gx, 2)`` ``(dy, dx)`` in px.

    A spatially smooth, temporally correlated (AR(1), time constant ``tau_s``) unit
    field with the spatial mean removed (so the rigid trajectory stays the field's
    mean), weighted larger toward the edges, then scaled per frame by the amplitude
    ``sqrt(quiet_px² + (coupling·m)²)`` where ``m`` is the (briefly smoothed) distance
    of the rigid trajectory from its mean: a quiet floor plus a part that grows with
    the rigid motion. Components are clipped to ``±max_px`` (the tissue-margin
    budget).
    """
    n = len(rigid_px)
    gy, gx = grid
    a = float(np.exp(-1.0 / (fps * tau_s)))
    burn = int(np.ceil(5 * tau_s * fps))
    white = rng.standard_normal((n + burn, gy, gx, 2))
    ar = np.asarray(lfilter([np.sqrt(1.0 - a * a)], [1.0, -a], white, axis=0))[burn:]
    ar = gaussian_filter(
        ar, sigma=(0, _NONRIGID_SPATIAL_SIGMA, _NONRIGID_SPATIAL_SIGMA, 0)
    )
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, gy) if gy > 1 else [0.0],
        np.linspace(-1, 1, gx) if gx > 1 else [0.0],
        indexing="ij",
    )
    weight = 1.0 + _NONRIGID_EDGE_BOOST * (yy**2 + xx**2) / 2.0
    weight /= np.sqrt(np.mean(weight**2))
    unit = ar * weight[None, :, :, None]
    unit -= unit.mean(axis=(1, 2), keepdims=True)
    rms = float(np.sqrt(np.mean(unit**2)))
    if rms > _EPS:
        unit /= rms
    dev = rigid_px - rigid_px.mean(axis=0)
    m = gaussian_filter1d(
        np.hypot(dev[:, 0], dev[:, 1]),
        sigma=max(_NONRIGID_COUPLING_SMOOTH_S * fps, 1e-6),
        mode="nearest",
    )
    env = np.sqrt(quiet_px**2 + (coupling * m) ** 2)
    return np.clip(env[:, None, None, None] * unit, -max_px, max_px)


def evaluate_nonrigid(
    field: np.ndarray,
    fov_shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    """Non-rigid displacement ``(frame, len(ys), len(xs), 2)`` on the grid ``ys × xs``.

    ``field`` is the ``(frame, gy, gx, 2)`` control-node array over a FOV of
    ``fov_shape`` (nodes from :func:`nonrigid_nodes`); ``ys``/``xs`` are 1-D pixel
    coordinates (``arange(H)`` / ``arange(W)`` gives the full-resolution field).
    """
    node_y, node_x = nonrigid_nodes(fov_shape, field.shape[1:3])
    my = grid_interp_matrix(node_y, ys)
    mx = grid_interp_matrix(node_x, xs)
    return np.einsum("yi,fijc,xj->fyxc", my, field, mx, optimize=True)


def brain_motion_fields(
    spec, acq, n_frames: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rigid ``(frame, 2)`` shifts and optional non-rigid node field for a spec.

    The generation half of :class:`BrainMotionStep`, factored out so the step *and*
    the streaming video writer (which renders frame-chunks without ever materializing
    a full movie) produce the **identical** motion from the same RNG draws - the
    property the streamer relies on to match ``simulate()`` bit-for-bit. The rigid
    trajectory is an explicit ``trajectory_um`` (µm→px) if given, else the
    ``model``-selected generator. The non-rigid field (``spec.nonrigid``) is drawn
    from a child generator seeded by one draw of ``rng`` *after* the rigid draws, so
    with ``nonrigid=False`` (the default) the RNG stream is untouched. Returns
    ``(shifts_px, nonrigid_px_or_None)``.
    """
    shifts = _rigid_shifts(spec, acq, n_frames, rng)
    if not spec.nonrigid:
        return shifts, None
    child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    field = nonrigid_node_field(
        shifts,
        acq.fps,
        tuple(spec.nonrigid_grid),
        quiet_px=acq.um_to_px(spec.nonrigid_quiet_um),
        coupling=spec.nonrigid_coupling,
        tau_s=spec.nonrigid_tau_s,
        max_px=acq.um_to_px(spec.nonrigid_max_um),
        rng=child,
    )
    return shifts, field


def brain_motion_shifts(
    spec, acq, n_frames: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-frame rigid ``(dy, dx)`` displacement in pixels for a ``BrainMotion`` spec.

    The rigid half of :func:`brain_motion_fields` (kept for callers that only need the
    trajectory). With ``nonrigid`` off it consumes the RNG identically.
    """
    return brain_motion_fields(spec, acq, n_frames, rng)[0]


def _rigid_shifts(spec, acq, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    if spec.trajectory_um is not None:
        if len(spec.trajectory_um) != n_frames:
            raise ValueError(
                f"trajectory_um has {len(spec.trajectory_um)} entries but the "
                f"recording has {n_frames} frames; they must match."
            )
        return np.array(
            [[acq.um_to_px(dy), acq.um_to_px(dx)] for dy, dx in spec.trajectory_um]
        )
    if spec.model == "miniscope":
        return miniscope_brain_motion(
            n_frames,
            acq.fps,
            rms_major_px=acq.um_to_px(spec.motion_rms_um),
            axis_ratio=spec.motion_axis_ratio,
            axis_deg=spec.motion_axis_deg,
            burst_rate_hz=spec.burst_rate_hz,
            burst_duration_s=(spec.burst_min_duration_s, spec.burst_max_duration_s),
            burst_peak_px=(
                acq.um_to_px(spec.burst_min_peak_um),
                acq.um_to_px(spec.burst_max_peak_um),
            ),
            drift_px_per_s=acq.um_to_px(spec.drift_um_per_min) / 60.0,
            max_px=acq.um_to_px(spec.max_shift_um),
            rng=rng,
        )
    if spec.model == "physical":
        return physical_brain_motion(
            n_frames,
            acq.fps,
            locomotion_freq_hz=spec.locomotion_freq_hz,
            resonance_freq_hz=spec.resonance_freq_hz,
            damping_ratio=spec.damping_ratio,
            locomotion_fraction=spec.locomotion_fraction,
            locomotion_axis=0 if spec.locomotion_axis == "y" else 1,
            amplitude_px=acq.um_to_px(spec.motion_amplitude_um),
            max_px=acq.um_to_px(spec.max_shift_um),
            rng=rng,
        )
    return bounded_random_walk(
        n_frames,
        acq.um_to_px(spec.walk_step_um),
        acq.um_to_px(spec.max_shift_um),
        rng,
    )


def shift_and_crop(
    canvas: np.ndarray,
    shifts_px: np.ndarray,
    fov_shape: tuple[int, int],
    nonrigid: np.ndarray | None = None,
) -> np.ndarray:
    """Shift each canvas frame by its ``(dy, dx)`` and crop the centered FOV.

    ``canvas`` is ``(frame, H, W)``; each frame is translated by ``shifts_px[f]``
    with bilinear interpolation (sub-pixel, no overshoot) and the centered
    ``fov_shape`` window is cropped out. Positive ``dy``/``dx`` move content toward
    higher indices (down/right). No edge fill is ever seen in the output: the margin
    ``(H − fov) / 2`` is ≥ the maximum shift, so the vacated strip always lies
    outside the crop window.

    The warp is ``cv2.warpAffine`` with a pure-translation matrix - ~4x faster than
    ``scipy.ndimage.shift`` for this bilinear shift (the same swap minian's
    motion-correction path made, in the inverse direction). The centered FOV crop is
    folded into the affine translation, so ``warpAffine`` emits the shifted FOV
    directly: ``dst(j, i) = canvas(left + j − dx, top + i − dy)``. cv2 warps in
    float32; the ~1e-2-scale interpolation difference from the old spline path is far
    below the sensor's per-pixel noise and quantization (the recording is
    noise-dominated), and it is applied identically by ``simulate`` and the streaming
    writer, so the two stay in lock-step.

    With ``nonrigid`` (``(frame, gy, gx, 2)`` control-node displacements, px) each
    frame is instead warped by ``cv2.remap`` with the per-pixel displacement
    ``rigid + interpolated field`` evaluated at the *output* pixel (same bilinear
    sampling and zero border as the affine path): ``dst(y, x) = canvas(top + y − dy,
    left + x − dx)`` with ``(dy, dx)`` varying over the image. ``nonrigid=None`` takes
    the affine path unchanged, bit-for-bit.
    """
    n_frames, canvas_h, canvas_w = canvas.shape
    fov_h, fov_w = fov_shape
    top = (canvas_h - fov_h) // 2
    left = (canvas_w - fov_w) // 2
    frames32 = canvas.astype(np.float32, copy=False)
    out = np.empty((n_frames, fov_h, fov_w))
    if nonrigid is not None:
        node_y, node_x = nonrigid_nodes(fov_shape, nonrigid.shape[1:3])
        my = grid_interp_matrix(node_y, np.arange(fov_h))
        mx = grid_interp_matrix(node_x, np.arange(fov_w))
        yy = (top + np.arange(fov_h, dtype=np.float64))[:, None]
        xx = (left + np.arange(fov_w, dtype=np.float64))[None, :]
        for f in range(n_frames):
            dense_y = my @ nonrigid[f, :, :, 0] @ mx.T + shifts_px[f, 0]
            dense_x = my @ nonrigid[f, :, :, 1] @ mx.T + shifts_px[f, 1]
            out[f] = cv2.remap(
                frames32[f],
                (xx - dense_x).astype(np.float32),
                (yy - dense_y).astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        return out
    for f in range(n_frames):
        dy, dx = float(shifts_px[f, 0]), float(shifts_px[f, 1])
        warp = np.array([[1.0, 0.0, dx - left], [0.0, 1.0, dy - top]], dtype=np.float32)
        out[f] = cv2.warpAffine(
            frames32[f],
            warp,
            (fov_w, fov_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    return out


class BrainMotionStep(Step["BrainMotion"]):
    """Rigidly translate the brain-frame canvas per frame, then crop the sensor FOV.

    Resolves the per-frame ``(dy, dx)`` displacement - an explicit
    ``trajectory_um`` (converted µm→px) if given, else the ``model``-selected
    generator: :func:`physical_brain_motion` (the default driven damped oscillator)
    or :func:`bounded_random_walk` - then :func:`shift_and_crop`s the canvas down to
    the sensor FOV. The canvas must carry a tissue margin
    (``Scene.zeros(acq, margin_px=…)``) at least as large as the maximum shift;
    otherwise the crop would expose the canvas edge and this step raises (rather
    than silently filling with fabricated tissue). ``simulate()`` sizes
    the margin automatically from this spec.

    With ``spec.nonrigid`` a smooth per-frame deformation field (control-node grid,
    see :func:`nonrigid_node_field`) is applied on top via ``cv2.remap`` and written
    to ground truth as ``nonrigid_field``; ``shifts`` stays the rigid component.

    Writes ``shifts`` ``(n_frames, 2)`` to ground truth: the applied content
    displacement ``(dy, dx)`` in **pixels**, matching minian's ``shift_dim``
    order. A motion-correction stage estimates the *correction* - the negation of
    this - so a motion-correction RMSE test compares against ``−shifts``.
    """

    name = "brain_motion"
    domain = "motion"
    consumes_rng = (
        True  # trajectory generators draw noise (unless trajectory_um is given)
    )

    def __call__(self, scene: Scene) -> None:
        acq = self.acq
        canvas = scene.movie.values
        n_frames, canvas_h, canvas_w = canvas.shape
        fov = (acq.image_sensor.n_px_height, acq.image_sensor.n_px_width)
        margin_h, margin_w = (canvas_h - fov[0]) // 2, (canvas_w - fov[1]) // 2
        if (
            margin_h < 0
            or margin_w < 0
            or canvas_h - 2 * margin_h != fov[0]
            or canvas_w - 2 * margin_w != fov[1]
        ):
            raise ValueError(
                f"canvas {(canvas_h, canvas_w)} is not the sensor FOV {fov} plus a "
                "symmetric margin; allocate the scene with Scene.zeros(acq, margin_px=…)."
            )

        shifts_px, nonrigid = brain_motion_fields(
            self.spec, self.acq, n_frames, self.rng
        )
        self._check_within_margin(shifts_px, margin_h, margin_w, nonrigid)

        cropped = shift_and_crop(canvas, shifts_px, fov, nonrigid)
        scene.movie = xr.DataArray(
            cropped,
            dims=list(MOVIE_DIMS),
            coords={
                "frame": np.arange(n_frames),
                "height": np.arange(fov[0]),
                "width": np.arange(fov[1]),
            },
            name="movie",
        )
        scene.truth.shifts = shifts_px
        if nonrigid is not None:
            scene.truth.nonrigid_field = nonrigid
            scene.truth.nonrigid_nodes_y, scene.truth.nonrigid_nodes_x = nonrigid_nodes(
                fov, nonrigid.shape[1:3]
            )

    @staticmethod
    def _check_within_margin(
        shifts_px: np.ndarray,
        margin_h: int,
        margin_w: int,
        nonrigid: np.ndarray | None = None,
    ) -> None:
        """Fail fast if any shift exceeds the tissue margin (the crop would expose
        the canvas edge), telling the caller to enlarge the margin."""
        if len(shifts_px) == 0:
            return
        max_dy = float(np.abs(shifts_px[:, 0]).max())
        max_dx = float(np.abs(shifts_px[:, 1]).max())
        if nonrigid is not None and nonrigid.size:
            # Nodes bound the (spline-interpolated) field up to a small overshoot, which
            # the +1 px guard in the margin absorbs.
            max_dy += float(np.abs(nonrigid[..., 0]).max())
            max_dx += float(np.abs(nonrigid[..., 1]).max())
        if max_dy > margin_h + 1e-9 or max_dx > margin_w + 1e-9:
            raise ValueError(
                f"shift up to (dy={max_dy:.2f}, dx={max_dx:.2f}) px exceeds the tissue "
                f"margin ({margin_h}, {margin_w}) px; allocate the scene with a larger "
                "margin_px (simulate() sizes it from max_shift_um automatically)."
            )

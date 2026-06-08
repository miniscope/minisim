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

from typing import TYPE_CHECKING

import cv2
import numpy as np
import xarray as xr
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
        1.0 + _LOCOMOTION_FREQ_CV * _lowpass(rng.standard_normal(n_hr), _LOCOMOTION_FREQ_TAU_S, dt)
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
        pos = _integrate_dho(accel, dt, w0, zeta).reshape(n_frames, bins, 2).mean(axis=1)
        pos -= pos[0]
        rms = float(np.sqrt(np.mean(pos[:, 0] ** 2 + pos[:, 1] ** 2)))
        return pos / rms if rms > _EPS else pos

    traj = locomotion_fraction * _frames(a_loco) + (1.0 - locomotion_fraction) * _frames(a_noise)

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


def brain_motion_shifts(
    spec, acq, n_frames: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-frame ``(dy, dx)`` displacement in pixels for a ``BrainMotion`` spec.

    The shift-generation half of :class:`BrainMotionStep`, factored out so the step
    *and* the streaming video writer (which renders frame-chunks without ever
    materializing a full movie) produce the **identical** trajectory from the same
    RNG draws - the property the streamer relies on to match ``simulate()``
    bit-for-bit. An explicit ``trajectory_um`` (µm→px) takes precedence, else the
    ``model``-selected generator (:func:`physical_brain_motion` or
    :func:`bounded_random_walk`).
    """
    if spec.trajectory_um is not None:
        if len(spec.trajectory_um) != n_frames:
            raise ValueError(
                f"trajectory_um has {len(spec.trajectory_um)} entries but the "
                f"recording has {n_frames} frames; they must match."
            )
        return np.array(
            [[acq.um_to_px(dy), acq.um_to_px(dx)] for dy, dx in spec.trajectory_um]
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
    canvas: np.ndarray, shifts_px: np.ndarray, fov_shape: tuple[int, int]
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
    """
    n_frames, canvas_h, canvas_w = canvas.shape
    fov_h, fov_w = fov_shape
    top = (canvas_h - fov_h) // 2
    left = (canvas_w - fov_w) // 2
    frames32 = canvas.astype(np.float32, copy=False)
    out = np.empty((n_frames, fov_h, fov_w))
    for f in range(n_frames):
        dy, dx = float(shifts_px[f, 0]), float(shifts_px[f, 1])
        warp = np.array([[1.0, 0.0, dx - left], [0.0, 1.0, dy - top]], dtype=np.float32)
        out[f] = cv2.warpAffine(
            frames32[f], warp, (fov_w, fov_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
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

    Writes ``shifts`` ``(n_frames, 2)`` to ground truth: the applied content
    displacement ``(dy, dx)`` in **pixels**, matching minian's ``shift_dim``
    order. A motion-correction stage estimates the *correction* - the negation of
    this - so a motion-correction RMSE test compares against ``−shifts``.
    """

    name = "brain_motion"
    domain = "motion"
    consumes_rng = True  # trajectory generators draw noise (unless trajectory_um is given)

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

        shifts_px = brain_motion_shifts(self.spec, self.acq, n_frames, self.rng)
        self._check_within_margin(shifts_px, margin_h, margin_w)

        cropped = shift_and_crop(canvas, shifts_px, fov)
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

    @staticmethod
    def _check_within_margin(shifts_px: np.ndarray, margin_h: int, margin_w: int) -> None:
        """Fail fast if any shift exceeds the tissue margin (the crop would expose
        the canvas edge), telling the caller to enlarge the margin."""
        if len(shifts_px) == 0:
            return
        max_dy = float(np.abs(shifts_px[:, 0]).max())
        max_dx = float(np.abs(shifts_px[:, 1]).max())
        if max_dy > margin_h + 1e-9 or max_dx > margin_w + 1e-9:
            raise ValueError(
                f"shift up to (dy={max_dy:.2f}, dx={max_dx:.2f}) px exceeds the tissue "
                f"margin ({margin_h}, {margin_w}) px; allocate the scene with a larger "
                "margin_px (simulate() sizes it from max_shift_um automatically)."
            )

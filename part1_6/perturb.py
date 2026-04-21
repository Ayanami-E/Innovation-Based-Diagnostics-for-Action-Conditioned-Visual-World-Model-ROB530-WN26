"""Synthetic perturbations of GT pose sequences.

All perturbations preserve kinematic plausibility (continuous, smooth)
but inject a physical error of a controlled magnitude. The four types
mirror plan §1.1.
"""

from __future__ import annotations

import numpy as np

from part1.se2 import wrap_angle


def _wrap_theta(arr: np.ndarray) -> np.ndarray:
    """Wrap the theta column of a (T, 3) pose array to (-pi, pi]."""
    out = arr.copy()
    out[:, 2] = np.array([wrap_angle(a) for a in out[:, 2]])
    return out


def apply_perturbation(gt_poses: np.ndarray, kind: str, magnitude: float,
                       window: tuple[int, int] | None = None) -> np.ndarray:
    """Apply one of P1-P4 to a (T, 3) GT pose sequence.

    Parameters
    ----------
    gt_poses : (T, 3) array of (x, y, theta), world-frame, meters/radians.
    kind : 'P1' (constant velocity drift), 'P2' (constant angular bias),
           'P3' (slow drift ramp / constant accel), 'P4' (sine bump).
    magnitude : scalar, units depend on kind. m=0 must reproduce gt_poses.
    window : optional [t1, t2] for P4. Otherwise full episode.

    Returns
    -------
    perturbed observations of shape (T, 3), with theta wrapped.
    """
    gt = np.asarray(gt_poses, dtype=float)
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt_poses must be (T,3); got {gt.shape}")
    T = gt.shape[0]
    t = np.arange(T, dtype=float)

    # Magnitude units (per plan §1.1, intentionally heterogeneous across
    # perturbation types so each m_max = 0.05 still triggers the relevant
    # kinematic order):
    #   P1: m = constant per-step velocity offset (m/step)
    #   P2: m = constant angular bias (radians)
    #   P3: m = constant acceleration (m/step^2)
    #   P4: m = peak sine displacement on the segment (meters)
    # Consequence: cumulative end-drift differs wildly across types and
    # is exposed via the `pos_err_mean / pos_err_max` columns of
    # summary_1_6.csv. The perturbation-vs-metric story is per-type
    # ("how does NIS respond to *this* type's growing magnitude"), not
    # cross-type.

    if kind == "P1":
        # Constant velocity drift in +x: y_t = gt_t + t * (m, 0, 0).
        delta = np.zeros((T, 3))
        delta[:, 0] = magnitude * t
        out = gt + delta

    elif kind == "P2":
        # Constant angular bias.   y_t = gt_t + (0, 0, m)  with m in radians.
        delta = np.zeros((T, 3))
        delta[:, 2] = magnitude
        out = gt + delta

    elif kind == "P3":
        # Constant acceleration in +x: y_t = gt_t + 0.5 * m * t^2 * (1, 0, 0).
        delta = np.zeros((T, 3))
        delta[:, 0] = 0.5 * magnitude * t * t
        out = gt + delta

    elif kind == "P4":
        # Sine bump on the windowed segment, isotropic in (x, y, theta=0.3*).
        if window is None:
            t1, t2 = T // 4, 3 * T // 4
        else:
            t1, t2 = window
            t1 = max(0, int(t1)); t2 = min(T, int(t2))
        if t2 <= t1 + 2:
            return _wrap_theta(gt)
        period = float(t2 - t1)
        tau = (t[t1:t2] - t1) / period
        bump = magnitude * np.sin(2 * np.pi * tau)
        delta = np.zeros((T, 3))
        delta[t1:t2, 0] = bump
        delta[t1:t2, 1] = bump
        delta[t1:t2, 2] = 0.3 * bump
        out = gt + delta

    else:
        raise ValueError(f"unknown perturbation kind {kind!r}")

    return _wrap_theta(out)


def perturbation_severity(perturbed: np.ndarray, gt_poses: np.ndarray
                          ) -> dict:
    """Per-trajectory ground-truth-vs-observation error metrics.

    Returns mean / max positional error in meters and the same for
    orientation in radians (wrapped to (-pi, pi]).
    """
    p = np.asarray(perturbed); g = np.asarray(gt_poses)
    pos_err = np.sqrt((p[:, 0] - g[:, 0]) ** 2 + (p[:, 1] - g[:, 1]) ** 2)
    ori_err = np.array([abs(wrap_angle(a - b))
                        for a, b in zip(p[:, 2], g[:, 2])])
    return dict(
        pos_err_mean=float(pos_err.mean()),
        pos_err_max=float(pos_err.max()),
        ori_err_mean=float(ori_err.mean()),
        ori_err_max=float(ori_err.max()),
    )

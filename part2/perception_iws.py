"""IWS-PushT perception: HSV pink-T detector with PCA orientation.

The IWS PushT scene shows a saturated pink T-block on a white tabletop with
two orange end-effectors and visible robot arms in the background. The
existing Part 1 detector (`part1.perception.detect_pixel`) was tuned for
MuJoCo's purple T at 256x256; the bounds need slight relaxation for the
real-camera pink (slightly higher H, lower S floor) and the morphology
needs a larger kernel to fight JPEG/mp4v ringing.

Path A (HSV + PCA) is what we use here. Returns pose in **image-pixel**
units `(cx_px, cy_px, theta_rad)` -- no homography to world coords, since
the IWS camera is third-person with perspective. NIS in pixel space is
still well-defined: only the `(z - ẑ)ᵀ S⁻¹ (z - ẑ)` quadratic form matters.

Public API:
    detect(rgb_uint8) -> np.ndarray | None    # returns [x_px, y_px, theta]
    visualize_detection(rgb, pose) -> rgb_with_overlay
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from part1.se2 import wrap_angle

# HSV bounds for the pink T. Wider than Part 1's MuJoCo bounds because
# the real-world pink under lab lighting + mp4v compression spreads in S/V.
# Reference values cross-checked against `physics_metrics.extract_state`
# (which uses [130,40,80]-[175,255,255]) and Part 1 (`[135,30,40]-[170,255,255]`).
HSV_LO = np.array([135, 60, 60], dtype=np.uint8)
HSV_HI = np.array([175, 255, 255], dtype=np.uint8)

# Minimum mask area in pixels (256x256 frame) to consider a detection valid.
# The T-block at this scale is ~1500-3000 px; 250 catches partial occlusion
# but rejects spurious texture.
MIN_AREA_PX = 250

# Morphology kernel size. mp4v ringing breaks the mask into 2-3 pieces with
# 3x3; 5x5 closes them reliably.
MORPH_KERNEL = np.ones((5, 5), np.uint8)


def detect_pixel(rgb: np.ndarray,
                 prev_theta: Optional[float] = None) -> Optional[tuple]:
    """Low-level: HSV + PCA on the largest pink mask. Pixel coords."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) RGB; got {rgb.shape}")

    img = cv2.GaussianBlur(rgb, (3, 3), 0.8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < MIN_AREA_PX:
        return None
    mask = (labels == largest).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_AREA_PX:
        return None

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    pts = np.column_stack([xs - cx, ys - cy]).astype(np.float64)
    cov = pts.T @ pts / len(pts)
    _, evecs = np.linalg.eigh(cov)
    stem_axis = evecs[:, 1]
    proj = pts @ stem_axis
    if proj.max() < -proj.min():
        stem_axis = -stem_axis

    # PCA gives an axis in image (x, y_down) coords. Convert to a "world-like"
    # angle with y-up by flipping y, then subtract pi/2 so theta=0 means stem
    # pointing right (matches Part 1 convention).
    theta = np.arctan2(-stem_axis[1], stem_axis[0]) - np.pi / 2
    theta = wrap_angle(theta)

    if prev_theta is not None:
        diff1 = abs(wrap_angle(theta - prev_theta))
        diff2 = abs(wrap_angle(theta + np.pi - prev_theta))
        if diff2 < diff1:
            theta = wrap_angle(theta + np.pi)

    return cx, cy, theta, area


def detect(rgb: np.ndarray,
           prev_theta: Optional[float] = None) -> Optional[np.ndarray]:
    """Public detector. Returns np.array([x_px, y_px, theta_rad]) or None."""
    res = detect_pixel(rgb, prev_theta=prev_theta)
    if res is None:
        return None
    cx, cy, theta, _ = res
    return np.array([cx, cy, theta], dtype=float)


def visualize_detection(rgb: np.ndarray,
                        pose: Optional[np.ndarray],
                        color=(0, 255, 0)) -> np.ndarray:
    """Overlay detection: centroid dot + stem-direction arrow.

    Color is BGR for cv2.line/circle, applied on a BGR copy then converted
    back. Returns RGB so callers can imshow without an extra cvt.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if pose is not None:
        cx, cy, theta = float(pose[0]), float(pose[1]), float(pose[2])
        cv2.circle(bgr, (int(round(cx)), int(round(cy))), 4, color, -1)
        # Arrow direction: stem points along (cos(theta+pi/2), -sin(theta+pi/2))
        # in image coords (y-down inversion).
        arrow_len = 30.0
        dx = arrow_len * np.cos(theta + np.pi / 2)
        dy = -arrow_len * np.sin(theta + np.pi / 2)
        cv2.arrowedLine(bgr, (int(round(cx)), int(round(cy))),
                        (int(round(cx + dx)), int(round(cy + dy))),
                        color, 2, tipLength=0.25)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detection_drop_rate(detections) -> float:
    """Fraction of frames where detect(...) returned None."""
    if len(detections) == 0:
        return 1.0
    n_failed = sum(1 for d in detections if d is None)
    return n_failed / len(detections)


def detect_episode(frames: np.ndarray) -> tuple[list, np.ndarray]:
    """Run detect on a sequence; return (raw_list_with_None, dense_filled).

    `dense_filled` carries the last-known pose forward across drops so
    downstream filters can run without gaps. Callers should still consult
    the raw list to skip update steps on dropped frames.
    """
    raw = []
    dense = np.zeros((len(frames), 3), dtype=float)
    prev_theta = None
    last = None
    for t, frame in enumerate(frames):
        det = detect(frame, prev_theta=prev_theta)
        raw.append(det)
        if det is not None:
            prev_theta = float(det[2])
            last = det
        if last is not None:
            dense[t] = last
        else:
            dense[t] = np.array([np.nan, np.nan, np.nan])
    return raw, dense

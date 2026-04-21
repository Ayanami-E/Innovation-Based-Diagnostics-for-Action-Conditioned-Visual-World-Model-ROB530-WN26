"""T-block perception: HSV segmentation + convex-hull orientation.

Extracted from the original part1_mujoco_kf.py so the same detector can be
reused by filter drivers. Operates on a (uint8) RGB image and returns a
world-frame pose (x, y, theta) or None on failure.
"""

import numpy as np
import cv2

from part1.se2 import wrap_angle

IMG_SIZE = 256

# Visual-centroid offset of the composite T-block in body-frame coords,
# derived analytically from pusht_scene.xml geometry:
#   bar  area = 0.1 * 0.02 m^2 at y = -0.045 m
#   stem area = 0.02 * 0.08 m^2 at y = +0.005 m
# -> body-frame centroid = (0, -0.02278) m.
# The HSV mask centroid measures this visual centroid in world coords; to
# recover the body-origin pose used by MuJoCo qpos we subtract R(theta) *
# CENTROID_OFFSET_BODY.
CENTROID_OFFSET_BODY = np.array([0.0, -0.02278])


def pixel_to_world(cx, cy, img_size=IMG_SIZE, fov_half_width=None):
    if fov_half_width is None:
        fov_half_width = 0.5 * np.tan(np.radians(22.5))
    wx = (cx / img_size - 0.5) * 2 * fov_half_width
    wy = -(cy / img_size - 0.5) * 2 * fov_half_width
    return float(wx), float(wy)


def world_to_pixel(wx, wy, img_size=IMG_SIZE, fov_half_width=None):
    if fov_half_width is None:
        fov_half_width = 0.5 * np.tan(np.radians(22.5))
    cx = (wx / (2 * fov_half_width) + 0.5) * img_size
    cy = (-wy / (2 * fov_half_width) + 0.5) * img_size
    return float(cx), float(cy)


def detect_pixel(image_rgb, prev_theta=None):
    """Low-level detector. Returns (cx_px, cy_px, theta_rad) or None.

    A small fixed-radius Gaussian blur denoises the input (standard in any
    real perception pipeline). HSV segmentation, morphology, and largest-
    connected-component selection isolate the T-block mask. Orientation is
    recovered from principal-component analysis of the mask pixels: for the
    T-block geometry the stem dominates the y-body spread so the major PCA
    axis aligns with the stem direction, and the sign (stem tip side) is
    disambiguated by the extent of projections onto that axis.

    Corruption noise should be injected upstream via
    part1.corruption.apply_corruption.
    """
    img = cv2.GaussianBlur(image_rgb, (3, 3), 0.8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([135, 30, 40])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component so scattered noise pixels
    # elsewhere in the image don't poison the PCA.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    if stats[largest, cv2.CC_STAT_AREA] < 100:
        return None
    mask = (labels == largest).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
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

    theta = np.arctan2(-stem_axis[1], stem_axis[0]) - np.pi / 2
    theta = wrap_angle(theta)

    if prev_theta is not None:
        diff1 = abs(wrap_angle(theta - prev_theta))
        diff2 = abs(wrap_angle(theta + np.pi - prev_theta))
        if diff2 < diff1:
            theta = wrap_angle(theta + np.pi)

    return cx, cy, theta


def detect_tblock(image_rgb, prev_theta=None, fov_half_width=None,
                  compensate_centroid=True):
    """High-level detector returning world-frame pose (x, y, theta) or None.

    When compensate_centroid is True, the reported (x, y) is the T-block
    body origin (what MuJoCo qpos[0:2] stores), not the mask centroid.
    """
    res = detect_pixel(image_rgb, prev_theta=prev_theta)
    if res is None:
        return None
    cx, cy, theta = res
    wx, wy = pixel_to_world(cx, cy, fov_half_width=fov_half_width)
    if compensate_centroid:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        off = R @ CENTROID_OFFSET_BODY
        wx -= off[0]
        wy -= off[1]
    return np.array([wx, wy, theta])

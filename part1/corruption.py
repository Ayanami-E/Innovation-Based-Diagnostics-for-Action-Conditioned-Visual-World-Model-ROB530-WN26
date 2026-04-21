"""Image corruption stage for observation noise sweeps."""

import numpy as np
import cv2


def apply_corruption(img, sigma_pixel=0.0, blur_px=0, brightness_shift=0,
                     rng=None):
    """Inject sensor imperfections into an RGB (or BGR) uint8 image.

    - sigma_pixel: std of additive Gaussian noise (pixel intensities).
    - blur_px: kernel size for Gaussian blur; must be odd and >= 3 to apply.
    - brightness_shift: additive intensity offset applied to all channels.
                        If a scalar is 0, no shift. If rng is provided and
                        brightness_shift > 0, a random offset in
                        [-brightness_shift, +brightness_shift] is drawn.
    Returns uint8 array, same shape as input.
    """
    out = img.astype(np.float32, copy=True)

    if sigma_pixel and sigma_pixel > 0:
        if rng is None:
            noise = np.random.randn(*out.shape) * sigma_pixel
        else:
            noise = rng.standard_normal(out.shape) * sigma_pixel
        out = out + noise

    k = int(blur_px)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        out_u8 = cv2.GaussianBlur(out_u8, (k, k), 0)
        out = out_u8.astype(np.float32)

    if brightness_shift:
        if rng is not None and brightness_shift > 0:
            shift = rng.uniform(-brightness_shift, brightness_shift)
        else:
            shift = brightness_shift
        out = out + shift

    return np.clip(out, 0, 255).astype(np.uint8)

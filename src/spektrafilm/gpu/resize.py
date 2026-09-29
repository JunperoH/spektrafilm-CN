"""B4 GPU-accelerated anti-aliased downscale for the auto-exposure preview.

The auto-exposure stage downsizes the full-resolution image to a small preview
(ResizingService.small_preview -> skimage.transform.rescale, order=0), whose
only purpose is to feed measure_autoexposure_ev, which returns a single scalar
EV. The dominant cost is skimage's anti-aliasing gaussian (scipy correlate1d,
~4.4 s on 24 MP, two separable passes).

This module replaces that gaussian with the validated B1 GPU blur, then does the
cheap order-0 resample on CPU (already anti-aliased, so anti_aliasing=False).

PARITY is on the EV SCALAR, not per-pixel: the EV is a (center-weighted) mean of
luminance and is insensitive to small per-pixel downscale differences. We mirror
skimage's anti-alias sigma = (1/scale - 1)/2 exactly; the B1 IIR gaussian's
boundary handling differs slightly from skimage's at the frame edge, which is
negligible in an averaged statistic (validated in test_autoexposure_gpu_parity).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def antialiased_downscale_dispatch(image, scale_factor: float, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU anti-aliased downscale matching skimage.rescale(order=0). Returns the
    downscaled preview (float64, HxWx3) or None when the CPU path should run
    (no GPU, unsupported input, below the blur's pixel gate, or any error)."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    h, w = arr.shape[:2]
    out_h = int(round(h * scale_factor))
    out_w = int(round(w * scale_factor))
    if out_h < 1 or out_w < 1 or (out_h >= h and out_w >= w):
        return None

    # skimage's anti-alias sigma per spatial axis; near-isotropic for a scalar
    # scale, so use one sigma (the row-axis factor) for the B1 gaussian.
    factor = h / out_h
    sigma = max(0.0, (factor - 1.0) / 2.0)
    if sigma <= 0.0:
        return None

    try:
        from spektrafilm.gpu.blur import gaussian_dispatch
    except ImportError:
        return None
    blurred = gaussian_dispatch(arr, sigma, backend=backend)
    if blurred is None:
        return None  # no GPU / below gate -> let the caller run skimage's AA rescale

    try:
        from skimage.transform import resize

        preview = resize(
            blurred, (out_h, out_w), order=0, anti_aliasing=False, preserve_range=True
        )
    except Exception:
        return None
    return np.asarray(preview, dtype=np.float64)

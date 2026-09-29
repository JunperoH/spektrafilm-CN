from __future__ import annotations

import numpy as np
from skimage.transform import rescale

from spektrafilm.utils.crop_resize import crop_image


class ResizingService:
    """Shared resize/crop operations used by runtime stages."""

    def __init__(self, io_params, film_format_mm: float):
        self._io = io_params
        self.film_format_mm = film_format_mm
        self.pixel_size_um = None

    def crop_and_rescale(self, image: np.ndarray) -> np.ndarray:
        # The INPUT crop is a camera/framing operation: the cropped region
        # becomes the full film gate (pixel_size_um below), so grain, halation
        # and diffusion rescale with the framing. Skew/desqueeze/crop-into-
        # the-grain live at the SCAN end instead (scan_finishing geometry).
        self.pixel_size_um = self.film_format_mm * 1000 / np.max(image.shape[0:2])

        if self._io.crop:
            image = crop_image(image, center=self._io.crop_center, size=self._io.crop_size)

        if self._io.upscale_factor != 1.0:
            self.pixel_size_um /= self._io.upscale_factor
            image = rescale(
                image,
                self._io.upscale_factor,
                channel_axis=2,
                order=3,
            )
        return image
    
    def small_preview(self, image: np.ndarray,
                      max_size: int = 256) -> np.ndarray:
        if max(image.shape[0:2]) > max_size:
            scale_factor = max_size / max(image.shape[0:2])
            # GPU anti-aliased downscale (B1 gaussian + order-0 resample) for the
            # auto-exposure preview; the EV it feeds is parity-tested. Falls back
            # to skimage's AA rescale on any miss.
            try:
                from spektrafilm.gpu.resize import antialiased_downscale_dispatch
                preview = antialiased_downscale_dispatch(image, scale_factor)
                if preview is not None:
                    return preview
            except ImportError:
                pass
            return rescale(
                image,
                scale_factor,
                channel_axis=2,
                order=0,
            )
        return image
    
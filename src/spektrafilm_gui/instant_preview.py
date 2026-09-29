"""Instant input preview for the sorting/input mode (NAT 2026-07-25).

Sorting a roll must not pay the ~5s full-quality input load (2.4s RAW decode +
2.4s colour transform on a 24MP frame). Instead we show the fastest honest
representation of the frame:

  * RAW  -> the JPEG the camera embedded in the file (rawpy extract_thumb,
            ~0.02s). It is the camera's OWN rendering, not the flat negative
            input, but it is instant and plenty to judge keep/toss/framing.
  * RGB  -> the file itself, downscaled on load (PIL draft) so a big TIFF
            still opens fast.

Returns a uint8 (H, W, 3) RGB array the gpu canvas can display directly. This
is a DISPLAY convenience only — it never feeds the simulation, so it can't
affect parity.
"""
from __future__ import annotations

import io

import numpy as np

from spektrafilm_gui.roll_session import classify_path


def load_instant_preview(path: str, *, max_long_edge: int = 2048) -> np.ndarray | None:
    """A fast display image for ``path``; None if it can't be read."""
    try:
        kind = classify_path(path)
        if kind == "raw":
            return _raw_embedded_thumb(path, max_long_edge)
        if kind == "rgb":
            return _rgb_downscaled(path, max_long_edge)
    except Exception:
        return None
    return None


def _raw_embedded_thumb(path: str, max_long_edge: int) -> np.ndarray | None:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        thumb = raw.extract_thumb()
    if thumb.format == rawpy.ThumbFormat.JPEG:
        from PIL import Image

        image = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        image = _pil_capped(image, max_long_edge)
        return np.asarray(image)
    # BITMAP thumbnails are already decoded RGB arrays.
    data = np.asarray(thumb.data)
    return data[..., :3] if data.ndim == 3 else None


def _rgb_downscaled(path: str, max_long_edge: int) -> np.ndarray | None:
    from PIL import Image

    with Image.open(str(path)) as image:
        # draft() lets the decoder skip full-res work for big JPEGs.
        try:
            image.draft("RGB", (max_long_edge, max_long_edge))
        except Exception:
            pass
        image = image.convert("RGB")
        image = _pil_capped(image, max_long_edge)
        return np.asarray(image)


def _pil_capped(image, max_long_edge: int):
    long_edge = max(image.width, image.height)
    if long_edge > max_long_edge:
        scale = max_long_edge / float(long_edge)
        new_size = (max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)))
        from PIL import Image as _Image

        image = image.resize(new_size, _Image.BILINEAR)
    return image

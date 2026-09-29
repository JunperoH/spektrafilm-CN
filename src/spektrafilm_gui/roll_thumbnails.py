"""Film-strip thumbnails for the roll (B1 follow-up).

Small UNEDITED source previews: for RAWs the camera's embedded JPEG thumbnail
(rawpy.extract_thumb, NegPy's approach -- fast, no demosaic) with a half-size
postprocess fallback; for RGB files PIL, with the app's OIIO loader as a
fallback for formats PIL cannot read (e.g. EXR).

Loading runs on the shared QThreadPool via ThumbnailTask so adding a 36-frame
roll never blocks the GUI; the task emits (path, uint8 RGB array) back to the
GUI thread (same signals pattern as controller_runtime.ExportWorker).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import QObject, QRunnable, Signal

MAX_DIM = 256  # >= strip icon width (125) x hidpi ratio 2, keeps icons crisp


def _fit(img: "object") -> "object":
    """PIL thumbnail-fit to MAX_DIM, RGB."""
    from PIL import Image

    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.BILINEAR)
    return img


def _raw_thumbnail(path: str) -> Optional[np.ndarray]:
    import io

    import rawpy
    from PIL import Image

    with rawpy.imread(path) as raw:
        img = None
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
                img.load()
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                img = Image.fromarray(thumb.data)
        except Exception:
            img = None
        if img is None:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,
                no_auto_bright=True,
                user_flip=None,
            )
            img = Image.fromarray(rgb)
    return np.asarray(_fit(img), dtype=np.uint8)


def _rgb_thumbnail(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(_fit(img), dtype=np.uint8)
    except Exception:
        pass
    # PIL could not read it (e.g. EXR) -> the app's OIIO loader + display encode.
    from spektrafilm.utils.io import load_image_oiio

    data = np.asarray(load_image_oiio(path), dtype=np.float64)[..., :3]
    h, w = data.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / MAX_DIM)))
    small = data[::step, ::step]
    small = np.clip(small, 0.0, 1.0) ** (1.0 / 2.2)  # rough display encode
    return (small * 255.0 + 0.5).astype(np.uint8)


def load_thumbnail_array(path: str, kind: str) -> Optional[np.ndarray]:
    """uint8 (h, w, 3) thumbnail with max dimension MAX_DIM, or None."""
    try:
        # NAT 2026-07-22: lift PIL's 89 MP "decompression bomb" ceiling — a
        # server-side DoS guard that only spams a warning on the user's own
        # large frames (same rationale as the display transform).
        from PIL import Image as _PIL_Image
        if _PIL_Image.MAX_IMAGE_PIXELS is not None:
            _PIL_Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
    try:
        arr = _raw_thumbnail(path) if kind == "raw" else _rgb_thumbnail(path)
    except Exception:
        return None
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return np.ascontiguousarray(arr[..., :3])


class ThumbnailSignals(QObject):
    ready = Signal(str, object)  # path, uint8 ndarray or None


class ThumbnailTask(QRunnable):
    def __init__(self, path: str, kind: str):
        super().__init__()
        self.path = path
        self.kind = kind
        self.signals = ThumbnailSignals()

    def run(self) -> None:  # pragma: no cover - thread entry
        arr = load_thumbnail_array(self.path, self.kind)
        self.signals.ready.emit(self.path, arr)

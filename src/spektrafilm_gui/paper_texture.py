"""Paper-texture compositing (NAT 2026-07-26 v1.0.2).

A display-referred finish: user paper-scan textures are blended onto the
RENDERED scan output (and, with padding burn, onto the padding border too). It
never touches the film science — it runs after the pipeline on the display /
export image, so with no active textures the output is bit-identical.

Blend maths operate on float [0, 1] RGB. The 'Photographic set' of fusion
modes covers real paper looks (darkening stains, tooth, light leaks).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

FUSION_MODES: tuple[str, ...] = (
    "Multiply", "Screen", "Overlay", "Soft Light", "Linear Burn", "Grain Merge",
)
DEFAULT_FUSION_MODE = "Multiply"


def _multiply(b, t):
    return b * t


def _screen(b, t):
    return 1.0 - (1.0 - b) * (1.0 - t)


def _overlay(b, t):
    return np.where(b <= 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))


def _soft_light(b, t):
    # W3C / Photoshop soft light.
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(np.clip(b, 0.0, None)))
    return np.where(t <= 0.5, b - (1.0 - 2.0 * t) * b * (1.0 - b),
                    b + (2.0 * t - 1.0) * (d - b))


def _linear_burn(b, t):
    return b + t - 1.0


def _grain_merge(b, t):
    return b + t - 0.5


_BLEND = {
    "Multiply": _multiply,
    "Screen": _screen,
    "Overlay": _overlay,
    "Soft Light": _soft_light,
    "Linear Burn": _linear_burn,
    "Grain Merge": _grain_merge,
}


def blend(base: np.ndarray, texture: np.ndarray, mode: str,
          opacity: float = 1.0) -> np.ndarray:
    """Blend ``texture`` onto ``base`` (both float [0, 1], same shape) with the
    named fusion mode; result clipped to [0, 1]. Unknown mode = Multiply.
    ``opacity`` (0..1) lerps between the base and the fully-blended result, so
    0 = no effect, 1 = full blend (standard layer opacity)."""
    fn = _BLEND.get(mode, _multiply)
    mixed = np.clip(fn(base, texture), 0.0, 1.0)
    o = float(np.clip(opacity, 0.0, 1.0))
    if o >= 1.0:
        return mixed
    if o <= 0.0:
        return base
    return np.clip(base * (1.0 - o) + mixed * o, 0.0, 1.0)


def _fit_texture(texture: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a texture (uint8/float, gray or RGB) to (height, width, 3) float
    [0, 1] via nearest sampling (dependency-free, adequate for paper grain)."""
    tex = np.asarray(texture)
    if tex.ndim == 2:
        tex = tex[:, :, None]
    tex = tex[:, :, :3]
    if tex.shape[2] == 1:
        tex = np.repeat(tex, 3, axis=2)
    if not np.issubdtype(tex.dtype, np.floating):
        tex = tex.astype(np.float32) / 255.0
    else:
        tex = tex.astype(np.float32)
    th, tw = tex.shape[:2]
    if (th, tw) != (height, width):
        ys = (np.linspace(0, th - 1, height)).astype(np.int64)
        xs = (np.linspace(0, tw - 1, width)).astype(np.int64)
        tex = tex[ys][:, xs]
    return np.clip(tex, 0.0, 1.0)


def composite_textures(base: np.ndarray,
                       textures: Iterable[tuple]) -> np.ndarray:
    """Composite an ordered list of (texture_array, fusion_mode[, opacity]) onto
    ``base`` (float [0, 1] RGB). Opacity defaults to 1.0 when the entry omits
    it. Returns a new float [0, 1] array. An empty list returns ``base``
    unchanged (the identity that keeps default output bit-identical)."""
    out = np.asarray(base, dtype=np.float32)
    h, w = out.shape[:2]
    for entry in textures:
        texture, mode = entry[0], entry[1]
        opacity = float(entry[2]) if len(entry) > 2 else 1.0
        if texture is None:
            continue
        out = blend(out, _fit_texture(texture, h, w), mode, opacity)
    return out

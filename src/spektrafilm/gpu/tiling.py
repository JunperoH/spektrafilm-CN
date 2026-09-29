"""Phase 6E: dynamic tiling — the 'fit large work into the budget' path.

Turns gpu/budget.plan_residency's 'will not fit' answer into tile-to-fit:
the image is cut into core tiles, each rendered from a PADDED sub-image (core
+ halo wide enough for the widest neighborhood kernel), and the cores are
stitched. The kernels are unchanged — tiling wraps ABOVE the dispatches,
which already accept arbitrary (H, W, 3) arrays. RNG kernels (grain, glare)
take the tile's GLOBAL origin so their per-pixel streams are tile-invariant;
per-pixel and blur kernels need only the halo.

Tile size is DERIVED, not hardcoded: the largest square tile whose padded
footprint keeps every buffer under max-storage-buffer-binding-size /
max-buffer-size AND the chain total under the VRAM budget — so a 2 GB
integrated GPU tiles finer than an 8 GB card automatically (NegPy's 2048 px
tiles are the reference point for capable hardware).

Seam correctness: the halo must cover the largest blur radius. Gaussian and
exponential-tail blurs decay but never truncate exactly, so the halo uses
HALO_SIGMAS x sigma (+margin); the tiled-vs-untiled seam test pins the
residual within the documented parity tolerance.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from spektrafilm.gpu import budget as _budget
from spektrafilm.gpu.device import GPUDevice, logger

# Halo width per unit blur sigma. 6 sigma leaves a ~1e-8 relative gaussian
# residual at the seam — below f32 noise, so even the DISCRETE grain sampler
# (where any input difference can flip a particle) sees seam pixels no
# differently from interior ones; +16 px guards the IIR blur's warm-up region.
HALO_SIGMAS = 6.0
HALO_MARGIN_PX = 16
MIN_TILE_PX = 256          # below this, tiling overhead dwarfs the work: use CPU
PREFERRED_TILE_PX = 2048   # NegPy reference tile edge on capable hardware


def _limit(limits: dict, *names: str, default: int) -> int:
    for name in names:
        value = limits.get(name)
        if value:
            return int(value)
    return default


def halo_from_sigmas(*sigmas: float) -> int:
    """Halo width covering the widest blur in a chain (0 for pure per-pixel)."""
    widest = max([float(s) for s in sigmas if s and s > 0], default=0.0)
    if widest <= 0:
        return 0
    return int(math.ceil(HALO_SIGMAS * widest)) + HALO_MARGIN_PX


def max_tile_edge(buffers_per_pixel: float, plane_buffers_per_pixel: float = 0.0) -> Optional[int]:
    """Largest square tile edge (px) whose chain footprint fits the device.

    buffers_per_pixel: number of full-image-sized f32 RGB buffers the chain
    holds simultaneously (12 for the fused develop+grain graph). Plane
    buffers (single-channel) count at 1/3 weight via plane_buffers_per_pixel.
    Returns None when even MIN_TILE_PX will not fit (caller uses CPU).
    """
    gpu = GPUDevice.get()
    if gpu.device is None:
        return None
    limits = gpu.limits or {}
    max_binding = _limit(limits, 'max-storage-buffer-binding-size',
                         'maxStorageBufferBindingSize', default=128 * 1024 * 1024)
    max_buffer = _limit(limits, 'max-buffer-size', 'maxBufferSize',
                        default=256 * 1024 * 1024)
    budget = _budget.gpu_budget_bytes()
    per_pixel_one_buffer = 3 * 4                    # f32 RGB
    per_pixel_chain = per_pixel_one_buffer * (buffers_per_pixel + plane_buffers_per_pixel / 3.0)

    # per-buffer limit: edge^2 * 12 B <= min(max_binding, max_buffer)
    edge_binding = int(math.sqrt(min(max_binding, max_buffer) / per_pixel_one_buffer))
    # chain budget: edge^2 * chain B <= budget
    edge_budget = int(math.sqrt(budget / per_pixel_chain))
    edge = min(edge_binding, edge_budget, PREFERRED_TILE_PX)
    if edge < MIN_TILE_PX:
        return None
    return edge


def tile_layout(h: int, w: int, tile_edge: int, halo: int):
    """Yield (core, padded) slice pairs covering the (h, w) image.

    core:   the region this tile OWNS in the output.
    padded: core expanded by the halo, clamped to the image — the sub-image
            the chain actually renders. The core is recovered from the padded
            result at [core.start - padded.start : ... + core span].
    """
    step = tile_edge - 2 * halo
    if step <= 0:
        raise ValueError("tile edge must exceed twice the halo")
    tiles = []
    for y0 in range(0, h, step):
        y1 = min(y0 + step, h)
        py0, py1 = max(0, y0 - halo), min(h, y1 + halo)
        for x0 in range(0, w, step):
            x1 = min(x0 + step, w)
            px0, px1 = max(0, x0 - halo), min(w, x1 + halo)
            tiles.append(((slice(y0, y1), slice(x0, x1)),
                          (slice(py0, py1), slice(px0, px1))))
    return tiles


def run_or_tile(
    image: np.ndarray,
    plan_sizes: list,
    buffers_per_pixel: float,
    plane_buffers_per_pixel: float,
    halo: int,
    render: Callable[[np.ndarray, tuple], Optional[np.ndarray]],
    label: str = "chain",
) -> Optional[np.ndarray]:
    """The 6E ladder in one call: whole-frame when plan_residency(plan_sizes)
    says 'gpu', else tiled with `halo`, else None (CPU rung). `render` is the
    unchanged whole-image chain closure (sub-image, global origin)."""
    if _budget.plan_residency(plan_sizes) == 'gpu':
        return render(image, (0, 0))
    edge = max_tile_edge(buffers_per_pixel, plane_buffers_per_pixel)
    if edge is None or edge <= 2 * halo + 64:
        return None
    logger.info("%s: tiling engaged (edge %d, halo %d)", label, edge, halo)
    return run_tiled(image, edge, halo, render)


def run_tiled(
    image: np.ndarray,
    tile_edge: int,
    halo: int,
    render: Callable[[np.ndarray, tuple], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    """Render `image` tile by tile. `render(padded_subimage, origin_xy)` is
    the UNCHANGED whole-image chain (origin is the padded tile's global (x, y)
    for RNG kernels). Any tile returning None aborts to the caller's fallback
    — never a partially-GPU image."""
    h, w = image.shape[0], image.shape[1]
    out = None
    for core, padded in tile_layout(h, w, tile_edge, halo):
        sub = np.ascontiguousarray(image[padded[0], padded[1]])
        origin = (padded[1].start, padded[0].start)   # (x, y)
        rendered = render(sub, origin)
        if rendered is None:
            logger.info("tiled render: a tile declined; falling back whole")
            return None
        if out is None:
            out = np.empty((h, w) + rendered.shape[2:], dtype=rendered.dtype)
        ys = core[0].start - padded[0].start
        xs = core[1].start - padded[1].start
        out[core[0], core[1]] = rendered[
            ys: ys + (core[0].stop - core[0].start),
            xs: xs + (core[1].stop - core[1].start),
        ]
    return out

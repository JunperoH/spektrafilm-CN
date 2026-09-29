"""Phase 6D: fused develop+grain resident graph — ONE submit, one upload of
log_raw, one readback of the grained density.

The develop chain's output buffer feeds the grain chain directly on device
(the grain passes only ever READ their density input, so the hand-off is a
plain buffer reuse — no copy, no hazard). Before this fusion the two islands
each did their own upload + readback of the full frame; at 24 MP that is
2 x ~292 MB of PCIe traffic and two queue waits saved per render.

Math and pass order are UNCHANGED: the fused graph encodes exactly what
develop_resident.encode and grain_resident.encode encode standalone, so
resident-vs-standalone parity is exact given identical f32 inputs and seeds
(same parity model as gpu/grain_resident.py: statistical vs CPU because the
GPU RNG is its own stream, exact vs the standalone GPU islands).

Eligibility is the INTERSECTION of both islands' gates (each island's plan
helper is reused, not duplicated). Any miss returns None and model/develop.py
runs the existing separate-island path, which remains fully intact.

Safety rails (gpu/budget.py): plan_residency over the SUMMED footprint of
both chains BEFORE any allocation; uncaptured-error trap consumed after the
readback — a trapped error discards the result.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from spektrafilm.gpu.device import logger


def _fused_enabled() -> bool:
    """SPEKTRAFILM_RESIDENT_FUSED env toggle (default ON). 0/off/false/no
    forces the separate develop/grain islands — A/B lever + safety hatch."""
    val = os.environ.get("SPEKTRAFILM_RESIDENT_FUSED", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def develop_grain_fused_dispatch(
    log_raw,
    pixel_size_um: float,
    log_exposure,
    normalized_density_curves,
    dir_couplers,
    grain,
    density_curves_layers,
    profile_type: str,
    gamma_factor=1.0,
    use_fast_stats: bool = False,
    *,
    backend: str = "auto",
) -> Optional[np.ndarray]:
    """Fused GPU dispatch for model/develop.develop's deterministic core PLUS
    apply_grain's sublayers path in one submit. Returns the final grained
    density as float64 (H, W, 3), or None -> the caller's existing
    separate-island / CPU path runs unchanged."""
    if not _fused_enabled():
        return None
    arr = np.asarray(log_raw)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    # Couplers must be active (same gate as the standalone develop island).
    if not getattr(dir_couplers, "active", False):
        return None

    try:
        from spektrafilm.gpu.backend import Backend, resolve_backend
        from spektrafilm.gpu.spectral import MIN_GPU_PIXELS
    except Exception:
        return None
    requested = (backend or "auto").strip().lower()
    if resolve_backend(requested) is not Backend.GPU:
        return None
    if requested == "auto" and arr.shape[0] * arr.shape[1] < MIN_GPU_PIXELS:
        return None

    # Phase 6D rail 1 / 6E ladder: summed footprint of BOTH chains (develop:
    # log + A + B + up to 5 diffusion scratch; grain: accum + optional blur
    # pair + 4 planes; + one staging), checked BEFORE any allocation.
    # 'gpu' -> whole-frame; anything else -> TILE to fit (6E); tiling
    # impossible -> None (CPU).
    try:
        from spektrafilm.gpu.budget import consume_device_error, plan_residency
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        whole_frame = plan_residency([nbytes] * 12 + [nbytes // 3] * 4) == 'gpu'
    except Exception:
        return None

    try:
        from spektrafilm.gpu.develop_resident import develop_plan, get_develop_core
        from spektrafilm.gpu.grain_resident import grain_plan, _get_core as get_grain_core
    except Exception:
        return None

    dev_plan = develop_plan(
        pixel_size_um, log_exposure, normalized_density_curves,
        dir_couplers, profile_type, gamma_factor,
    )
    if dev_plan is None:
        return None
    g_plan = grain_plan(
        pixel_size_um, grain, normalized_density_curves, density_curves_layers,
        profile_type, use_fast_stats,
    )
    if g_plan is None:
        return None

    dev_core = get_develop_core()
    grain_core = get_grain_core()
    if dev_core is None or grain_core is None:
        return None

    density_min = g_plan.pop("density_min")

    def _render(sub: np.ndarray, origin: tuple) -> Optional[np.ndarray]:
        """The unchanged fused chain on one (sub-)image; origin feeds the
        grain RNG so tiles reproduce the whole-frame stream exactly."""
        try:
            import wgpu

            device = dev_core.gpu.device
            img = np.ascontiguousarray(sub, dtype=np.float32)
            h, w, _ = img.shape
            nb = img.nbytes

            log_buf = device.create_buffer(
                size=nb, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
            device.queue.write_buffer(log_buf, 0, img)

            consume_device_error()  # drop stale errors from other work
            encoder = device.create_command_encoder()
            density_buf, dev_owned = dev_core.encode(
                encoder, log_buf, h, w, nb, **dev_plan
            )
            grain_out, grain_owned = grain_core.encode(
                encoder, density_buf, h, w, nb, origin=origin, **g_plan
            )
            staging = device.create_buffer(
                size=nb, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
            )
            encoder.copy_buffer_to_buffer(grain_out, 0, staging, 0, nb)
            device.queue.submit([encoder.finish()])  # ONE submit: both chains + copy
            staging.map_sync(mode=wgpu.MapMode.READ)
            try:
                out = (
                    np.frombuffer(staging.read_mapped(0, nb), dtype=np.float32)
                    .reshape(h, w, 3)
                    .copy()
                )
            finally:
                staging.unmap()
            for b in [log_buf, staging] + dev_owned + grain_owned:
                try:
                    b.destroy()
                except Exception:
                    pass
            # Phase 6D rail 2: discard the readback if the device trapped an error.
            error = consume_device_error()
            if error is not None:
                logger.warning("fused develop+grain trapped '%s'; result discarded", error)
                return None
            return out
        except Exception:
            logger.exception("fused develop+grain render failed; falling back")
            return None

    if whole_frame:
        out = _render(arr, (0, 0))
    else:
        # 6E: tile to fit. Halo covers the widest blur in the chain: couplers
        # diffusion (the exponential tail is encoded as 3 gaussians, widest
        # ~2x the tail sigma), grain dye-cloud and final blurs.
        from spektrafilm.gpu.tiling import halo_from_sigmas, max_tile_edge, run_tiled

        # The tail blur only runs when BOTH its weight and sigma are > 0 (and
        # only downstream of the gaussian): mirror the encode's own condition
        # so an inactive tail does not inflate the halo. The EXPONENTIAL tail
        # decays like exp(-r/lambda), far slower than a gaussian: the halo
        # must reach ~16 lambda for a ~1e-7 residual (halo_from_sigmas
        # multiplies by HALO_SIGMAS=6, so pass 16/6 lambda as the 'sigma') —
        # anything larger than ~1e-7 gets AMPLIFIED by the discrete grain
        # sampler downstream into wide bands of one-quantum flips.
        tail_sigma = 0.0
        if (dev_plan.get("diffusion_tail_weight", 0.0) > 0.0
                and dev_plan.get("diffusion_size_pixel", 0.0) > 0.0):
            tail_sigma = (16.0 / 6.0) * dev_plan.get("diffusion_tail_pixel", 0.0)
        halo = halo_from_sigmas(
            dev_plan.get("diffusion_size_pixel", 0.0),
            tail_sigma,
            float(np.max(g_plan["dye_blur_sigma"])) if np.size(g_plan["dye_blur_sigma"]) else 0.0,
            g_plan.get("final_blur_sigma", 0.0),
        )
        edge = max_tile_edge(12, 4)
        if edge is None or edge <= 2 * halo + 64:
            return None  # cannot tile sensibly on this device: CPU ladder rung
        logger.info("fused develop+grain: tiling engaged (edge %d, halo %d)", edge, halo)
        out = run_tiled(arr, edge, halo, _render)
    if out is None:
        return None
    return np.asarray(out, dtype=np.float64) - np.asarray(density_min, dtype=np.float64)

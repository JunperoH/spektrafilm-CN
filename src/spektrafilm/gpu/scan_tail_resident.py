"""Phase 6D tail: fused scanning tail — exp10 -> glare -> XYZ->RGB in ONE
submit (one upload of log_xyz, one readback of linear RGB).

Reuses the ALREADY-VALIDATED pipelines of the three existing impls
(gpu/elementwise.ElementwiseGPU 'exp10_scale', gpu/glare.GlareGPU field/add +
BlurGPU, gpu/color.ColorGPU matrix pass with cctf off) — no new shader code,
only buffer plumbing: before this fusion each step was its own upload +
readback (3 round-trips, ~0.25 s + dtype churn at 24 MP).

Engages only when black_white_xyz_correction is an identity (the caller
checks the service state) — otherwise the unchanged per-step path runs.
Glare passes are included only when glare is active (parity model then is
glare's: statistical, deterministic run-to-run); with glare off the chain is
deterministic and STRICTLY parity-testable against the per-step GPU path.

Safety rails (gpu/budget.py): plan_residency before allocation,
uncaptured-error trap consumed after readback.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from spektrafilm.gpu.budget import consume_device_error, plan_residency
from spektrafilm.gpu.device import GPUDevice, logger


def _enabled() -> bool:
    val = os.environ.get("SPEKTRAFILM_RESIDENT_SCANTAIL", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def scan_tail_dispatch(
    log_xyz,
    glare,
    illuminant_xyz,
    colourspace,
    illuminant_xy,
    *,
    backend: str = "auto",
) -> Optional[np.ndarray]:
    """xyz = 10**log_xyz; add_glare(xyz); XYZ_to_RGB(xyz) — fused. Returns
    linear RGB float64 (H,W,3) or None (caller runs the per-step path)."""
    if not _enabled():
        return None
    arr = np.asarray(log_xyz)
    if arr.ndim != 3 or arr.shape[2] != 3:
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

    glare_on = (
        glare is not None
        and getattr(glare, "active", False)
        and float(glare.percent) > 0
    )

    try:
        import wgpu

        from spektrafilm.gpu.color import _get_color_gpu, _xyz_to_rgb_matrix
        from spektrafilm.gpu.elementwise import _get_impl as _get_elementwise
        from spektrafilm.gpu.glare import _get_impl as _get_glare, _GLARE_SEED
        from spektrafilm.gpu.tiling import halo_from_sigmas, run_or_tile

        ew = _get_elementwise()
        col = _get_color_gpu()
        gl = _get_glare() if glare_on else None
        if ew is None or col is None or (glare_on and gl is None):
            return None
        M = _xyz_to_rgb_matrix(colourspace, illuminant_xy)
        if M is None or not np.all(np.isfinite(M)):
            return None
    except Exception:
        logger.exception("scan tail setup failed; falling back")
        return None

    def _render(sub: np.ndarray, origin: tuple) -> Optional[np.ndarray]:
        device = GPUDevice.get().device
        img = np.ascontiguousarray(sub, dtype=np.float32)
        h, w, _ = img.shape
        nbytes = img.nbytes
        gx, gy = (w + 7) // 8, (h + 7) // 8

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        UNI = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        src = device.create_buffer(size=nbytes, usage=STO)
        xyz_buf = device.create_buffer(size=nbytes, usage=STO)
        rgb_buf = device.create_buffer(size=nbytes, usage=STO)
        device.queue.write_buffer(src, 0, img)
        owned = [src, xyz_buf, rgb_buf]

        def ubuf(raw):
            b = device.create_buffer(size=raw.nbytes, usage=UNI)
            device.queue.write_buffer(b, 0, np.ascontiguousarray(raw))
            owned.append(b)
            return b

        def bg(layout, entries):
            return device.create_bind_group(
                layout=layout,
                entries=[
                    {"binding": bi, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                    for bi, buf in entries
                ],
            )

        consume_device_error()  # drop stale errors from other work
        encoder = device.create_command_encoder()

        def dispatch(pipe, bind):
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

        # 1. exp10_scale (scale = 1): src -> xyz_buf
        eraw = np.zeros(8, dtype=np.uint32)
        eraw[0:4] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32).view(np.uint32)
        eraw[4] = np.uint32(w)
        eraw[5] = np.uint32(h)
        dispatch(ew._pipes["exp10_scale"], bg(ew._bgl, [(0, ubuf(eraw)), (1, src), (2, xyz_buf)]))

        # 2. glare (optional): field -> blur -> add, in place on xyz_buf
        if glare_on:
            m = float(glare.percent)
            s = float(glare.roughness) * m
            sigma2 = math.log(1.0 + (s * s) / (m * m))
            sigma = math.sqrt(sigma2)
            mu = math.log(m) - sigma2 / 2.0
            nbytes1 = nbytes // 3
            field = device.create_buffer(size=nbytes1, usage=STO)
            owned.append(field)
            fraw = np.zeros(8, dtype=np.uint32)
            fraw[0:3] = [w, h, _GLARE_SEED]
            fraw[3] = np.uint32(int(origin[0]))   # 6E: tile-invariant RNG
            fraw[4:6] = np.array([mu, sigma], dtype=np.float32).view(np.uint32)
            fraw[6] = np.uint32(int(origin[1]))
            dispatch(gl._field_pipe, bg(gl._field_bgl, [(0, ubuf(fraw)), (1, field)]))
            blurred = field
            if float(glare.blur) > 0.0:
                tmp1 = device.create_buffer(size=nbytes1, usage=STO)
                fblur = device.create_buffer(size=nbytes1, usage=STO)
                owned += [tmp1, fblur]
                transient: list = []
                gl._blur.encode_gaussian(
                    encoder, transient, field, tmp1, fblur,
                    float(glare.blur), 3.0, h, w, 1, nbytes1,
                )
                owned += transient
                blurred = fblur
            araw = np.zeros(8, dtype=np.uint32)
            araw[0:2] = [w, h]
            araw[4:7] = (np.asarray(illuminant_xyz, dtype=np.float64) / 100.0).astype(
                np.float32).view(np.uint32)
            dispatch(gl._add_pipe, bg(
                gl._add_bgl, [(2, ubuf(araw)), (3, blurred), (4, xyz_buf)]
            ))

        # 3. XYZ->RGB matrix (colour's own; cctf off, clip off): xyz -> rgb
        craw = np.zeros(16, dtype=np.uint32)
        cfl = np.zeros(12, dtype=np.float32)
        M32 = np.asarray(M, dtype=np.float32)
        cfl[0:3] = M32[0]
        cfl[4:7] = M32[1]
        cfl[8:11] = M32[2]
        craw[0:12] = cfl.view(np.uint32)
        craw[12] = np.uint32(w)
        craw[13] = np.uint32(h)
        dispatch(col._pipeline, bg(col._bgl, [(0, xyz_buf), (1, rgb_buf), (2, ubuf(craw))]))

        staging = device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        owned.append(staging)
        encoder.copy_buffer_to_buffer(rgb_buf, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in owned:
            try:
                b.destroy()
            except Exception:
                pass
        error = consume_device_error()
        if error is not None:
            logger.warning("scan tail trapped '%s'; result discarded", error)
            return None
        return out

    try:
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        # 6E ladder: whole -> tiled -> CPU. Per-pixel except the glare field
        # blur; the glare RNG is tile-invariant via the origin.
        halo = halo_from_sigmas(float(glare.blur) if glare_on else 0.0)
        out = run_or_tile(
            arr, [nbytes] * 3 + [nbytes // 3] * 3, 3, 3, halo, _render,
            label="scan tail",
        )
        if out is None:
            return None
        return np.asarray(out, dtype=np.float64)
    except Exception:
        logger.exception("scan tail fused dispatch failed; falling back")
        return None

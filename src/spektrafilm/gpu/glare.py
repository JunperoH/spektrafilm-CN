"""Phase 6D tail: GPU glare — model/glare.add_glare in one submit.

Chain: lognormal field (seeded pcg4d, glare.wgsl) -> BlurGPU gaussian on the
single-channel field -> add into xyz scaled by illuminant_xyz/100. The
(mean, std) -> (mu, sigma) inversion is done host-side with the EXACT
fast_lognormal_from_mean_std formulas (both are scalars for glare).

PARITY MODEL: statistical, like grain — the CPU path draws from numpy's
UNSEEDED global stream, so values cannot match; the distribution and the blur
do (test_glare_gpu_parity.py: field mean/std vs the analytic moments, and the
added-energy statistics vs CPU). The GPU field uses a FIXED seed, so GPU
renders are deterministic run-to-run — deliberately better than the CPU path.

Safety rails (gpu/budget.py): plan_residency before allocation,
uncaptured-error trap consumed after readback.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.budget import (
    consume_device_error,
    install_error_trap,
    plan_residency,
)
from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "glare.wgsl")
_GLARE_SEED = 0x61A7E  # fixed: GPU glare is reproducible run-to-run


class GlareGPU:
    def __init__(self) -> None:
        import wgpu

        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        install_error_trap()
        self._wgpu = wgpu
        device = self.gpu.device
        with open(_SHADER_PATH, "r", encoding="utf-8") as fh:
            module = device.create_shader_module(code=fh.read())

        un = wgpu.BufferBindingType.uniform
        ro = wgpu.BufferBindingType.read_only_storage
        st = wgpu.BufferBindingType.storage

        def bgl(entries):
            return device.create_bind_group_layout(
                entries=[
                    {"binding": b, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
                    for b, t in entries
                ]
            )

        self._field_bgl = bgl([(0, un), (1, st)])
        self._field_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._field_bgl]),
            compute={"module": module, "entry_point": "glare_field"},
        )
        self._add_bgl = bgl([(2, un), (3, ro), (4, st)])
        self._add_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._add_bgl]),
            compute={"module": module, "entry_point": "glare_add"},
        )
        from spektrafilm.gpu.blur import BlurGPU

        self._blur = BlurGPU()

    def _buf(self, nbytes: int, usage) -> Any:
        return self.gpu.device.create_buffer(size=nbytes, usage=usage)

    def run(self, xyz: np.ndarray, illum_scaled: np.ndarray,
            mu: float, sigma: float, blur_sigma: float) -> np.ndarray:
        wgpu = self._wgpu
        device = self.gpu.device

        img = np.ascontiguousarray(xyz, dtype=np.float32)
        h, w, _ = img.shape
        nbytes3 = img.nbytes
        nbytes1 = nbytes3 // 3
        gx, gy = (w + 7) // 8, (h + 7) // 8

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        UNI = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

        xyz_buf = self._buf(nbytes3, STO)
        field_buf = self._buf(nbytes1, STO)
        device.queue.write_buffer(xyz_buf, 0, img)

        fpar = np.zeros(8, dtype=np.uint32)
        fpar[0:3] = [w, h, _GLARE_SEED]
        fpar[4:6] = np.array([mu, sigma], dtype=np.float32).view(np.uint32)
        fp_buf = self._buf(32, UNI)
        device.queue.write_buffer(fp_buf, 0, fpar)

        apar = np.zeros(8, dtype=np.uint32)
        apar[0:2] = [w, h]
        apar[4:7] = np.asarray(illum_scaled, dtype=np.float32).view(np.uint32)
        ap_buf = self._buf(32, UNI)
        device.queue.write_buffer(ap_buf, 0, apar)

        def bg(layout, entries):
            return device.create_bind_group(
                layout=layout,
                entries=[
                    {"binding": b, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                    for b, buf in entries
                ],
            )

        transient: list = []
        extra: list = []
        encoder = device.create_command_encoder()

        def dispatch(pipe, bind):
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

        dispatch(self._field_pipe, bg(self._field_bgl, [(0, fp_buf), (1, field_buf)]))

        blurred_src = field_buf
        if blur_sigma > 0.0:
            tmp1 = self._buf(nbytes1, STO)
            fblur = self._buf(nbytes1, STO)
            extra += [tmp1, fblur]
            self._blur.encode_gaussian(
                encoder, transient, field_buf, tmp1, fblur,
                float(blur_sigma), 3.0, h, w, 1, nbytes1,
            )
            blurred_src = fblur

        dispatch(self._add_pipe, bg(self._add_bgl, [(2, ap_buf), (3, blurred_src), (4, xyz_buf)]))

        staging = self._buf(nbytes3, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(xyz_buf, 0, staging, 0, nbytes3)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes3), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in [xyz_buf, field_buf, fp_buf, ap_buf, staging] + extra + transient:
            try:
                b.destroy()
            except Exception:
                pass
        return out


_IMPL: Optional[GlareGPU] = None
_IMPL_FAILED = False


def _get_impl() -> Optional[GlareGPU]:
    global _IMPL, _IMPL_FAILED
    if _IMPL_FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = GlareGPU()
        except Exception as exc:
            logger.warning("Glare GPU kernel unavailable (%s); using CPU", exc)
            _IMPL_FAILED = True
            return None
    return _IMPL


def _enabled() -> bool:
    val = os.environ.get("SPEKTRAFILM_GPU_GLARE", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def add_glare_dispatch(xyz, illuminant_xyz, glare, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for model/glare.add_glare. float64 (H,W,3) or None (the
    caller's CPU path runs). Statistical parity model — see module docstring."""
    if not _enabled():
        return None
    if glare is None or not getattr(glare, "active", False) or not float(glare.percent) > 0:
        return None  # CPU add_glare is a no-op then; keep it byte-exact
    arr = np.asarray(xyz)
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

    try:
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        # xyz + staging (image-sized) + up to 3 single-channel planes
        if plan_residency([nbytes] * 2 + [nbytes // 3] * 3) != 'gpu':
            return None
    except Exception:
        return None

    impl = _get_impl()
    if impl is None:
        return None

    try:
        # Host-side (m, s) -> (mu, sigma): exactly fast_lognormal_from_mean_std.
        m = float(glare.percent)
        s = float(glare.roughness) * m
        sigma2 = math.log(1.0 + (s * s) / (m * m))
        sigma = math.sqrt(sigma2)
        mu = math.log(m) - sigma2 / 2.0
        illum_scaled = np.asarray(illuminant_xyz, dtype=np.float64) / 100.0

        consume_device_error()  # drop stale errors from other work
        out = impl.run(arr, illum_scaled, mu, sigma, float(glare.blur))
        error = consume_device_error()
        if error is not None:
            logger.warning("GPU glare trapped '%s'; result discarded", error)
            return None
        return np.asarray(out, dtype=np.float64)
    except Exception:
        logger.exception("GPU glare dispatch failed; falling back to CPU")
        return None

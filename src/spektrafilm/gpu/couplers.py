"""G2 GPU DIR-coupler exposure correction (the per-pixel matvec).

Ports the matvec inside model/couplers.compute_exposure_correction_dir_couplers:
    silver = positive ? density_max - density_cmy : density_cmy
    silver += shift * silver^2            (shift default 0 in the develop path)
    correction = contract('ijk,km->ijm', silver, couplers_matrix)

couplers_matrix is pre-scaled by amount on the host. The spatial diffusion
(gaussian + exponential mix) and the final `log_raw - correction` subtract are
applied separately (existing blur + a pointwise op); this kernel emits the
pre-diffusion correction. Strict per-pixel max_abs parity (pure matvec, ~1e-6
f32). Standalone kernel + parity gate (Phase 3 cadence); assembled into the
resident develop chain in G2c.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "couplers.wgsl")
_PARAMS_NBYTES = 80  # 4*u32 + 4*vec4<f32> (density_max + 3 matrix rows)


class CouplersGPU:
    def __init__(self) -> None:
        import wgpu

        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        device = self.gpu.device
        with open(_SHADER_PATH, "r", encoding="utf-8") as fh:
            module = device.create_shader_module(code=fh.read())
        self._bgl = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
            ]
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": module, "entry_point": "couplers_correction"}
        )

    @staticmethod
    def _pack_params(width: int, height: int, positive: bool, density_max, matrix, shift: float) -> np.ndarray:
        dmax = np.asarray(density_max, dtype=np.float32).reshape(-1)[:3]
        M = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
        raw = np.zeros(20, dtype=np.uint32)
        raw[0] = np.uint32(width)
        raw[1] = np.uint32(height)
        raw[2] = np.uint32(1 if positive else 0)
        fl = np.zeros(16, dtype=np.float32)
        fl[0:3] = dmax
        fl[3] = np.float32(shift)
        fl[4:7] = M[0]
        fl[8:11] = M[1]
        fl[12:15] = M[2]
        raw[4:20] = fl.view(np.uint32)
        return raw

    def correction(self, density_cmy: np.ndarray, matrix: np.ndarray, *,
                   positive: bool, density_max, high_exposure_shift: float = 0.0) -> np.ndarray:
        import wgpu

        img = np.ascontiguousarray(density_cmy, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("couplers correction expects an (H, W, 3) density_cmy array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        device = self.gpu.device

        in_buf = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        out_buf = device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        device.queue.write_buffer(in_buf, 0, img)
        raw = self._pack_params(w, h, positive, density_max, matrix, high_exposure_shift)
        params = device.create_buffer(size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        device.queue.write_buffer(params, 0, raw)

        bind_group = device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
                {"binding": 1, "resource": {"buffer": in_buf, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": out_buf, "offset": 0, "size": nbytes}},
            ],
        )
        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        cpass.end()
        staging = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in (in_buf, out_buf, params, staging):
            try:
                b.destroy()
            except Exception:
                pass
        return out


_COUPLERS_GPU: Optional[CouplersGPU] = None
_COUPLERS_GPU_FAILED = False


def _get_couplers_gpu() -> Optional[CouplersGPU]:
    global _COUPLERS_GPU, _COUPLERS_GPU_FAILED
    if _COUPLERS_GPU_FAILED:
        return None
    if _COUPLERS_GPU is None:
        try:
            _COUPLERS_GPU = CouplersGPU()
        except Exception as exc:
            logger.warning("Couplers GPU kernel unavailable (%s); using CPU", exc)
            _COUPLERS_GPU_FAILED = True
            return None
    return _COUPLERS_GPU


def couplers_correction_dispatch(
    density_cmy, couplers_matrix, *, positive: bool, density_max, high_exposure_shift: float = 0.0, backend: str = "auto"
) -> Optional[np.ndarray]:
    """GPU pre-diffusion DIR-coupler correction = contract('ijk,km->ijm', silver, M).
    Returns float64 (H,W,3) or None (caller runs CPU). Diffusion + subtract handled
    by the caller (existing blur + pointwise)."""
    from spektrafilm.gpu.backend import Backend, resolve_backend
    from spektrafilm.gpu.spectral import MIN_GPU_PIXELS

    arr = np.asarray(density_cmy)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    requested = (backend or "auto").strip().lower()
    if resolve_backend(requested) is not Backend.GPU:
        return None
    if requested == "auto" and arr.shape[0] * arr.shape[1] < MIN_GPU_PIXELS:
        return None
    impl = _get_couplers_gpu()
    if impl is None:
        return None
    try:
        return impl.correction(
            arr, couplers_matrix, positive=positive, density_max=density_max,
            high_exposure_shift=high_exposure_shift,
        ).astype(np.float64)
    except Exception:
        logger.exception("couplers correction GPU dispatch failed; falling back to CPU")
        return None

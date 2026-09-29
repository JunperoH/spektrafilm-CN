"""GPU kernel for the Phase 11C 'synthesis' grain model (NAT 2026-07-22).

Mirrors model/grain_synthesis.py:_synthesis_channel: one (channel, depth
layer) Boolean field per dispatch; the caller sums layers exactly like the
CPU path.

PARITY MODEL: statistical (the house grain standard) — WGSL has no 64-bit
integers, so the counter hash is a 32-bit mix instead of the CPU's 64-bit
murmur; the calibration (lam * E[pi R^2] = D_max * ln10) is identical, so
the expected density matches the CPU field and the texture statistics agree
within the tolerance documented in tests/test_grain_synthesis_gpu.py.
Deterministic run-to-run (pure counter hash). Returns None on any miss so
the numba path runs unchanged.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "grain_synthesis.wgsl")
_PARAMS_NBYTES = 80   # 8 x u32/i32 + 12 x f32, must match struct Params


class GrainSynthesisGPU:
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
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = device.create_compute_pipeline(
            layout=layout,
            compute={"module": module, "entry_point": "synthesis_grain"},
        )
        self._pool: dict[str, tuple[int, Any]] = {}
        self._staging: Optional[tuple[int, Any]] = None

    def _pooled(self, label: str, nbytes: int, usage: int) -> Any:
        entry = self._pool.get(label)
        if entry is not None and entry[0] >= nbytes:
            return entry[1]
        if entry is not None:
            try:
                entry[1].destroy()
            except Exception:
                pass
        buf = self.gpu.device.create_buffer(size=nbytes, usage=usage)
        self._pool[label] = (nbytes, buf)
        return buf

    def _staging_for(self, nbytes: int) -> Any:
        import wgpu

        if self._staging is not None and self._staging[0] >= nbytes:
            return self._staging[1]
        if self._staging is not None:
            try:
                self._staging[1].destroy()
            except Exception:
                pass
        buf = self.gpu.device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        self._staging = (nbytes, buf)
        return buf

    def run(self, p_map: np.ndarray, *, pixel_um: float, mu_r: float,
            sigma_r: float, r_max: float, lam_area: float,
            aperture_sigma_um: float, samples: int, cell_um: float,
            max_grains: int, epsilon: float, seed: int) -> np.ndarray:
        import wgpu

        field = np.ascontiguousarray(p_map, dtype=np.float32)
        h, w = field.shape
        nbytes = field.nbytes
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                 | wgpu.BufferUsage.COPY_SRC)
        src = self._pooled("p_map", nbytes, usage)
        dst = self._pooled("out", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, field)

        inv_cell = 1.0 / cell_um
        rings = 1 + int(r_max * inv_cell)
        raw = np.zeros(20, dtype=np.uint32)
        raw[0] = np.uint32(w)
        raw[1] = np.uint32(h)
        raw[2] = np.uint32(max(1, int(samples)))
        raw[3] = np.uint32(max(1, int(max_grains)))
        raw[4] = np.uint32(int(seed) & 0xFFFFFFFF)
        raw[5] = np.int32(rings).view(np.uint32)
        floats = np.array([pixel_um, mu_r, sigma_r, r_max,
                           lam_area * cell_um * cell_um, inv_cell, cell_um,
                           aperture_sigma_um, epsilon, 0.0, 0.0, 0.0],
                          dtype=np.float32)
        raw[8:20] = floats.view(np.uint32)
        params = self.gpu.device.create_buffer(
            size=_PARAMS_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.gpu.device.queue.write_buffer(params, 0, raw)

        bind_group = self.gpu.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
            ])
        encoder = self.gpu.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        cpass.end()
        staging = self._staging_for(nbytes)
        encoder.copy_buffer_to_buffer(dst, 0, staging, 0, nbytes)
        self.gpu.device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = (np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32)
                   .reshape(h, w).copy())
        finally:
            staging.unmap()
        try:
            params.destroy()
        except Exception:
            pass
        return out


_IMPL: Optional[GrainSynthesisGPU] = None
_FAILED = False


def _get_impl() -> Optional[GrainSynthesisGPU]:
    global _IMPL, _FAILED
    if _FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = GrainSynthesisGPU()
        except Exception as exc:
            logger.warning("Synthesis grain GPU kernel unavailable (%s); using CPU", exc)
            _FAILED = True
            return None
    return _IMPL


def synthesis_grain_dispatch(p_map, *, pixel_um: float, mu_r: float,
                             sigma_r: float, r_max: float, lam_area: float,
                             aperture_sigma_um: float, samples: int,
                             cell_um: float, max_grains: int, epsilon: float,
                             seed: int, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU hook for apply_grain_synthesis: ONE channel/layer field, float64
    out or None. No minimum-pixel gate — the model is expensive per pixel,
    so the GPU pays off at any size."""
    from spektrafilm.gpu.backend import Backend, resolve_backend

    field = np.asarray(p_map)
    if field.ndim != 2:
        return None
    chosen = resolve_backend((backend or "auto").strip().lower())
    if chosen is not Backend.GPU:
        return None
    impl = _get_impl()
    if impl is None:
        return None
    try:
        out = impl.run(field, pixel_um=float(pixel_um), mu_r=float(mu_r),
                       sigma_r=float(sigma_r), r_max=float(r_max),
                       lam_area=float(lam_area),
                       aperture_sigma_um=float(aperture_sigma_um),
                       samples=int(samples), cell_um=float(cell_um),
                       max_grains=int(max_grains), epsilon=float(epsilon),
                       seed=int(seed))
    except Exception:
        logger.exception("Synthesis grain GPU dispatch failed; falling back to CPU")
        return None
    return np.asarray(out, dtype=np.float64)

"""GPU kernel for the Phase 11A 'preview' grain model (2026-07-21 polish).

Mirrors model/grain.py:apply_grain_preview: per-channel Gaussian noise with
the production model's closed-form std, all three channels in one dispatch;
the final grain blur reuses the B1 fast_gaussian_filter chokepoint (itself
GPU-accelerated), exactly like the CPU path.

PARITY MODEL: statistical (the house grain standard) — the mean field and
per-pixel std are deterministic functions of density and match the CPU
closed form tightly; the noise stream is the GPU's own pcg4d counter hash
(deterministic run-to-run). Monochrome (B&W) draws ONE shared field.
Returns None on any miss so the CPU path runs unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger
from spektrafilm.gpu.spectral import MIN_GPU_PIXELS

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "grain_preview.wgsl")
_PARAMS_NBYTES = 80   # 4 u32 + 4 vec4<f32>, must match struct Params


class GrainPreviewGPU:
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
            compute={"module": module, "entry_point": "preview_grain"},
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

    def run(self, density_cmy: np.ndarray, density_max, n_particles,
            uniformity, density_min, seed: int, monochrome: bool) -> np.ndarray:
        import wgpu

        img = np.ascontiguousarray(density_cmy, dtype=np.float32)
        h, w, _ = img.shape
        nbytes = img.nbytes
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                 | wgpu.BufferUsage.COPY_SRC)
        src = self._pooled("density", nbytes, usage)
        dst = self._pooled("out", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, img)

        raw = np.zeros(20, dtype=np.uint32)
        raw[0] = np.uint32(w)
        raw[1] = np.uint32(h)
        raw[2] = np.uint32(int(seed) & 0xFFFFFFFF)
        raw[3] = np.uint32(1 if monochrome else 0)
        for offset, values in ((4, density_max), (8, n_particles),
                               (12, uniformity), (16, density_min)):
            vec = np.zeros(4, dtype=np.float32)
            vec[:3] = np.asarray(values, dtype=np.float32)
            raw[offset:offset + 4] = vec.view(np.uint32)
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
                   .reshape(h, w, 3).copy())
        finally:
            staging.unmap()
        try:
            params.destroy()
        except Exception:
            pass
        return out


_IMPL: Optional[GrainPreviewGPU] = None
_FAILED = False


def _get_impl() -> Optional[GrainPreviewGPU]:
    global _IMPL, _FAILED
    if _FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = GrainPreviewGPU()
        except Exception as exc:
            logger.warning("Preview grain GPU kernel unavailable (%s); using CPU", exc)
            _FAILED = True
            return None
    return _IMPL


def preview_grain_dispatch(density_cmy, density_max, n_particles, uniformity,
                           density_min, seed: int = 1000,
                           monochrome: bool = False, *,
                           backend: str = "auto") -> Optional[np.ndarray]:
    """GPU hook for apply_grain_preview (pre final-blur). float64 out or None."""
    from spektrafilm.gpu.backend import Backend, resolve_backend

    img = np.asarray(density_cmy)
    if img.ndim != 3 or img.shape[2] != 3:
        return None
    requested = (backend or "auto").strip().lower()
    chosen = resolve_backend(requested)
    if chosen is not Backend.GPU:
        return None
    if requested == "auto" and img.shape[0] * img.shape[1] < MIN_GPU_PIXELS:
        return None
    impl = _get_impl()
    if impl is None:
        return None
    try:
        out = impl.run(img, density_max, n_particles, uniformity, density_min,
                       seed, monochrome)
    except Exception:
        logger.exception("Preview grain GPU dispatch failed; falling back to CPU")
        return None
    return np.asarray(out, dtype=np.float64)

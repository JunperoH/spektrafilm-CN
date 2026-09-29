"""Task 7 elementwise transcendental dispatches (shaders/elementwise.wgsl).

GPU drop-ins for the per-pixel transcendental tails that stayed on CPU:
    scale_log10_dispatch:        log10(fmax(x * c, 0) + 1e-10)   (filming tail)
    exp10_scale_log10_dispatch:  log10(fmax(10**x * c, 0) + 1e-10) (printing tail)
    exp10_scale_dispatch:        10**x * c                       (scanning)

Precision: the software exp2/log2 (Gotcha 5 fix) gives ~3e-7 relative, so the
log10-domain outputs match the CPU f64 within ~1e-6 (parity budget 1e-5).
Each returns float64 (H, W, 3) or None (caller runs the CPU path unchanged).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger
from spektrafilm.gpu.spectral import MIN_GPU_PIXELS

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "elementwise.wgsl")
_PARAMS_NBYTES = 32  # vec4<f32> + 4 u32
_ENTRIES = ("scale_log10", "exp10_scale_log10", "exp10_scale")


class ElementwiseGPU:
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
        self._pipes = {
            name: device.create_compute_pipeline(
                layout=layout, compute={"module": module, "entry_point": name}
            )
            for name in _ENTRIES
        }

    def apply(self, entry: str, image: np.ndarray, scale) -> np.ndarray:
        import wgpu

        img = np.ascontiguousarray(image, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("elementwise kernel expects an (H, W, 3) array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        device = self.gpu.device

        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        src = device.create_buffer(size=nbytes, usage=usage)
        dst = device.create_buffer(size=nbytes, usage=usage)
        device.queue.write_buffer(src, 0, img)

        raw = np.zeros(8, dtype=np.uint32)
        s3 = np.broadcast_to(np.asarray(scale, dtype=np.float32).reshape(-1), (3,)) \
            if np.ndim(scale) == 0 or np.size(scale) == 1 \
            else np.asarray(scale, dtype=np.float32).reshape(-1)[:3]
        fl = np.zeros(4, dtype=np.float32)
        fl[0:3] = s3
        raw[0:4] = fl.view(np.uint32)
        raw[4] = np.uint32(w)
        raw[5] = np.uint32(h)
        params = device.create_buffer(size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        device.queue.write_buffer(params, 0, raw)

        bind = device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
                {"binding": 1, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
            ],
        )
        encoder = device.create_command_encoder()
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self._pipes[entry])
        cp.set_bind_group(0, bind)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        cp.end()
        staging = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(dst, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in (src, dst, params, staging):
            try:
                b.destroy()
            except Exception:
                pass
        return out


_IMPL: Optional[ElementwiseGPU] = None
_IMPL_FAILED = False


def _get_impl() -> Optional[ElementwiseGPU]:
    global _IMPL, _IMPL_FAILED
    if _IMPL_FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = ElementwiseGPU()
        except Exception as exc:
            logger.warning("Elementwise GPU kernels unavailable (%s); using CPU", exc)
            _IMPL_FAILED = True
            return None
    return _IMPL


def _dispatch(entry: str, image, scale, backend: str) -> Optional[np.ndarray]:
    from spektrafilm.gpu.backend import Backend, resolve_backend

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    requested = (backend or "auto").strip().lower()
    if resolve_backend(requested) is not Backend.GPU:
        return None
    if requested == "auto" and arr.shape[0] * arr.shape[1] < MIN_GPU_PIXELS:
        return None
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    if s.size not in (1, 3) or not np.all(np.isfinite(s)):
        return None
    impl = _get_impl()
    if impl is None:
        return None
    try:
        return impl.apply(entry, arr, s).astype(np.float64)
    except Exception:
        logger.exception("Elementwise GPU dispatch failed; falling back to CPU")
        return None


def scale_log10_dispatch(image, scale, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for: log10(fmax(image * scale, 0) + 1e-10)."""
    return _dispatch("scale_log10", image, scale, backend)


def exp10_scale_log10_dispatch(image, scale, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for: log10(fmax((10 ** image) * scale, 0) + 1e-10)."""
    return _dispatch("exp10_scale_log10", image, scale, backend)


def exp10_scale_dispatch(image, scale=1.0, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for: (10 ** image) * scale."""
    return _dispatch("exp10_scale", image, scale, backend)

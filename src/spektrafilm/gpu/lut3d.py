"""Phase 6D tail: 3D PCHIP LUT apply on the GPU.

GPU drop-in for utils/fast_interp_lut.apply_lut_pchip_3d as used by
utils/lut.compute_with_lut (the enlarger/scanner spectral LUT applies). The
monotone slopes and per-cell bounds come from the SAME host prepare the CPU
uses (prepare_lut_pchip_3d) and are cached per LUT content together with
their device buffers, so repeated applies of one LUT upload nothing but the
image.

PARITY: strict per-pixel max_abs (deterministic; same prepared slopes, only
f32 interpolation arithmetic differs). Tolerance documented in
tests/test_lut3d_gpu_parity.py.

Safety rails (gpu/budget.py): plan_residency before allocation,
uncaptured-error trap consumed after readback.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.budget import (
    consume_device_error,
    install_error_trap,
    plan_residency,
)
from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "lut3d_pchip.wgsl")


class Lut3dGPU:
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
        ro = wgpu.BufferBindingType.read_only_storage
        entries = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform}}]
        for b in range(1, 8):
            entries.append({"binding": b, "visibility": wgpu.ShaderStage.COMPUTE,
                            "buffer": {"type": ro}})
        entries.append({"binding": 8, "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.storage}})
        self._bgl = device.create_bind_group_layout(entries=entries)
        self._pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bgl]),
            compute={"module": module, "entry_point": "lut3d_pchip"},
        )
        # prepared-LUT cache: content digest -> (size, [6 device buffers])
        self._cache: dict[int, tuple[int, list]] = {}

    def _prepared_buffers(self, lut: np.ndarray) -> Optional[tuple[int, list]]:
        import wgpu

        lut64 = np.ascontiguousarray(lut, dtype=np.float64)
        key = hash(lut64.tobytes())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        from spektrafilm.utils.fast_interp_lut import prepare_lut_pchip_3d

        prepared = prepare_lut_pchip_3d(lut64)  # lut, sx, sy, sz, cmin, cmax
        size = int(prepared[0].shape[0])
        if size < 2:
            return None
        bufs = []
        for arr in prepared:
            a32 = np.ascontiguousarray(arr, dtype=np.float32)
            buf = self.gpu.device.create_buffer(
                size=a32.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
            self.gpu.device.queue.write_buffer(buf, 0, a32)
            bufs.append(buf)
        if len(self._cache) > 8:
            for _, (_, old) in self._cache.items():
                for b in old:
                    try:
                        b.destroy()
                    except Exception:
                        pass
            self._cache.clear()
        self._cache[key] = (size, bufs)
        return self._cache[key]

    def run(self, image: np.ndarray, lut: np.ndarray,
            xmin: np.ndarray, xmax: np.ndarray) -> Optional[np.ndarray]:
        wgpu = self._wgpu
        device = self.gpu.device

        prepared = self._prepared_buffers(lut)
        if prepared is None:
            return None
        size, lut_bufs = prepared

        img = np.ascontiguousarray(image, dtype=np.float32)
        h, w, _ = img.shape
        nbytes = img.nbytes

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        src = device.create_buffer(size=nbytes, usage=STO)
        dst = device.create_buffer(size=nbytes, usage=STO)
        device.queue.write_buffer(src, 0, img)

        raw = np.zeros(12, dtype=np.uint32)
        raw[0:3] = [w, h, size]
        fl = np.zeros(8, dtype=np.float32)
        fl[0:3] = np.asarray(xmin, dtype=np.float64)
        fl[4:7] = 1.0 / (np.asarray(xmax, dtype=np.float64) - np.asarray(xmin, dtype=np.float64))
        raw[4:12] = fl.view(np.uint32)
        params = device.create_buffer(
            size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        device.queue.write_buffer(params, 0, raw)

        entries = [{"binding": 0, "resource": {"buffer": params, "offset": 0, "size": 48}},
                   {"binding": 1, "resource": {"buffer": src, "offset": 0, "size": nbytes}}]
        for b, buf in enumerate(lut_bufs, start=2):
            entries.append({"binding": b, "resource": {"buffer": buf, "offset": 0, "size": buf.size}})
        entries.append({"binding": 8, "resource": {"buffer": dst, "offset": 0, "size": nbytes}})
        bind = device.create_bind_group(layout=self._bgl, entries=entries)

        encoder = device.create_command_encoder()
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(self._pipeline)
        cp.set_bind_group(0, bind)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        cp.end()
        staging = device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
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


_IMPL: Optional[Lut3dGPU] = None
_IMPL_FAILED = False


def _get_impl() -> Optional[Lut3dGPU]:
    global _IMPL, _IMPL_FAILED
    if _IMPL_FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = Lut3dGPU()
        except Exception as exc:
            logger.warning("3D LUT GPU kernel unavailable (%s); using CPU", exc)
            _IMPL_FAILED = True
            return None
    return _IMPL


def _enabled() -> bool:
    val = os.environ.get("SPEKTRAFILM_GPU_LUT3D", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def apply_lut_pchip_3d_dispatch(
    lut, image, *, xmin=(0.0, 0.0, 0.0), xmax=(1.0, 1.0, 1.0), backend: str = "auto"
) -> Optional[np.ndarray]:
    """GPU drop-in for apply_lut_pchip_3d(lut, (image-xmin)/(xmax-xmin)).
    The [xmin, xmax] -> [0, 1] normalization is folded INTO the kernel so the
    caller never materializes the normalized f64 array (0.6 s at 24 MP).
    float64 (H,W,3) or None (the caller's CPU path runs unchanged)."""
    if not _enabled():
        return None
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    lut_arr = np.asarray(lut)
    if lut_arr.ndim != 4 or lut_arr.shape[3] != 3 or lut_arr.shape[0] < 2:
        return None
    if not (lut_arr.shape[0] == lut_arr.shape[1] == lut_arr.shape[2]):
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

    impl = _get_impl()
    if impl is None:
        return None
    try:
        from spektrafilm.gpu.tiling import run_or_tile

        xmin_a = np.asarray(xmin, dtype=np.float64).reshape(-1)
        xmax_a = np.asarray(xmax, dtype=np.float64).reshape(-1)
        if xmin_a.size == 1:
            xmin_a = np.full(3, float(xmin_a))
        if xmax_a.size == 1:
            xmax_a = np.full(3, float(xmax_a))
        if not (np.all(np.isfinite(lut_arr)) and np.all(xmax_a > xmin_a)):
            return None

        def _render(sub, _origin):
            consume_device_error()  # drop stale errors from other work
            res = impl.run(sub, lut_arr, xmin_a, xmax_a)
            if res is None:
                return None
            error = consume_device_error()
            if error is not None:
                logger.warning("3D LUT GPU trapped '%s'; result discarded", error)
                return None
            return res

        # 6E ladder: pure per-pixel (halo 0)
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        lut_nbytes = int(lut_arr.size * 4)
        out = run_or_tile(arr, [nbytes] * 3 + [lut_nbytes] * 6, 3, 0, 0,
                          _render, label="3D LUT")
        if out is None:
            return None
        return np.asarray(out, dtype=np.float64)
    except Exception:
        logger.exception("3D LUT GPU dispatch failed; falling back to CPU")
        return None

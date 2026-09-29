"""Phase 6D tail: fused spectral-upsampling apply on the GPU.

GPU drop-in for the per-pixel runtime of rgb_to_raw_hanatos2025 (and any
tc-LUT method with the same runtime shape): RGB->XYZ (colour's own matrix +
CAT16, recovered host-side exactly as gpu/color.py does), chromaticity + b,
tri2quad, Mitchell-Netravali bicubic 2D LUT sample with reflect boundaries,
scale by b — ONE kernel, one upload, one readback, replacing a GPU matrix
round-trip plus ~1.4 s of CPU numpy/numba tc math at 24 MP.

PARITY: strict per-pixel max_abs vs the f64 CPU composition (deterministic,
no RNG). The LUT itself is bit-identical input data on both paths; only f32
arithmetic separates the results. Tolerance documented in
tests/test_tc_lut_gpu_parity.py.

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

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "tc_lut.wgsl")
_PARAMS_NBYTES = 64


class TcLutGPU:
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
        self._bgl = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": ro}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": ro}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
            ]
        )
        self._pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bgl]),
            compute={"module": module, "entry_point": "rgb_to_raw_tc"},
        )
        # LUTs are small and stable across a render session: cache the device
        # buffer keyed on the array's identity + a content digest.
        self._lut_cache: dict[int, tuple[int, Any]] = {}

    def _lut_buffer(self, lut32: np.ndarray) -> Any:
        import wgpu

        key = hash(lut32.tobytes())
        entry = self._lut_cache.get(key)
        if entry is not None:
            return entry[1]
        buf = self.gpu.device.create_buffer(
            size=lut32.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.gpu.device.queue.write_buffer(buf, 0, lut32)
        if len(self._lut_cache) > 8:   # a render uses 1-2 LUTs; keep it tiny
            for _, (_, old) in self._lut_cache.items():
                try:
                    old.destroy()
                except Exception:
                    pass
            self._lut_cache.clear()
        self._lut_cache[key] = (lut32.nbytes, buf)
        return buf

    def run(self, rgb: np.ndarray, lut: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        wgpu = self._wgpu
        device = self.gpu.device

        img = np.ascontiguousarray(rgb, dtype=np.float32)
        h, w, _ = img.shape
        nbytes = img.nbytes
        lut32 = np.ascontiguousarray(lut, dtype=np.float32)
        size = lut32.shape[0]

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        src = device.create_buffer(size=nbytes, usage=STO)
        dst = device.create_buffer(size=nbytes, usage=STO)
        device.queue.write_buffer(src, 0, img)
        lut_buf = self._lut_buffer(lut32)

        raw = np.zeros(16, dtype=np.uint32)
        fl = np.zeros(12, dtype=np.float32)
        M = np.asarray(matrix, dtype=np.float32)
        fl[0:3] = M[0]
        fl[4:7] = M[1]
        fl[8:11] = M[2]
        raw[0:12] = fl.view(np.uint32)
        raw[12] = np.uint32(w)
        raw[13] = np.uint32(h)
        raw[14] = np.uint32(size)
        params = device.create_buffer(
            size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        device.queue.write_buffer(params, 0, raw)

        bind = device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
                {"binding": 1, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": lut_buf, "offset": 0, "size": lut32.nbytes}},
                {"binding": 3, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
            ],
        )
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


_IMPL: Optional[TcLutGPU] = None
_IMPL_FAILED = False


def _get_impl() -> Optional[TcLutGPU]:
    global _IMPL, _IMPL_FAILED
    if _IMPL_FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = TcLutGPU()
        except Exception as exc:
            logger.warning("tc-LUT GPU kernel unavailable (%s); using CPU", exc)
            _IMPL_FAILED = True
            return None
    return _IMPL


def _enabled() -> bool:
    val = os.environ.get("SPEKTRAFILM_GPU_TCLUT", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def rgb_to_raw_tc_dispatch(
    rgb, tc_lut, color_space, illuminant_xy, *, backend: str = "auto"
) -> Optional[np.ndarray]:
    """Fused GPU apply for the tc-LUT runtime. float64 (H,W,3) or None (the
    caller's CPU composition runs unchanged). Matrix-only path: the caller
    must have apply_cctf_decoding False."""
    if not _enabled():
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    lut = np.asarray(tc_lut)
    if lut.ndim != 3 or lut.shape[2] != 3 or lut.shape[0] != lut.shape[1] or lut.shape[0] < 2:
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
        from spektrafilm.gpu.color import _rgb_to_xyz_matrix
        from spektrafilm.gpu.tiling import run_or_tile
        M = _rgb_to_xyz_matrix(color_space, illuminant_xy, "CAT16")
        if M is None:
            return None
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(lut))):
            return None
        impl = _get_impl()
        if impl is None:
            return None

        def _render(sub, _origin):
            consume_device_error()  # drop stale errors from other work
            res = impl.run(sub, lut, M)
            error = consume_device_error()
            if error is not None:
                logger.warning("tc-LUT GPU trapped '%s'; result discarded", error)
                return None
            return res

        # 6E ladder: pure per-pixel (halo 0)
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        out = run_or_tile(arr, [nbytes] * 3 + [int(lut.nbytes)], 3, 0, 0,
                          _render, label="tc-lut")
        if out is None:
            return None
        return np.asarray(out, dtype=np.float64)
    except Exception:
        logger.exception("tc-LUT GPU dispatch failed; falling back to CPU")
        return None

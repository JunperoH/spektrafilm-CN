"""G2 GPU density-curve interpolation: log-exposure -> density, per channel.

Drop-in GPU port of model/density_curves.interpolate_exposure_to_density, which
calls utils/fast_interp.fast_interp with a channel-specific x-axis
(x_axis = log_exposure[:, None] / gamma[None, :], y_vals = density_curves).

PARITY: strict per-pixel max_abs (like spectral/blur/colour), NOT statistical.
The kernel reproduces fast_interp exactly: left/right endpoint clamps,
searchsorted(side='right')-1 bracket, linear lerp, t=0 when dx==0. The x-grid is
precomputed on the host in f64 then cast to f32, so the only difference from CPU
is f32 lerp rounding (~1e-6). This is a standalone kernel + parity gate (the
Phase 3 cadence); it is assembled into the resident develop chain in G2c.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "density_curve.wgsl")
_PARAMS_NBYTES = 16  # 4 * u32 (width, height, k, pad)


def build_x_grid(log_exposure: np.ndarray, gamma_factor) -> np.ndarray:
    """Host-side x-axis exactly as interpolate_exposure_to_density builds it:
    log_exposure[:, None] / gamma[None, :], shape (K, 3), f32 C-contiguous."""
    le = np.asarray(log_exposure, dtype=np.float64)
    g = np.asarray(gamma_factor, dtype=np.float64)
    if g.size == 1:
        g = np.array([float(g), float(g), float(g)], dtype=np.float64)
    x_grid = le[:, None] / g[None, :]
    return np.ascontiguousarray(x_grid, dtype=np.float32)


class DensityCurveGPU:
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
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
            ]
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": module, "entry_point": "density_curve_interp"}
        )

    def interpolate(self, log_raw: np.ndarray, x_grid: np.ndarray, density_curves: np.ndarray) -> np.ndarray:
        import wgpu

        img = np.ascontiguousarray(log_raw, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("density-curve interp expects an (H, W, 3) log_raw array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        xg = np.ascontiguousarray(x_grid, dtype=np.float32)
        yc = np.ascontiguousarray(density_curves, dtype=np.float32)
        if xg.shape != yc.shape or xg.ndim != 2 or xg.shape[1] != 3:
            raise ValueError("x_grid and density_curves must both be (K, 3)")
        k = xg.shape[0]

        device = self.gpu.device
        st_usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        in_buf = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        out_buf = device.create_buffer(size=nbytes, usage=st_usage)
        xg_buf = device.create_buffer(size=xg.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        yc_buf = device.create_buffer(size=yc.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        device.queue.write_buffer(in_buf, 0, img)
        device.queue.write_buffer(xg_buf, 0, xg)
        device.queue.write_buffer(yc_buf, 0, yc)

        raw = np.zeros(4, dtype=np.uint32)
        raw[0] = np.uint32(w)
        raw[1] = np.uint32(h)
        raw[2] = np.uint32(k)
        params = device.create_buffer(size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        device.queue.write_buffer(params, 0, raw)

        bind_group = device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
                {"binding": 1, "resource": {"buffer": in_buf, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": xg_buf, "offset": 0, "size": xg.nbytes}},
                {"binding": 3, "resource": {"buffer": yc_buf, "offset": 0, "size": yc.nbytes}},
                {"binding": 4, "resource": {"buffer": out_buf, "offset": 0, "size": nbytes}},
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
        for b in (in_buf, out_buf, xg_buf, yc_buf, params, staging):
            try:
                b.destroy()
            except Exception:
                pass
        return out


_DC_GPU: Optional[DensityCurveGPU] = None
_DC_GPU_FAILED = False


def _get_dc_gpu() -> Optional[DensityCurveGPU]:
    global _DC_GPU, _DC_GPU_FAILED
    if _DC_GPU_FAILED:
        return None
    if _DC_GPU is None:
        try:
            _DC_GPU = DensityCurveGPU()
        except Exception as exc:
            logger.warning("Density-curve GPU kernel unavailable (%s); using CPU", exc)
            _DC_GPU_FAILED = True
            return None
    return _DC_GPU


def interpolate_exposure_to_density_dispatch(
    log_exposure_rgb, density_curves, log_exposure, gamma_factor, *, backend: str = "auto"
) -> Optional[np.ndarray]:
    """GPU drop-in for model.density_curves.interpolate_exposure_to_density.
    Returns float64 (H,W,3) or None (caller runs the CPU path)."""
    from spektrafilm.gpu.backend import Backend, resolve_backend
    from spektrafilm.gpu.spectral import MIN_GPU_PIXELS

    arr = np.asarray(log_exposure_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    requested = (backend or "auto").strip().lower()
    chosen = resolve_backend(requested)
    if chosen is not Backend.GPU:
        return None
    if requested == "auto" and arr.shape[0] * arr.shape[1] < MIN_GPU_PIXELS:
        return None
    impl = _get_dc_gpu()
    if impl is None:
        return None
    try:
        x_grid = build_x_grid(log_exposure, gamma_factor)
        return impl.interpolate(arr, x_grid, np.asarray(density_curves)).astype(np.float64)
    except Exception:
        logger.exception("density-curve interp GPU dispatch failed; falling back to CPU")
        return None

"""Phase 6D: SCAN-tab finishing (linear + display groups) on the GPU.

GPU drop-ins for model/scan_finishing.py's apply_scan_finishing_linear and
apply_scan_finishing_display. Parameters are reduced host-side exactly as the
CPU does (gains from EVs in f64, the row-normalized separation matrix, master
+ per-channel levels, clamps, zero-sum tint vectors) and the shader replicates
the per-pixel ops in the same order with the same skip-conditions, using the
precision-refined software pow/exp/log (Gotcha 5).

First consumer of the two 6D safety rails in gpu/budget.py:
- plan_residency() is consulted BEFORE any allocation (src + dst + staging);
  a 'cpu' answer returns None so the caller's unchanged CPU path runs.
- install_error_trap() at init + consume_device_error() after readback: if
  the device reported an uncaptured error during our submit, the readback is
  DISCARDED and None is returned — a wrong image never survives.

PARITY: strict per-pixel max_abs vs the f64 CPU functions on identical f32
inputs; tolerance 1e-5 (linear group) / 5e-5 (display group — the longer
chained luma/hue path accumulates a few more f32 roundings). Proven by
tests/test_scan_finishing_gpu.py.
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
from spektrafilm.gpu.spectral import MIN_GPU_PIXELS
from spektrafilm.model.scan_finishing import (
    _SEPARATION_MATRIX,
    _display_is_identity,
    _linear_is_identity,
    _tint_vector,
    build_display_curve_lut,
    ScanFinishingParams,
)

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "scan_finishing.wgsl")
_PARAMS_NBYTES = 192  # 11*vec4<f32> + 4*u32, must match struct Params in the wgsl

MODE_LINEAR = 0
MODE_DISPLAY = 1


class ScanFinishingGPU:
    def __init__(self) -> None:
        import wgpu

        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        install_error_trap()
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
                {"binding": 9, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            ]
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": module, "entry_point": "scan_finishing"}
        )
        # Micro-contrast passes (luma extract -> BlurGPU gaussian -> add).
        ro = wgpu.BufferBindingType.read_only_storage
        st = wgpu.BufferBindingType.storage
        un = wgpu.BufferBindingType.uniform

        def bgl(entries):
            return device.create_bind_group_layout(
                entries=[
                    {"binding": b, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
                    for b, t in entries
                ]
            )

        self._luma_bgl = bgl([(3, un), (4, ro), (5, st)])
        self._luma_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._luma_bgl]),
            compute={"module": module, "entry_point": "luma_extract"},
        )
        self._micro_bgl = bgl([(3, un), (6, st), (7, ro), (8, ro)])
        self._micro_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._micro_bgl]),
            compute={"module": module, "entry_point": "micro_add"},
        )
        from spektrafilm.gpu.blur import BlurGPU
        self._blur = BlurGPU()
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
            size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        self._staging = (nbytes, buf)
        return buf

    def run(self, image: np.ndarray, uniform: np.ndarray,
            micro_amount: float = 0.0,
            curve_lut: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """One finishing pass (+ the micro-contrast tail when micro_amount is
        non-zero), all in ONE submit. Returns None (caller falls back to CPU)
        if the device trapped an uncaptured error during this submit."""
        import wgpu

        img = np.ascontiguousarray(image, dtype=np.float32)
        h, w, _ = img.shape
        nbytes = img.nbytes

        usage = (
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        src = self._pooled("src", nbytes, usage)
        dst = self._pooled("dst", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, img)

        params = self.gpu.device.create_buffer(
            size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.gpu.device.queue.write_buffer(params, 0, np.ascontiguousarray(uniform))

        # Curve LUT (binding 9) must always be bound; without a curve a tiny
        # dummy is bound and the shader never reads it (curve_n == 0).
        if curve_lut is None:
            lut_data = np.zeros((2, 3), dtype=np.float32)
        else:
            lut_data = np.ascontiguousarray(curve_lut, dtype=np.float32)
        lut_buf = self._pooled(
            "curve_lut", lut_data.nbytes,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.gpu.device.queue.write_buffer(lut_buf, 0, lut_data)

        bind_group = self.gpu.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
                {"binding": 9, "resource": {"buffer": lut_buf, "offset": 0, "size": lut_data.nbytes}},
            ],
        )
        # Drop stale errors from OTHER work so we only judge this submit.
        consume_device_error()
        encoder = self.gpu.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        cpass.end()

        micro_owned: list = []
        if micro_amount != 0.0:
            from spektrafilm.model.scan_finishing import MICRO_CONTRAST_SIGMA

            nbytes1 = nbytes // 3
            STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            luma = self.gpu.device.create_buffer(size=nbytes1, usage=STO)
            tmp1 = self.gpu.device.create_buffer(size=nbytes1, usage=STO)
            lblur = self.gpu.device.create_buffer(size=nbytes1, usage=STO)
            mraw = np.zeros(8, dtype=np.uint32)
            mraw[0:2] = [w, h]
            mraw[4:5] = np.array([micro_amount], dtype=np.float32).view(np.uint32)
            mp = self.gpu.device.create_buffer(
                size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
            )
            self.gpu.device.queue.write_buffer(mp, 0, mraw)
            micro_owned += [luma, tmp1, lblur, mp]

            def mbg(layout, entries):
                return self.gpu.device.create_bind_group(
                    layout=layout,
                    entries=[
                        {"binding": b, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                        for b, buf in entries
                    ],
                )

            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self._luma_pipe)
            cp.set_bind_group(0, mbg(self._luma_bgl, [(3, mp), (4, dst), (5, luma)]))
            cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
            cp.end()
            self._blur.encode_gaussian(
                encoder, micro_owned, luma, tmp1, lblur,
                float(MICRO_CONTRAST_SIGMA), 3.0, h, w, 1, nbytes1,
            )
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self._micro_pipe)
            cp.set_bind_group(0, mbg(self._micro_bgl, [(3, mp), (6, dst), (7, luma), (8, lblur)]))
            cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
            cp.end()

        staging = self._staging_for(nbytes)
        encoder.copy_buffer_to_buffer(dst, 0, staging, 0, nbytes)
        self.gpu.device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = (
                np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32)
                .reshape(h, w, 3)
                .copy()
            )
        finally:
            staging.unmap()
        for b in [params] + micro_owned:
            try:
                b.destroy()
            except Exception:
                pass
        error = consume_device_error()
        if error is not None:
            logger.warning("scan finishing GPU submit trapped '%s'; result discarded", error)
            return None
        return out


# ----------------------------------------------------------------- packing
def _pack_uniform(p: ScanFinishingParams, width: int, height: int, mode: int,
                  curve_n: int = 0) -> np.ndarray:
    """Host-side parameter reduction, mirroring the CPU functions in f64
    before the final f32 quantization."""
    fl = np.zeros(44, dtype=np.float32)

    gains = 2.0 ** (float(p.exposure_ev) + np.array(
        [float(p.gain_red_ev), float(p.gain_green_ev), float(p.gain_blue_ev)]))
    separation = min(max(float(p.color_separation), 0.0), 1.0)
    fl[0:3] = gains
    fl[3] = 1.0 if separation > 0.0 else 0.0
    if separation > 0.0:
        matrix = np.eye(3) * (1.0 - separation) + _SEPARATION_MATRIX * separation
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-6)
        fl[4:7] = matrix[0]
        fl[8:11] = matrix[1]
        fl[12:15] = matrix[2]

    black = float(p.black_point) + np.asarray(p.black_point_rgb, dtype=float)
    white = float(p.white_point) + np.asarray(p.white_point_rgb, dtype=float)
    fl[16:19] = black
    fl[19] = max(float(p.gamma), 0.05)
    fl[20:23] = white
    fl[23] = min(max(float(p.contrast), -1.0), 1.0)
    fl[24] = min(max(float(p.toe), 0.0), 1.0)
    fl[25] = min(max(float(p.shoulder), 0.0), 1.0)
    fl[26] = max(float(p.saturation), 0.0)
    fl[27] = max(float(p.vibrance), 0.0)
    scales = tuple(float(v) for v in p.hue_saturation_ryg) + tuple(
        float(v) for v in p.hue_saturation_cbm)
    fl[28:34] = [max(s, 0.0) for s in scales]
    fl[34] = 0.25 * min(max(float(p.shadow_tint_strength), 0.0), 1.0)
    fl[35] = 0.25 * min(max(float(p.highlight_tint_strength), 0.0), 1.0)
    fl[36:39] = _tint_vector(p.shadow_tint_hue)
    fl[40:43] = _tint_vector(p.highlight_tint_hue)

    raw = np.zeros(48, dtype=np.uint32)
    raw[0:44] = fl.view(np.uint32)
    raw[44] = np.uint32(width)
    raw[45] = np.uint32(height)
    raw[46] = np.uint32(mode)
    raw[47] = np.uint32(curve_n)
    return raw


# ----------------------------------------------------------------- dispatch
_IMPL: Optional[ScanFinishingGPU] = None
_IMPL_FAILED = False


def _get_impl() -> Optional[ScanFinishingGPU]:
    global _IMPL, _IMPL_FAILED
    if _IMPL_FAILED:
        return None
    if _IMPL is None:
        try:
            _IMPL = ScanFinishingGPU()
        except Exception as exc:
            logger.warning("Scan finishing GPU kernel unavailable (%s); using CPU", exc)
            _IMPL_FAILED = True
            return None
    return _IMPL


def _dispatch(rgb, p: ScanFinishingParams, mode: int, backend: str) -> Optional[np.ndarray]:
    from spektrafilm.gpu.backend import Backend, resolve_backend

    # Mirror the CPU's exact-identity guard: the CPU returns the input array
    # UNTOUCHED then, and running the kernel would quantize it to f32.
    if not p.active:
        return None
    if mode == MODE_LINEAR and _linear_is_identity(p):
        return None
    if mode == MODE_DISPLAY and _display_is_identity(p):
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    requested = (backend or "auto").strip().lower()
    chosen = resolve_backend(requested)
    if chosen is not Backend.GPU:
        return None
    if requested == "auto" and arr.shape[0] * arr.shape[1] < MIN_GPU_PIXELS:
        return None
    impl = _get_impl()
    if impl is None:
        return None
    try:
        from spektrafilm.gpu.tiling import halo_from_sigmas, run_or_tile
        from spektrafilm.model.scan_finishing import MICRO_CONTRAST_SIGMA

        micro = 0.0
        curve_lut = None
        if mode == MODE_DISPLAY:
            micro = min(max(float(getattr(p, 'micro_contrast', 0.0)), -1.0), 1.0)
            curve_lut = build_display_curve_lut(p)   # None when identity
        curve_n = 0 if curve_lut is None else curve_lut.shape[0]

        def _render(sub, _origin):
            uniform = _pack_uniform(p, sub.shape[1], sub.shape[0], mode, curve_n)
            return impl.run(sub, uniform, micro_amount=micro, curve_lut=curve_lut)

        # 6E ladder: per-pixel (halo 0) except when micro-contrast adds its
        # small-radius luma blur; src + dst + staging (+ 3 luma planes then)
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        halo = halo_from_sigmas(MICRO_CONTRAST_SIGMA) if micro != 0.0 else 0
        planes = 3 if micro != 0.0 else 0
        out = run_or_tile(arr, [nbytes] * 3 + [nbytes // 3] * planes, 3, planes,
                          halo, _render, label="scan finishing")
        if out is None:
            return None
        return out.astype(np.float64)
    except Exception:
        logger.exception("scan finishing GPU dispatch failed; falling back to CPU")
        return None


def scan_finishing_linear_dispatch(rgb, p, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for apply_scan_finishing_linear. float64 (H,W,3) or None."""
    return _dispatch(rgb, p, MODE_LINEAR, backend)


def scan_finishing_display_dispatch(rgb, p, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for apply_scan_finishing_display. float64 (H,W,3) or None."""
    return _dispatch(rgb, p, MODE_DISPLAY, backend)

"""Task 7: ScanningStage lens blur + unsharp mask as ONE GPU submit.

CPU path: apply_gaussian_blur(rgb, lens_blur) then
apply_unsharp_mask: rgb + amount * (rgb - gauss(rgb, sigma))
                  = (1 + amount) * rgb - amount * gauss(rgb, sigma).
Each blur previously dispatched individually (own upload + readback) and the
unsharp mix ran in f64 numpy. This chain: upload once -> [lens blur] ->
[unsharp blur + axpby3 mix] -> read back once, reusing the validated
BlurGPU encodings and the axpby3 per-channel mix kernel.

Env toggle: SPEKTRAFILM_RESIDENT_SCANBLUR (default ON).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_AXPBY3_WGSL = os.path.join(os.path.dirname(__file__), "shaders", "axpby3.wgsl")
_PARAMS_NBYTES = 48


def _pack(a3, b3, w, h) -> np.ndarray:
    raw = np.zeros(12, dtype=np.uint32)
    fl = np.zeros(8, dtype=np.float32)
    fl[0:3] = np.asarray(a3, dtype=np.float32).reshape(-1)[:3]
    fl[4:7] = np.asarray(b3, dtype=np.float32).reshape(-1)[:3]
    raw[0:8] = fl.view(np.uint32)
    raw[8] = np.uint32(w)
    raw[9] = np.uint32(h)
    return raw


class ScanBlurResident:
    def __init__(self) -> None:
        import wgpu

        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        self._wgpu = wgpu
        device = self.gpu.device
        with open(_AXPBY3_WGSL, "r", encoding="utf-8") as fh:
            module = device.create_shader_module(code=fh.read())
        un = wgpu.BufferBindingType.uniform
        ro = wgpu.BufferBindingType.read_only_storage
        st = wgpu.BufferBindingType.storage
        self._bgl = device.create_bind_group_layout(
            entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
                for i, t in enumerate([un, ro, ro, st])
            ]
        )
        self._pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bgl]),
            compute={"module": module, "entry_point": "axpby3"},
        )
        from spektrafilm.gpu.blur import BlurGPU

        self._blur = BlurGPU()

    def run(self, rgb: np.ndarray, lens_sigma: float, unsharp_sigma: float, amount: float,
            truncate: float = 3.0) -> np.ndarray:
        wgpu = self._wgpu
        device = self.gpu.device
        img = np.ascontiguousarray(rgb, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("scan blur chain expects an (H, W, 3) array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        gx, gy = (w + 7) // 8, (h + 7) // 8

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        UNI = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

        src = self._buf = self.gpu.device.create_buffer(size=nbytes, usage=STO)
        tmp = device.create_buffer(size=nbytes, usage=STO)
        device.queue.write_buffer(src, 0, img)
        transient: list = []
        extra: list = [tmp]
        encoder = device.create_command_encoder()

        cur = src
        if lens_sigma > 0:
            lens_dst = device.create_buffer(size=nbytes, usage=STO)
            extra.append(lens_dst)
            self._blur.encode_gaussian(
                encoder, transient, cur, tmp, lens_dst, float(lens_sigma), float(truncate), h, w, 3, nbytes,
            )
            cur = lens_dst
        if unsharp_sigma > 0 and amount > 0:
            g_dst = device.create_buffer(size=nbytes, usage=STO)
            out_dst = device.create_buffer(size=nbytes, usage=STO)
            extra += [g_dst, out_dst]
            self._blur.encode_gaussian(
                encoder, transient, cur, tmp, g_dst, float(unsharp_sigma), float(truncate), h, w, 3, nbytes,
            )
            p = device.create_buffer(size=_PARAMS_NBYTES, usage=UNI)
            ones = np.ones(3)
            device.queue.write_buffer(p, 0, _pack((1.0 + amount) * ones, -amount * ones, w, h))
            extra.append(p)
            bind = device.create_bind_group(
                layout=self._bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": p, "offset": 0, "size": _PARAMS_NBYTES}},
                    {"binding": 1, "resource": {"buffer": cur, "offset": 0, "size": nbytes}},
                    {"binding": 2, "resource": {"buffer": g_dst, "offset": 0, "size": nbytes}},
                    {"binding": 3, "resource": {"buffer": out_dst, "offset": 0, "size": nbytes}},
                ],
            )
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self._pipe)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()
            cur = out_dst

        staging = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(cur, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in [src, staging] + extra + transient:
            try:
                b.destroy()
            except Exception:
                pass
        return out


_CORE: Optional[ScanBlurResident] = None
_CORE_FAILED = False


def _get_core() -> Optional[ScanBlurResident]:
    global _CORE, _CORE_FAILED
    if _CORE_FAILED:
        return None
    if _CORE is None:
        try:
            _CORE = ScanBlurResident()
        except Exception as exc:
            logger.warning("Resident scan blur unavailable (%s); using existing path", exc)
            _CORE_FAILED = True
            return None
    return _CORE


def _enabled() -> bool:
    val = os.environ.get("SPEKTRAFILM_RESIDENT_SCANBLUR", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def scan_blur_unsharp_dispatch(rgb, lens_sigma, unsharp_sigma, amount, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU dispatch for the scan-stage lens blur + unsharp tail. float64 or None."""
    if not _enabled():
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    lens = float(lens_sigma)
    sig = float(unsharp_sigma)
    amt = float(amount)
    do_lens = lens > 0
    do_unsharp = sig > 0 and amt > 0
    if not do_lens and not do_unsharp:
        return None  # nothing to do; CPU path is a no-op
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
        from spektrafilm.gpu.budget import consume_device_error
        from spektrafilm.gpu.tiling import halo_from_sigmas, run_or_tile
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
    except Exception:
        return None
    core = _get_core()
    if core is None:
        return None
    try:
        def _render(sub, _origin):
            try:
                consume_device_error()  # drop stale errors from other work
                res = core.run(sub, lens, sig, amt)
                # Phase 6D rail 2: discard the readback on a trapped error.
                error = consume_device_error()
                if error is not None:
                    logger.warning("resident scan blur trapped '%s'; discarded", error)
                    return None
                return res
            except Exception:
                logger.exception("resident scan blur render failed")
                return None

        # 6E ladder: deterministic chain; halo covers lens + unsharp gaussians.
        halo = halo_from_sigmas(lens if do_lens else 0.0, sig if do_unsharp else 0.0)
        out = run_or_tile(arr, [nbytes] * 6, 6, 0, halo, _render, label="scan blur")
        if out is None:
            return None
        return np.asarray(out, dtype=np.float64)
    except Exception:
        logger.exception("Resident scan blur dispatch failed; falling back to existing path")
        return None

"""Resident halation chain (Task 6, 2026-07-06): model/diffusion.apply_halation_um
run as ONE GPU submit -- single upload, all blurs and per-channel mixes chained
buffer-to-buffer on device, single readback.

The CPU path already dispatches its 2 + N blurs to the GPU individually (each
paying its own upload + readback of the 24 MP image) and does the elementwise
mixes in f64 numpy (~1.9 s at 24 MP). This chain removes both costs while
reusing the exact validated blur encodings (BlurGPU.encode_gaussian /
encode_exponential) plus a per-channel axpby (axpby3.wgsl) for the mixes:

    scatter:  scattered = (1 - w_s) (.) G(sigma_c) raw + w_s (.) Exp(lambda_t) raw
              raw1      = (1 - s) raw + s * scattered
    halation: accum     = sum_k w_k G(sigma_h sqrt(k)) raw1        (ping-pong)
              out       = raw1 + a_tot (.) accum
              out       = out / (1 + a_tot)                        (renormalize)

(.) is per-channel multiply. Host-side parameter math replicates
apply_halation_um line-for-line (same activation conditions, same 1e-6 sigma
floors, same bounce-decay normalization).

Env toggle: SPEKTRAFILM_RESIDENT_HALATION (default ON; 0/off/false/no forces
the existing per-stage path), mirroring SPEKTRAFILM_RESIDENT_DEVELOP.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_AXPBY3_WGSL = os.path.join(_SHADER_DIR, "axpby3.wgsl")
_PARAMS_NBYTES = 48  # 2*vec4<f32> + 4*u32


def _pack_axpby3(a3, b3, w: int, h: int) -> np.ndarray:
    raw = np.zeros(12, dtype=np.uint32)
    fl = np.zeros(8, dtype=np.float32)
    fl[0:3] = np.asarray(a3, dtype=np.float32).reshape(-1)[:3]
    fl[4:7] = np.asarray(b3, dtype=np.float32).reshape(-1)[:3]
    raw[0:8] = fl.view(np.uint32)
    raw[8] = np.uint32(w)
    raw[9] = np.uint32(h)
    return raw


class HalationResident:
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

    def _buf(self, nbytes: int, usage) -> Any:
        return self.gpu.device.create_buffer(size=nbytes, usage=usage)

    def run(
        self,
        raw: np.ndarray,
        *,
        s_amount: float,
        w_s: np.ndarray,
        sigma_c_px: np.ndarray,
        lambda_t_px: np.ndarray,
        a_tot: np.ndarray,
        sigma_h_px: np.ndarray,
        n_bounces: int,
        rho: float,
        renormalize: bool,
        truncate: float = 3.0,
    ) -> np.ndarray:
        wgpu = self._wgpu
        device = self.gpu.device

        img = np.ascontiguousarray(raw, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("halation chain expects an (H, W, 3) array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        gx, gy = (w + 7) // 8, (h + 7) // 8

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        UNI = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

        raw_buf = self._buf(nbytes, STO)
        tmp = self._buf(nbytes, STO)
        device.queue.write_buffer(raw_buf, 0, img)

        transient: list = []
        extra: list = [tmp]
        encoder = device.create_command_encoder()

        def mix(a3, b3, x_buf, y_buf, out_buf) -> None:
            p = self._buf(_PARAMS_NBYTES, UNI)
            device.queue.write_buffer(p, 0, _pack_axpby3(a3, b3, w, h))
            extra.append(p)
            bg = device.create_bind_group(
                layout=self._bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": p, "offset": 0, "size": _PARAMS_NBYTES}},
                    {"binding": 1, "resource": {"buffer": x_buf, "offset": 0, "size": nbytes}},
                    {"binding": 2, "resource": {"buffer": y_buf, "offset": 0, "size": nbytes}},
                    {"binding": 3, "resource": {"buffer": out_buf, "offset": 0, "size": nbytes}},
                ],
            )
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(self._pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

        ones = np.ones(3)
        zeros = np.zeros(3)

        # ---- scatter pass (same activation condition as the CPU) ----
        cur = raw_buf
        scatter_on = s_amount > 0 and (np.any(sigma_c_px > 0) or np.any(lambda_t_px > 0))
        if scatter_on:
            g_buf = self._buf(nbytes, STO)
            e_dst = self._buf(nbytes, STO)
            e_acc = self._buf(nbytes, STO)
            s_buf = self._buf(nbytes, STO)
            r1_buf = self._buf(nbytes, STO)
            extra += [g_buf, e_dst, e_acc, s_buf, r1_buf]
            self._blur.encode_gaussian(
                encoder, transient, raw_buf, tmp, g_buf,
                np.maximum(sigma_c_px, 1e-6), float(truncate), h, w, 3, nbytes,
            )
            self._blur.encode_exponential(
                encoder, transient, raw_buf, tmp, e_dst, e_acc,
                np.maximum(lambda_t_px, 1e-6), n_gaussians=3, truncate=float(truncate),
                h=h, w=w, c=3, nbytes=nbytes,
            )
            mix(1.0 - w_s, w_s, g_buf, e_acc, s_buf)          # scattered
            mix((1.0 - s_amount) * ones, s_amount * ones, raw_buf, s_buf, r1_buf)
            cur = r1_buf

        # ---- halation pass (same activation condition as the CPU) ----
        halation_on = n_bounces >= 1 and np.any(a_tot > 0) and np.any(sigma_h_px > 0)
        if halation_on:
            decay = np.array([rho ** (k - 1) for k in range(1, n_bounces + 1)], dtype=np.float64)
            decay /= decay.sum()
            b_buf = self._buf(nbytes, STO)
            acc_a = self._buf(nbytes, STO)   # zero-initialized by WebGPU
            acc_b = self._buf(nbytes, STO)
            out_buf = self._buf(nbytes, STO)
            extra += [b_buf, acc_a, acc_b, out_buf]
            acc_cur, acc_next = acc_a, acc_b
            for k, wk in zip(range(1, n_bounces + 1), decay):
                sigma_k = np.maximum(sigma_h_px * np.sqrt(k), 1e-6)
                self._blur.encode_gaussian(
                    encoder, transient, cur, tmp, b_buf,
                    sigma_k, float(truncate), h, w, 3, nbytes,
                )
                mix(float(wk) * ones, ones, b_buf, acc_cur, acc_next)
                acc_cur, acc_next = acc_next, acc_cur
            mix(ones, a_tot, cur, acc_cur, out_buf)
            if renormalize:
                final = self._buf(nbytes, STO)
                extra.append(final)
                mix(1.0 / (1.0 + a_tot), zeros, out_buf, out_buf, final)
                cur = final
            else:
                cur = out_buf

        staging = self._buf(nbytes, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(cur, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in [raw_buf, staging] + extra + transient:
            try:
                b.destroy()
            except Exception:
                pass
        return out


_CORE: Optional[HalationResident] = None
_CORE_FAILED = False


def _get_core() -> Optional[HalationResident]:
    global _CORE, _CORE_FAILED
    if _CORE_FAILED:
        return None
    if _CORE is None:
        try:
            _CORE = HalationResident()
        except Exception as exc:
            logger.warning("Resident halation unavailable (%s); using existing path", exc)
            _CORE_FAILED = True
            return None
    return _CORE


def _resident_halation_enabled() -> bool:
    """SPEKTRAFILM_RESIDENT_HALATION env toggle (default ON). Set to
    0/off/false/no to force the legacy per-stage path -- the A/B lever for
    timing and a safety hatch, mirroring SPEKTRAFILM_RESIDENT_DEVELOP."""
    val = os.environ.get("SPEKTRAFILM_RESIDENT_HALATION", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def halation_dispatch(raw, halation, pixel_size_um, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU dispatch for model/diffusion.apply_halation_um. Returns the processed
    image as float64, or None when the caller should run the existing path
    (toggle off, halation inactive, nothing to do, no GPU, image below
    MIN_GPU_PIXELS, unsupported input, non-finite parameters, or any error)."""
    if not _resident_halation_enabled():
        return None
    if not getattr(halation, "active", False):
        return None
    arr = np.asarray(raw)
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
        from spektrafilm.gpu.budget import consume_device_error
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
    except Exception:
        return None

    try:
        # Host parameter math: line-for-line apply_halation_um.
        s_amount = float(halation.scatter_amount)
        s_scale = float(halation.scatter_spatial_scale)
        w_s = np.asarray(halation.scatter_tail_weight, dtype=np.float64)
        sigma_c_px = np.asarray(halation.scatter_core_um, dtype=np.float64) * s_scale / pixel_size_um
        lambda_t_px = np.asarray(halation.scatter_tail_um, dtype=np.float64) * s_scale / pixel_size_um

        h_amount = float(halation.halation_amount)
        h_scale = float(halation.halation_spatial_scale)
        a_tot = np.asarray(halation.halation_strength, dtype=np.float64) * h_amount
        sigma_h_px = np.asarray(halation.halation_first_sigma_um, dtype=np.float64) * h_scale / pixel_size_um
        n_bounces = int(halation.halation_n_bounces)
        rho = float(halation.halation_bounce_decay)

        scatter_on = s_amount > 0 and (np.any(sigma_c_px > 0) or np.any(lambda_t_px > 0))
        halation_on = n_bounces >= 1 and np.any(a_tot > 0) and np.any(sigma_h_px > 0)
        if not scatter_on and not halation_on:
            return None  # CPU path is a cheap no-op
        for v in (w_s, sigma_c_px, lambda_t_px, a_tot, sigma_h_px):
            if not np.all(np.isfinite(v)):
                return None

        core = _get_core()
        if core is None:
            return None

        def _render(sub, _origin):
            try:
                consume_device_error()  # drop stale errors from other work
                res = core.run(
                    sub,
                    s_amount=s_amount,
                    w_s=w_s,
                    sigma_c_px=sigma_c_px,
                    lambda_t_px=lambda_t_px,
                    a_tot=a_tot,
                    sigma_h_px=sigma_h_px,
                    n_bounces=n_bounces,
                    rho=rho,
                    renormalize=bool(halation.halation_renormalize),
                )
                # Phase 6D rail 2: discard the readback on a trapped error.
                error = consume_device_error()
                if error is not None:
                    logger.warning("resident halation trapped '%s'; discarded", error)
                    return None
                return res
            except Exception:
                logger.exception("resident halation render failed")
                return None

        # 6E ladder: whole -> tiled -> CPU. The chain is deterministic (no
        # RNG); halo covers the widest blur: scatter core gaussian, scatter
        # exponential tail (~2x lambda), and the widest halation bounce
        # (sigma_h * sqrt(n_bounces)).
        from spektrafilm.gpu.tiling import halo_from_sigmas, run_or_tile

        halo = halo_from_sigmas(
            float(np.max(sigma_c_px)) if s_amount > 0 else 0.0,
            # exponential tail: reach ~16 lambda (see develop_grain_resident —
            # the grain sampler downstream quantizes any residual > ~1e-7)
            (16.0 / 6.0) * float(np.max(lambda_t_px)) if s_amount > 0 and np.any(w_s > 0) else 0.0,
            float(np.max(sigma_h_px)) * (n_bounces ** 0.5) if n_bounces >= 1 else 0.0,
        )
        out = run_or_tile(arr, [nbytes] * 8, 8, 0, halo, _render, label="halation")
        if out is None:
            return None
    except Exception:
        logger.exception("Resident halation dispatch failed; falling back to existing path")
        return None
    return np.asarray(out, dtype=np.float64)

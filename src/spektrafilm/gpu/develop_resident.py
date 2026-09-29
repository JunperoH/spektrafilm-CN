"""G2c: resident develop-core chain (deterministic part) on the GPU.

Chains the develop stage's per-pixel deterministic passes on persistent
on-device buffers in ONE command encoder with a single upload and single
readback (the architecture proven in G1):

    density   = density_curve_interp(log_raw, x_grid, curves_norm)     [G2a kernel]
    correction= couplers_correction(density, matrix, ...)             [G2b kernel]
    log_raw_0 = log_raw - correction                                  [subtract]
    density   = density_curve_interp(log_raw_0, x_grid, curves_0)     [G2a kernel]

This is model/emulsion.develop with the spatial diffusion OFF and grain bypassed
-- the fully deterministic core, exactly parity-checkable (max_abs). The
diffusion blur and grain (multi-pass, buffer-to-buffer) are added in the next
steps; the orchestrator below is structured to slot them in between passes 2 and
3 (blur on the correction buffer) and after pass 4 (grain on the density buffer).

All curve/matrix precomputes are pixel-independent and supplied by the caller
(computed with the SAME CPU helpers as model/emulsion.develop, so the host math
is identical and only f32 GPU rounding separates the result from CPU).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_INTERP_WGSL = os.path.join(_SHADER_DIR, "density_curve.wgsl")
_COUPLERS_WGSL = os.path.join(_SHADER_DIR, "couplers.wgsl")
_SUBTRACT_WGSL = os.path.join(_SHADER_DIR, "subtract.wgsl")
_AXPBY_WGSL = os.path.join(_SHADER_DIR, "axpby.wgsl")


def _pack_axpby_params(width, height, a, b) -> np.ndarray:
    raw = np.zeros(8, dtype=np.uint32)
    raw[0] = np.uint32(width)
    raw[1] = np.uint32(height)
    fl = np.zeros(4, dtype=np.float32)
    fl[0] = np.float32(a)
    fl[1] = np.float32(b)
    raw[4:8] = fl.view(np.uint32)
    return raw


def _u32(*vals) -> np.ndarray:
    a = np.zeros(len(vals), dtype=np.uint32)
    for i, v in enumerate(vals):
        a[i] = np.uint32(v)
    return a


def _pack_couplers_params(width, height, positive, density_max, matrix, shift) -> np.ndarray:
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


class ResidentDevelopCore:
    def __init__(self) -> None:
        import wgpu

        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        from spektrafilm.gpu.budget import install_error_trap
        install_error_trap()
        self._wgpu = wgpu
        device = self.gpu.device

        def module(path):
            with open(path, "r", encoding="utf-8") as fh:
                return device.create_shader_module(code=fh.read())

        ro = wgpu.BufferBindingType.read_only_storage
        st = wgpu.BufferBindingType.storage
        un = wgpu.BufferBindingType.uniform

        def bgl(types):
            return device.create_bind_group_layout(
                entries=[
                    {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
                    for i, t in enumerate(types)
                ]
            )

        # interp: uniform, log_raw(ro), x_grid(ro), curves(ro), out(storage)
        self._interp_bgl = bgl([un, ro, ro, ro, st])
        self._interp_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._interp_bgl]),
            compute={"module": module(_INTERP_WGSL), "entry_point": "density_curve_interp"},
        )
        # couplers: uniform, density(ro), correction(storage)
        self._coup_bgl = bgl([un, ro, st])
        self._coup_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._coup_bgl]),
            compute={"module": module(_COUPLERS_WGSL), "entry_point": "couplers_correction"},
        )
        # subtract: uniform, a(ro), b(ro), out(storage)
        self._sub_bgl = bgl([un, ro, ro, st])
        self._sub_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._sub_bgl]),
            compute={"module": module(_SUBTRACT_WGSL), "entry_point": "subtract"},
        )
        # axpby mix (out = a*x + b*y): uniform, x(ro), y(ro), out(storage)
        self._mix_bgl = bgl([un, ro, ro, st])
        self._mix_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._mix_bgl]),
            compute={"module": module(_AXPBY_WGSL), "entry_point": "axpby"},
        )
        # Blur for the couplers diffusion, encoded buffer-to-buffer into our chain.
        from spektrafilm.gpu.blur import BlurGPU
        self._blur = BlurGPU()

    def _buf(self, nbytes, usage):
        return self.gpu.device.create_buffer(size=nbytes, usage=usage)

    def encode(
        self,
        encoder,
        log_buf,
        h: int,
        w: int,
        nbytes: int,
        x_grid: np.ndarray,
        curves_norm: np.ndarray,
        curves_0: np.ndarray,
        couplers_matrix: np.ndarray,
        density_max,
        *,
        positive: bool,
        high_exposure_shift: float = 0.0,
        diffusion_size_pixel: float = 0.0,
        diffusion_tail_pixel: float = 0.0,
        diffusion_tail_weight: float = 0.0,
        truncate: float = 3.0,
    ):
        """Phase 6D: encode the develop chain into an EXISTING encoder, reading
        log_buf (already uploaded). Returns (out_buf, owned): out_buf holds the
        density after submit; owned lists every buffer created here — the caller
        submits and destroys. This is the fusion primitive that lets grain read
        the develop output without a readback/re-upload round-trip."""
        wgpu = self._wgpu
        device = self.gpu.device
        xg = np.ascontiguousarray(x_grid, dtype=np.float32)
        cn = np.ascontiguousarray(curves_norm, dtype=np.float32)
        c0 = np.ascontiguousarray(curves_0, dtype=np.float32)
        if not (xg.shape == cn.shape == c0.shape and xg.ndim == 2 and xg.shape[1] == 3):
            raise ValueError("x_grid, curves_norm, curves_0 must all be (K, 3)")
        k = xg.shape[0]

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        STIN = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        UNI = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

        buf_a = self._buf(nbytes, STO)
        buf_b = self._buf(nbytes, STO)
        xg_buf = self._buf(xg.nbytes, STIN)
        cn_buf = self._buf(cn.nbytes, STIN)
        c0_buf = self._buf(c0.nbytes, STIN)
        device.queue.write_buffer(xg_buf, 0, xg)
        device.queue.write_buffer(cn_buf, 0, cn)
        device.queue.write_buffer(c0_buf, 0, c0)

        interp_params = self._buf(16, UNI)
        device.queue.write_buffer(interp_params, 0, _u32(w, h, k, 0))
        sub_params = self._buf(16, UNI)
        device.queue.write_buffer(sub_params, 0, _u32(w, h, 0, 0))
        coup_params = self._buf(80, UNI)
        device.queue.write_buffer(
            coup_params, 0,
            _pack_couplers_params(w, h, positive, density_max, couplers_matrix, high_exposure_shift),
        )

        def bg(layout, buffers):
            return device.create_bind_group(
                layout=layout,
                entries=[
                    {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                    for i, b in enumerate(buffers)
                ],
            )

        bg_interp1 = bg(self._interp_bgl, [interp_params, log_buf, xg_buf, cn_buf, buf_a])
        bg_coup = bg(self._coup_bgl, [coup_params, buf_a, buf_b])

        gx, gy = (w + 7) // 8, (h + 7) // 8
        transient: list = []   # taps/params from the blur encodes; freed after submit
        extra: list = []       # diffusion scratch buffers; freed after submit

        def dispatch(pipe, bind):
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

        dispatch(self._interp_pipe, bg_interp1)   # log_raw -> A (density)
        dispatch(self._coup_pipe, bg_coup)        # A -> B (correction, pre-diffusion)

        # Couplers diffusion on the correction (B). The blurred result goes to a
        # SEPARATE buffer so B is never overwritten while the blur passes are
        # still reading it (no write-after-read hazard). corr_src is what the
        # subtract reads: the raw correction (no diffusion), the gaussian (no
        # tail), or the (1-w)*gaussian + w*exponential mix.
        corr_src = buf_b
        if diffusion_size_pixel and diffusion_size_pixel > 0:
            g_buf = self._buf(nbytes, STO)
            tmp_blur = self._buf(nbytes, STO)
            extra += [g_buf, tmp_blur]
            self._blur.encode_gaussian(
                encoder, transient, buf_b, tmp_blur, g_buf,
                float(diffusion_size_pixel), float(truncate), h, w, 3, nbytes,
            )
            corr_src = g_buf
            if diffusion_tail_weight and diffusion_tail_weight > 0 and diffusion_tail_pixel and diffusion_tail_pixel > 0:
                e_dst = self._buf(nbytes, STO)
                e_accum = self._buf(nbytes, STO)
                blurred = self._buf(nbytes, STO)
                extra += [e_dst, e_accum, blurred]
                self._blur.encode_exponential(
                    encoder, transient, buf_b, tmp_blur, e_dst, e_accum,
                    float(diffusion_tail_pixel), n_gaussians=3, truncate=float(truncate),
                    h=h, w=w, c=3, nbytes=nbytes,
                )
                mix_params = self._buf(32, UNI)
                device.queue.write_buffer(
                    mix_params, 0, _pack_axpby_params(w, h, 1.0 - diffusion_tail_weight, diffusion_tail_weight)
                )
                extra.append(mix_params)
                dispatch(self._mix_pipe, bg(self._mix_bgl, [mix_params, g_buf, e_accum, blurred]))
                corr_src = blurred

        bg_sub = bg(self._sub_bgl, [sub_params, log_buf, corr_src, buf_a])
        bg_interp2 = bg(self._interp_bgl, [interp_params, buf_a, xg_buf, c0_buf, buf_b])
        dispatch(self._sub_pipe, bg_sub)          # log_raw - corr_src -> A (log_raw_0)
        dispatch(self._interp_pipe, bg_interp2)   # A -> B (density_out, curves_0)

        owned = [buf_a, buf_b, xg_buf, cn_buf, c0_buf,
                 interp_params, sub_params, coup_params] + extra + transient
        return buf_b, owned

    def run(
        self,
        log_raw: np.ndarray,
        x_grid: np.ndarray,
        curves_norm: np.ndarray,
        curves_0: np.ndarray,
        couplers_matrix: np.ndarray,
        density_max,
        *,
        positive: bool,
        high_exposure_shift: float = 0.0,
        diffusion_size_pixel: float = 0.0,
        diffusion_tail_pixel: float = 0.0,
        diffusion_tail_weight: float = 0.0,
        truncate: float = 3.0,
    ) -> np.ndarray:
        wgpu = self._wgpu
        device = self.gpu.device

        img = np.ascontiguousarray(log_raw, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("develop core expects an (H, W, 3) log_raw array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        log_buf = self._buf(nbytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        device.queue.write_buffer(log_buf, 0, img)
        encoder = device.create_command_encoder()
        out_buf, owned = self.encode(
            encoder, log_buf, h, w, nbytes,
            x_grid, curves_norm, curves_0, couplers_matrix, density_max,
            positive=positive,
            high_exposure_shift=high_exposure_shift,
            diffusion_size_pixel=diffusion_size_pixel,
            diffusion_tail_pixel=diffusion_tail_pixel,
            diffusion_tail_weight=diffusion_tail_weight,
            truncate=truncate,
        )
        staging = self._buf(nbytes, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])  # single submit: whole chain + copy
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in [log_buf, staging] + owned:
            try:
                b.destroy()
            except Exception:
                pass
        return out


_CORE: Optional[ResidentDevelopCore] = None
_CORE_FAILED = False


def get_develop_core() -> Optional[ResidentDevelopCore]:
    global _CORE, _CORE_FAILED
    if _CORE_FAILED:
        return None
    if _CORE is None:
        try:
            _CORE = ResidentDevelopCore()
        except Exception as exc:
            logger.warning("Resident develop core unavailable (%s); using existing path", exc)
            _CORE_FAILED = True
            return None
    return _CORE


def _resident_develop_enabled() -> bool:
    """SPEKTRAFILM_RESIDENT_DEVELOP env toggle (default ON). Set to 0/off/false/no
    to force the legacy per-stage develop path -- the A/B lever for timing and a
    safety hatch, mirroring the SPEKTRAFILM_GPU toggle.

    G2d-3 gate closed 2026-07-06: the resident chain matches the identical
    shipped GPU kernels to ~2e-6 (test_gpu_resident_diffusion_matches_gpu_dispatch);
    the earlier 4.4e-3 vs the CPU-f64 reference was the GPU IIR blur's intrinsic
    f32 cancellation noise floor (~1.4e-4 at the diffusion tail's sigma ~96 px)
    amplified by the synthetic test curves' ~30x slope -- the same noise the live
    per-stage path already ships via the blur chokepoint dispatch."""
    val = os.environ.get("SPEKTRAFILM_RESIDENT_DEVELOP", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def develop_deterministic_dispatch(
    log_raw,
    pixel_size_um: float,
    log_exposure,
    normalized_density_curves,
    dir_couplers,
    profile_type: str,
    gamma_factor=1.0,
    *,
    backend: str = "auto",
) -> Optional[np.ndarray]:
    """GPU dispatch for the DETERMINISTIC core of model/emulsion.develop, i.e.
    develop_simple + apply_density_correction_dir_couplers collapsed into ONE
    resident chain (single upload, single readback). Grain is applied by the
    caller on the result, exactly as before.

    Reproduces the CPU path bit-for-algorithm (only f32 GPU rounding differs):
        density   = interpolate_exposure_to_density(log_raw, curves_norm)
        correction= couplers(density) @ matrix, then (1-w)*gauss + w*exp diffusion
        log_raw_0 = log_raw - correction
        density   = interpolate_exposure_to_density(log_raw_0, curves_0)
    using the SAME host precomputes as apply_density_correction_dir_couplers.

    Returns density_cmy as float64 (H, W, 3), or None when the caller should run
    the existing CPU/per-stage path (toggle off, couplers inactive, no GPU,
    image below MIN_GPU_PIXELS, unsupported input, or any error)."""
    if not _resident_develop_enabled():
        return None

    # Couplers must be active: with dir_couplers.active False the CPU
    # apply_density_correction_dir_couplers is an identity pass-through, which the
    # resident core (always couplers+diffusion) does not model -> defer to CPU.
    if not getattr(dir_couplers, "active", False):
        return None

    arr = np.asarray(log_raw)
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

    # Phase 6D rail 1: the develop chain allocates up to 8 image-sized buffers
    # (log + A + B + 5 diffusion scratch) plus staging — decline BEFORE
    # allocating when they will not fit (6E turns this signal into tiling).
    try:
        from spektrafilm.gpu.budget import consume_device_error, plan_residency
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        if plan_residency([nbytes] * 9) != 'gpu':
            return None
    except Exception:
        return None

    core = get_develop_core()
    if core is None:
        return None

    plan = develop_plan(
        pixel_size_um, log_exposure, normalized_density_curves,
        dir_couplers, profile_type, gamma_factor,
    )
    if plan is None:
        return None

    try:
        consume_device_error()  # drop stale errors from other work
        out = core.run(arr, **plan)
        # Phase 6D rail 2: an uncaptured device error during OUR submit means
        # the readback may be zeros/garbage — discard it, run the CPU path.
        error = consume_device_error()
        if error is not None:
            logger.warning("resident develop trapped '%s'; result discarded", error)
            return None
    except Exception:
        logger.exception("Resident develop dispatch failed; falling back to CPU path")
        return None

    return np.asarray(out, dtype=np.float64)


def develop_plan(
    pixel_size_um: float,
    log_exposure,
    normalized_density_curves,
    dir_couplers,
    profile_type: str,
    gamma_factor=1.0,
) -> Optional[dict]:
    """Host precomputes for the develop chain, shared by the standalone
    dispatch above and the fused develop+grain dispatch (Phase 6D). Returns
    kwargs for ResidentDevelopCore.run/encode (minus the image), or None when
    the inputs are unsupported."""
    try:
        from spektrafilm.model.couplers import (
            compute_dir_couplers_matrix,
            compute_density_curves_before_dir_couplers,
        )
        from spektrafilm.gpu.density_curve import build_x_grid

        positive = profile_type == "positive"
        curves_norm = np.asarray(normalized_density_curves)
        # Same precomputes as apply_density_correction_dir_couplers.
        matrix = compute_dir_couplers_matrix(dir_couplers) * dir_couplers.amount
        curves_0 = compute_density_curves_before_dir_couplers(
            curves_norm, log_exposure, matrix, positive=positive
        )
        density_max = np.nanmax(curves_norm, axis=0)
        x_grid = build_x_grid(log_exposure, gamma_factor)
        # NaN in the curve grids would make GPU interp diverge from the CPU
        # fast_interp; that is rare for real profiles but defer safely if present.
        if not (
            np.all(np.isfinite(x_grid))
            and np.all(np.isfinite(curves_norm))
            and np.all(np.isfinite(curves_0))
            and np.all(np.isfinite(matrix))
        ):
            return None

        px = float(pixel_size_um)
        return dict(
            x_grid=x_grid,
            curves_norm=curves_norm,
            curves_0=curves_0,
            couplers_matrix=matrix,
            density_max=density_max,
            positive=positive,
            diffusion_size_pixel=float(dir_couplers.diffusion_size_um) / px,
            diffusion_tail_pixel=float(dir_couplers.diffusion_tail_um) / px,
            diffusion_tail_weight=float(dir_couplers.diffusion_tail_weight),
        )
    except Exception:
        logger.exception("develop plan precompute failed")
        return None

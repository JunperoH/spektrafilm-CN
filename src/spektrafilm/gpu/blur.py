"""B1 GPU blur primitive: Gaussian (FIR + Young-van Vliet IIR) and the
exponential (Gaussian-mixture) filter, mirroring utils/fast_gaussian_filter.py.

Parity-by-construction choices:
- The DISPATCH RULE is the CPU's own: per channel, FIR for sigma <
  SMALL_SIGMA_MAX, YvV IIR for sigma >= SMALL_SIGMA_MAX. The GPU does not
  substitute a wide FIR for the IIR: that would be a different approximation
  of the Gaussian (the IIR's documented ~1e-3 model error vs analytic) and
  parity against the CPU output would be bounded by that model difference
  instead of by f32 rounding.
- FIR taps come from the CPU's _gaussian_kernel_1d and IIR coefficients from
  the CPU's _yvv_coeffs, so weights are bit-identical (then cast to f32).
- Boundary handling mirrors the CPU exactly: scipy 'reflect' for FIR,
  edge-sample replication for the IIR sweeps.
- Pass order mirrors the CPU: FIR vertical -> horizontal, IIR horizontal ->
  vertical.

All passes for a call are recorded into ONE command encoder and submitted
once; intermediates never leave VRAM (NegPy multi-pass pattern, on storage
buffers). Kernels run in f32 (inputs are cast once on upload); the dispatch
hooks cast results back to float64 so downstream numba consumers keep the
dtype the CPU path always produced. The parity test bounds the f32 effect.
IIR coefficients are made DC-exact in f32 (see _encode_gaussian_channel).

Buffers are pooled per (label) and reused across calls while big enough,
with persistent readback staging (NegPy resources.py pattern).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger
from spektrafilm.utils.fast_gaussian_filter import (
    SMALL_SIGMA_MAX,
    _EXPONENTIAL_GAUSSIAN_FITS,
    _gaussian_kernel_1d,
    _yvv_coeffs,
    fast_exponential_filter_cpu as _cpu_exponential_filter,
    fast_gaussian_filter_cpu as _cpu_gaussian_filter,
)

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "blur.wgsl")

# Same rationale and value as spectral.MIN_GPU_PIXELS: below this, dispatch +
# transfer overhead beats the compute and backend="auto" stays on CPU.
from spektrafilm.gpu.spectral import MIN_GPU_PIXELS  # noqa: E402

_ENTRY_POINTS = ("fir_pass", "iir_h", "iir_v", "copy_chan", "accumulate")
_PARAMS_NBYTES = 64  # 8 u32 + 8 f32, must match struct Params in blur.wgsl


def _sigmas_per_channel(sigma, channels: int) -> np.ndarray:
    if np.ndim(sigma) == 0:
        return np.full(channels, float(sigma), dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64).ravel()
    if s.shape[0] != channels:
        raise ValueError(
            f"sigma length {s.shape[0]} does not match channel count {channels}"
        )
    return s


class BlurGPU:
    """Compiles the blur pipelines once; dispatches whole calls in one submit."""

    def __init__(self) -> None:
        import wgpu

        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        device = self.gpu.device
        with open(_SHADER_PATH, "r", encoding="utf-8") as fh:
            module = device.create_shader_module(code=fh.read())
        # Explicit shared layout: layout="auto" culls bindings an entry point
        # does not use (iir_*/copy_chan never read `taps`), which would reject
        # our uniform 4-entry bind groups. One explicit layout serves all five
        # entry points.
        self._bgl = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipelines = {
            name: device.create_compute_pipeline(
                layout=pipeline_layout, compute={"module": module, "entry_point": name}
            )
            for name in _ENTRY_POINTS
        }
        # Buffer pool: label -> (alloc_nbytes, buffer). Reused while
        # alloc_nbytes >= requested; bind sizes are passed explicitly.
        self._pool: dict[str, tuple[int, Any]] = {}
        self._staging: Optional[tuple[int, Any]] = None
        # Tiny one-shot buffers destroyed after each submit.
        self._transient: list[Any] = []

    # ----------------------------------------------------------------- pool
    def _pooled(self, label: str, nbytes: int, usage: int) -> Any:
        import wgpu  # noqa: F401

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

    def _params_buf(self, ints: tuple[int, ...], floats: tuple[float, ...]) -> Any:
        import wgpu

        raw = np.zeros(16, dtype=np.uint32)
        raw[:8] = np.array(list(ints) + [0] * (8 - len(ints)), dtype=np.uint32)
        raw[8:] = (
            np.array(list(floats) + [0.0] * (8 - len(floats)), dtype=np.float32)
            .view(np.uint32)
        )
        buf = self.gpu.device.create_buffer(
            size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.gpu.device.queue.write_buffer(buf, 0, raw)
        self._transient.append(buf)
        return buf

    def _taps_buf(self, taps: np.ndarray) -> Any:
        import wgpu

        t = np.ascontiguousarray(taps, dtype=np.float32)
        buf = self.gpu.device.create_buffer(
            size=max(t.nbytes, 4), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        if t.nbytes:
            self.gpu.device.queue.write_buffer(buf, 0, t)
        self._transient.append(buf)
        return buf

    # ------------------------------------------------------------- dispatch
    def _bind_and_dispatch(
        self,
        encoder: Any,
        entry: str,
        src: Any,
        dst: Any,
        taps: Any,
        params: Any,
        nbytes: int,
        groups: tuple[int, int, int],
    ) -> None:
        pipeline = self._pipelines[entry]
        bind_group = self.gpu.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": taps, "offset": 0, "size": taps.size}},
                {"binding": 3, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
            ],
        )
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(*groups)
        cpass.end()

    def _encode_gaussian_channel(
        self,
        encoder: Any,
        ch: int,
        sigma: float,
        truncate: float,
        h: int,
        w: int,
        c: int,
        nbytes: int,
        src: Any,
        tmp: Any,
        dst: Any,
        empty_taps: Any,
    ) -> None:
        """Record the two passes for one channel, mirroring _dispatch_2d."""
        gx88 = ((w + 7) // 8, (h + 7) // 8, 1)
        if sigma <= 0.0:
            params = self._params_buf((w, h, c, ch, 0, 0, 0), ())
            self._bind_and_dispatch(encoder, "copy_chan", src, dst, empty_taps, params, nbytes, gx88)
            return
        if sigma < SMALL_SIGMA_MAX:
            kernel, radius = _gaussian_kernel_1d(float(sigma), float(truncate))
            taps = self._taps_buf(kernel)
            p_v = self._params_buf((w, h, c, ch, int(radius), 0, 0), ())
            p_h = self._params_buf((w, h, c, ch, int(radius), 1, 0), ())
            self._bind_and_dispatch(encoder, "fir_pass", src, tmp, taps, p_v, nbytes, gx88)
            self._bind_and_dispatch(encoder, "fir_pass", tmp, dst, taps, p_h, nbytes, gx88)
            return
        # IIR path (sigma >= SMALL_SIGMA_MAX, so the CPU's sigma<0.5 FIR
        # fallback inside _gaussian_filter_2d_large is unreachable here).
        # DC-EXACT f32 coefficients: rounding all four f64 coefficients to f32
        # independently breaks the unit-DC-gain identity B = 1 - (B1+B2+B3)
        # that holds in the CPU's f64 math, and that gain bias compounds over
        # the recurrence (dominant f32 error at large sigma). Rounding the
        # three feedback terms and DERIVING B in f32 restores the identity:
        # measured 5-6x error reduction at sigma 20-95 (see parity test).
        _, B1d, B2d, B3d = _yvv_coeffs(float(sigma))
        b1 = np.float32(B1d)
        b2 = np.float32(B2d)
        b3 = np.float32(B3d)
        b0 = np.float32(1.0) - (b1 + b2 + b3)
        coeffs = (float(b0), float(b1), float(b2), float(b3))
        p_h = self._params_buf((w, h, c, ch, 0, 0, 0), coeffs)
        p_v = self._params_buf((w, h, c, ch, 0, 0, 0), coeffs)
        self._bind_and_dispatch(
            encoder, "iir_h", src, tmp, empty_taps, p_h, nbytes, ((h + 63) // 64, 1, 1)
        )
        self._bind_and_dispatch(
            encoder, "iir_v", tmp, dst, empty_taps, p_v, nbytes, ((w + 63) // 64, 1, 1)
        )

    # ----------------------------------------------- buffer-to-buffer (resident)
    def encode_gaussian(
        self,
        encoder: Any,
        transient: list,
        src: Any,
        tmp: Any,
        dst: Any,
        sigma,
        truncate: float,
        h: int,
        w: int,
        c: int,
        nbytes: int,
    ) -> None:
        """Record the Gaussian passes for an (h,w,c) image into the caller's
        ``encoder``, reading caller buffer ``src``, scratch ``tmp``, writing
        ``dst`` (all GPU buffers). Taps/params buffers are appended to the
        caller-owned ``transient`` list (destroy them after the caller submits).

        This lets the Gaussian run INSIDE a larger resident chain (e.g. the
        develop diffusion) with no host round-trip. It reuses the exact, validated
        per-channel FIR/IIR encoding of gaussian(); only buffer ownership and the
        submit move to the caller. Implementation: temporarily point the internal
        transient list at the caller's so the existing _taps_buf/_params_buf
        helpers append there, leaving every existing method unchanged."""
        sigmas = _sigmas_per_channel(sigma, c)
        saved = self._transient
        self._transient = transient
        try:
            empty_taps = self._taps_buf(np.zeros(1, dtype=np.float32))
            for ch in range(c):
                self._encode_gaussian_channel(
                    encoder, ch, float(sigmas[ch]), float(truncate),
                    h, w, c, nbytes, src, tmp, dst, empty_taps,
                )
        finally:
            self._transient = saved

    def encode_exponential(
        self,
        encoder: Any,
        transient: list,
        src: Any,
        tmp: Any,
        dst: Any,
        accum: Any,
        decay_constant,
        *,
        n_gaussians: int = 3,
        truncate: float = 3.0,
        h: int,
        w: int,
        c: int,
        nbytes: int,
    ) -> None:
        """Record the exponential (Gaussian-mixture) filter into the caller's
        ``encoder`` on caller buffers (src input, tmp/dst scratch, result in
        ``accum``). Same component loop + accumulate as exponential(); only buffer
        ownership and the submit move to the caller. Taps/params append to the
        caller-owned ``transient`` list."""
        if n_gaussians not in _EXPONENTIAL_GAUSSIAN_FITS:
            raise ValueError(
                f"No hardcoded fit for n_gaussians={n_gaussians}; "
                f"available: {sorted(_EXPONENTIAL_GAUSSIAN_FITS)}"
            )
        fit = _EXPONENTIAL_GAUSSIAN_FITS[n_gaussians]
        decay = np.asarray(decay_constant, dtype=np.float64)
        total = h * w * c
        saved = self._transient
        self._transient = transient
        try:
            empty_taps = self._taps_buf(np.zeros(1, dtype=np.float32))
            for comp_i, (amplitude, sigma_ratio) in enumerate(fit):
                sigmas = _sigmas_per_channel(sigma_ratio * decay, c)
                for ch in range(c):
                    self._encode_gaussian_channel(
                        encoder, ch, float(sigmas[ch]), float(truncate),
                        h, w, c, nbytes, src, tmp, dst, empty_taps,
                    )
                gx = min(65535, (total + 255) // 256)
                grid_x = gx * 256
                gy = (total + grid_x - 1) // grid_x
                p_acc = self._params_buf(
                    (w, h, c, 0, 0, 0, 1 if comp_i == 0 else 0, grid_x),
                    (0.0, 0.0, 0.0, 0.0, float(amplitude)),
                )
                self._bind_and_dispatch(
                    encoder, "accumulate", dst, accum, empty_taps, p_acc, nbytes, (gx, gy, 1),
                )
        finally:
            self._transient = saved

    # ---------------------------------------------------------------- calls
    def gaussian(self, image: np.ndarray, sigma, truncate: float = 3.0) -> np.ndarray:
        import wgpu

        img = np.ascontiguousarray(image, dtype=np.float32)
        squeeze = img.ndim == 2
        if squeeze:
            img = img[:, :, None]
        h, w, c = img.shape
        sigmas = _sigmas_per_channel(sigma, c)
        nbytes = img.nbytes

        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        src = self._pooled("in", nbytes, usage)
        tmp = self._pooled("tmp", nbytes, usage)
        dst = self._pooled("out", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, img)

        empty_taps = self._taps_buf(np.zeros(1, dtype=np.float32))
        encoder = self.gpu.device.create_command_encoder()
        for ch in range(c):
            self._encode_gaussian_channel(
                encoder, ch, float(sigmas[ch]), float(truncate),
                h, w, c, nbytes, src, tmp, dst, empty_taps,
            )
        out = self._submit_and_read(encoder, dst, nbytes, (h, w, c))
        return out[:, :, 0] if squeeze else out

    def exponential(
        self, image: np.ndarray, decay_constant, *, n_gaussians: int = 3, truncate: float = 3.0
    ) -> np.ndarray:
        import wgpu

        if n_gaussians not in _EXPONENTIAL_GAUSSIAN_FITS:
            raise ValueError(
                f"No hardcoded fit for n_gaussians={n_gaussians}; "
                f"available: {sorted(_EXPONENTIAL_GAUSSIAN_FITS)}"
            )
        fit = _EXPONENTIAL_GAUSSIAN_FITS[n_gaussians]
        decay = np.asarray(decay_constant, dtype=np.float64)

        img = np.ascontiguousarray(image, dtype=np.float32)
        squeeze = img.ndim == 2
        if squeeze:
            img = img[:, :, None]
        h, w, c = img.shape
        nbytes = img.nbytes
        total = h * w * c

        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        src = self._pooled("in", nbytes, usage)
        tmp = self._pooled("tmp", nbytes, usage)
        dst = self._pooled("out", nbytes, usage)
        accum = self._pooled("accum", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, img)

        empty_taps = self._taps_buf(np.zeros(1, dtype=np.float32))
        encoder = self.gpu.device.create_command_encoder()
        for comp_i, (amplitude, sigma_ratio) in enumerate(fit):
            sigmas = _sigmas_per_channel(sigma_ratio * decay, c)
            for ch in range(c):
                self._encode_gaussian_channel(
                    encoder, ch, float(sigmas[ch]), float(truncate),
                    h, w, c, nbytes, src, tmp, dst, empty_taps,
                )
            # accumulate is a flat per-element pass. A 1D grid of
            # ceil(total/256) workgroups overflows WebGPU's hard
            # maxComputeWorkgroupsPerDimension limit (65535) at full
            # resolution: 24 MP x 3 ch -> 285012 > 65535, a validation error
            # at submit. Spill into a 2D grid and pass the X stride (grid_x =
            # gx*256) so the shader rebuilds the flat index. gx*gy*256 >=
            # total; the i>=total guard in the shader drops the excess.
            gx = min(65535, (total + 255) // 256)
            grid_x = gx * 256
            gy = (total + grid_x - 1) // grid_x
            p_acc = self._params_buf(
                (w, h, c, 0, 0, 0, 1 if comp_i == 0 else 0, grid_x),
                (0.0, 0.0, 0.0, 0.0, float(amplitude)),
            )
            self._bind_and_dispatch(
                encoder, "accumulate", dst, accum, empty_taps, p_acc, nbytes,
                (gx, gy, 1),
            )
        out = self._submit_and_read(encoder, accum, nbytes, (h, w, c))
        return out[:, :, 0] if squeeze else out

    def _submit_and_read(
        self, encoder: Any, result_buf: Any, nbytes: int, shape: tuple[int, int, int]
    ) -> np.ndarray:
        import wgpu

        staging = self._staging_for(nbytes)
        encoder.copy_buffer_to_buffer(result_buf, 0, staging, 0, nbytes)
        self.gpu.device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = (
                np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32)
                .reshape(shape)
                .copy()
            )
        finally:
            staging.unmap()
        for buf in self._transient:
            try:
                buf.destroy()
            except Exception:
                pass
        self._transient.clear()
        return out


# ----------------------------------------------------------------- dispatch
_BLUR_GPU: Optional[BlurGPU] = None
_BLUR_GPU_FAILED = False


def _get_blur_gpu() -> Optional[BlurGPU]:
    global _BLUR_GPU, _BLUR_GPU_FAILED
    if _BLUR_GPU_FAILED:
        return None
    if _BLUR_GPU is None:
        try:
            _BLUR_GPU = BlurGPU()
        except Exception as exc:
            logger.warning("Blur GPU kernels unavailable (%s); using CPU", exc)
            _BLUR_GPU_FAILED = True
            return None
    return _BLUR_GPU


def _resolve(image: np.ndarray, backend: str):
    from spektrafilm.gpu.backend import Backend, resolve_backend

    requested = (backend or "auto").strip().lower()
    chosen = resolve_backend(requested)
    if (
        chosen is Backend.GPU
        and requested == "auto"
        and image.shape[0] * image.shape[1] < MIN_GPU_PIXELS
    ):
        chosen = Backend.CPU
    return chosen


def gaussian_dispatch(
    image, sigma, truncate: float = 3.0, *, backend: str = "auto"
) -> Optional[np.ndarray]:
    """GPU dispatch hook for the utils.fast_gaussian_filter chokepoint.

    Returns the blurred image as float64 (so downstream numba consumers keep
    seeing the dtype the CPU path always produced), or None when the CPU path
    should run instead (CPU backend, image below MIN_GPU_PIXELS, no device,
    unsupported input, or any GPU error)."""
    from spektrafilm.gpu.backend import Backend

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        return None
    if _resolve(img, backend) is not Backend.GPU:
        return None
    impl = _get_blur_gpu()
    if impl is None:
        return None
    try:
        return impl.gaussian(image, sigma, truncate).astype(np.float64)
    except Exception:
        logger.exception("Gaussian GPU dispatch failed; falling back to CPU")
        return None


def exponential_dispatch(
    image, decay_constant, *, n_gaussians: int = 3, truncate: float = 3.0, backend: str = "auto"
) -> Optional[np.ndarray]:
    """GPU dispatch hook for the utils.fast_exponential_filter chokepoint.

    Same contract as gaussian_dispatch (float64 result or None). Invalid
    n_gaussians returns None so the CPU path raises its own ValueError."""
    from spektrafilm.gpu.backend import Backend

    if n_gaussians not in _EXPONENTIAL_GAUSSIAN_FITS:
        return None
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        return None
    if _resolve(img, backend) is not Backend.GPU:
        return None
    impl = _get_blur_gpu()
    if impl is None:
        return None
    try:
        return impl.exponential(
            image, decay_constant, n_gaussians=n_gaussians, truncate=truncate
        ).astype(np.float64)
    except Exception:
        logger.exception("Exponential GPU dispatch failed; falling back to CPU")
        return None


def fast_gaussian_filter(image, sigma, truncate: float = 3.0, *, backend: str = "auto"):
    """GPU/CPU dispatching drop-in for utils.fast_gaussian_filter.

    Returns float64 on both paths (GPU results are cast at the boundary)."""
    out = gaussian_dispatch(image, sigma, truncate, backend=backend)
    if out is not None:
        return out
    return _cpu_gaussian_filter(image, sigma, truncate=truncate)


def fast_exponential_filter(
    image, decay_constant, *, n_gaussians: int = 3, truncate: float = 3.0, backend: str = "auto"
):
    """GPU/CPU dispatching drop-in for utils.fast_exponential_filter."""
    out = exponential_dispatch(
        image, decay_constant, n_gaussians=n_gaussians, truncate=truncate, backend=backend
    )
    if out is not None:
        return out
    if n_gaussians not in _EXPONENTIAL_GAUSSIAN_FITS:
        raise ValueError(
            f"No hardcoded fit for n_gaussians={n_gaussians}; "
            f"available: {sorted(_EXPONENTIAL_GAUSSIAN_FITS)}"
        )
    return _cpu_exponential_filter(
        image, decay_constant, n_gaussians=n_gaussians, truncate=truncate
    )

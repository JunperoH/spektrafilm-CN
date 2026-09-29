"""Fused spectral reduction: the Phase 2 first GPU kernel.

Ports the per-pixel chain `compute_density_spectral -> density_to_light ->
contract` (3 -> 81 -> 3) that dominates the full-precision render. The 81-band
cube is never materialized. One kernel serves both printing.expose and
scanning.scan; only the matrices/illuminant differ.

Wired into the live pipeline in Phase 3 (2026-06-10): scanning.cmy_to_log_xyz
and printing._film_cmy_to_print_log_raw call spectral_reduce. This module
provides a CPU reference (delegating to the real pipeline functions, so it is
parity-exact by construction) and a wgpu implementation, plus a dispatcher with
automatic CPU fallback. Stage-level parity: tests/test_stage_spectral_wiring.py.

Contract (mirrors the CPU call sites exactly):
    spectral_reduce(density_cmy, channel_density, base_density, illuminant,
                    output_matrix) -> (H, W, 3)

where
    density_cmy    : (H, W, 3) float
    channel_density: (nbands, 3)   = A,    matches contract('ijk,lk->ijl', cmy, A)
    base_density   : (nbands,) or None
    illuminant     : (nbands,)
    output_matrix  : (nbands, 3)   = B,    matches contract('ijk,kl->ijl', light, B)

Returns the result of the second contraction (the heavy ported block). The
cheap per-stage scalar ops (/normalization, *exposure_factor, +preflash, log10)
remain on the host and are unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "spectral.wgsl")


# --------------------------------------------------------------------------- #
# CPU reference - delegates to the real pipeline math so it is parity-exact.
# --------------------------------------------------------------------------- #
def spectral_reduce_cpu(
    density_cmy: np.ndarray,
    channel_density: np.ndarray,
    base_density: Optional[np.ndarray],
    illuminant: np.ndarray,
    output_matrix: np.ndarray,
) -> np.ndarray:
    """Exact CPU reference, using spektrafilm's own functions.

    This is the parity target. It calls the same `compute_density_spectral` and
    `density_to_light` the shipped pipeline uses, so the GPU kernel is validated
    against the real implementation, not a re-derivation.
    """
    from spektrafilm.model.develop import compute_density_spectral
    from spektrafilm.utils.conversions import density_to_light

    density_spectral = compute_density_spectral(channel_density, density_cmy, base_density)
    light = density_to_light(density_spectral, np.asarray(illuminant))
    # Mirror the pipeline's contract('ijk,kl->ijl', light, output_matrix) exactly
    # (same subscripts, same accumulation intent as opt_einsum's contract).
    return np.einsum("ijk,kl->ijl", light, np.asarray(output_matrix))


# --------------------------------------------------------------------------- #
# GPU implementation
# --------------------------------------------------------------------------- #
class SpectralGPU:
    """Compiles the fused spectral kernel once and dispatches it per call.

    Holds the compiled pipeline (shader compilation is the expensive setup).
    Input/output buffers are sized to the image and allocated per call; the
    constant matrices are small and re-uploaded each call for simplicity and to
    avoid stale-cache bugs in this first kernel.
    """

    def __init__(self) -> None:
        self.gpu = GPUDevice.get()
        if not self.gpu.is_available:
            raise RuntimeError(f"GPU not available: {self.gpu.init_error}")
        self._pipeline: Optional[Any] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        with open(_SHADER_PATH, "r", encoding="utf-8") as fh:
            code = fh.read()
        device = self.gpu.device
        module = device.create_shader_module(code=code)
        self._pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main"},
        )

    def run(
        self,
        density_cmy: np.ndarray,
        channel_density: np.ndarray,
        base_density: Optional[np.ndarray],
        illuminant: np.ndarray,
        output_matrix: np.ndarray,
    ) -> np.ndarray:
        import wgpu  # type: ignore

        device = self.gpu.device
        h, w, c = density_cmy.shape
        if c != 3:
            raise ValueError(f"density_cmy must be (H, W, 3), got {density_cmy.shape}")

        a = np.ascontiguousarray(channel_density, dtype=np.float32)
        nbands = a.shape[0]
        if a.shape != (nbands, 3):
            raise ValueError(f"channel_density must be (nbands, 3), got {a.shape}")
        b = np.ascontiguousarray(output_matrix, dtype=np.float32)
        if b.shape != (nbands, 3):
            raise ValueError(f"output_matrix must be (nbands, 3), got {b.shape}")
        illum = np.ascontiguousarray(illuminant, dtype=np.float32).reshape(-1)
        if illum.shape[0] != nbands:
            raise ValueError(f"illuminant length {illum.shape[0]} != nbands {nbands}")
        if base_density is None:
            base = np.zeros(nbands, dtype=np.float32)
        else:
            base = np.ascontiguousarray(base_density, dtype=np.float32).reshape(-1)
            if base.shape[0] != nbands:
                raise ValueError(f"base_density length {base.shape[0]} != nbands {nbands}")

        cmy = np.ascontiguousarray(density_cmy, dtype=np.float32)

        usage = wgpu.BufferUsage
        cmy_buf = self._make_buffer(cmy, usage.STORAGE | usage.COPY_DST)
        a_buf = self._make_buffer(a, usage.STORAGE | usage.COPY_DST)
        base_buf = self._make_buffer(base, usage.STORAGE | usage.COPY_DST)
        illum_buf = self._make_buffer(illum, usage.STORAGE | usage.COPY_DST)
        b_buf = self._make_buffer(b, usage.STORAGE | usage.COPY_DST)

        # Per-pixel NaN mask (host-computed). The only NaN source in the pipeline
        # is cmy (the constant matrices are finite); a NaN in any cmy channel
        # makes the whole pixel's density NaN, which density_to_light zeroes.
        # This integer mask is the compiler-proof NaN guard in the shader.
        nan_mask = np.ascontiguousarray(np.isnan(cmy).any(axis=2)).astype(np.uint32)
        nanmask_buf = device.create_buffer(size=nan_mask.nbytes, usage=usage.STORAGE | usage.COPY_DST)
        device.queue.write_buffer(nanmask_buf, 0, nan_mask)

        out_nbytes = h * w * 3 * 4
        out_buf = device.create_buffer(size=out_nbytes, usage=usage.STORAGE | usage.COPY_SRC)

        params = np.array([w, h, nbands, 0], dtype=np.uint32)
        params_buf = device.create_buffer(size=params.nbytes, usage=usage.UNIFORM | usage.COPY_DST)
        device.queue.write_buffer(params_buf, 0, params)

        bind_group = device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": cmy_buf, "offset": 0, "size": cmy.nbytes}},
                {"binding": 1, "resource": {"buffer": a_buf, "offset": 0, "size": a.nbytes}},
                {"binding": 2, "resource": {"buffer": base_buf, "offset": 0, "size": base.nbytes}},
                {"binding": 3, "resource": {"buffer": illum_buf, "offset": 0, "size": illum.nbytes}},
                {"binding": 4, "resource": {"buffer": b_buf, "offset": 0, "size": b.nbytes}},
                {"binding": 5, "resource": {"buffer": out_buf, "offset": 0, "size": out_nbytes}},
                {"binding": 6, "resource": {"buffer": params_buf, "offset": 0, "size": params.nbytes}},
                {"binding": 7, "resource": {"buffer": nanmask_buf, "offset": 0, "size": nan_mask.nbytes}},
            ],
        )

        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        gx = (w + 7) // 8
        gy = (h + 7) // 8
        cpass.dispatch_workgroups(gx, gy, 1)
        cpass.end()

        staging = device.create_buffer(size=out_nbytes, usage=usage.COPY_DST | usage.MAP_READ)
        encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, out_nbytes)
        device.queue.submit([encoder.finish()])

        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            mapped = staging.read_mapped()
            result = np.frombuffer(mapped, dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()

        for buf in (cmy_buf, a_buf, base_buf, illum_buf, b_buf, nanmask_buf, out_buf, params_buf, staging):
            try:
                buf.destroy()
            except Exception:
                pass

        return result

    def _make_buffer(self, arr: np.ndarray, usage: int) -> Any:
        device = self.gpu.device
        buf = device.create_buffer(size=arr.nbytes, usage=usage)
        device.queue.write_buffer(buf, 0, arr)
        return buf


# Below this pixel count, backend="auto" stays on CPU: dispatch + transfer
# overhead exceeds the compute. Covers the LUT identity probe (2x2) and the
# black/white reference calls (1x1). Explicit backend="gpu" is not gated.
MIN_GPU_PIXELS = 4096

# Module-level cache so the pipeline is compiled once per process.
_SPECTRAL_GPU: Optional[SpectralGPU] = None
_SPECTRAL_GPU_FAILED = False


def _get_spectral_gpu() -> Optional[SpectralGPU]:
    global _SPECTRAL_GPU, _SPECTRAL_GPU_FAILED
    if _SPECTRAL_GPU_FAILED:
        return None
    if _SPECTRAL_GPU is None:
        try:
            _SPECTRAL_GPU = SpectralGPU()
        except Exception as exc:
            logger.warning("Spectral GPU kernel unavailable (%s); using CPU", exc)
            _SPECTRAL_GPU_FAILED = True
            return None
    return _SPECTRAL_GPU


def spectral_reduce(
    density_cmy: np.ndarray,
    channel_density: np.ndarray,
    base_density: Optional[np.ndarray],
    illuminant: np.ndarray,
    output_matrix: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Dispatch the fused spectral reduction to GPU or CPU.

    backend: "auto" (GPU if available, else CPU), "gpu" (GPU, fall back to CPU
    on any error), or "cpu" (force the reference path).

    The CPU path is always available and is the parity reference. Any GPU
    failure (no device, shader error, dispatch fault) silently falls back to CPU
    with a logged warning, mirroring NegPy's ImageProcessor.run_pipeline.
    """
    from spektrafilm.gpu.backend import Backend, resolve_backend

    requested = (backend or "auto").strip().lower()
    chosen = resolve_backend(requested)
    if (
        chosen is Backend.GPU
        and requested == "auto"
        and density_cmy.shape[0] * density_cmy.shape[1] < MIN_GPU_PIXELS
    ):
        chosen = Backend.CPU
    if chosen is Backend.GPU:
        impl = _get_spectral_gpu()
        if impl is not None:
            try:
                return impl.run(density_cmy, channel_density, base_density, illuminant, output_matrix)
            except Exception:
                logger.exception("Spectral GPU dispatch failed; falling back to CPU")
    return spectral_reduce_cpu(density_cmy, channel_density, base_density, illuminant, output_matrix)

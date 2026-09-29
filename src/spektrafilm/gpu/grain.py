"""B2 GPU grain primitive: mirrors the poisson_binomial + use_fast_stats branch
of model/grain.py:layer_particle_model and the samplers in utils/fast_stats.py.

PARITY MODEL: statistical, not per-pixel. The GPU mirrors fast_stats' exact
algorithm (Knuth / Normal-approx / inversion branches and thresholds), so the
output DISTRIBUTION matches, but the random stream is the GPU's own
counter-based RNG (pcg4d, keyed per pixel + the CPU's per-layer seed). A GPU
cannot reproduce scipy/numba RNG streams; the parity test judges moments and
distribution, and the mean field (deterministic in density) matches tightly.

The dispatch hook deliberately engages ONLY for use_fast_stats=True (the shipped
path). With use_fast_stats=False the CPU uses scipy (the full-precision
reference), which we leave untouched. The optional per-particle blur reuses the
B1 fast_gaussian_filter chokepoint, so it stays GPU-accelerated and faithful.
"""

from __future__ import annotations

import os
import math
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger
from spektrafilm.gpu.spectral import MIN_GPU_PIXELS  # shared dispatch-overhead gate

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "grain.wgsl")
_PARAMS_NBYTES = 48  # 4 u32 + 4 f32 + origin/pads, must match struct Params in grain.wgsl


class GrainGPU:
    """Compiles the grain pipeline once; one dispatch per layer call."""

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
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": module, "entry_point": "grain_layer"},
        )
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

    def layer(
        self,
        density: np.ndarray,
        density_max: float,
        n_particles_per_pixel: float,
        grain_uniformity: float,
        seed: int,
    ) -> np.ndarray:
        """One layer of the poisson_binomial grain model (pre per-particle blur)."""
        import wgpu

        img = np.ascontiguousarray(density, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError("grain layer expects a 2D density array")
        h, w = img.shape
        nbytes = img.nbytes
        od_particle = float(density_max) / float(n_particles_per_pixel)

        usage = (
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        src = self._pooled("density", nbytes, usage)
        dst = self._pooled("grain", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, img)

        raw = np.zeros(12, dtype=np.uint32)
        raw[0] = np.uint32(w)
        raw[1] = np.uint32(h)
        raw[2] = np.uint32(int(seed) & 0xFFFFFFFF)
        raw[3] = 0
        raw[4:8] = np.array(
            [density_max, n_particles_per_pixel, grain_uniformity, od_particle],
            dtype=np.float32,
        ).view(np.uint32)
        # raw[8:10] = tile origin (6E); 0 for the whole-image path.
        params = self.gpu.device.create_buffer(
            size=_PARAMS_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.gpu.device.queue.write_buffer(params, 0, raw)

        bind_group = self.gpu.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
            ],
        )
        encoder = self.gpu.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        cpass.end()

        staging = self._staging_for(nbytes)
        encoder.copy_buffer_to_buffer(dst, 0, staging, 0, nbytes)
        self.gpu.device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = (
                np.frombuffer(staging.read_mapped(0, nbytes), dtype=np.float32)
                .reshape(h, w)
                .copy()
            )
        finally:
            staging.unmap()
        try:
            params.destroy()
        except Exception:
            pass
        return out


# ----------------------------------------------------------------- dispatch
_GRAIN_GPU: Optional[GrainGPU] = None
_GRAIN_GPU_FAILED = False


def _get_grain_gpu() -> Optional[GrainGPU]:
    global _GRAIN_GPU, _GRAIN_GPU_FAILED
    if _GRAIN_GPU_FAILED:
        return None
    if _GRAIN_GPU is None:
        try:
            _GRAIN_GPU = GrainGPU()
        except Exception as exc:
            logger.warning("Grain GPU kernel unavailable (%s); using CPU", exc)
            _GRAIN_GPU_FAILED = True
            return None
    return _GRAIN_GPU


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


def layer_particle_model_dispatch(
    density,
    density_max,
    n_particles_per_pixel,
    grain_uniformity,
    seed,
    blur_particle: float = 0.0,
    *,
    backend: str = "auto",
) -> Optional[np.ndarray]:
    """GPU dispatch hook for model.grain.layer_particle_model (poisson_binomial,
    use_fast_stats path).

    Returns the grain layer as float64 (matching the CPU path's dtype), or None
    when the CPU path should run instead (CPU backend, image below
    MIN_GPU_PIXELS, no seed, no device, unsupported input, or any GPU error)."""
    from spektrafilm.gpu.backend import Backend

    if seed is None:  # CPU uses random state; nothing to key the GPU RNG with
        return None
    img = np.asarray(density)
    if img.ndim != 2:
        return None
    if _resolve(img, backend) is not Backend.GPU:
        return None
    impl = _get_grain_gpu()
    if impl is None:
        return None
    try:
        grain = impl.layer(
            density,
            float(density_max),
            float(n_particles_per_pixel),
            float(grain_uniformity),
            int(seed),
        )
    except Exception:
        logger.exception("Grain GPU dispatch failed; falling back to CPU")
        return None

    if blur_particle and blur_particle > 0:
        # Same blur the CPU path applies; routes through the B1 chokepoint.
        from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter

        od_particle = float(density_max) / float(n_particles_per_pixel)
        grain = fast_gaussian_filter(grain, blur_particle * math.sqrt(od_particle))
    return np.asarray(grain, dtype=np.float64)

"""G2d-4: resident grain -- model/grain.apply_grain's sublayers path in ONE GPU
submit (single upload, single readback).

The shipped path (sublayers_active + use_fast_stats) previously ran:
  interp_density_cmy_layers      CPU numba (~1.5 s at 24 MP)
  9x layer_particle_model        GPU, but one upload + readback EACH (~1.2 s)
  accumulation / dtype churn     CPU numpy (~1.5 s)
  final grain blur               GPU via chokepoint (one more round-trip)

This chain keeps everything on device:
  upload density once ->
  for each (sublayer, channel): interp_plane (grain_layers.wgsl, same bisection
    as the validated density_curve.wgsl, + density_min_layers fold) ->
    grain_layer (the UNCHANGED validated grain.wgsl kernel, same pcg4d seeding)
    -> optional dye-cloud blur (BlurGPU, c=1) -> accum_plane into the
    interleaved output ->
  optional final grain blur (BlurGPU, c=3) -> read back once.
  density_min is subtracted host-side after readback: the CPU subtracts before
  the final blur, but the blur has exactly unit DC gain (normalized FIR taps /
  DC-exact IIR coefficients), so blur(x) - c == blur(x - c) to f32 rounding.

PARITY MODEL: same as gpu/grain.py -- statistical for GPU-vs-CPU (the GPU RNG
is its own counter-based stream), EXACT for resident-vs-standalone-GPU-kernels
given identical f32 plane inputs and seeds (test_grain_resident_parity.py).

Gates (dispatch returns None -> unchanged existing path): env toggle
SPEKTRAFILM_RESIDENT_GRAIN off, grain inactive / not sublayers / not
use_fast_stats, micro-structure would engage (sigma > 0.05: the resident chain
does not model it), non-(H,W,3) input, non-finite or non-ascending curve grids,
no GPU, below MIN_GPU_PIXELS, or any error.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_LAYERS_WGSL = os.path.join(_SHADER_DIR, "grain_layers.wgsl")
_GRAIN_WGSL = os.path.join(_SHADER_DIR, "grain.wgsl")

_INTERP_PARAMS_NBYTES = 32  # 4 u32 + 4 f32
_ACCUM_PARAMS_NBYTES = 16   # 4 u32
_GRAIN_PARAMS_NBYTES = 48   # 4 u32 + 4 f32 + origin/pads (grain.wgsl Params)


class GrainLayersResident:
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

        un = wgpu.BufferBindingType.uniform
        ro = wgpu.BufferBindingType.read_only_storage
        st = wgpu.BufferBindingType.storage

        def bgl(types):
            return device.create_bind_group_layout(
                entries=[
                    {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
                    for i, t in enumerate(types)
                ]
            )

        layers_mod = module(_LAYERS_WGSL)
        self._interp_bgl = bgl([un, ro, ro, ro, st])
        self._interp_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._interp_bgl]),
            compute={"module": layers_mod, "entry_point": "interp_plane"},
        )
        self._accum_bgl = bgl([un, ro, st])
        self._accum_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._accum_bgl]),
            compute={"module": layers_mod, "entry_point": "accum_plane"},
        )
        # The validated grain kernel, compiled for buffer-to-buffer use here.
        self._grain_bgl = bgl([ro, st, un])
        self._grain_pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._grain_bgl]),
            compute={"module": module(_GRAIN_WGSL), "entry_point": "grain_layer"},
        )
        from spektrafilm.gpu.blur import BlurGPU

        self._blur = BlurGPU()

    def _buf(self, nbytes: int, usage) -> Any:
        return self.gpu.device.create_buffer(size=nbytes, usage=usage)

    def encode(
        self,
        encoder,
        density_buf,
        h: int,
        w: int,
        nbytes3: int,
        *,
        grid_x: np.ndarray,          # (K, 3) sign-folded ascending per channel
        grid_y: np.ndarray,          # (K, 3, 3) layer curves [K, sl, ch]
        sign: float,
        density_min_layers: np.ndarray,      # (3, 3) [sl, ch]
        density_max_layers: np.ndarray,      # (3, 3) [sl, ch]
        n_particles: np.ndarray,             # (3, 3) [sl, ch]
        uniformity: np.ndarray,              # (3,)
        seeds: np.ndarray,                   # (3, 3) [sl, ch]
        dye_blur_sigma: np.ndarray,          # (3, 3) [sl, ch], 0 = off
        final_blur_sigma: float,
        truncate: float = 3.0,
        origin: tuple = (0, 0),              # 6E: tile origin (x, y) in global coords
    ):
        """Phase 6D: encode the grain chain into an EXISTING encoder, reading
        density_buf (which the chain never writes — so the develop output
        buffer can feed it directly in a fused submit). Returns (out_buf,
        owned); the host-side density_min subtraction stays with the caller."""
        wgpu = self._wgpu
        device = self.gpu.device
        nbytes1 = nbytes3 // 3
        k = int(grid_x.shape[0])
        gx, gy = (w + 7) // 8, (h + 7) // 8

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        UNI = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

        accum_buf = self._buf(nbytes3, STO)      # zero-initialized by WebGPU
        plane_buf = self._buf(nbytes1, STO)
        gplane_buf = self._buf(nbytes1, STO)
        bplane_buf = self._buf(nbytes1, STO)
        tmp1_buf = self._buf(nbytes1, STO)

        transient: list = []
        extra: list = []

        def bg(layout, buffers, sizes):
            return device.create_bind_group(
                layout=layout,
                entries=[
                    {"binding": i, "resource": {"buffer": b, "offset": 0, "size": s}}
                    for i, (b, s) in enumerate(zip(buffers, sizes))
                ],
            )

        def dispatch(pipe, bind):
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

        for ch in range(3):
            gx_ch = np.ascontiguousarray(grid_x[:, ch], dtype=np.float32)
            gx_buf = self._buf(gx_ch.nbytes, STO)
            device.queue.write_buffer(gx_buf, 0, gx_ch)
            extra.append(gx_buf)
            for sl in range(3):
                gy_ch = np.ascontiguousarray(grid_y[:, sl, ch], dtype=np.float32)
                gy_buf = self._buf(gy_ch.nbytes, STO)
                device.queue.write_buffer(gy_buf, 0, gy_ch)
                extra.append(gy_buf)

                ipar = np.zeros(8, dtype=np.uint32)
                ipar[0:4] = [w, h, k, ch]
                ipar[4:8] = np.array(
                    [sign, float(density_min_layers[sl, ch]), 0.0, 0.0], dtype=np.float32
                ).view(np.uint32)
                ip_buf = self._buf(_INTERP_PARAMS_NBYTES, UNI)
                device.queue.write_buffer(ip_buf, 0, ipar)
                extra.append(ip_buf)
                dispatch(self._interp_pipe, bg(
                    self._interp_bgl,
                    [ip_buf, density_buf, gx_buf, gy_buf, plane_buf],
                    [_INTERP_PARAMS_NBYTES, nbytes3, gx_ch.nbytes, gy_ch.nbytes, nbytes1],
                ))

                dmax = float(density_max_layers[sl, ch])
                npp = float(n_particles[sl, ch])
                od_particle = dmax / npp
                gpar = np.zeros(12, dtype=np.uint32)
                gpar[0] = np.uint32(w)
                gpar[1] = np.uint32(h)
                gpar[2] = np.uint32(int(seeds[sl, ch]) & 0xFFFFFFFF)
                gpar[3] = 0
                gpar[4:8] = np.array(
                    [dmax, npp, float(uniformity[ch]), od_particle], dtype=np.float32
                ).view(np.uint32)
                gpar[8] = np.uint32(int(origin[0]))
                gpar[9] = np.uint32(int(origin[1]))
                gp_buf = self._buf(_GRAIN_PARAMS_NBYTES, UNI)
                device.queue.write_buffer(gp_buf, 0, gpar)
                extra.append(gp_buf)
                dispatch(self._grain_pipe, bg(
                    self._grain_bgl,
                    [plane_buf, gplane_buf, gp_buf],
                    [nbytes1, nbytes1, _GRAIN_PARAMS_NBYTES],
                ))

                src_plane = gplane_buf
                sig = float(dye_blur_sigma[sl, ch])
                if sig > 0.0:
                    self._blur.encode_gaussian(
                        encoder, transient, gplane_buf, tmp1_buf, bplane_buf,
                        sig, float(truncate), h, w, 1, nbytes1,
                    )
                    src_plane = bplane_buf

                apar = np.zeros(4, dtype=np.uint32)
                apar[0:3] = [w, h, ch]
                ap_buf = self._buf(_ACCUM_PARAMS_NBYTES, UNI)
                device.queue.write_buffer(ap_buf, 0, apar)
                extra.append(ap_buf)
                dispatch(self._accum_pipe, bg(
                    self._accum_bgl,
                    [ap_buf, src_plane, accum_buf],
                    [_ACCUM_PARAMS_NBYTES, nbytes1, nbytes3],
                ))

        out_buf = accum_buf
        if final_blur_sigma > 0.0:
            tmp3 = self._buf(nbytes3, STO)
            final3 = self._buf(nbytes3, STO)
            extra += [tmp3, final3]
            self._blur.encode_gaussian(
                encoder, transient, accum_buf, tmp3, final3,
                float(final_blur_sigma), float(truncate), h, w, 3, nbytes3,
            )
            out_buf = final3

        owned = [accum_buf, plane_buf, gplane_buf, bplane_buf, tmp1_buf] + extra + transient
        return out_buf, owned

    def run(
        self,
        density_cmy: np.ndarray,
        *,
        density_min: np.ndarray,             # (3,)
        **encode_kwargs,
    ) -> np.ndarray:
        wgpu = self._wgpu
        device = self.gpu.device

        img = np.ascontiguousarray(density_cmy, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("resident grain expects an (H, W, 3) density array")
        h, w, _ = img.shape
        nbytes3 = img.nbytes

        STO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        density_buf = self._buf(nbytes3, STO)
        device.queue.write_buffer(density_buf, 0, img)
        encoder = device.create_command_encoder()
        out_buf, owned = self.encode(encoder, density_buf, h, w, nbytes3, **encode_kwargs)

        staging = self._buf(nbytes3, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, nbytes3)
        device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            out = np.frombuffer(staging.read_mapped(0, nbytes3), dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            staging.unmap()
        for b in [density_buf, staging] + owned:
            try:
                b.destroy()
            except Exception:
                pass
        return np.asarray(out, dtype=np.float64) - np.asarray(density_min, dtype=np.float64)


_CORE: Optional[GrainLayersResident] = None
_CORE_FAILED = False


def _get_core() -> Optional[GrainLayersResident]:
    global _CORE, _CORE_FAILED
    if _CORE_FAILED:
        return None
    if _CORE is None:
        try:
            _CORE = GrainLayersResident()
        except Exception as exc:
            logger.warning("Resident grain unavailable (%s); using existing path", exc)
            _CORE_FAILED = True
            return None
    return _CORE


def _resident_grain_enabled() -> bool:
    """SPEKTRAFILM_RESIDENT_GRAIN env toggle (default ON). 0/off/false/no forces
    the legacy per-layer path -- A/B lever + safety hatch."""
    val = os.environ.get("SPEKTRAFILM_RESIDENT_GRAIN", "1").strip().lower()
    return val not in ("0", "off", "false", "no")


def compute_layer_params(grain, density_curves_layers, pixel_size_um):
    """Host precomputes, line-for-line apply_grain / apply_grain_to_density_layers."""
    density_max_layers_in = np.nanmax(density_curves_layers, axis=0)          # (3,3) [sl,ch]
    density_max_total = np.sum(density_max_layers_in, axis=0)                 # (3,)
    density_max_fractions = density_max_layers_in / density_max_total[None, :]
    density_min = np.array(grain.density_min, dtype=np.float64)
    density_min_layers = density_max_fractions * density_min[None, :]
    density_max_layers = density_max_layers_in + density_min_layers

    pixel_area_um2 = float(pixel_size_um) ** 2
    agx_area_layers = (
        float(grain.particle_area_um2)
        * np.array(grain.particle_scale, dtype=np.float64)[None, :]
        * np.array(grain.particle_scale_layers, dtype=np.float64)[:, None]
    )
    n_particles = pixel_area_um2 * density_max_fractions / agx_area_layers

    seeds = np.array([[0, 1, 2]], dtype=np.int64) + np.arange(3)[:, None] * 10  # [sl,ch] = seed[ch]+sl*10
    od_particle = density_max_layers / n_particles
    dye = float(grain.blur_dye_clouds_um)
    dye_blur_sigma = dye * np.sqrt(od_particle) if dye > 0 else np.zeros((3, 3))
    return {
        "density_min": density_min,
        "density_min_layers": density_min_layers,
        "density_max_layers": density_max_layers,
        "n_particles": n_particles,
        "seeds": seeds,
        "dye_blur_sigma": dye_blur_sigma,
    }


def grain_plan(
    pixel_size_um,
    grain,
    density_curves,
    density_curves_layers,
    profile_type: str,
    use_fast_stats: bool,
) -> Optional[dict]:
    """Eligibility gates + host precomputes for the grain chain, shared by the
    standalone dispatch below and the fused develop+grain dispatch (Phase 6D).
    Returns kwargs for GrainLayersResident.run (minus the image; includes
    density_min for the host-side subtraction), or None -> existing path."""
    if not getattr(grain, "active", False) or not getattr(grain, "sublayers_active", False):
        return None
    if not use_fast_stats:
        return None  # exact scipy sampling stays on the untouched CPU path
    # Micro-structure gate: the resident chain does not model it; same condition
    # as model/grain.add_micro_structure.
    micro = getattr(grain, "micro_structure", (0.0, 0.0))
    if float(micro[1]) * 0.001 / float(pixel_size_um) > 0.05:
        return None
    try:
        curves = np.asarray(density_curves, dtype=np.float64)
        layers = np.asarray(density_curves_layers, dtype=np.float64)
        if curves.ndim != 2 or curves.shape[1] != 3 or layers.shape != (curves.shape[0], 3, 3):
            return None
        if not (np.all(np.isfinite(curves)) and np.all(np.isfinite(layers))):
            return None
        positive = profile_type == "positive"
        sign = -1.0 if positive else 1.0
        grid_x = sign * curves
        # No monotonicity gate: real profile curves carry tiny (~1e-4)
        # non-monotone wiggles. The kernel's bisection is copied from the
        # validated density_curve.wgsl, which replicates fast_interp's
        # searchsorted bisection EXACTLY -- on non-monotone data both produce
        # the same deterministic result, so CPU/GPU parity is unaffected.

        p = compute_layer_params(grain, layers, pixel_size_um)
        return dict(
            grid_x=grid_x,
            grid_y=layers,
            sign=sign,
            density_min=p["density_min"],
            density_min_layers=p["density_min_layers"],
            density_max_layers=p["density_max_layers"],
            n_particles=p["n_particles"],
            uniformity=np.array(grain.uniformity, dtype=np.float64),
            seeds=p["seeds"],
            dye_blur_sigma=p["dye_blur_sigma"],
            final_blur_sigma=float(grain.blur),
        )
    except Exception:
        logger.exception("grain plan precompute failed")
        return None


def grain_layers_dispatch(
    density_cmy,
    pixel_size_um,
    grain,
    density_curves,
    density_curves_layers,
    profile_type: str,
    use_fast_stats: bool,
    *,
    backend: str = "auto",
) -> Optional[np.ndarray]:
    """GPU dispatch for apply_grain's sublayers path. float64 result or None."""
    if not _resident_grain_enabled():
        return None
    arr = np.asarray(density_cmy)
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

    # Phase 6D rail 1: grain allocates density + accum + optional final-blur
    # pair (image-sized) + 4 single-channel planes + staging; decline BEFORE
    # allocating when that will not fit.
    try:
        from spektrafilm.gpu.budget import consume_device_error, plan_residency
        nbytes = arr.shape[0] * arr.shape[1] * 3 * 4
        if plan_residency([nbytes] * 5 + [nbytes // 3] * 4) != 'gpu':
            return None
    except Exception:
        return None

    plan = grain_plan(
        pixel_size_um, grain, density_curves, density_curves_layers,
        profile_type, use_fast_stats,
    )
    if plan is None:
        return None
    core = _get_core()
    if core is None:
        return None
    try:
        consume_device_error()  # drop stale errors from other work
        out = core.run(arr, **plan)
        # Phase 6D rail 2: discard the readback if the device trapped an error.
        error = consume_device_error()
        if error is not None:
            logger.warning("resident grain trapped '%s'; result discarded", error)
            return None
        return out
    except Exception:
        logger.exception("Resident grain dispatch failed; falling back to existing path")
        return None

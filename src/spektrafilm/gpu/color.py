"""B3 GPU colour-conversion primitive: per-pixel 3x3 matrix + optional sRGB
cctf encode + optional clip, mirroring the `colour` calls in the scan stage.

PARITY: strict per-pixel max_abs (like spectral and blur), NOT statistical.
- The matrix step pulls `colour`'s OWN combined matrix (chromatic adaptation +
  XYZ->RGB) by transforming the basis through the identical colour.XYZ_to_RGB
  call, so the linear transform is colour's exact math; the GPU only adds f32
  matvec rounding (~1e-6).
- The sRGB cctf is replicated analytically (piecewise, signed toe), matching
  colour.RGB_to_RGB(..., apply_cctf_encoding=True) for the sRGB colourspace.

The GPU cctf path is ENABLED for sRGB (2026-07-06): color.wgsl now implements a
precision-refined software pow (double-single log2/exp2 with fma, avoiding the
~4.4e-4 hardware pow of Gotcha 5) measured at ~2.8e-7 max_abs vs colour's f64
sRGB encode -- well inside the 1e-5 parity budget. The MATRIX path (XYZ_to_RGB,
cctf off) is unchanged and parity-exact since the matrix comes from colour
itself.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger
from spektrafilm.gpu.spectral import MIN_GPU_PIXELS

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "color.wgsl")
_PARAMS_NBYTES = 64  # 3*vec4<f32> + 4*u32, must match struct Params in color.wgsl

# Colourspaces whose cctf_encoding runs on the GPU. sRGB enabled 2026-07-06:
# color.wgsl's precise_pow (software double-single log2/exp2, Gotcha 5 fix)
# measures ~2.8e-7 max_abs vs colour's f64 sRGB encode across [0, 1.5], the
# toe boundary, negatives, and large values -- far inside the 1e-5 parity
# budget (the old hardware pow was ~4.4e-4). Add other colourspaces only with
# their exact piecewise curves replicated in the shader + a parity test.
_GPU_CCTF: set = {"sRGB"}


class ColorGPU:
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
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
            ]
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": module, "entry_point": "color_transform"}
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

    def transform(self, image: np.ndarray, matrix: np.ndarray, cctf: int, clip: int,
                  decode: int = 0) -> np.ndarray:
        import wgpu

        img = np.ascontiguousarray(image, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("color transform expects an (H, W, 3) array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        M = np.asarray(matrix, dtype=np.float32)

        usage = (
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        src = self._pooled("src", nbytes, usage)
        dst = self._pooled("dst", nbytes, usage)
        self.gpu.device.queue.write_buffer(src, 0, img)

        raw = np.zeros(16, dtype=np.uint32)
        fl = np.zeros(12, dtype=np.float32)
        fl[0:3] = M[0]
        fl[3] = np.float32(decode)   # input cctf decode packed in row0.w (unused)
        fl[4:7] = M[1]
        fl[8:11] = M[2]
        raw[0:12] = fl.view(np.uint32)
        raw[12] = np.uint32(w)
        raw[13] = np.uint32(h)
        raw[14] = np.uint32(cctf)
        raw[15] = np.uint32(clip)
        params = self.gpu.device.create_buffer(
            size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
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
                .reshape(h, w, 3)
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
_COLOR_GPU: Optional[ColorGPU] = None
_COLOR_GPU_FAILED = False
_MATRIX_CACHE: dict[tuple, np.ndarray] = {}


def _get_color_gpu() -> Optional[ColorGPU]:
    global _COLOR_GPU, _COLOR_GPU_FAILED
    if _COLOR_GPU_FAILED:
        return None
    if _COLOR_GPU is None:
        try:
            _COLOR_GPU = ColorGPU()
        except Exception as exc:
            logger.warning("Color GPU kernel unavailable (%s); using CPU", exc)
            _COLOR_GPU_FAILED = True
            return None
    return _COLOR_GPU


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


def _xyz_to_rgb_matrix(colourspace, illuminant_xy) -> Optional[np.ndarray]:
    """The exact combined (chromatic-adaptation + XYZ->RGB) linear transform that
    colour.XYZ_to_RGB applies, recovered by transforming the basis through the
    identical call (linear when apply_cctf_encoding=False)."""
    key = (str(colourspace), tuple(np.round(np.asarray(illuminant_xy, dtype=np.float64), 10)))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import colour

        eye = np.eye(3, dtype=np.float64)
        cols = [
            np.asarray(
                colour.XYZ_to_RGB(
                    eye[k], colourspace=colourspace,
                    apply_cctf_encoding=False, illuminant=illuminant_xy,
                ),
                dtype=np.float64,
            )
            for k in range(3)
        ]
        M = np.column_stack(cols)
    except Exception:
        return None
    _MATRIX_CACHE[key] = M
    return M


def xyz_to_rgb_dispatch(xyz, colourspace, illuminant_xy, *, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for colour.XYZ_to_RGB(..., apply_cctf_encoding=False). Returns
    float64 (H,W,3) or None (CPU path should run)."""
    from spektrafilm.gpu.backend import Backend

    arr = np.asarray(xyz)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if _resolve(arr, backend) is not Backend.GPU:
        return None
    M = _xyz_to_rgb_matrix(colourspace, illuminant_xy)
    if M is None:
        return None
    impl = _get_color_gpu()
    if impl is None:
        return None
    try:
        return impl.transform(arr, M, cctf=0, clip=0).astype(np.float64)
    except Exception:
        logger.exception("XYZ_to_RGB GPU dispatch failed; falling back to CPU")
        return None


def _rgb_to_xyz_matrix(colourspace, illuminant_xy, cat) -> Optional[np.ndarray]:
    """The exact RGB->XYZ linear transform colour.RGB_to_XYZ applies (matrix +
    chromatic adaptation), recovered by transforming the basis through the
    identical call. Linear only when apply_cctf_decoding is False."""
    key = ("rgb2xyz", str(colourspace), str(cat),
           tuple(np.round(np.asarray(illuminant_xy, dtype=np.float64), 10)))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import colour

        eye = np.eye(3, dtype=np.float64)
        cols = [
            np.asarray(
                colour.RGB_to_XYZ(
                    eye[k], colourspace=colourspace, apply_cctf_decoding=False,
                    illuminant=illuminant_xy, chromatic_adaptation_transform=cat,
                ),
                dtype=np.float64,
            )
            for k in range(3)
        ]
        M = np.column_stack(cols)
    except Exception:
        return None
    _MATRIX_CACHE[key] = M
    return M


def rgb_to_xyz_dispatch(
    rgb, colourspace, illuminant_xy, *,
    chromatic_adaptation_transform="CAT02", apply_cctf_decoding=False, backend: str = "auto",
) -> Optional[np.ndarray]:
    """GPU drop-in for colour.RGB_to_XYZ(..., apply_cctf_decoding=False). Returns
    float64 (H,W,3) or None. Declines when apply_cctf_decoding is True: that
    adds a power-law decode, which on the target GPU is below the parity budget
    (Gotcha 5), so the whole call stays on CPU in that case."""
    from spektrafilm.gpu.backend import Backend

    if apply_cctf_decoding:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if _resolve(arr, backend) is not Backend.GPU:
        return None
    M = _rgb_to_xyz_matrix(colourspace, illuminant_xy, chromatic_adaptation_transform)
    if M is None:
        return None
    impl = _get_color_gpu()
    if impl is None:
        return None
    try:
        return impl.transform(arr, M, cctf=0, clip=0).astype(np.float64)
    except Exception:
        logger.exception("RGB_to_XYZ GPU dispatch failed; falling back to CPU")
        return None


def _rgb_to_rgb_matrix(colourspace) -> Optional[np.ndarray]:
    """The exact linear transform colour.RGB_to_RGB(cs, cs) applies BEFORE the
    cctf encode, recovered by transforming the basis through the identical call
    with the cctfs off. NOT the identity: colour composes the colourspace's
    published (rounded) RGB->XYZ and XYZ->RGB matrices, whose product deviates
    from identity by ~1e-4 -- the GPU must replicate that for strict parity."""
    key = ("rgb2rgb", str(colourspace))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import colour

        eye = np.eye(3, dtype=np.float64)
        cols = [
            np.asarray(
                colour.RGB_to_RGB(
                    eye[k], colourspace, colourspace,
                    apply_cctf_decoding=False, apply_cctf_encoding=False,
                ),
                dtype=np.float64,
            )
            for k in range(3)
        ]
        M = np.column_stack(cols)
    except Exception:
        return None
    _MATRIX_CACHE[key] = M
    return M


# -- Phase 16D: the display transform's dominant stage on the GPU -------------
# apply_display_transform's first stage -- colour.RGB_to_RGB(output_space, sRGB,
# decode=True, encode=True) -- is ~108ms of numpy pow on a 1024px preview (72%
# of the whole preview render). It is deterministic matrix + transfer-function
# math, so it ports to the GPU at strict parity: DECODE(output cctf) -> the
# exact linear output->sRGB matrix -> ENCODE(sRGB). Only output spaces whose
# decode curve the shader replicates are handled; others return None (CPU path).
_DISPLAY_DECODE_MODE: dict[str, int] = {"ProPhoto RGB": 1}

# The color kernel holds src + dst as full-image f32 RGB storage buffers (12
# bytes/px each). An image whose buffer exceeds the device's storage-binding
# limit would crash wgpu (a device fault, NOT a catchable Python exception), so
# oversized frames MUST be declined proactively -> CPU. This bounds the PREVIEW
# (the 16D target, always small); full-res exports simply stay on the CPU.
_DISPLAY_BYTES_PER_PIXEL = 12


def _display_gpu_pixel_cap() -> int:
    try:
        from spektrafilm.gpu.tiling import _limit
        limits = GPUDevice.get().limits or {}
        max_binding = _limit(limits, "max-storage-buffer-binding-size",
                             "maxStorageBufferBindingSize", default=128 * 1024 * 1024)
    except Exception:
        max_binding = 128 * 1024 * 1024
    # 0.9 margin for the staging buffer + driver overhead.
    return int(0.9 * max_binding / _DISPLAY_BYTES_PER_PIXEL)


def _rgb_to_rgb_cross_matrix(src_space: str, dst_space: str) -> Optional[np.ndarray]:
    """The exact linear transform colour.RGB_to_RGB(src, dst) applies BETWEEN
    the input decode and the output encode, recovered by pushing the basis
    through the identical call with both cctfs off."""
    key = ("rgb2rgb_cross", str(src_space), str(dst_space))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import colour

        eye = np.eye(3, dtype=np.float64)
        cols = [
            np.asarray(
                colour.RGB_to_RGB(
                    eye[k], src_space, dst_space,
                    apply_cctf_decoding=False, apply_cctf_encoding=False,
                ),
                dtype=np.float64,
            )
            for k in range(3)
        ]
        M = np.column_stack(cols)
    except Exception:
        return None
    _MATRIX_CACHE[key] = M
    return M


def display_transform_stage1_dispatch(
    image, output_color_space: str, *, backend: str = "auto"
) -> Optional[np.ndarray]:
    """GPU drop-in for colour.RGB_to_RGB(image, output_color_space, 'sRGB',
    apply_cctf_decoding=True, apply_cctf_encoding=True). Returns the sRGB-encoded
    (H, W, 3) float32 in the same shape, or None so the CPU path runs (unknown
    output space, sub-threshold image, or no GPU)."""
    from spektrafilm.gpu.backend import Backend

    decode_mode = _DISPLAY_DECODE_MODE.get(str(output_color_space))
    if decode_mode is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if _resolve(arr, backend) is not Backend.GPU:
        return None
    # Decline oversized frames (a big export/upscale) so wgpu never faults on a
    # buffer past the device limit; the CPU path handles them.
    if arr.shape[0] * arr.shape[1] > _display_gpu_pixel_cap():
        return None
    M = _rgb_to_rgb_cross_matrix(output_color_space, "sRGB")
    if M is None:
        return None
    impl = _get_color_gpu()
    if impl is None:
        return None
    try:
        return impl.transform(arr, M, cctf=1, clip=1, decode=decode_mode)
    except Exception:
        logger.exception("display transform GPU dispatch failed; using CPU")
        return None


def cctf_encode_clip_dispatch(rgb, colourspace, *, clip: bool = True, backend: str = "auto") -> Optional[np.ndarray]:
    """GPU drop-in for colour.RGB_to_RGB(rgb, cs, cs, decode=False, encode=True)
    followed by clip([0,1]). Only the colourspaces in _GPU_CCTF are replicated
    exactly; others return None. float64 (H,W,3) or None."""
    from spektrafilm.gpu.backend import Backend

    if str(colourspace) not in _GPU_CCTF:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if _resolve(arr, backend) is not Backend.GPU:
        return None
    M = _rgb_to_rgb_matrix(colourspace)
    if M is None:
        return None
    impl = _get_color_gpu()
    if impl is None:
        return None
    try:
        return impl.transform(arr, M, cctf=1, clip=1 if clip else 0).astype(np.float64)
    except Exception:
        logger.exception("cctf encode GPU dispatch failed; falling back to CPU")
        return None

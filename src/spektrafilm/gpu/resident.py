"""Phase G1 spike: GPU-resident pipeline foundation.

Goal: prove the architecture the fast references use (references/spektrafilm.rs
`wgpu_backend.rs::run_film_chain`, and NegPy's single-encoder pattern):

    upload the image ONCE  ->  run N compute passes chained in ONE command
    encoder, ping-ponging between TWO persistent on-device buffers  ->  read
    back ONCE.

Our current GPU code instead does, per stage: write numpy -> dispatch -> read
back to numpy (f64) -> next stage. Those host<->device round-trips are why our
per-kernel speedups (1.4x overall) do not compound. This module isolates and
measures the cost of removing them.

To keep the measurement honest, both run modes use the SAME validated kernel
(`shaders/color.wgsl` per-pixel 3x3 matvec), so the ONLY difference is the
transfer pattern. The absolute numbers here are an architecture proof, not the
final pipeline speed (the real stages move onto this foundation in phases
G2-G4). Nothing here is on the default path; callers opt in and fall back.

The resident pipeline reuses GPUDevice (CPU fallback preserved) and never
raises into the pipeline: construction failure -> caller uses the existing path.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from spektrafilm.gpu.device import GPUDevice, logger

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders", "color.wgsl")
_PARAMS_NBYTES = 64  # 3*vec4<f32> + 4*u32, must match struct Params in color.wgsl


def _pack_matvec_params(matrix: np.ndarray, width: int, height: int, cctf: int = 0, clip: int = 0) -> np.ndarray:
    """Pack a 3x3 matrix + dims into the 16-u32 Params layout color.wgsl expects."""
    M = np.asarray(matrix, dtype=np.float32)
    raw = np.zeros(16, dtype=np.uint32)
    fl = np.zeros(12, dtype=np.float32)
    fl[0:3] = M[0]
    fl[4:7] = M[1]
    fl[8:11] = M[2]
    raw[0:12] = fl.view(np.uint32)
    raw[12] = np.uint32(width)
    raw[13] = np.uint32(height)
    raw[14] = np.uint32(cctf)
    raw[15] = np.uint32(clip)
    return raw


class GpuResidentPipeline:
    """Minimal on-device pass chain over two persistent ping-pong buffers.

    A pass is one dispatch of the reused matvec kernel reading the current
    buffer and writing the other. `run_resident` encodes the whole chain into a
    single command buffer with one upload and one readback; `run_per_stage`
    mimics our current pattern (upload/dispatch/readback per pass) for an A/B.
    """

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
        # Persistent ping-pong image buffers + a persistent readback staging buffer.
        self._buf_a: Optional[tuple[int, Any]] = None
        self._buf_b: Optional[tuple[int, Any]] = None
        self._staging: Optional[tuple[int, Any]] = None

    # ---------------------------------------------------------------- buffers
    def _image_buffer(self, slot: str, nbytes: int) -> Any:
        import wgpu

        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        entry = self._buf_a if slot == "a" else self._buf_b
        if entry is not None and entry[0] >= nbytes:
            return entry[1]
        if entry is not None:
            try:
                entry[1].destroy()
            except Exception:
                pass
        buf = self.gpu.device.create_buffer(size=nbytes, usage=usage)
        if slot == "a":
            self._buf_a = (nbytes, buf)
        else:
            self._buf_b = (nbytes, buf)
        return buf

    def _staging_buffer(self, nbytes: int) -> Any:
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

    def _uniform(self, raw: np.ndarray) -> Any:
        import wgpu

        buf = self.gpu.device.create_buffer(
            size=_PARAMS_NBYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.gpu.device.queue.write_buffer(buf, 0, raw)
        return buf

    def _bind_group(self, src: Any, dst: Any, params: Any, nbytes: int) -> Any:
        return self.gpu.device.create_bind_group(
            layout=self._bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": nbytes}},
                {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": nbytes}},
                {"binding": 2, "resource": {"buffer": params, "offset": 0, "size": _PARAMS_NBYTES}},
            ],
        )

    def _finish_and_read(self, encoder: Any, source_buf: Any, nbytes: int, h: int, w: int) -> np.ndarray:
        """Append the readback copy to THIS encoder, submit it ONCE, map, read.

        Folding the copy into the same encoder as the compute passes is what makes
        the resident path a single submit (and is required: the compute passes
        only execute when their encoder is submitted)."""
        import wgpu

        staging = self._staging_buffer(nbytes)
        encoder.copy_buffer_to_buffer(source_buf, 0, staging, 0, nbytes)
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
        return out

    # ------------------------------------------------------------- run modes
    def run_resident(self, image: np.ndarray, matrices: List[np.ndarray]) -> np.ndarray:
        """Upload ONCE, run all passes chained in ONE encoder (ping-pong), read ONCE."""
        img = np.ascontiguousarray(image, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("resident pipeline expects an (H, W, 3) array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        if not matrices:
            return img.copy()

        buf_a = self._image_buffer("a", nbytes)
        buf_b = self._image_buffer("b", nbytes)
        self.gpu.device.queue.write_buffer(buf_a, 0, img)  # single upload

        # One uniform buffer per pass (a shared one would be overwritten before
        # the single submit, so every pass would see the last matrix).
        param_bufs = [self._uniform(_pack_matvec_params(M, w, h)) for M in matrices]

        gx, gy = (w + 7) // 8, (h + 7) // 8
        encoder = self.gpu.device.create_command_encoder()
        current = buf_a
        other = buf_b
        for k in range(len(matrices)):
            bind_group = self._bind_group(current, other, param_bufs[k], nbytes)
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._pipeline)
            cpass.set_bind_group(0, bind_group)
            cpass.dispatch_workgroups(gx, gy, 1)
            cpass.end()
            current, other = other, current  # ping-pong
        # All passes encoded; `current` now holds the final result. Fold the
        # readback into the SAME encoder and submit once (this also actually runs
        # the passes -- a separate encoder would leave them unsubmitted).
        result = self._finish_and_read(encoder, current, nbytes, h, w)
        for pb in param_bufs:
            try:
                pb.destroy()
            except Exception:
                pass
        return result

    def run_per_stage(self, image: np.ndarray, matrices: List[np.ndarray]) -> np.ndarray:
        """Per-pass upload/dispatch/readback to numpy (mimics the CURRENT pattern)."""
        import wgpu

        img = np.ascontiguousarray(image, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("resident pipeline expects an (H, W, 3) array")
        h, w, _ = img.shape
        nbytes = img.nbytes
        gx, gy = (w + 7) // 8, (h + 7) // 8
        cur = img
        for M in matrices:
            buf_a = self._image_buffer("a", nbytes)
            buf_b = self._image_buffer("b", nbytes)
            self.gpu.device.queue.write_buffer(buf_a, 0, np.ascontiguousarray(cur, dtype=np.float32))
            params = self._uniform(_pack_matvec_params(M, w, h))
            bind_group = self._bind_group(buf_a, buf_b, params, nbytes)
            encoder = self.gpu.device.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._pipeline)
            cpass.set_bind_group(0, bind_group)
            cpass.dispatch_workgroups(gx, gy, 1)
            cpass.end()
            # One submit per stage (compute + copy), then read back -> mimics the
            # current per-kernel host round-trip we are trying to eliminate.
            cur = self._finish_and_read(encoder, buf_b, nbytes, h, w)
            try:
                params.destroy()
            except Exception:
                pass
        return cur


_RESIDENT: Optional[GpuResidentPipeline] = None
_RESIDENT_FAILED = False


def get_resident_pipeline() -> Optional[GpuResidentPipeline]:
    """Lazily construct the resident pipeline, or None if the GPU is unavailable."""
    global _RESIDENT, _RESIDENT_FAILED
    if _RESIDENT_FAILED:
        return None
    if _RESIDENT is None:
        try:
            _RESIDENT = GpuResidentPipeline()
        except Exception as exc:
            logger.warning("Resident GPU pipeline unavailable (%s); using existing path", exc)
            _RESIDENT_FAILED = True
            return None
    return _RESIDENT

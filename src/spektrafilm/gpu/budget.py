"""Phase 6D: proactive residency guard + uncaptured-error trap.

Two safety rails the resident path allocates against (spec: REQUIRED in 6D,
reused by 6E's tiling):

1. plan_residency(buffer_sizes) — BEFORE allocating, sum the footprint and
   check it against the device limits (max-buffer-size,
   max-storage-buffer-binding-size) plus a CONSERVATIVE VRAM budget. Returns
   'gpu' or 'cpu' ('cpu' is the interim decline; 6E turns the same signal into
   tiling). The budget is deliberately modest — device limits do NOT reflect
   actual VRAM (wgpu-native reports effectively-unbounded max-buffer-size on
   the dev GPU), so a fixed cap keeps low-VRAM/integrated adapters safe.
   Override with SPEKTRAFILM_GPU_BUDGET_MB.

2. install_error_trap() / consume_device_error() — WebGPU surfaces many
   OOM/device errors ASYNCHRONOUSLY, so per-op try/except does not catch them
   and a readback can silently return zeros/garbage. The trap registers the
   device's uncaptured-error callback; dispatchers call consume_device_error()
   AFTER their readback and, if an error fired during the submit, DISCARD the
   result and fall back to CPU — a wrong image must never survive.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from spektrafilm.gpu.device import GPUDevice, logger

# Adapter-aware default budget. A 24 MP frame is ~288 MB in f32 RGB and the
# largest resident chain transiently holds ~8 image-sized buffers (~2.3 GB),
# so a flat conservative number would decline full-resolution work that a
# discrete card handles easily. The guard is the DETERMINISTIC decline for
# clearly-oversized work; the uncaptured-error trap below is the backstop
# that catches the unusual machine the heuristic misjudges — either way the
# result is a detected clean CPU fallback, never a silent wrong image.
_DISCRETE_BUDGET_MB = 4096
_MODEST_BUDGET_MB = 1024   # integrated / virtual / unknown adapters

_error_lock = threading.Lock()
_last_device_error: Optional[str] = None
_trap_installed = False


def _default_budget_mb() -> int:
    try:
        info = getattr(GPUDevice.get().adapter, 'info', None)
        if info is not None and str(info.get('adapter_type', '')).lower().startswith('discrete'):
            return _DISCRETE_BUDGET_MB
    except Exception:
        pass
    return _MODEST_BUDGET_MB


def gpu_budget_bytes() -> int:
    default = _default_budget_mb()
    try:
        megabytes = int(os.environ.get('SPEKTRAFILM_GPU_BUDGET_MB', default))
    except ValueError:
        megabytes = default
    return max(megabytes, 16) * 1024 * 1024


def _limit(limits: dict, *names: str, default: int) -> int:
    for name in names:
        value = limits.get(name)
        if value:
            return int(value)
    return default


def plan_residency(buffer_sizes: list[int]) -> str:
    """'gpu' when every buffer and the total footprint fit; else 'cpu'.

    Checked BEFORE any allocation so an over-budget graph is never built
    (deterministic ladder: 6E adds 'tile' between these two answers).
    """
    gpu = GPUDevice.get()
    if gpu.device is None:
        return 'cpu'
    limits = gpu.limits or {}
    max_binding = _limit(limits, 'max-storage-buffer-binding-size',
                         'maxStorageBufferBindingSize', default=128 * 1024 * 1024)
    max_buffer = _limit(limits, 'max-buffer-size', 'maxBufferSize',
                        default=256 * 1024 * 1024)
    budget = gpu_budget_bytes()
    total = 0
    for size in buffer_sizes:
        if size > max_binding or size > max_buffer:
            logger.info('residency guard: buffer %d B exceeds device limits; CPU', size)
            return 'cpu'
        total += size
    if total > budget:
        logger.info('residency guard: footprint %d B exceeds budget %d B; CPU', total, budget)
        return 'cpu'
    return 'gpu'


def _on_uncaptured_error(error_type=None, message=None) -> None:  # signature varies
    global _last_device_error
    text = f'{error_type}: {message}' if message is not None else str(error_type)
    with _error_lock:
        _last_device_error = text
    logger.warning('wgpu uncaptured device error trapped: %s', text)


def install_error_trap() -> bool:
    """Register the device uncaptured-error callback (idempotent)."""
    global _trap_installed
    if _trap_installed:
        return True
    gpu = GPUDevice.get()
    device = gpu.device
    if device is None:
        return False
    try:
        # wgpu-py 0.31 exposes the slot as an attribute.
        device._uncaptured_error_callback = _on_uncaptured_error
        _trap_installed = True
        return True
    except Exception as exc:
        logger.warning('could not install wgpu error trap: %s', exc)
        return False


def consume_device_error() -> Optional[str]:
    """The error (if any) trapped since the last call; clears the flag.
    Dispatchers call this after readback: non-None => discard the result."""
    global _last_device_error
    with _error_lock:
        error, _last_device_error = _last_device_error, None
    return error

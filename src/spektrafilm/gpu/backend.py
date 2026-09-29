"""Backend selection and automatic CPU fallback.

Models NegPy's split: the CPU path is always present and authoritative; the GPU
path is an optional accelerator chosen only when a device is actually available,
with automatic fallback on any failure. This module is the single place that
decides CPU vs GPU so the policy is consistent across kernels.

Selection precedence for backend="auto":
    1. SPEKTRAFILM_GPU env var, if set: "0"/"cpu"/"false" -> force CPU,
       "1"/"gpu"/"true" -> prefer GPU, "auto" -> hardware probe.
    2. Otherwise: GPU if GPUDevice.get().is_available else CPU.

Even when GPU is chosen, individual kernels fall back to CPU on any exception
(see spectral.spectral_reduce), so selecting GPU is never a correctness risk -
only a performance and parity-tolerance choice.
"""

from __future__ import annotations

import enum
import os

from spektrafilm.gpu.device import GPUDevice


class Backend(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


def _env_preference() -> str | None:
    raw = os.environ.get("SPEKTRAFILM_GPU")
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in ("0", "cpu", "false", "off", "no"):
        return "cpu"
    if val in ("1", "gpu", "true", "on", "yes"):
        return "gpu"
    return "auto"


def gpu_available() -> bool:
    """True if a wgpu compute device initialized on this machine."""
    return GPUDevice.get().is_available


def resolve_backend(backend: str = "auto") -> Backend:
    """Resolve a requested backend string to a concrete Backend.

    "cpu"  -> CPU (always).
    "gpu"  -> GPU (kernels still fall back to CPU on error).
    "auto" -> env override, else hardware probe.
    """
    backend = (backend or "auto").strip().lower()
    if backend == "cpu":
        return Backend.CPU
    if backend == "gpu":
        return Backend.GPU
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'cpu', or 'gpu'")

    pref = _env_preference()
    if pref == "cpu":
        return Backend.CPU
    if pref == "gpu":
        return Backend.GPU
    # pref is None or "auto" -> probe hardware
    return Backend.GPU if gpu_available() else Backend.CPU


def backend_report() -> dict:
    """Human-readable snapshot for logging / the device probe."""
    dev = GPUDevice.get()
    return {
        "gpu_available": dev.is_available,
        "backend_name": dev.backend_name,
        "init_error": dev.init_error,
        "env_SPEKTRAFILM_GPU": os.environ.get("SPEKTRAFILM_GPU"),
        "resolved_auto": resolve_backend("auto").value,
    }

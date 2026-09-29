"""wgpu adapter/device manager.

Adapted from references/NegPy/.../infrastructure/gpu/device.py (PyQt6 app),
stripped of any Qt coupling and using stdlib logging so it can be imported by
the headless pipeline, tests, and the GUI alike.

Initialization is wrapped so that *any* failure (no compatible adapter, driver
fault, wgpu import error) leaves the singleton in an "unavailable" state rather
than raising. Callers check `GPUDevice.get().is_available` and fall back to CPU.

wgpu API note: NegPy targets wgpu==0.31.0. The calls used here
(`wgpu.gpu.request_adapter_sync`, `adapter.request_device_sync`) are the
0.31.x synchronous API. `verify_wgpu_api()` checks the installed version at
runtime and logs a warning if the surface differs, so a future wgpu bump is
caught loudly instead of failing deep in a dispatch.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("spektrafilm.gpu")


def verify_wgpu_api() -> tuple[bool, str]:
    """Best-effort check that the installed wgpu exposes the API we use.

    Returns (ok, message). Does not raise. Logs a warning on mismatch.
    """
    try:
        import wgpu  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time environment issue
        return False, f"wgpu not importable: {exc}"

    version = getattr(wgpu, "__version__", "unknown")
    gpu = getattr(wgpu, "gpu", None)
    has_request_adapter = hasattr(gpu, "request_adapter_sync")
    if not has_request_adapter:
        msg = (
            f"installed wgpu {version} does not expose wgpu.gpu.request_adapter_sync; "
            "the device-init path was written against the 0.31.x sync API and may "
            "need adapting."
        )
        logger.warning(msg)
        return False, msg
    if version != "0.31.0":
        logger.info(
            "wgpu %s installed; NegPy reference used 0.31.0. Sync adapter API "
            "present, proceeding. Re-verify buffer-mapping calls if anything fails.",
            version,
        )
    return True, f"wgpu {version} API surface OK"


class GPUDevice:
    """Hardware adapter and device manager.

    Singleton: one initialization per process. Never raises on init failure;
    `is_available` reports whether a compute device was actually obtained.
    """

    _instance: Optional["GPUDevice"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if GPUDevice._instance is not None:
            raise RuntimeError("GPUDevice is a singleton; use GPUDevice.get()")
        self.adapter: Optional[Any] = None
        self.device: Optional[Any] = None
        self.limits: dict[str, Any] = {}
        self.init_error: Optional[str] = None
        self._initialize()

    @classmethod
    def get(cls) -> "GPUDevice":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = GPUDevice()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (tests only). Releases the device reference."""
        with cls._lock:
            cls._instance = None

    def _initialize(self) -> None:
        ok, msg = verify_wgpu_api()
        if not ok:
            self.init_error = msg
            return
        try:
            import wgpu  # type: ignore

            self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if self.adapter is not None:
                self.device = self._request_device_modest(self.adapter)
                # limits may be a mapping or an object depending on wgpu version
                raw_limits = getattr(self.device, "limits", {}) or {}
                try:
                    self.limits = dict(raw_limits)
                except (TypeError, ValueError):
                    self.limits = {}
                logger.info("wgpu backend ready: %s", self._summary())
            else:
                self.init_error = "no compatible GPU adapter found"
                logger.warning("No compatible GPU adapter found; CPU fallback will be used")
        except Exception as exc:
            self.init_error = f"device initialization failed: {exc}"
            logger.warning("GPU device init failed (%s); CPU fallback will be used", exc)
            self.adapter = None
            self.device = None

    # Phase 6D: request MODEST explicit limits instead of the adapter maximum.
    # With required_limits omitted, wgpu-native grants the adapter's best
    # (measured: ~4.5 PB max-buffer-size on the dev GPU) — numbers no budget
    # can be sized against. Capping the two buffer limits keeps the residency
    # guard meaningful and matches the broad-compatibility goal: we request
    # min(adapter, cap), so low-end adapters always qualify, and 6E tiling
    # sizes the work to whatever was actually granted.
    _MAX_BUFFER_CAP = 1024 * 1024 * 1024  # 1 GiB: > a 24 MP f32 RGB frame (~292 MB)

    @staticmethod
    def _request_device_modest(adapter: Any) -> Any:
        try:
            adapter_limits = dict(getattr(adapter, "limits", {}) or {})
            required: dict[str, int] = {}
            for key in ("max-buffer-size", "max-storage-buffer-binding-size"):
                value = adapter_limits.get(key)
                if value:
                    required[key] = min(int(value), GPUDevice._MAX_BUFFER_CAP)
            if required:
                return adapter.request_device_sync(required_limits=required)
        except Exception as exc:
            logger.warning(
                "modest-limits device request failed (%s); retrying with defaults", exc
            )
        return adapter.request_device_sync()

    def _summary(self) -> str:
        if self.adapter is None:
            return "no adapter"
        summary = getattr(self.adapter, "summary", None)
        if summary is not None:
            return str(summary)
        info = getattr(self.adapter, "info", None)
        return str(info) if info is not None else "adapter (no summary)"

    @property
    def is_available(self) -> bool:
        return self.device is not None

    @property
    def backend_name(self) -> Optional[str]:
        if self.adapter is None:
            return None
        summary = self._summary()
        if "(" in summary:
            return summary.split("(")[-1].replace(")", "").strip()
        parts = summary.split()
        return parts[-1] if parts else summary

    def poll(self) -> None:
        """Force queue processing (names differ across wgpu versions)."""
        if self.device is None:
            return
        if hasattr(self.device, "poll"):
            self.device.poll()
        elif hasattr(self.device, "_poll"):
            self.device._poll()

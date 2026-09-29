from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qtpy import QtCore

# NAT 2026-07-22: the worker except blocks below emit only the exception's
# type + message to the UI, discarding the traceback — so a caught crash
# never reached sys.excepthook and was NOT recorded in the rotating log,
# making failures hard to diagnose (the None-None scan bug took static
# analysis to find). Log the full traceback here to the app log.
_worker_logger = logging.getLogger('spektrafilm.simulation')


DISPLAY_PREVIEW_COLOR_SPACE = 'sRGB'
QObject = getattr(QtCore, 'QObject')
QRunnable = getattr(QtCore, 'QRunnable')
Signal = getattr(QtCore, 'Signal')


@dataclass(slots=True)
class SimulationRequest:
    mode_label: str
    image: np.ndarray
    params: object
    output_color_space: str
    use_display_transform: bool


@dataclass(slots=True)
class SimulationResult:
    mode_label: str
    display_image: np.ndarray
    float_image: np.ndarray
    output_color_space: str
    use_display_transform: bool
    status_message: str


class SimulationWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class SimulationWorker(QRunnable):
    def __init__(self, request: SimulationRequest, *, execute_request: Callable[[SimulationRequest], SimulationResult]):
        super().__init__()
        self._request = request
        self._execute_request = execute_request
        self.signals = SimulationWorkerSignals()

    def run(self) -> None:
        try:
            result = self._execute_request(self._request)
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _worker_logger.exception('worker failed: %s', type(exc).__name__)
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self.signals.finished.emit(result)


@dataclass(slots=True)
class ExportRequest:
    """Immutable payload for a file-export render+write job.

    Two modes:
      - full_precision=True: re-render ``image`` through ``params`` (LUT-off,
        produced directly in the saving space/encoding -> no colour conversion).
      - full_precision=False: reuse ``prerendered`` (the buffer from the last
        Scan/Preview, in ``source_color_space``/``source_cctf_encoding``) and
        only convert it to the saving space/encoding. No pipeline recompute.

    ``bit_depth`` is chosen at write time so 16-bit always comes from a float
    buffer, never an 8-bit one promoted.
    """

    image: object
    params: object
    filepath: str
    saving_color_space: str
    saving_cctf_encoding: bool
    bit_depth: int
    source_metadata: object
    full_precision: bool = True
    prerendered: object = None
    source_color_space: str = "sRGB"
    source_cctf_encoding: bool = True
    custom_tags: object = None


class ExportWorkerSignals(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)


class ExportWorker(QRunnable):
    """Runs a full-resolution export render then writes the file, off the UI
    thread. Cancellation is cooperative at the coarse pre-render / pre-write
    boundaries only (a single full-res render is one long call and is not
    interrupted mid-pass); this matches the chosen 'pre-render / between-image'
    cancel granularity and keeps the simulation pipeline untouched.
    """

    def __init__(
        self,
        request: ExportRequest,
        *,
        render_fn: Callable[[ExportRequest], np.ndarray],
        write_fn: Callable[[ExportRequest, np.ndarray], str],
    ):
        super().__init__()
        self._request = request
        self._render_fn = render_fn
        self._write_fn = write_fn
        self.signals = ExportWorkerSignals()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        if self._cancel.is_set():
            self.signals.failed.emit('Export cancelled')
            return
        try:
            full_precision = getattr(self._request, 'full_precision', True)
            self.signals.progress.emit(
                'Rendering full precision...' if full_precision else 'Preparing export from current scan...'
            )
            buffer = self._render_fn(self._request)
            if self._cancel.is_set():
                self.signals.failed.emit('Export cancelled')
                return
            self.signals.progress.emit('Writing file...')
            status_message = self._write_fn(self._request, buffer)
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _worker_logger.exception('worker failed: %s', type(exc).__name__)
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self.signals.finished.emit(status_message)


class LutExportSignals(QObject):
    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)


class LutExportWorker(QRunnable):
    """Bakes and writes a look LUT off the UI thread (LUT sampling runs the
    deterministic pipeline over a grid; small but can take a couple of seconds
    at high resolution)."""

    def __init__(self, run_fn: Callable[[], str]):
        super().__init__()
        self._run_fn = run_fn
        self.signals = LutExportSignals()

    def run(self) -> None:
        try:
            self.signals.progress.emit('Baking LUT...')
            status = self._run_fn()
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _worker_logger.exception('worker failed: %s', type(exc).__name__)
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self.signals.finished.emit(status)


def unique_export_path(out_dir: str, stem: str, ext: str) -> Path:
    """First non-existing '<stem>.<ext>' in out_dir, counting '<stem>_2' etc.
    (NegPy's no-overwrite naming)."""
    target = Path(out_dir) / f'{stem}.{ext}'
    counter = 2
    while target.exists():
        target = Path(out_dir) / f'{stem}_{counter}.{ext}'
        counter += 1
    return target


@dataclass(slots=True)
class BatchFrameJob:
    """Immutable per-frame payload for a roll export (NegPy ExportTask pattern):
    everything is resolved on the GUI thread BEFORE the worker starts, so the
    worker never touches widgets or the session."""

    name: str                 # source basename, stem used for the output name
    path: str                 # source file path
    kind: str                 # 'raw' | 'rgb'
    state: object             # this frame's GuiState (deep copy; the LOOK)
    saving_color_space: str
    saving_cctf_encoding: bool
    ext: str
    bit_depth: int
    custom_tags: object = None


class BatchExportWorker(QRunnable):
    """Renders and writes every frame of a roll sequentially, off the UI thread.

    Cancellation is cooperative BETWEEN frames and at each frame's pre-render /
    pre-write boundaries (a single full-res render is one long call). A frame
    that fails is recorded and the batch continues (NegPy's behavior); the
    summary reports both counts.
    """

    def __init__(
        self,
        jobs: list,
        *,
        render_fn: Callable[[BatchFrameJob], np.ndarray],
        write_fn: Callable[[BatchFrameJob, np.ndarray], str],
    ):
        super().__init__()
        self._jobs = list(jobs)
        self._render_fn = render_fn
        self._write_fn = write_fn
        self.signals = ExportWorkerSignals()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        total = len(self._jobs)
        done = 0
        failures: list[str] = []
        for index, job in enumerate(self._jobs, start=1):
            if self._cancel.is_set():
                self.signals.failed.emit(f'Export cancelled after {done} of {total} frame(s)')
                return
            label = f'frame {index}/{total} ({job.name})'
            try:
                self.signals.progress.emit(f'Roll export: {label} rendering...')
                buffer = self._render_fn(job)
                if self._cancel.is_set():
                    self.signals.failed.emit(f'Export cancelled after {done} of {total} frame(s)')
                    return
                self.signals.progress.emit(f'Roll export: {label} writing...')
                self._write_fn(job, buffer)
                done += 1
            except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(f'{job.name}: {type(exc).__name__}: {exc}')
                continue
        if failures:
            summary = f'Roll export: {done} of {total} frame(s) written; {len(failures)} failed ({failures[0]}' + (
                f' and {len(failures) - 1} more)' if len(failures) > 1 else ')'
            )
        else:
            summary = f'Roll export complete: {done} frame(s) written'
        self.signals.finished.emit(summary)


class RollAnalysisSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)  # {'evs': {path: ev}, 'failures': ['name: err', ...]}
    failed = Signal(str)


class RollAnalysisWorker(QRunnable):
    """Measures the auto-exposure EV of every roll frame off the UI thread
    (NegPy 'Batch Analysis' pattern: per-file measurement, worker loop with
    progress, cancel between frames, continue past per-frame failures)."""

    def __init__(self, jobs: list, *, measure_fn: Callable[[object], float]):
        super().__init__()
        self._jobs = list(jobs)
        self._measure_fn = measure_fn
        self.signals = RollAnalysisSignals()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        total = len(self._jobs)
        evs: dict[str, float] = {}
        failures: list[str] = []
        for index, job in enumerate(self._jobs, start=1):
            if self._cancel.is_set():
                self.signals.failed.emit(f'Roll analysis cancelled after {len(evs)} of {total} frame(s)')
                return
            self.signals.progress.emit(f'Roll analysis: frame {index}/{total} ({job.name})...')
            try:
                evs[job.path] = float(self._measure_fn(job))
            except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(f'{job.name}: {type(exc).__name__}: {exc}')
                continue
        self.signals.finished.emit({'evs': evs, 'failures': failures})


def normalized_image_data(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image, 0.0, 1.0)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        if max_value == 0:
            return image.astype(np.float32)
        return image.astype(np.float32) / max_value
    return image.astype(np.float32)


def apply_white_padding(image_data: np.ndarray, padding_pixels: float) -> np.ndarray:
    padding = max(0, int(round(padding_pixels)))
    if padding == 0:
        return np.asarray(image_data)

    image = np.asarray(image_data)
    if image.ndim < 2:
        return image

    fill_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    pad_width = [(padding, padding), (padding, padding)]
    pad_width.extend((0, 0) for _ in range(image.ndim - 2))
    return np.pad(image, pad_width, mode='constant', constant_values=fill_value)


def apply_border_padding(
    image_data: np.ndarray,
    padding_pixels: float,
    fill: 'tuple[float, ...] | None' = None,
) -> np.ndarray:
    """Solid border around an (H, W, C) image. ``fill`` is a per-channel color
    in NORMALIZED 0-1 terms (scaled to the dtype's range for integer images);
    None = white — the negative view/compare gets a film-base border."""
    padding = max(0, int(round(padding_pixels)))
    if padding == 0:
        return np.asarray(image_data)
    image = np.asarray(image_data)
    if fill is None or image.ndim != 3:
        return apply_white_padding(image, padding)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        values = [int(round(float(fill[i % len(fill)]) * max_value))
                  for i in range(image.shape[2])]
    else:
        values = [float(fill[i % len(fill)]) for i in range(image.shape[2])]
    channels = [
        np.pad(image[..., index], padding, mode='constant',
               constant_values=np.asarray(values[index], dtype=image.dtype))
        for index in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


def padding_pixels_for_image(image_data: np.ndarray, padding_fraction: float) -> int:
    image = np.asarray(image_data)
    if image.ndim < 2:
        return 0

    padding_fraction = max(0.0, float(padding_fraction))
    long_edge = max(int(image.shape[0]), int(image.shape[1]))
    return int(np.floor(long_edge * padding_fraction))


def display_profile_name(display_profile: object, *, imagecms_module: Any) -> str:
    try:
        profile_name = imagecms_module.getProfileName(display_profile)
    except (AttributeError, OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        profile_name = None

    if isinstance(profile_name, str):
        cleaned_name = profile_name.replace('\x00', ' ').strip()
        if cleaned_name:
            return ' '.join(cleaned_name.split())

    profile_filename = getattr(display_profile, 'filename', None)
    if isinstance(profile_filename, str) and profile_filename.strip():
        return Path(profile_filename).stem

    return type(display_profile).__name__


def display_profile_details(*, imagecms_module: Any) -> tuple[object | None, str | None]:
    try:
        display_profile = imagecms_module.get_display_profile()
    except (OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        return None, None
    if display_profile is None:
        return None, None
    return display_profile, display_profile_name(display_profile, imagecms_module=imagecms_module)


def display_profile_available(*, imagecms_module: Any) -> bool:
    try:
        return imagecms_module.get_display_profile() is not None
    except (OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        return False


def display_transform_status_message(enabled: bool, *, imagecms_module: Any) -> str:
    if not enabled:
        return 'Display transform: disabled'
    display_profile, profile_name = display_profile_details(imagecms_module=imagecms_module)
    if display_profile is None:
        return 'Display transform: no display profile, using raw preview'
    return f'Display transform: display profile found ({profile_name})'


def prepare_input_color_preview_image(
    image_data: np.ndarray,
    *,
    input_color_space: str,
    apply_cctf_decoding: bool,
    colour_module: Any,
) -> np.ndarray:
    normalized_image = normalized_image_data(np.asarray(image_data)[..., :3])
    try:
        srgb_preview = colour_module.RGB_to_RGB(
            normalized_image,
            input_color_space,
            DISPLAY_PREVIEW_COLOR_SPACE,
            apply_cctf_decoding=apply_cctf_decoding,
            apply_cctf_encoding=True,
        )
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
        return np.asarray(np.clip(normalized_image, 0.0, 1.0), dtype=np.float32)
    return np.asarray(np.clip(srgb_preview, 0.0, 1.0), dtype=np.float32)


def _gpu_display_stage1(image_data: np.ndarray, output_color_space: str):
    """Phase 16D: the colour-space + cctf display conversion on the GPU, or None
    to fall back to the CPU (unsupported space, sub-threshold size, GPU off, or
    no device). Never raises — the display path must not break on GPU trouble."""
    try:
        from spektrafilm.gpu.color import display_transform_stage1_dispatch
        return display_transform_stage1_dispatch(image_data, output_color_space)
    except Exception:
        return None


def apply_display_transform(
    image_data: np.ndarray,
    *,
    output_color_space: str,
    colour_module: Any,
    imagecms_module: Any,
    pil_image_module: Any,
) -> tuple[np.ndarray, str]:
    display_profile, profile_name = display_profile_details(imagecms_module=imagecms_module)
    if display_profile is None:
        return np.uint8(np.clip(image_data, 0.0, 1.0) * 255), 'Display transform: no display profile, using raw preview'

    if output_color_space == DISPLAY_PREVIEW_COLOR_SPACE:
        # Already sRGB-encoded: RGB_to_RGB(sRGB->sRGB) is an identity round-trip
        # (decode then re-encode the same cctf) but still runs two pow passes over
        # the full image. Skip it; only the monitor ICC transform below is needed.
        srgb_preview = image_data
    else:
        # Phase 16D: this colour-space + cctf conversion is ~72% of the preview
        # render (~108ms on a 1024px frame of numpy pow). Run it on the GPU when
        # available (strict parity, uint8 within 1 LSB); the CPU path below is
        # the automatic fallback for unsupported spaces / no-GPU machines.
        srgb_preview = _gpu_display_stage1(image_data, output_color_space)
        if srgb_preview is None:
            srgb_preview = colour_module.RGB_to_RGB(
                image_data,
                output_color_space,
                DISPLAY_PREVIEW_COLOR_SPACE,
                apply_cctf_decoding=True,
                apply_cctf_encoding=True,
            )
    srgb_preview_uint8 = np.uint8(np.clip(srgb_preview, 0.0, 1.0) * 255)
    source_profile = imagecms_module.createProfile(DISPLAY_PREVIEW_COLOR_SPACE)
    # NAT 2026-07-22: this is a desktop app on the user's OWN large camera
    # frames (a 100+ MP export/upscale trips PIL's 89 MP "decompression bomb"
    # ceiling — a server-side DoS guard that only spams a warning here). Lift
    # it before fromarray so legitimate high-res displays stay quiet.
    if getattr(pil_image_module, 'MAX_IMAGE_PIXELS', 0) is not None:
        pil_image_module.MAX_IMAGE_PIXELS = None
    source_image = pil_image_module.fromarray(srgb_preview_uint8, mode='RGB')
    transformed_image = imagecms_module.profileToProfile(source_image, source_profile, display_profile, outputMode='RGB')
    return np.asarray(transformed_image, dtype=np.uint8), f'Display transform: active ({profile_name})'


def prepare_output_display_image(
    image_data: np.ndarray,
    *,
    output_color_space: str,
    use_display_transform: bool,
    padding_pixels: float = 0.0,
    imagecms_module: Any,
    colour_module: Any,
    pil_image_module: Any,
) -> tuple[np.ndarray, str]:
    del padding_pixels
    normalized_image = normalized_image_data(np.asarray(image_data)[..., :3])
    preview_image = np.uint8(np.clip(normalized_image, 0.0, 1.0) * 255)
    if not use_display_transform:
        return preview_image, display_transform_status_message(False, imagecms_module=imagecms_module)
    try:
        transformed_image, status = apply_display_transform(
            normalized_image,
            output_color_space=output_color_space,
            colour_module=colour_module,
            imagecms_module=imagecms_module,
            pil_image_module=pil_image_module,
        )
        return transformed_image, status
    except (OSError, ValueError, TypeError, imagecms_module.PyCMSError):
        return preview_image, 'Display transform: transform failed, using raw preview'


def execute_simulation_request(
    request: SimulationRequest,
    *,
    run_simulation_fn: Callable[[np.ndarray, object], np.ndarray],
    prepare_output_display_image_fn: Callable[..., tuple[np.ndarray, str]],
) -> SimulationResult:
    import os
    import time

    timing = bool(os.environ.get('SPEKTRAFILM_TIMING'))
    t0 = time.perf_counter()
    scan = run_simulation_fn(request.image, request.params)
    t1 = time.perf_counter()
    scan_display, display_status = prepare_output_display_image_fn(
        scan,
        output_color_space=request.output_color_space,
        use_display_transform=request.use_display_transform,
    )
    t2 = time.perf_counter()
    if timing:
        print(
            f"[spektrafilm-timing] {request.mode_label}: render(total)={t1 - t0:.2f}s "
            f"display_transform={t2 - t1:.2f}s",
            flush=True,
        )
    return SimulationResult(
        mode_label=request.mode_label,
        display_image=scan_display,
        float_image=np.asarray(scan),
        output_color_space=request.output_color_space,
        use_display_transform=request.use_display_transform,
        status_message=display_status,
    )
from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets

from spektrafilm_gui import controller_persistence as persistence_actions
from spektrafilm_gui import controller_profile_sync as profile_sync
from spektrafilm_gui import controller_runtime as runtime
from spektrafilm_gui.controller_layers import (
    INPUT_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    ViewerLayerService,
)
from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    load_dialog_dir,
    load_gui_state_from_path,
    save_default_gui_state,
    save_dialog_dir,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, digest_after_selection, gui_state_from_params
from spektrafilm_gui.napari_layout import dialog_parent, reset_viewer_camera, set_canvas_background, set_status
from spektrafilm_gui.params_mapper import build_params_from_state, configure_export_params
from spektrafilm_gui.options import export_format_spec
from spektrafilm_gui.roll_session import RollSession
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import WidgetBundle

OUTPUT_FLOAT_DATA_KEY = 'pipeline_float_output'
OUTPUT_COLOR_SPACE_KEY = 'pipeline_output_color_space'
OUTPUT_CCTF_ENCODING_KEY = 'pipeline_output_cctf_encoding'
OUTPUT_DISPLAY_TRANSFORM_KEY = 'pipeline_use_display_transform'
PROFILE_SYNC_FIELDS = profile_sync.PROFILE_SYNC_FIELDS
if TYPE_CHECKING:
    pass


QThreadPool = getattr(QtCore, 'QThreadPool')
QTimer = getattr(QtCore, 'QTimer')
QSettings = getattr(QtCore, 'QSettings')
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
SimulationRequest = runtime.SimulationRequest
SimulationResult = runtime.SimulationResult
ExportRequest = runtime.ExportRequest
ExportWorker = runtime.ExportWorker
STARTUP_PREVIEW_ASPECT_RATIO = (3, 2)


class _DirMemoryDialog:
    """Wraps QFileDialog to open in the last-used directory via QSettings."""

    def __init__(self, key: str) -> None:
        self._key = key

    def get_save_file_name(self, parent, title, filename, file_filter):
        last_dir = load_dialog_dir(self._key)
        initial = str(Path(last_dir) / Path(filename).name) if last_dir else filename
        path, fmt = QFileDialog.getSaveFileName(parent, title, initial, file_filter)
        if path:
            save_dialog_dir(self._key, str(Path(path).parent))
        return path, fmt

    def get_open_file_name(self, parent, title, _initial, file_filter):
        path, fmt = QFileDialog.getOpenFileName(
            parent, title, load_dialog_dir(self._key), file_filter
        )
        if path:
            save_dialog_dir(self._key, str(Path(path).parent))
        return path, fmt


class _LazyModuleProxy:
    def __init__(self, loader):
        self._loader = loader
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = self._loader()
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


def _import_colour_module():
    return import_module('colour')


def _import_pil_image_module():
    return import_module('PIL.Image')


def _import_imagecms_module():
    return import_module('PIL.ImageCms')


def runtime_simulator(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').Simulator(*args, **kwargs)


def digest_params(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').digest_params(*args, **kwargs)


def load_image_oiio(*args, **kwargs):
    return import_module('spektrafilm.utils.io').load_image_oiio(*args, **kwargs)


def save_image_oiio(*args, **kwargs):
    return import_module("spektrafilm.utils.io").save_image_oiio(*args, **kwargs)


def read_image_metadata(*args, **kwargs):
    return import_module("spektrafilm.utils.io").read_image_metadata(*args, **kwargs)


def write_image_metadata(*args, **kwargs):
    return import_module("spektrafilm.utils.io").write_image_metadata(*args, **kwargs)


def load_and_process_raw_file(*args, **kwargs):
    return import_module('spektrafilm.utils.raw_file_processor').load_and_process_raw_file(*args, **kwargs)


def apply_rgb_white_balance(*args, **kwargs):
    return import_module('spektrafilm.utils.rgb_white_balance').apply_rgb_white_balance(*args, **kwargs)




def resize_for_preview(*args, **kwargs):
    return import_module('spektrafilm.utils.preview').resize_for_preview(*args, **kwargs)


colour = _LazyModuleProxy(_import_colour_module)
PILImage = _LazyModuleProxy(_import_pil_image_module)
ImageCms = _LazyModuleProxy(_import_imagecms_module)


class GuiController:
    # Class-level default so partially-built controllers (tests use __new__)
    # see the closed gate too; __init__ re-asserts it per instance.
    _first_render_done = False

    def __init__(self, *, viewer: Any, widgets: WidgetBundle):
        self._viewer = viewer
        self._widgets = widgets
        self._layers = ViewerLayerService(
            viewer=viewer,
            output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
            output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
            output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
            output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
        )
        self._thread_pool = QThreadPool.globalInstance()
        self._active_simulation_worker: runtime.SimulationWorker | None = None
        self._active_simulation_label: str | None = None
        self._active_export_worker: runtime.ExportWorker | None = None
        self._runtime_simulator = None
        self._next_runtime_digest_applies_stock_specifics = True
        self._current_input_image: np.ndarray | None = None
        self._current_input_path: str | None = None
        self._current_preview_image: np.ndarray | None = None
        self._auto_preview_scheduled = False
        self._pending_auto_preview = False
        # Phase 16B: while any slider/knob/fader is being dragged, preview
        # frames render at reduced resolution for a live stream; when the
        # drag ends, one full-preview-resolution frame follows.
        self._last_preview_reduced = False
        self._drag_preview_divisor = 2.0
        from spektrafilm_gui.widget_slider import set_drag_end_callback
        set_drag_end_callback(self._on_drag_ended)
        # Phase 16C: stage-dirty partial re-render. Two byte-identical
        # tiers reuse the latest still-valid pipeline tap since the last full
        # preview: SCAN (only scan_finishing changed -> re-run scan, ~4x) and
        # DEVELOP (only develop-and-later sections changed -> re-run from
        # LOG_E_FILM, skipping preprocess+expose, ~1.7x).
        self._stage_cache: dict | None = None
        self._pending_stage_cache_keys: tuple | None = None
        self._preview_image_version = 0
        # Auto preview stays dormant until the first render of the session
        # (Preview/F5, Scan, or Ctrl+R before/after): opening an image shows
        # the input untouched, one explicit render arms the live loop
        # (NAT 2026-07-18).
        self._first_render_done = False
        # Phase 10C stock editor: the modeless dialog + the UNSAVED recipe it
        # is currently editing (rendered live in place of the selected film).
        self._stock_editor = None
        self._live_stock_recipe = None
        self._pending_live_render = False
        self._active_simulation_reports_status = True
        self._roll = RollSession()
        self._roll_thumbs: dict[str, Any] = {}   # path -> QPixmap (GUI thread only)
        self._thumb_tasks: dict[str, Any] = {}   # path -> ThumbnailTask until delivered
        self._roll_analysis_ev: float | None = None
        self._roll_analysis_running = False
        self._last_film_curves = None

    def show_startup_placeholder(self) -> None:
        if self._white_border_layer() is not None:
            return

        state = collect_gui_state(widgets=self._widgets)
        preview_height = max(int(state.display.preview_max_size), 1)
        preview_width = max(
            int(round(preview_height * STARTUP_PREVIEW_ASPECT_RATIO[1] / STARTUP_PREVIEW_ASPECT_RATIO[0])),
            1,
        )
        placeholder_preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
        self._layers.set_or_add_input_preview_layer(
            placeholder_preview,
            watermark_source_size=(preview_height, preview_width),
            white_padding=state.display.white_padding,
            hide_output=True,
            set_active=True,
        )
        self._home_input_stack()

    def load_input_image(self, path: str) -> None:
        gui_state = collect_gui_state(widgets=self._widgets)
        set_status(self._viewer, "Loading image...", timeout_ms=0)
        try:
            image = load_image_oiio(path)[..., :3]
            # White balance/tint like the raw path, applied in the linear
            # light of the declared input colour space so the stored image
            # keeps the encoding the rest of the pipeline expects.
            image = apply_rgb_white_balance(
                image,
                white_balance=gui_state.load_rgb.white_balance,
                temperature=gui_state.load_rgb.temperature,
                tint=gui_state.load_rgb.tint,
                color_space=gui_state.input_image.input_color_space,
                cctf_encoded=gui_state.input_image.apply_cctf_decoding,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Import RGB', f'Failed to load RGB image.\n\n{exc}')
            set_status(self._viewer, 'Import RGB failed')
            return
        self._current_input_path = path
        self._set_or_add_input_stack(image)
        wb_adjusted = gui_state.load_rgb.white_balance not in ('none', 'as_shot')
        if wb_adjusted:
            set_status(self._viewer, 'Loaded RGB with white balance adjustment')
        else:
            set_status(self._viewer, 'Loaded RGB')
        self._request_auto_preview_if_enabled()

    def load_raw_image(self, path: str) -> None:
        gui_state = collect_gui_state(widgets=self._widgets)
        set_status(self._viewer, "Loading raw...", timeout_ms=0)
        lens_info: dict[str, str] = {}
        try:
            image = load_and_process_raw_file(
                path,
                white_balance=gui_state.load_raw.white_balance,
                temperature=gui_state.load_raw.temperature,
                tint=gui_state.load_raw.tint,
                lens_correction=gui_state.load_raw.lens_correction,
                output_colorspace=gui_state.input_image.input_color_space,
                output_cctf_encoding=gui_state.input_image.apply_cctf_decoding,
                lens_info_out=lens_info,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Load raw', f'Failed to load RAW image.\n\n{exc}')
            set_status(self._viewer, 'Load raw failed')
            return

        self._current_input_path = path
        self._set_or_add_input_stack(image)

        lens_summary = lens_info.get('summary')
        if lens_summary:
            set_status(
                self._viewer,
                f"Loaded raw and applied lens correction: {lens_summary}",
            )
        elif gui_state.load_raw.lens_correction:
            set_status(self._viewer, "Loaded raw, lens correction not applied")
        else:
            set_status(self._viewer, "Loaded raw")
        self._request_auto_preview_if_enabled()

    # ------------------------------------------------------------------ roll
    def roll_add_files(self, paths: list) -> None:
        was_empty = self._roll.current is None
        added = self._roll.add_files(list(paths))
        skipped = len(paths) - len(added)
        self._roll_refresh_view()
        self._roll_request_thumbnails(added)
        if was_empty and self._roll.current is not None:
            # First files of the roll: load frame 1. It has no stored look yet,
            # so it inherits whatever the controls currently show.
            self._roll_load_current(restore_state=False)
        message = f'Added {len(added)} file(s) to the roll'
        if skipped:
            message += f' ({skipped} skipped: duplicate or unsupported)'
        set_status(self._viewer, message)

    def roll_activate(self, index: int) -> None:
        if index == self._roll.current_index:
            return
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            set_status(self._viewer, 'Busy rendering; wait before switching frames')
            self._roll_refresh_view()
            return
        # Snapshot the outgoing frame's look, switch, restore the incoming one.
        self._roll.store_state_for_current(collect_gui_state(widgets=self._widgets))
        if self._roll.select(index) is None:
            return
        self._roll_load_current(restore_state=True)
        self._roll_refresh_view()

    def roll_remove(self, index: int) -> None:
        previous = self._roll.current
        removed = self._roll.remove(index)
        if removed is None:
            return
        self._roll_refresh_view()
        if self._roll.current is None:
            set_status(self._viewer, f'Removed {removed.name}; roll is empty (loaded image kept)')
            return
        if previous is not None and removed.path == previous.path:
            self._roll_load_current(restore_state=True)
        set_status(self._viewer, f'Removed {removed.name} from the roll')

    def roll_remove_current(self) -> None:
        """Remove the CURRENT frame from the roll (Ctrl+Del). Same semantics
        as removing it from the film strip: the next frame (or the previous
        one at the end of the roll) becomes current; per-frame look dropped."""
        if not self._roll.files:
            set_status(self._viewer, 'The roll is empty')
            return
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            set_status(self._viewer, 'Busy rendering; wait before removing frames')
            return
        self.roll_remove(self._roll.current_index)

    def roll_clear(self) -> None:
        """Empty the whole roll (Clear button). The loaded image stays on the
        canvas; per-frame looks are dropped with their files."""
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            set_status(self._viewer, 'Busy rendering; wait before clearing the roll')
            return
        count = self._roll.clear()
        self._roll_refresh_view()
        if count:
            set_status(self._viewer, f'Cleared the roll ({count} file(s) removed; loaded image kept)')
        else:
            set_status(self._viewer, 'The roll is already empty')

    def roll_apply_look_to_all(self) -> None:
        from spektrafilm_gui.section_picker import pick_settings_sections
        from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE

        if not self._roll.files:
            set_status(self._viewer, 'The roll is empty — nothing to apply to')
            return
        sections = pick_settings_sections(
            dialog_parent(self._viewer),
            title='Apply to all frames',
            intro='Copy the ticked settings from THIS frame onto every frame '
                  'in the roll. Unticked settings keep each frame\'s own edit.')
        if sections is None:
            return  # cancelled
        if not sections:
            set_status(self._viewer, 'Nothing selected — no frames changed')
            return
        state = collect_gui_state(widgets=self._widgets)
        count = self._roll.apply_sections_to_all(
            state, sections, default_state=PROJECT_DEFAULT_GUI_STATE)
        if count:
            set_status(self._viewer,
                       f'Applied the selected settings to all {count} frame(s)')

    # ------------------------------------------------------------------
    # Copy/paste look (Ctrl+Shift+C / Ctrl+Shift+V): the in-app clipboard
    # between 'this one frame' and 'apply to all'.
    # ------------------------------------------------------------------
    _look_clipboard = None
    _look_clipboard_sections = None

    def copy_current_look(self) -> None:
        """Ctrl+Shift+C: pick WHICH settings to copy (NAT 2026-07-25), then
        stash them. Paste replays exactly that selection — so you can carry
        'just the grain' without disturbing the rest of the target frame."""
        from spektrafilm_gui.section_picker import pick_settings_sections

        sections = pick_settings_sections(
            dialog_parent(self._viewer),
            title='Copy settings',
            intro='Tick the settings to copy. Ctrl+Shift+V pastes exactly '
                  'this selection onto another frame.')
        if sections is None:
            return  # cancelled
        if not sections:
            set_status(self._viewer, 'Nothing selected — nothing copied')
            return
        self._look_clipboard = collect_gui_state(widgets=self._widgets)
        self._look_clipboard_sections = tuple(sections)
        set_status(self._viewer,
                   f'Copied {len(sections)} setting group(s) — Ctrl+Shift+V pastes them')

    def paste_look(self) -> None:
        if self._look_clipboard is None:
            set_status(self._viewer, 'No settings copied yet (Ctrl+Shift+C copies them)')
            return
        from spektrafilm_gui.state import clone_gui_state
        from spektrafilm_gui.state_bridge import (
            GUI_STATE_SECTION_NAMES, apply_gui_state_sections)

        sections = self._look_clipboard_sections or GUI_STATE_SECTION_NAMES
        # Same semantics as restoring a roll frame's stored look, but limited
        # to the copied sections; suppress the RGB loader's auto-reprocess (the
        # paste is not a file change), refresh.
        apply_gui_state_sections(
            clone_gui_state(self._look_clipboard),
            widgets=self._widgets, section_names=tuple(sections))
        self._widgets.filepicker.cancel_scheduled_reprocess()
        self._sync_canvas_background()
        set_status(self._viewer, 'Settings pasted')
        self.request_view_refresh()

    # ------------------------------------------------------------------
    # Logs + roll autosave / crash recovery
    # ------------------------------------------------------------------
    def open_logs(self) -> None:
        from spektrafilm_gui.app_logging import open_log_directory
        if not open_log_directory():
            set_status(self._viewer, 'Could not open the log folder')

    _autosave_timer = None
    AUTOSAVE_INTERVAL_MS = 3 * 60 * 1000

    @staticmethod
    def roll_recovery_path() -> str:
        base = os.environ.get('APPDATA') or os.path.join(os.path.expanduser('~'), '.spektrafilm')
        directory = os.path.join(base, 'spektrafilm')
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, 'recovery.sfroll')

    def start_roll_autosave(self) -> None:
        """Every few minutes, snapshot the roll (locations + looks, NO
        thumbnails — fast) into a recovery file. The file is removed on a
        clean exit, so its presence at startup means a crash."""
        if self._autosave_timer is not None:
            return
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(self.AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self.autosave_roll_now)
        self._autosave_timer.start()

    def autosave_roll_now(self) -> None:
        try:
            from spektrafilm_gui.roll_file import save_roll
            if not self._roll.files:
                return
            self._roll.store_state_for_current(collect_gui_state(widgets=self._widgets))
            save_roll(self._roll, self.roll_recovery_path(), name='crash recovery')
        except Exception:
            import logging
            logging.getLogger('spektrafilm.app').exception('roll autosave failed')

    def clear_roll_recovery(self) -> None:
        """Clean exit: the recovery file must not outlive the session."""
        try:
            path = self.roll_recovery_path()
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def offer_roll_recovery(self) -> None:
        """Startup: a leftover recovery file means the last session crashed —
        offer to restore the roll from it."""
        path = self.roll_recovery_path()
        if not os.path.exists(path):
            return
        answer = QMessageBox.question(
            dialog_parent(self._viewer), 'Restore roll',
            'The last session did not close cleanly.\n\n'
            'Restore the roll (files and per-frame edits) from the crash recovery?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.open_roll_path(path)
        else:
            self.clear_roll_recovery()

    # ------------------------------------------------------------------
    # Roll session file (Phase 8): save/open the roll as a .sfroll file.
    # ------------------------------------------------------------------
    def roll_save_to_file(self) -> None:
        """Save the roll (photo locations + per-frame edits + roll EV +
        embedded thumbnails) as a .sfroll session file. Non-destructive:
        the sources are referenced by path, never copied."""
        from spektrafilm_gui.roll_file import SFROLL_DIALOG_FILTER, SFROLL_EXTENSION, save_roll

        if not self._roll.files:
            set_status(self._viewer, 'The roll is empty; add files before saving a roll')
            return
        # The file must carry what is on screen: snapshot the current frame.
        self._roll.store_state_for_current(collect_gui_state(widgets=self._widgets))
        start_dir = load_dialog_dir('roll_file') or load_dialog_dir('roll_input')
        filepath, _ = QFileDialog.getSaveFileName(
            dialog_parent(self._viewer), 'Save roll',
            os.path.join(start_dir, 'roll' + SFROLL_EXTENSION) if start_dir else 'roll' + SFROLL_EXTENSION,
            SFROLL_DIALOG_FILTER,
        )
        if not filepath:
            return
        if not filepath.lower().endswith(SFROLL_EXTENSION):
            filepath += SFROLL_EXTENSION
        save_dialog_dir('roll_file', str(Path(filepath).parent))
        from spektrafilm_gui.roll_thumbnails import load_thumbnail_array

        roll_settings = {
            'canvas_background': str(
                QSettings('spektrafilm', 'spektrafilm').value('canvas_background', '') or ''),
        }
        try:
            save_roll(self._roll, filepath,
                      roll_settings=roll_settings, thumbnail_fn=load_thumbnail_array)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save roll',
                                 f'Failed to save the roll file.\n\n{exc}')
            return
        size_kb = os.path.getsize(filepath) / 1024.0
        set_status(self._viewer,
                   f'Saved roll ({len(self._roll.files)} frames, {size_kb:.0f} KB) to {filepath}')

    def roll_open_from_file(self) -> None:
        """Open a .sfroll file chosen in a dialog (replaces the current roll)."""
        from spektrafilm_gui.roll_file import SFROLL_DIALOG_FILTER

        filepath, _ = QFileDialog.getOpenFileName(
            dialog_parent(self._viewer), 'Open roll',
            load_dialog_dir('roll_file'), SFROLL_DIALOG_FILTER,
        )
        if filepath:
            save_dialog_dir('roll_file', str(Path(filepath).parent))
            self.open_roll_path(filepath)

    def open_roll_path(self, filepath: str) -> None:
        """Load a .sfroll roll session (also the double-click / argv entry).
        Sources are resolved relative to the roll file FIRST, then by their
        recorded absolute path; still-missing files get one relink-folder
        prompt and are otherwise skipped (reported, never a crash)."""
        import json as _json

        from spektrafilm_gui.roll_file import load_roll_dict, restore_session

        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            set_status(self._viewer, 'Busy rendering; wait before opening a roll')
            return
        try:
            data = load_roll_dict(filepath)
        except (OSError, ValueError, _json.JSONDecodeError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Open roll',
                                 f'Failed to open the roll file.\n\n{exc}')
            return
        roll_dir = os.path.dirname(os.path.abspath(filepath))
        session, missing = restore_session(data, roll_dir)
        if missing:
            names = ', '.join(str(m.get('name', '?')) for m in missing[:8])
            answer = QMessageBox.question(
                dialog_parent(self._viewer), 'Missing files',
                f'{len(missing)} source file(s) of this roll were not found '
                f'({names}{", ..." if len(missing) > 8 else ""}).\n\n'
                'Locate a folder to search for them by filename?',
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                directory = QFileDialog.getExistingDirectory(
                    dialog_parent(self._viewer), 'Locate missing roll files',
                    load_dialog_dir('roll_input'))
                if directory:
                    session, missing = restore_session(
                        data, roll_dir, relink_dirs=[directory])
        if not session.files:
            QMessageBox.warning(dialog_parent(self._viewer), 'Open roll',
                                'None of the roll\'s source files could be found.')
            return
        self._roll = session
        # Film strip: seed instantly from the roll file's EMBEDDED thumbnails
        # (small unedited input previews), then fetch fresh ones for any frame
        # the file did not carry — opening a roll bypasses roll_add_files, the
        # only other place thumbnails were requested.
        self._seed_roll_thumbnails(data)
        self._roll_request_thumbnails(session.files)
        self._roll_refresh_view()
        self._roll_load_current(restore_state=True)
        background = (data.get('roll_settings') or {}).get('canvas_background')
        if background:
            self.set_canvas_background_color(str(background))
        message = f'Opened roll "{data.get("name", "")}" ({len(session.files)} frames)'
        if missing:
            message += f'; {len(missing)} missing file(s) skipped'
        set_status(self._viewer, message)

    def roll_analyze(self) -> None:
        """B3 (NegPy Batch Analysis): measure every frame's auto-exposure EV off
        the UI thread and compute the roll average with outliers discarded.
        Read-only -- nothing is applied until roll_apply_exposure."""
        if self._active_export_worker is not None:
            if self._roll_analysis_running:
                self._active_export_worker.cancel()
                set_status(self._viewer, 'Roll analysis cancel requested...', timeout_ms=0)
            else:
                set_status(self._viewer, 'An export is running; wait before analyzing.')
            return
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'A render is already running; wait before analyzing.')
            return
        if not self._roll.files:
            QMessageBox.warning(dialog_parent(self._viewer), 'Analyze roll', 'The roll is empty. Add files first.')
            return

        from copy import deepcopy

        current_state = collect_gui_state(widgets=self._widgets)
        self._roll.store_state_for_current(current_state)
        jobs = []
        for roll_file in self._roll.files:
            frame_state = self._roll.state_for(roll_file) or deepcopy(current_state)
            jobs.append(runtime.BatchFrameJob(
                name=roll_file.name, path=roll_file.path, kind=roll_file.kind, state=frame_state,
                saving_color_space='sRGB', saving_cctf_encoding=True, ext='', bit_depth=8,
            ))

        worker = runtime.RollAnalysisWorker(jobs, measure_fn=self._roll_measure_frame)
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_roll_analysis_finished)
        worker.signals.failed.connect(self._on_roll_analysis_failed)
        self._active_export_worker = worker
        self._roll_analysis_running = True
        self._set_export_controls_enabled(False)
        self._widgets.roll.set_analyzing(True)
        set_status(self._viewer, f'Analyzing roll: {len(jobs)} frame(s)...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _roll_measure_frame(self, job) -> float:
        """One frame's auto-exposure EV, measured exactly like the pipeline's
        auto_exposure stage (same measure function, same method / colour-space
        semantics from the frame's own state) on a decimated preview -- the EV
        measurement is downscale-robust (see test_autoexposure_gpu_parity)."""
        from spektrafilm.utils.autoexposure import measure_autoexposure_ev

        state = job.state
        if job.kind == 'raw':
            image = load_and_process_raw_file(
                job.path,
                white_balance=state.load_raw.white_balance,
                temperature=state.load_raw.temperature,
                tint=state.load_raw.tint,
                lens_correction=state.load_raw.lens_correction,
                output_colorspace=state.input_image.input_color_space,
                output_cctf_encoding=state.input_image.apply_cctf_decoding,
            )
        else:
            image = load_image_oiio(job.path)[..., :3]
            image = apply_rgb_white_balance(
                image,
                white_balance=state.load_rgb.white_balance,
                temperature=state.load_rgb.temperature,
                tint=state.load_rgb.tint,
                color_space=state.input_image.input_color_space,
                cctf_encoded=state.input_image.apply_cctf_decoding,
            )
        image = np.asarray(image)
        step = max(1, int(np.ceil(max(image.shape[0], image.shape[1]) / 640)))
        small = np.ascontiguousarray(image[::step, ::step])
        return float(measure_autoexposure_ev(
            small,
            state.input_image.input_color_space,
            state.input_image.apply_cctf_decoding,
            method=state.simulation.auto_exposure_method,
        ))

    def _on_roll_analysis_finished(self, payload) -> None:
        from spektrafilm_gui.roll_session import robust_roll_ev

        self._active_export_worker = None
        self._roll_analysis_running = False
        self._set_export_controls_enabled(True)
        self._widgets.roll.set_analyzing(False)
        evs = dict(payload.get('evs', {}))
        failures = list(payload.get('failures', []))
        roll_ev, used, rejected = robust_roll_ev(list(evs.values()))
        self._roll_analysis_ev = roll_ev if used else None
        pieces = [f'roll EV {roll_ev:+.2f} from {used} frame(s)']
        if rejected:
            pieces.append(f'{rejected} outlier(s) discarded')
        if failures:
            pieces.append(f'{len(failures)} failed ({failures[0]})')
        text = 'Roll analysis: ' + ', '.join(pieces)
        self._widgets.roll.set_analysis_result(text, apply_enabled=used > 0)
        set_status(self._viewer, text)
        self._replay_pending_auto_preview()

    def _on_roll_analysis_failed(self, message: str) -> None:
        self._active_export_worker = None
        self._roll_analysis_running = False
        self._set_export_controls_enabled(True)
        self._widgets.roll.set_analyzing(False)
        set_status(self._viewer, message)
        self._replay_pending_auto_preview()

    def roll_apply_exposure(self) -> None:
        if self._roll_analysis_ev is None:
            set_status(self._viewer, 'Run "Analyze roll" first.')
            return
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            set_status(self._viewer, 'Busy rendering; wait before applying the roll exposure.')
            return
        current_state = collect_gui_state(widgets=self._widgets)
        self._roll.store_state_for_current(current_state)
        count = self._roll.apply_roll_exposure(self._roll_analysis_ev, current_state)
        # The current frame's stored look changed: reflect it in the widgets.
        current = self._roll.current
        if current is not None:
            stored = self._roll.state_for(current)
            if stored is not None:
                apply_gui_state(stored, widgets=self._widgets)
                self._sync_canvas_background()
        set_status(
            self._viewer,
            f'Roll exposure {self._roll_analysis_ev:+.2f} EV applied to {count} frame(s) '
            f'(auto exposure off, one shared exposure for the roll)',
        )
        self._request_auto_preview_if_enabled()

    def _roll_load_current(self, *, restore_state: bool) -> None:
        roll_file = self._roll.current
        if roll_file is None:
            return
        # Sorting mode: skip the full decode + render, just show the instant
        # thumbnail (NAT 2026-07-25). The real load happens when the mode is
        # left. Track that we navigated so the exit knows to reload.
        if self._input_mode:
            self._current_input_path = roll_file.path
            self._input_mode_navigated = True
            self._widgets.filepicker.cancel_scheduled_reprocess()
            image = self._show_instant_input()
            if image is not None:
                self._update_banner_and_metadata(image)
            return
        if restore_state:
            stored = self._roll.state_for(roll_file)
            if stored is not None:
                apply_gui_state(stored, widgets=self._widgets)
                self._sync_canvas_background()
                self._warn_if_missing_custom_stock()
        # Applying state may have armed the RGB picker's debounced reprocess
        # with the OUTGOING file's path; the explicit load below supersedes it.
        self._widgets.filepicker.cancel_scheduled_reprocess()
        if roll_file.kind == 'raw':
            self._widgets.load_raw.set_path(roll_file.path)
            self._widgets.filepicker.set_path('')
            self.load_raw_image(roll_file.path)
        else:
            self._widgets.filepicker.set_path(roll_file.path)
            self.load_input_image(roll_file.path)

    # ------------------------------------------------------------------
    # Star ratings (per frame) + the nav-bar minimum-rating filter
    # ------------------------------------------------------------------
    _rating_filter = 0
    _single_rating = 0    # rating of the loaded image when there is no roll

    def _current_rating(self) -> int:
        current = self._roll.current
        if current is not None:
            return self._roll.rating_for(current)
        return int(self._single_rating)

    def set_current_rating(self, rating: int) -> None:
        """Metadata panel stars: rate the CURRENT frame (0-5)."""
        rating = min(max(int(rating), 0), 5)
        current = self._roll.current
        if current is not None:
            self._roll.set_rating(current, rating)
            if self._rating_filter:
                self._roll_refresh_view()
        else:
            self._single_rating = rating
        self._sync_rating_panel()          # reflect keyboard shortcuts in the stars
        set_status(self._viewer,
                   f'Rating: {"★" * rating}{"☆" * (5 - rating)}' if rating else 'Rating cleared')

    def set_rating_filter(self, minimum: int) -> None:
        """Nav-bar star filter: frames below ``minimum`` gray out, prev/next
        skips them, Export roll writes only the rest (the export panel shows
        a live 'exports N of M' note). 0 = everything."""
        self._rating_filter = min(max(int(minimum), 0), 5)
        self._roll_refresh_view()
        if self._rating_filter:
            kept = sum(1 for f in self._roll.files if self._passes_rating_filter(f))
            set_status(self._viewer,
                       f'Star filter: >= {self._rating_filter} '
                       f'({kept}/{len(self._roll.files)} frames)')
        else:
            set_status(self._viewer, 'Star filter off')

    def _passes_rating_filter(self, roll_file) -> bool:
        return self._roll.rating_for(roll_file) >= self._rating_filter

    def _sync_rating_filter_ui(self) -> None:
        """Refresh the export panel's 'exports N of M' line from the nav-bar
        star filter (the filter's single control)."""
        minimum = self._rating_filter
        export_section = getattr(self._widgets, 'export', None)
        info = getattr(export_section, 'rating_filter_info', None)
        if info is not None:
            if not minimum:
                info.setText('Exports every frame in the roll.')
            elif not self._roll.files:
                info.setText(f'Exports frames rated {minimum}+ stars.')
            else:
                kept = sum(1 for f in self._roll.files if self._passes_rating_filter(f))
                info.setText(f'Exports {kept} of {len(self._roll.files)} frames '
                             f'(rated {minimum}+ stars).')

    def _sync_rating_panel(self) -> None:
        panel = getattr(self._widgets, 'metadata', None)
        set_rating = getattr(panel, 'set_rating', None)
        if callable(set_rating):
            set_rating(self._current_rating())

    def _roll_refresh_view(self) -> None:
        roll_section = getattr(self._widgets, 'roll', None)
        if roll_section is not None:
            roll_section.set_files(
                [f.name for f in self._roll.files],
                self._roll.current_index,
                self._roll.position_label(),
            )
        strip = getattr(self._widgets, 'film_strip', None)
        if strip is not None:
            strip.set_entries(
                [f.name for f in self._roll.files],
                [self._roll_thumbs.get(f.path) for f in self._roll.files],
                self._roll.current_index,
                dimmed=[not self._passes_rating_filter(f) for f in self._roll.files]
                if self._rating_filter else None,
            )
        self._sync_rating_panel()
        self._sync_rating_filter_ui()

    def _seed_roll_thumbnails(self, roll_document: dict) -> None:
        """Populate the film strip from the .sfroll's embedded thumbnails
        (matched by file name), so a reopened roll shows its previews
        immediately, before (or instead of) the background loaders."""
        try:
            from qtpy.QtGui import QImage, QPixmap

            from spektrafilm_gui.roll_file import decode_thumbnail
        except ImportError:
            return
        embedded = {}
        for frame in roll_document.get('frames', []):
            if isinstance(frame, dict) and frame.get('thumbnail') and frame.get('name'):
                embedded[str(frame['name'])] = frame['thumbnail']
        for roll_file in self._roll.files:
            if roll_file.path in self._roll_thumbs:
                continue
            text = embedded.get(roll_file.name)
            if not text:
                continue
            array = decode_thumbnail(text)
            if array is None:
                continue
            arr = np.ascontiguousarray(array)
            h, w, _ = arr.shape
            image = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self._roll_thumbs[roll_file.path] = QPixmap.fromImage(image)

    def _roll_request_thumbnails(self, roll_files: list) -> None:
        try:
            from spektrafilm_gui.roll_thumbnails import ThumbnailTask
        except ImportError:
            return
        for roll_file in roll_files:
            if roll_file.path in self._roll_thumbs or roll_file.path in self._thumb_tasks:
                continue
            task = ThumbnailTask(roll_file.path, roll_file.kind)
            task.signals.ready.connect(self._on_roll_thumbnail)
            self._thumb_tasks[roll_file.path] = task
            self._thread_pool.start(task)

    def _on_roll_thumbnail(self, path: str, array) -> None:
        self._thumb_tasks.pop(path, None)
        if array is None:
            return
        from qtpy.QtGui import QImage, QPixmap

        arr = np.ascontiguousarray(array)
        h, w, _ = arr.shape
        image = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self._roll_thumbs[path] = pixmap
        strip = getattr(self._widgets, 'film_strip', None)
        if strip is None:
            return
        for index, roll_file in enumerate(self._roll.files):
            if roll_file.path == path:
                strip.set_thumbnail(index, pixmap)
                break

    def refresh_preview_cache(self, *_args) -> None:
        input_image = self._current_input_image
        if input_image is None:
            return
        self._update_preview_cache(
            input_image,
            home_input_stack=False,
            hide_output=False,
        )

    # ------------------------------------------------------------------
    # Interactive crop tool (NegPy's Manual/Move rubber-band, on napari)
    # ------------------------------------------------------------------
    _crop_tool_mode = 'off'                 # 'off' | 'draw' | 'move'
    _CROP_TOOL_LAYER_NAME = 'crop tool'

    def set_crop_tool(self, mode: str) -> None:
        """Arm or disarm the on-canvas crop tool. While armed the camera pan
        is suspended so the drag drives the crop instead. Phase 9 modes:
        'frame' (persistent LR crop frame) and 'line' (straighten); the
        legacy 'draw'/'move' rubber band remains for the napari-era path."""
        if mode in ('frame', 'line') and self._gpu_canvas is not None:
            self._crop_tool_mode = mode
            if mode == 'frame':
                self._arm_crop_frame()
            else:
                self._arm_straighten_line()
            return
        if mode == 'off' and self._frame_rect_uv is not None:
            # LR semantics: toggling the crop tool off APPLIES the frame
            self._frame_commit()
            return
        self._crop_tool_mode = mode if mode in ('draw', 'move') else 'off'
        # Phase 6B: when the wgpu canvas is active it owns the interaction
        # (overlay rubber band); napari's drag callbacks are not used.
        if self._gpu_canvas is not None:
            self._set_canvas_crop_tool(self._crop_tool_mode)
            return
        callbacks = self._viewer.mouse_drag_callbacks
        if self._crop_tool_mode == 'off':
            if self._crop_tool_drag in callbacks:
                callbacks.remove(self._crop_tool_drag)
            self._remove_crop_tool_overlay()
            self._set_camera_interactive(True)
            return
        if self._crop_tool_drag not in callbacks:
            callbacks.append(self._crop_tool_drag)
        self._set_camera_interactive(False)
        hint = ('Drag on the image to draw the crop rectangle'
                if self._crop_tool_mode == 'draw'
                else 'Drag to move the crop rectangle (Shift = fine)')
        set_status(self._viewer, hint, timeout_ms=0)

    # -- Phase 6B: crop tool on the wgpu canvas -----------------------------
    def _set_canvas_crop_tool(self, mode: str) -> None:
        canvas = self._gpu_canvas
        if mode == 'off':
            canvas.arm_crop_tool('off')
            return
        finishing_widgets = self._widgets.scan_finishing
        if mode == 'draw' and bool(finishing_widgets.crop.value):
            finishing_widgets.crop.value = False
            self._finish_crop_tool('Scan crop cleared - draw again on the full scan once it refreshes')
            return
        self._canvas_move_center0 = tuple(float(v) for v in finishing_widgets.crop_center.value)
        self._canvas_move_size0 = tuple(float(v) for v in finishing_widgets.crop_size.value)
        self._canvas_move_crop_active = bool(finishing_widgets.crop.value)
        canvas.arm_crop_tool(
            mode,
            on_draw=self._canvas_crop_drawn,
            on_move=self._canvas_crop_moved,
            on_finish=lambda: self._finish_crop_tool('Scan crop moved'),
            aspect_ratio=self._widgets.scan_frame.aspect_ratio,
        )
        hint = ('Drag on the image to draw the crop rectangle'
                if mode == 'draw'
                else 'Drag to move the crop rectangle (Shift = fine)')
        set_status(self._viewer, hint, timeout_ms=0)

    def _canvas_crop_drawn(self, u0: float, v0: float, u1: float, v1: float) -> None:
        """Rubber-band finished on the wgpu canvas: (u, v) are normalized
        image coordinates â€” the same frame the scan crop is defined in."""
        width, height = self._gpu_canvas.renderer.image_size
        long_edge = max(width, height, 1)
        rect_u, rect_v = abs(u1 - u0), abs(v1 - v0)
        size_x = rect_u * width / long_edge      # fraction of the long side
        size_y = rect_v * height / long_edge
        if size_x < 0.005 or size_y < 0.005:
            self._finish_crop_tool('Crop rectangle too small; kept previous crop')
            return
        finishing_widgets = self._widgets.scan_finishing
        finishing_widgets.crop_center.value = (
            min(max((u0 + u1) / 2.0, 0.0), 1.0),
            min(max((v0 + v1) / 2.0, 0.0), 1.0),
        )
        finishing_widgets.crop_size.value = (
            min(max(size_x, 0.005), 1.0),
            min(max(size_y, 0.005), 1.0),
        )
        finishing_widgets.crop.value = True
        self._finish_crop_tool('Scan crop set from the drawn rectangle')

    def _canvas_crop_moved(self, du: float, dv: float, fine: bool) -> None:
        """Streaming move deltas (fractions of the displayed image rect) â€”
        same sensitivity/span composition as the napari path."""
        sensitivity = 0.2 if fine else 0.5
        span_x = self._canvas_move_size0[0] if self._canvas_move_crop_active else 1.0
        span_y = self._canvas_move_size0[1] if self._canvas_move_crop_active else 1.0
        self._widgets.scan_finishing.crop_center.value = (
            min(max(self._canvas_move_center0[0] + du * sensitivity * span_x, 0.0), 1.0),
            min(max(self._canvas_move_center0[1] + dv * sensitivity * span_y, 0.0), 1.0),
        )

    # -- Phase 9: persistent Lightroom-style crop frame ----------------------
    _frame_rect_uv = None
    _frame_initial = None            # (crop_active, center, size) for cancel

    def _crop_image_wh(self) -> tuple[float, float]:
        width, height = self._gpu_canvas.renderer.image_size
        return (max(float(width), 1.0), max(float(height), 1.0))

    def _frame_ratio_uv(self):
        """The locked aspect (width/height, converted to uv units) or None.
        Any preset other than Free acts as the lock; Free is unlocked."""
        from spektrafilm_gui.crop_frame import parse_ratio

        image_wh = self._crop_image_wh()
        text = ''
        section = getattr(self._widgets, 'scan_frame', None)
        combo = getattr(section, 'ratio_combo', None)
        if combo is not None:
            text = str(combo.currentText())
        ratio = parse_ratio(text, image_wh)
        if ratio is None:
            return None
        width, height = image_wh
        return ratio * height / width

    def _arm_crop_frame(self) -> None:
        from spektrafilm_gui.crop_frame import crop_rect_uv

        finishing_widgets = self._widgets.scan_finishing
        center = tuple(float(v) for v in finishing_widgets.crop_center.value)
        size = tuple(float(v) for v in finishing_widgets.crop_size.value)
        crop_active = bool(finishing_widgets.crop.value)
        self._frame_initial = (crop_active, center, size)
        if crop_active:
            # LR shows the WHOLE image while cropping: lift the crop and
            # REFRESH THE VIEW EXPLICITLY (with auto preview off the editor
            # signal renders nothing and the frame would sit over the stale
            # cropped view — the 'crop tool not working' report).
            finishing_widgets.crop.value = False
            self._frame_rect_uv = crop_rect_uv(center, size, self._crop_image_wh())
            self.request_view_refresh()
        else:
            self._frame_rect_uv = (0.0, 0.0, 1.0, 1.0)
        self._gpu_canvas.arm_crop_tool(
            'frame',
            on_frame_drag=self._frame_dragged,
            on_frame_release=lambda: None,
            on_frame_commit=self._frame_commit,
            on_frame_cancel=self._frame_cancel,
        )
        self._gpu_canvas.set_crop_frame(self._frame_rect_uv)
        set_status(self._viewer,
                   'Crop: drag handles to resize, drag inside to move the frame '
                   '(Shift = fine) - Enter/double-click applies, Esc cancels',
                   timeout_ms=0)

    def _frame_dragged(self, hit: str, du: float, dv: float, fine: bool) -> None:
        from spektrafilm_gui.crop_frame import (
            clamp_rect_to_rotated_image, move_image_under_frame, resize_rect)

        if self._frame_rect_uv is None:
            return
        factor = 0.2 if fine else 1.0
        du, dv = du * factor, dv * factor
        if hit == 'inside':
            rect = move_image_under_frame(self._frame_rect_uv, du, dv)
        else:
            rect = resize_rect(self._frame_rect_uv, hit, du, dv,
                               ratio_uv=self._frame_ratio_uv())
        skew = float(self._widgets.scan_finishing.skew_deg.value)
        rect = clamp_rect_to_rotated_image(rect, skew, self._crop_image_wh())
        self._frame_rect_uv = rect
        self._gpu_canvas.set_crop_frame(rect)

    def _frame_commit(self) -> None:
        from spektrafilm_gui.crop_frame import MIN_CROP_SIZE, rect_to_center_size

        if self._frame_rect_uv is None:
            return
        u0, v0, u1, v1 = self._frame_rect_uv
        # full frame is judged on the uv spans (the persisted size is
        # long-edge-normalized, so a full-frame short axis is < 1.0)
        full_frame = (u1 - u0) >= 0.999 and (v1 - v0) >= 0.999
        center, size = rect_to_center_size(self._frame_rect_uv, self._crop_image_wh())
        finishing_widgets = self._widgets.scan_finishing
        finishing_widgets.crop_center.value = (
            min(max(center[0], 0.0), 1.0), min(max(center[1], 0.0), 1.0))
        finishing_widgets.crop_size.value = (
            min(max(size[0], MIN_CROP_SIZE), 1.0), min(max(size[1], MIN_CROP_SIZE), 1.0))
        finishing_widgets.crop.value = not full_frame
        self._frame_rect_uv = None
        self._frame_initial = None
        self._finish_crop_tool(
            'Scan crop applied' if not full_frame else 'Full frame - crop off')

    def _frame_cancel(self) -> None:
        finishing_widgets = self._widgets.scan_finishing
        if self._frame_initial is not None:
            crop_active, center, size = self._frame_initial
            finishing_widgets.crop_center.value = center
            finishing_widgets.crop_size.value = size
            finishing_widgets.crop.value = crop_active
        self._frame_rect_uv = None
        self._frame_initial = None
        self._finish_crop_tool('Crop cancelled - previous crop restored')

    def refit_crop_to_ratio(self) -> None:
        """Ratio preset selected while the frame is armed: re-fit the frame
        to the new aspect (centered, max-fit)."""
        from spektrafilm_gui.crop_frame import parse_ratio, refit_to_ratio

        if self._frame_rect_uv is None or self._gpu_canvas is None:
            return
        image_wh = self._crop_image_wh()
        section = getattr(self._widgets, 'scan_frame', None)
        combo = getattr(section, 'ratio_combo', None)
        text = str(combo.currentText()) if combo is not None else ''
        ratio = parse_ratio(text, image_wh)
        self._frame_rect_uv = refit_to_ratio(self._frame_rect_uv, ratio, image_wh)
        self._gpu_canvas.set_crop_frame(self._frame_rect_uv)

    def swap_crop_orientation(self) -> None:
        """The Frame section's Swap: landscape <-> portrait, LR-style — the
        crop's pixel dimensions swap (shrinking uniformly only when the
        swapped shape no longer fits), and a locked ratio preset flips to its
        counterpart (16:9 -> 9:16) so further handle drags stay consistent."""
        from spektrafilm_gui.crop_frame import (
            crop_rect_uv, rect_to_center_size, swap_ratio_label,
            swap_rect_orientation)

        section = getattr(self._widgets, 'scan_frame', None)
        combo = getattr(section, 'ratio_combo', None)
        if combo is not None:
            swapped_label = swap_ratio_label(combo.currentText())
            index = combo.findText(swapped_label)
            if index >= 0 and index != combo.currentIndex():
                was_blocked = combo.blockSignals(True)   # keep our size-preserving swap
                combo.setCurrentIndex(index)
                combo.blockSignals(was_blocked)
        image_wh = self._crop_image_wh() if self._gpu_canvas is not None else (1.0, 1.0)
        if self._frame_rect_uv is not None and self._gpu_canvas is not None:
            self._frame_rect_uv = swap_rect_orientation(self._frame_rect_uv, image_wh)
            self._gpu_canvas.set_crop_frame(self._frame_rect_uv)
            set_status(self._viewer, 'Crop frame orientation swapped')
            return
        finishing_widgets = self._widgets.scan_finishing
        if not bool(finishing_widgets.crop.value):
            set_status(self._viewer, 'No crop to swap - arm the Crop tool or set a crop first')
            return
        center = tuple(float(v) for v in finishing_widgets.crop_center.value)
        size = tuple(float(v) for v in finishing_widgets.crop_size.value)
        rect = swap_rect_orientation(crop_rect_uv(center, size, image_wh), image_wh)
        new_center, new_size = rect_to_center_size(rect, image_wh)
        finishing_widgets.crop_center.value = new_center
        finishing_widgets.crop_size.value = new_size
        set_status(self._viewer, 'Crop orientation swapped')
        self.request_view_refresh()

    def reset_crop(self) -> None:
        """The Frame section's Reset: back to the ORIGINAL photo format —
        full frame, ratio Free. Works on the armed frame or the persisted
        crop directly (view refreshed either way)."""
        section = getattr(self._widgets, 'scan_frame', None)
        combo = getattr(section, 'ratio_combo', None)
        if combo is not None:
            was_blocked = combo.blockSignals(True)
            combo.setCurrentIndex(0)            # 'Free'
            combo.blockSignals(was_blocked)
        if self._frame_rect_uv is not None and self._gpu_canvas is not None:
            self._frame_rect_uv = (0.0, 0.0, 1.0, 1.0)
            self._gpu_canvas.set_crop_frame(self._frame_rect_uv)
            set_status(self._viewer, 'Crop frame reset to the full photo')
            return
        finishing_widgets = self._widgets.scan_finishing
        finishing_widgets.crop.value = False
        finishing_widgets.crop_center.value = (0.5, 0.5)
        finishing_widgets.crop_size.value = (1.0, 1.0)
        set_status(self._viewer, 'Crop reset to the original photo format')
        self.request_view_refresh()

    # -- Phase 9: straighten line --------------------------------------------
    def _arm_straighten_line(self) -> None:
        self._gpu_canvas.arm_crop_tool('line', on_line=self._straighten_line_drawn)
        set_status(self._viewer,
                   'Straighten: drag a line along a horizon or vertical', timeout_ms=0)

    def _straighten_line_drawn(self, x0: float, y0: float, x1: float, y1: float) -> None:
        from spektrafilm_gui.crop_frame import (
            angle_from_line, clamp_rect_to_rotated_image, crop_rect_uv,
            rect_to_center_size)

        angle = angle_from_line(x0, y0, x1, y1)
        finishing_widgets = self._widgets.scan_finishing
        current = float(finishing_widgets.skew_deg.value)
        new_skew = max(min(current + angle, 45.0), -45.0)
        finishing_widgets.skew_deg.value = round(new_skew, 2)
        # auto-clamp the crop (armed frame or the persisted one) into the
        # rotated valid image region
        image_wh = self._crop_image_wh()
        if self._frame_rect_uv is not None:
            self._frame_rect_uv = clamp_rect_to_rotated_image(
                self._frame_rect_uv, new_skew, image_wh)
            self._gpu_canvas.set_crop_frame(self._frame_rect_uv)
        elif bool(finishing_widgets.crop.value):
            center = tuple(float(v) for v in finishing_widgets.crop_center.value)
            size = tuple(float(v) for v in finishing_widgets.crop_size.value)
            rect = clamp_rect_to_rotated_image(
                crop_rect_uv(center, size, image_wh), new_skew, image_wh)
            new_center, new_size = rect_to_center_size(rect, image_wh)
            finishing_widgets.crop_center.value = new_center
            finishing_widgets.crop_size.value = new_size
        self._finish_crop_tool(f'Straightened {angle:+.2f} deg (skew {new_skew:+.2f})')

    def _set_camera_interactive(self, interactive: bool) -> None:
        # napari >= 0.5 splits camera interaction into mouse_pan / mouse_zoom
        # ('interactive' is gone) â€” set whichever exists, or the drag pans the
        # canvas instead of driving the crop tool.
        camera = getattr(self._viewer, 'camera', None)
        if camera is None:
            return
        for attribute in ('interactive', 'mouse_pan', 'mouse_zoom'):
            if hasattr(camera, attribute):
                try:
                    setattr(camera, attribute, bool(interactive))
                except (AttributeError, ValueError):
                    pass

    def _crop_tool_drag(self, viewer, event):
        mode = self._crop_tool_mode
        if mode == 'off' or getattr(event, 'button', 1) != 1:
            return
        extent = self._crop_target_world_extent()
        if extent is None:
            return
        finishing_widgets = self._widgets.scan_finishing
        start = tuple(float(v) for v in event.position[-2:])   # (y, x) world
        if mode == 'move':
            editor = finishing_widgets.crop_center
            center0 = tuple(float(v) for v in editor.value)
            size0 = tuple(float(v) for v in finishing_widgets.crop_size.value)
            crop_active = bool(finishing_widgets.crop.value)
            ymin, xmin, ymax, xmax = extent
            yield
            while event.type == 'mouse_move':
                pos = tuple(float(v) for v in event.position[-2:])
                sensitivity = 0.2 if 'Shift' in tuple(getattr(event, 'modifiers', ())) else 0.5
                fx = (pos[1] - start[1]) / max(xmax - xmin, 1e-9) * sensitivity
                fy = (pos[0] - start[0]) / max(ymax - ymin, 1e-9) * sensitivity
                # With a crop active the display shows the cropped region, so
                # a full-width drag spans one crop width in pre-crop units.
                span_x = size0[0] if crop_active else 1.0
                span_y = size0[1] if crop_active else 1.0
                editor.value = (
                    min(max(center0[0] + fx * span_x, 0.0), 1.0),
                    min(max(center0[1] + fy * span_y, 0.0), 1.0),
                )
                yield
            self._finish_crop_tool('Scan crop moved')
            return
        # draw mode: the rect is defined against the FULL rendered scan; if a
        # scan crop is already active, clear it first (the preview refreshes
        # to full frame, then draw).
        if bool(finishing_widgets.crop.value):
            finishing_widgets.crop.value = False
            self._finish_crop_tool('Scan crop cleared - draw again on the full scan once it refreshes')
            return
        end = start
        yield
        while event.type == 'mouse_move':
            end = tuple(float(v) for v in event.position[-2:])
            end = self._constrain_crop_corner(start, end)
            self._update_crop_tool_overlay(start, end)
            yield
        self._remove_crop_tool_overlay()
        ymin, xmin, ymax, xmax = extent
        width_ws, height_ws = xmax - xmin, ymax - ymin
        long_ws = max(width_ws, height_ws, 1e-9)
        rect_w = abs(end[1] - start[1])
        rect_h = abs(end[0] - start[0])
        if rect_w / long_ws < 0.005 or rect_h / long_ws < 0.005:
            self._finish_crop_tool('Crop rectangle too small; kept previous crop')
            return
        cx = (start[1] + end[1]) / 2.0
        cy = (start[0] + end[0]) / 2.0
        finishing_widgets.crop_center.value = (
            min(max((cx - xmin) / max(width_ws, 1e-9), 0.0), 1.0),
            min(max((cy - ymin) / max(height_ws, 1e-9), 0.0), 1.0),
        )
        finishing_widgets.crop_size.value = (
            min(max(rect_w / long_ws, 0.005), 1.0),
            min(max(rect_h / long_ws, 0.005), 1.0),
        )
        finishing_widgets.crop.value = True
        self._finish_crop_tool('Scan crop set from the drawn rectangle')

    def _constrain_crop_corner(self, start, end):
        """Constrain the dragged corner to the selected aspect ratio, exactly
        as NegPy does during the drag (the smaller axis follows)."""
        ratio = self._widgets.scan_frame.aspect_ratio()
        if ratio is None:
            return end
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        if abs(dx) > abs(dy) * ratio:
            dx = abs(dy) * ratio * (1 if dx >= 0 else -1)
        else:
            dy = abs(dx) / ratio * (1 if dy >= 0 else -1)
        return (start[0] + dy, start[1] + dx)

    def _crop_target_world_extent(self):
        """World extent of the layer the SCAN crop tool draws on: the rendered
        output when visible (the crop is a cutout of the rendered film),
        otherwise the input preview. No geometry re-projection is needed â€”
        the displayed render already carries the scan desqueeze/skew, and the
        scan crop is defined in exactly that frame."""
        output_layer = self._output_layer()
        layer = (output_layer if output_layer is not None
                 and getattr(output_layer, 'visible', False) else None)
        if layer is None:
            layer = self._layers.preview_input_layer() or self._layers.image_layer(INPUT_LAYER_NAME)
        if layer is None:
            return None
        world = np.asarray(layer.extent.world)
        return (float(world[0][-2]), float(world[0][-1]),
                float(world[1][-2]), float(world[1][-1]))

    def _update_crop_tool_overlay(self, p0, p1) -> None:
        rect = np.array([
            [p0[0], p0[1]], [p0[0], p1[1]], [p1[0], p1[1]], [p1[0], p0[1]],
        ])
        layer = next((l for l in self._viewer.layers
                      if getattr(l, 'name', None) == self._CROP_TOOL_LAYER_NAME), None)
        previous_active = getattr(self._viewer.layers.selection, 'active', None)
        if layer is None:
            layer = self._viewer.add_shapes(
                [rect], shape_type='polygon',
                edge_color='white', face_color=[0.0, 0.0, 0.0, 0.25],
                edge_width=0.003, name=self._CROP_TOOL_LAYER_NAME,
            )
            layer.editable = False
            if previous_active is not None and previous_active is not layer:
                self._viewer.layers.selection.active = previous_active
        else:
            layer.data = [rect]

    def _remove_crop_tool_overlay(self) -> None:
        layer = next((l for l in self._viewer.layers
                      if getattr(l, 'name', None) == self._CROP_TOOL_LAYER_NAME), None)
        if layer is not None:
            self._viewer.layers.remove(layer)

    def _finish_crop_tool(self, message: str) -> None:
        self._widgets.scan_frame.release_tools()
        self.set_crop_tool('off')
        set_status(self._viewer, message)
        # Crop/straighten results are VIEW changes: render them even when
        # auto preview is off (same rule as the scan-film toggle).
        self.request_view_refresh()

    def rotate_input_image_clockwise(self) -> None:
        self._rotate_input_image(quarter_turns=-1)

    def rotate_input_image_counterclockwise(self) -> None:
        self._rotate_input_image(quarter_turns=1)

    def _rotate_input_image(self, *, quarter_turns: int) -> None:
        input_image = self._current_input_image
        if input_image is None:
            return

        rotated_image = np.rot90(np.asarray(input_image), k=int(quarter_turns))
        self._update_preview_cache(
            rotated_image,
            home_input_stack=True,
            hide_output=True,
        )
        self._request_auto_preview_if_enabled()

    def mirror_input_image_horizontal(self) -> None:
        """Mirror the source image left-right (like the NegPy flip-H toggle;
        applied to the working image, so the whole pipeline inherits it)."""
        self._mirror_input_image(axis=1)

    def mirror_input_image_vertical(self) -> None:
        """Mirror the source image top-bottom."""
        self._mirror_input_image(axis=0)

    def _mirror_input_image(self, *, axis: int) -> None:
        input_image = self._current_input_image
        if input_image is None:
            return
        data = np.asarray(input_image)
        mirrored = np.ascontiguousarray(np.flip(data, axis=axis))
        self._update_preview_cache(
            mirrored,
            home_input_stack=True,
            hide_output=True,
        )
        self._request_auto_preview_if_enabled()

    def roll_step(self, delta: int) -> None:
        """Prev/next frame from the bottom navigation bar (clamped). With the
        star filter active, frames below the minimum rating are skipped."""
        if not self._roll.files:
            set_status(self._viewer, 'The roll is empty')
            return
        step = 1 if int(delta) > 0 else -1
        index = self._roll.current_index
        for _ in range(len(self._roll.files)):
            index = index + step
            if not (0 <= index < len(self._roll.files)):
                if self._rating_filter:
                    set_status(self._viewer, 'No more frames matching the star filter')
                return
            if self._passes_rating_filter(self._roll.files[index]):
                self.roll_activate(index)
                return

    def set_canvas_background_color(self, color: str) -> None:
        """Apply one of the navigation-bar background swatches and persist the
        choice across sessions (NegPy persists its canvas_bg_index the same way)."""
        from spektrafilm_gui.napari_layout import set_canvas_background_color
        set_canvas_background_color(self._viewer, color)
        if self._gpu_canvas is not None:
            try:
                self._gpu_canvas.set_background_hex(color)
            except Exception:
                pass
        QSettings('spektrafilm', 'spektrafilm').setValue('canvas_background', color)

    def cycle_canvas_background(self) -> None:
        """Ctrl+L: step through the three nav-bar background swatches (the
        persisted choice is the cycle position; the swatch buttons follow)."""
        from spektrafilm_gui.napari_layout import CANVAS_BACKGROUND_OPTIONS
        current = str(QSettings('spektrafilm', 'spektrafilm').value('canvas_background', '') or '')
        colors = [color for _, color in CANVAS_BACKGROUND_OPTIONS]
        index = colors.index(current) if current in colors else -1
        next_index = (index + 1) % len(colors)
        label, color = CANVAS_BACKGROUND_OPTIONS[next_index]
        self.set_canvas_background_color(color)
        self._sync_background_swatches(next_index)
        set_status(self._viewer, f'Canvas background: {label}')

    def _sync_background_swatches(self, checked_index: int) -> None:
        """Reflect a programmatic background change on the nav-bar swatch
        buttons (cosmetic; never let a lookup failure break the shortcut)."""
        try:
            from qtpy import QtWidgets
            from spektrafilm_gui.napari_layout import CANVAS_BACKGROUND_OPTIONS
            host = dialog_parent(self._viewer)
            if host is None:
                return
            for index in range(len(CANVAS_BACKGROUND_OPTIONS)):
                button = host.findChild(QtWidgets.QToolButton, f'canvasSwatch{index}')
                if button is not None:
                    button.blockSignals(True)
                    button.setChecked(index == checked_index)
                    button.blockSignals(False)
        except Exception:
            pass

    def cycle_canvas_guides(self) -> None:
        """Ctrl+O: cycle the composition guide overlay on the canvas
        (off -> grid -> rule of thirds -> golden ratio -> diagonals ->
        triangle -> golden spiral -> center cross), persisted across sessions."""
        if self._gpu_canvas is None:
            set_status(self._viewer, 'Composition guides need the canvas')
            return
        try:
            mode = self._gpu_canvas.cycle_guide()
        except Exception:
            set_status(self._viewer, 'Composition guides are unavailable on this canvas')
            return
        QSettings('spektrafilm', 'spektrafilm').setValue('canvas_guides', mode)
        labels = {
            'off': 'off',
            'grid': 'grid',
            'thirds': 'rule of thirds',
            'golden': 'golden ratio',
            'diagonals': 'diagonals',
            'triangle': 'triangle',
            'golden_spiral': 'golden spiral',
            'center': 'center cross',
        }
        set_status(self._viewer, f'Composition guides: {labels.get(mode, mode)}')

    def flip_canvas_guides(self) -> None:
        """Shift+O: flip the orientation of the triangle / golden-spiral guide."""
        if self._gpu_canvas is None:
            set_status(self._viewer, 'Composition guides need the canvas')
            return
        try:
            orient = self._gpu_canvas.flip_guide()
        except Exception:
            set_status(self._viewer, 'Composition guides are unavailable on this canvas')
            return
        QSettings('spektrafilm', 'spektrafilm').setValue('canvas_guide_orient', orient)
        mode = getattr(self._gpu_canvas, 'guide_mode', 'off')
        if mode in ('triangle', 'golden_spiral'):
            set_status(self._viewer, f'Guide orientation: {int(orient) + 1}/4')
        else:
            set_status(self._viewer, 'Orientation flip applies to the triangle / golden spiral')

    def toggle_scan_film(self) -> None:
        """Ctrl+F: flip the bottom-bar 'scan film' toggle (show the scanned
        negative instead of the print), then refresh the preview."""
        sim = self._widgets.simulation
        try:
            new_value = not bool(sim.scan_film_value())
            sim.set_scan_film_value(new_value)
        except Exception:
            return
        self.request_view_refresh()
        set_status(self._viewer, f'Scan film: {"on" if new_value else "off"}')

    def request_view_refresh(self, *_args) -> None:
        """Scan-film / scan-for-print are VIEW switches, not parameter tweaks:
        re-render the preview even when auto preview is off or still dormant
        (routing them through request_auto_preview made the toggle and Ctrl+F
        look dead — NAT 2026-07-12)."""
        self._schedule_history_push()
        if self._current_preview_image is None:
            return
        if self._auto_preview_enabled():
            self.request_auto_preview()
            return
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            set_status(self._viewer, 'Busy rendering; the view switches after this render')
            return
        self._run_preview(report_status=False)

    def toggle_auto_preview(self) -> None:
        """Toggle live auto-preview on/off. Once the session's first render
        has happened, turning it on triggers one render via the checkbox's
        toggled signal; turning it off just stops auto-rendering (use F5 for
        a one-shot preview)."""
        sim = self._widgets.simulation
        try:
            new_value = not bool(sim.auto_preview_value())
            sim.set_auto_preview_value(new_value)
        except Exception:
            return
        set_status(self._viewer, f'Auto preview: {"on" if new_value else "off"}')

    def set_grain_in_preview(self, on: bool) -> None:
        """Grain-in-preview toggle (NAT 2026-07-26): off = the live preview
        skips grain for fast scrubbing; Scan/export always render grain. The
        cached partial-render tiers include grain, so toggling invalidates them
        and re-renders."""
        self._grain_in_preview = bool(on)
        self._stage_cache = None            # taps were computed with/without grain
        set_status(self._viewer,
                   f'Grain in preview: {"on" if self._grain_in_preview else "off (Scan/export still grains)"}')
        self.request_view_refresh()


    # Auto-levels safety margin: the measured floors/ceilings are backed off
    # by this fraction of the measured span on each side, so the mapped range
    # keeps foot- and headroom instead of pinning the 0.1/99.9 percentile
    # pixels to pure black/white (NAT feedback 2026-07-11: the exact mapping
    # sat too close to blown highlights and crushed shadows).
    AUTO_LEVELS_MARGIN = 0.05

    # Auto-Levels gamma: after the levels mapping the luma median should sit
    # near the middle of the encoded range; the master gamma (out**(1/g), >1
    # brightens) is set to put it there. Deadband so balanced images keep
    # g=1.0, clamped to a conservative auto range (the slider allows 0.2-3).
    AUTO_LEVELS_GAMMA_TARGET = 0.5
    AUTO_LEVELS_GAMMA_DEADBAND = 0.05
    AUTO_LEVELS_GAMMA_RANGE = (0.5, 2.0)

    def scan_auto_levels(self) -> None:
        """SCAN tab Auto levels: measure the rendered output's per-channel
        0.1 / 99.9 percentiles (scanner-software style floors/ceilings) and
        set the master black/white points from their envelope, backed off by
        AUTO_LEVELS_MARGIN of the span on each side, plus a master gamma that
        centers the luma median (NAT 2026-07-18). Luminance only: the
        per-channel offsets are reset to zero, never set — stretching each
        channel to its own floor/ceiling neutralized real casts and skewed
        grading, usually toward red (NAT 2026-07-18)."""
        output_layer = self._output_layer()
        if output_layer is None:
            set_status(self._viewer, 'Run a preview or scan before Auto levels')
            return
        data = np.asarray(output_layer.data)[..., :3]
        if not np.issubdtype(data.dtype, np.floating):
            data = runtime.normalized_image_data(data)
        floors = np.percentile(data.reshape(-1, 3), 0.1, axis=0)
        ceilings = np.percentile(data.reshape(-1, 3), 99.9, axis=0)
        # Envelope of the channel floors/ceilings: the same master points on
        # all three channels, so no channel clips and no cast is introduced.
        master_black = float(floors.min())
        master_white = float(ceilings.max())
        margin = self.AUTO_LEVELS_MARGIN * max(master_white - master_black, 1e-6)
        master_black -= margin
        master_white += margin
        gamma = self._auto_levels_gamma(data, master_black, master_white)
        section = self._widgets.scan_finishing
        section.black_point.value = round(master_black, 4)
        section.white_point.value = round(master_white, 4)
        section.black_point_rgb.value = (0.0, 0.0, 0.0)
        section.white_point_rgb.value = (0.0, 0.0, 0.0)
        section.gamma.value = gamma
        set_status(self._viewer, 'Scan levels measured from the rendered output (luminance only)')
        self._request_auto_preview_if_enabled()

    def _auto_levels_gamma(self, data: np.ndarray, master_black: float, master_white: float) -> float:
        """Master gamma putting the mapped luma median at the target: the
        finishing kernel applies out**(1/g) after the levels mapping, so
        pos**(1/g) = target -> g = ln(pos)/ln(target). Same curve on all
        three channels — no cast."""
        luma = data @ np.array([0.2126, 0.7152, 0.0722])   # Rec.709, as the kernel
        pos = (float(np.median(luma)) - master_black) / max(master_white - master_black, 1e-6)
        if not 0.02 < pos < 0.98:
            return 1.0
        gamma = float(np.log(pos) / np.log(self.AUTO_LEVELS_GAMMA_TARGET))
        if abs(gamma - 1.0) < self.AUTO_LEVELS_GAMMA_DEADBAND:
            return 1.0
        low, high = self.AUTO_LEVELS_GAMMA_RANGE
        return round(min(max(gamma, low), high), 2)

    # ------------------------------------------------------------------
    # Side-by-side compare (nav bar): working image vs input / negative,
    # as a second layer beside the output (shared zoom/pan). The pre-scan
    # mode was removed 2026-07-11 (NAT: useless in practice).
    # ------------------------------------------------------------------
    _COMPARE_LAYER_NAME = 'compare'

    def set_compare_mode(self, mode: str) -> None:
        mode = (mode or 'off').lower()
        if mode not in ('input', 'negative'):
            self._remove_compare_layer()
            self._last_compare = None
            if self._gpu_canvas is not None:
                self._gpu_canvas.clear_compare()
            if mode != 'off':
                set_status(self._viewer, 'Compare off')
            return
        preview_layer = self._layers.preview_input_layer()
        if preview_layer is None:
            set_status(self._viewer, 'Load an image before comparing')
            return
        if mode == 'input':
            image = np.asarray(preview_layer.data)[..., :3]
            compare_display = np.clip(image, 0.0, 1.0).astype(np.float32)
            self._push_compare_image(compare_display, 'input', 'base input')
            return
        self._start_compare_negative_render()

    def _push_compare_image(self, image: np.ndarray, mode: str, label: str) -> None:
        compare_display = np.asarray(image)
        self._set_compare_layer(compare_display)
        self._last_compare = (compare_display, mode)
        if self._gpu_canvas is not None:
            try:
                self._gpu_canvas.set_compare(self._padded_compare(compare_display, mode))
            except Exception:
                pass
        set_status(self._viewer, f'Comparing with the {label} (side by side)')

    def _build_compare_negative_request(self) -> SimulationRequest | None:
        """The compare-negative request, built EXACTLY like the scan-film
        view's: the LINEAR preview input (not the display-encoded layer), the
        same params builder + preview configuration, only io.scan_film forced.
        Executed by the same worker/executor as the main render — same
        exposure handling, same display preparation — so the panel colour
        matches the main output by construction. (Two earlier attempts that
        rendered with a bare Simulator on the encoded layer produced a colour
        shift and, with the display transform added, a WHITE panel.)"""
        source = self._simulation_input_image(source_layer_name=INPUT_PREVIEW_LAYER_NAME)
        if source is None:
            return None
        state = collect_gui_state(widgets=self._widgets)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=INPUT_PREVIEW_LAYER_NAME,
        )
        params.io.scan_film = True
        return SimulationRequest(
            mode_label='Negative compare',
            image=np.double(source),
            params=params,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
        )

    def _start_compare_negative_render(self) -> None:
        """Render the compare-negative through the SAME async worker path as
        the main preview (shared simulator cache mutation is safe here for the
        same reason it is between preview/scan renders: every render re-digests
        fresh params), routing the finished display image to the compare quad
        instead of the output layer."""
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'Simulation already running; retry the compare after it finishes')
            return
        if self._active_export_worker is not None:
            set_status(self._viewer, 'Export running; wait before comparing.')
            return
        request = self._build_compare_negative_request()
        if request is None:
            set_status(self._viewer, 'Load an image before comparing')
            return
        worker = runtime.SimulationWorker(request, execute_request=self._execute_simulation_request)
        worker.signals.finished.connect(self._on_compare_render_finished)
        worker.signals.failed.connect(self._on_compare_render_failed)
        self._active_simulation_worker = worker
        self._active_simulation_label = 'Negative compare'
        self._active_simulation_reports_status = False
        self._set_simulation_controls_enabled(False)
        set_status(self._viewer, 'Rendering negative compare...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _on_compare_render_finished(self, result: SimulationResult) -> None:
        self._active_simulation_worker = None
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        self._push_compare_image(result.display_image, 'negative', 'negative')
        self._replay_pending_auto_preview()

    def _on_compare_render_failed(self, message: str) -> None:
        self._active_simulation_worker = None
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        set_status(self._viewer, f'Compare render failed: {message}')
        self._replay_pending_auto_preview()

    # Blank-film fallback: kodak_gold_200's UNEXPOSED base rendered through
    # scan_film by the simulation itself (sRGB (204, 133, 94), measured
    # 2026-07-11) — used when the per-stock measurement below is unavailable.
    _FILM_BASE_FALLBACK = (0.7985, 0.5234, 0.3702)
    _blank_film_cache: dict = {}
    _last_compare = None

    def _blank_film_rgb(self) -> tuple:
        """The CURRENT film stock's blank-film color: a black (unexposed)
        frame developed and scanned as a negative at engine defaults — the
        base + orange mask exactly as the simulation renders it. Cached per
        stock (one tiny 16x16 render on first use); any failure falls back
        to the measured kodak_gold_200 constant."""
        try:
            stock = str(self._widgets.simulation.film_stock.value)
        except Exception:
            stock = ''
        cached = self._blank_film_cache.get(stock)
        if cached is not None:
            return cached
        color = self._FILM_BASE_FALLBACK
        if stock:
            try:
                from spektrafilm.runtime.params_builder import init_params
                from spektrafilm.runtime.process import simulate

                params = init_params(film_profile=stock)
                params.io.scan_film = True
                params.camera.auto_exposure = False
                params.debug.deactivate_spatial_effects = True
                params.debug.deactivate_stochastic_effects = True
                out = np.asarray(simulate(np.zeros((16, 16, 3)), params))
                color = tuple(float(v) for v in np.clip(
                    out[4:12, 4:12].reshape(-1, 3).mean(axis=0), 0.0, 1.0))
            except Exception:
                color = self._FILM_BASE_FALLBACK
        self._blank_film_cache[stock] = color
        return color

    def _scan_film_active(self) -> bool:
        try:
            return bool(self._widgets.simulation.scan_film_value())
        except Exception:
            return False

    def _padded_compare(self, image: np.ndarray, mode: str) -> np.ndarray:
        """The compare quad gets the SAME relative padding as the main image:
        white around the base input, blank-film color around the negative."""
        fraction = float(self._widgets.display.white_padding.value)
        padding = runtime.padding_pixels_for_image(image, fraction)
        if padding <= 0:
            return image
        fill = self._blank_film_rgb() if mode == 'negative' else None
        return runtime.apply_border_padding(image, padding, fill=fill)

    def _set_compare_layer(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        long_edge = max(height, width, 1)
        world_h, world_w = height / long_edge, width / long_edge
        output_layer = self._output_layer()
        ref = output_layer if output_layer is not None else self._layers.preview_input_layer()
        ref_world = np.asarray(ref.extent.world)
        gap = 0.05
        layer = next((l for l in self._viewer.layers
                      if getattr(l, 'name', None) == self._COMPARE_LAYER_NAME), None)
        if layer is None:
            layer = self._viewer.add_image(image, name=self._COMPARE_LAYER_NAME)
        else:
            layer.data = image
        layer.scale = (world_h / height, world_w / width)
        layer.translate = (-0.5 * world_h, float(ref_world[1][-1]) + gap)
        layer.visible = True
        reset_viewer_camera(self._viewer)

    def _remove_compare_layer(self) -> None:
        layer = next((l for l in self._viewer.layers
                      if getattr(l, 'name', None) == self._COMPARE_LAYER_NAME), None)
        if layer is not None:
            self._viewer.layers.remove(layer)
            set_status(self._viewer, 'Compare off')

    def adjust_print_filter_shift(self, channel: str, delta: float) -> None:
        """Keyboard nudge for the enlarger Y/M filter shifts (Ctrl+Shift+8/9/5/6)."""
        editor = (self._widgets.simulation.print_y_filter_shift if channel == 'y'
                  else self._widgets.simulation.print_m_filter_shift)
        editor.value = round(float(editor.value) + float(delta), 4)
        set_status(self._viewer,
                   f'Print {channel.upper()} filter shift: {float(editor.value):+.2f}')

    def show_fullscreen_output(self) -> None:
        """Full-screen the rendered output (white padding included) on a
        completely black background."""
        from spektrafilm_gui.napari_layout import show_fullscreen_image

        output_layer = self._output_layer()
        if output_layer is None:
            set_status(self._viewer, 'Run a preview or scan before going full screen')
            return
        data = np.asarray(output_layer.data)[..., :3]
        if np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, 0.0, 1.0)
        else:
            data = runtime.normalized_image_data(data)
        state = collect_gui_state(widgets=self._widgets)
        padding_pixels = float(state.display.white_padding) * max(data.shape[:2])
        fill = self._blank_film_rgb() if self._scan_film_active() else None
        padded = runtime.apply_border_padding(data, padding_pixels, fill=fill)
        image_rgb8 = (np.clip(np.asarray(padded, dtype=np.float32), 0.0, 1.0) * 255.0
                      + 0.5).astype(np.uint8)
        show_fullscreen_image(image_rgb8, parent=dialog_parent(self._viewer))

    def apply_profile_defaults(self, _selected_value: str) -> None:
        # Phase 10C: activating the 'Custom...' creator item is an editor
        # request, not a stock selection (the selector reverts itself).
        if _selected_value == 'Custom...':
            return
        state = collect_gui_state(widgets=self._widgets)
        if not state.simulation.film_stock or not state.simulation.print_paper:
            return

        params = build_params_from_state(state)
        synced_state = gui_state_from_params(
            digest_after_selection(params),
            film_stock=state.simulation.film_stock,
            print_paper=state.simulation.print_paper,
        )
        self._apply_profile_sync_state(synced_state)
        self._next_runtime_digest_applies_stock_specifics = True

    def apply_film_profile_defaults(self, film_stock: str) -> None:
        self.apply_profile_defaults(film_stock)

    # ------------------------------------------------------------------
    # Phase 10C: custom stock editor (modeless) + live recipe preview
    # ------------------------------------------------------------------
    def open_stock_editor(self, *_args) -> None:
        """Open (or focus) the modeless stock editor, seeded from the
        currently selected film stock — a custom stock reopens its recipe,
        a bundled one starts a fresh recipe on that base."""
        from spektrafilm_gui.stock_editor import StockEditorDialog

        seed = collect_gui_state(widgets=self._widgets).simulation.film_stock
        if self._stock_editor is None:
            editor = StockEditorDialog(dialog_parent(self._viewer))
            editor.recipeChanged.connect(self._on_live_recipe_changed)
            editor.stockSaved.connect(self._on_custom_stock_saved)
            editor.stockDeleted.connect(self._on_custom_stock_deleted)
            editor.finished.connect(self._on_stock_editor_closed)
            self._stock_editor = editor
        try:
            self._stock_editor.load_seed(seed)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Custom stock',
                                 f'Could not open the stock editor.\n\n{exc}')
            return
        self._stock_editor.show()
        self._stock_editor.raise_()
        self._stock_editor.activateWindow()

    def _on_live_recipe_changed(self, recipe) -> None:
        self._live_stock_recipe = recipe
        # look overrides land at digest time: force a specifics re-digest
        self._next_runtime_digest_applies_stock_specifics = True
        self._request_live_recipe_render()

    def _request_live_recipe_render(self) -> None:
        """Editor manipulation renders regardless of the auto-preview toggle
        or its first-render gate — live feedback IS the feature."""
        if self._current_preview_image is None:
            return
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            self._pending_live_render = True
            return
        self._run_preview(report_status=False)

    def _on_custom_stock_saved(self, slug: str) -> None:
        self._live_stock_recipe = None
        selector = getattr(self._widgets.simulation, 'film_stock', None)
        refresh = getattr(selector, 'refresh_custom_stocks', None)
        if callable(refresh):
            refresh()
        if selector is not None:
            selector.value = slug
        self._next_runtime_digest_applies_stock_specifics = True
        set_status(self._viewer, f'Custom stock saved: {slug}')
        self._request_live_recipe_render()

    def _on_custom_stock_deleted(self, slug: str) -> None:
        self._live_stock_recipe = None
        selector = getattr(self._widgets.simulation, 'film_stock', None)
        was_selected = selector is not None and selector.value == slug
        base = ''
        editor = self._stock_editor
        if editor is not None and editor.recipe is not None:
            base = editor.recipe.base_stock
        refresh = getattr(selector, 'refresh_custom_stocks', None)
        if callable(refresh):
            refresh()
        if was_selected and base:
            # Deleting a stock that is IN USE falls back to its base stock.
            selector.value = base
            self.apply_profile_defaults(base)
            set_status(self._viewer,
                       f'Custom stock {slug} deleted; fell back to its base {base}')
        else:
            set_status(self._viewer, f'Custom stock deleted: {slug}')
        self._request_live_recipe_render()

    def _on_stock_editor_closed(self, *_args) -> None:
        if self._live_stock_recipe is not None:
            # Discarded live edits: render once more with the saved state.
            self._live_stock_recipe = None
            self._next_runtime_digest_applies_stock_specifics = True
            self._request_live_recipe_render()

    def _warn_if_missing_custom_stock(self) -> None:
        """After a state apply: a roll/preset can reference a custom stock
        whose .sfstock is not installed — the selector kept its previous
        (valid) selection; tell the user instead of crashing (relink =
        install the .sfstock or pick another stock)."""
        selector = getattr(self._widgets.simulation, 'film_stock', None)
        missing = getattr(selector, 'last_missing_value', None)
        if not missing:
            return
        selector.last_missing_value = None
        current = selector.value if selector is not None else ''
        QMessageBox.warning(
            dialog_parent(self._viewer), 'Custom stock missing',
            f"This look references the custom stock '{missing}', which is not "
            f"installed on this machine.\n\nFalling back to '{current}'. To "
            "relink, install the .sfstock file into the custom stocks folder "
            "or pick another stock.")
        set_status(self._viewer, f'Custom stock {missing} missing; using {current}')

    def _apply_profile_sync_state(self, synced_state) -> None:
        profile_sync.apply_profile_sync_state(
            widgets=self._widgets,
            synced_state=synced_state,
            profile_sync_fields=PROFILE_SYNC_FIELDS,
        )

    def run_preview(self) -> None:
        self._run_preview(report_status=True)

    def _run_preview(self, *, report_status: bool) -> None:
        self._start_simulation(
            source_layer_name=INPUT_PREVIEW_LAYER_NAME,
            mode_label='Preview',
            report_status=report_status,
        )

    def run_scan(self) -> None:
        self._start_simulation(source_layer_name=INPUT_LAYER_NAME, mode_label='Scan')

    def request_auto_preview(self, *_args) -> None:
        # Every control change funnels through here, so it doubles as the
        # undo-history hook (debounced; skipped while undo/redo re-applies).
        self._schedule_history_push()
        if self._auto_preview_scheduled:
            return
        self._auto_preview_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_auto_preview)

    # ------------------------------------------------------------------
    # Undo / redo: a debounced stack of GUI-state snapshots. Covers every
    # parameter change (anything wired to auto-preview); loading files and
    # roll membership are not part of the stack.
    # ------------------------------------------------------------------
    _HISTORY_LIMIT = 100
    _history: list | None = None
    _history_index: int = -1
    _history_applying: bool = False
    _history_timer = None

    def history_capture_baseline(self) -> None:
        """Record the startup state so the first undo has somewhere to go."""
        self._history_push_now()

    def _schedule_history_push(self) -> None:
        if self._history_applying:
            return
        if self._history_timer is None:
            self._history_timer = QTimer()
            self._history_timer.setSingleShot(True)
            self._history_timer.timeout.connect(self._history_push_now)
        # Debounce: rapid slider/spinbox streams collapse into one snapshot.
        self._history_timer.start(700)

    def _history_push_now(self) -> None:
        from spektrafilm_gui.persistence import gui_state_to_dict

        snapshot = gui_state_to_dict(collect_gui_state(widgets=self._widgets))
        if self._history is None:
            self._history = []
        if (self._history and 0 <= self._history_index < len(self._history)
                and snapshot == self._history[self._history_index]):
            return
        del self._history[self._history_index + 1:]
        self._history.append(snapshot)
        if len(self._history) > self._HISTORY_LIMIT:
            del self._history[0]
        self._history_index = len(self._history) - 1

    def undo(self) -> None:
        # Flush a pending debounce first so the current edit is undoable.
        if self._history_timer is not None and self._history_timer.isActive():
            self._history_timer.stop()
            self._history_push_now()
        if not self._history or self._history_index <= 0:
            set_status(self._viewer, 'Nothing to undo')
            return
        self._history_index -= 1
        self._history_apply(f'Undo ({self._history_index + 1}/{len(self._history)})')

    def redo(self) -> None:
        if not self._history or self._history_index >= len(self._history) - 1:
            set_status(self._viewer, 'Nothing to redo')
            return
        self._history_index += 1
        self._history_apply(f'Redo ({self._history_index + 1}/{len(self._history)})')

    def _history_apply(self, message: str) -> None:
        from spektrafilm_gui.persistence import gui_state_from_dict

        self._history_applying = True
        try:
            apply_gui_state(gui_state_from_dict(self._history[self._history_index]),
                            widgets=self._widgets)
        finally:
            self._history_applying = False
        set_status(self._viewer, message)
        self._request_auto_preview_if_enabled()

    def _request_auto_preview_if_enabled(self) -> None:
        # Sorting mode never auto-renders (the whole point is instant, no
        # pipeline) — NAT 2026-07-25.
        if self._input_mode:
            return
        if not self._auto_preview_enabled() or self._current_preview_image is None:
            return
        self.request_auto_preview()

    def report_display_transform_status(self, enabled: bool) -> None:
        if enabled and not self.sync_display_transform_availability(report_status=True):
            return
        set_status(self._viewer, runtime.display_transform_status_message(enabled, imagecms_module=ImageCms))

    def set_gray_18_canvas_enabled(self, enabled: bool) -> None:
        set_canvas_background(self._viewer, gray_18_canvas=enabled)

    def set_output_interpolation_mode(self, mode: str) -> None:
        output_layer = self._output_layer()
        if output_layer is None:
            return
        self._layers.set_output_layer_interpolation(output_layer, mode)

    def sync_display_transform_availability(self, *, report_status: bool) -> bool:
        if runtime.display_profile_available(imagecms_module=ImageCms):
            return True

        self._set_display_transform_checked(False)
        if report_status:
            set_status(self._viewer, 'Display transform unavailable: no display profile detected, disabled')
        return False

    def save_output_layer(self) -> None:
        output_layer = self._output_layer()
        if output_layer is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Save output', 'Run a simulation before saving the output layer.')
            return

        if self._current_input_path is not None:
            default_name = Path(self._current_input_path).stem + '.jpg'
        else:
            default_name = 'output.jpg'

        filepath, _ = _DirMemoryDialog('save_output').get_save_file_name(
            dialog_parent(self._viewer),
            'Save output image',
            default_name,
            'Images (*.jpg *.jpeg *.png *.tif *.tiff *.exr)',
        )
        if not filepath:
            return

        gui_state = collect_gui_state(widgets=self._widgets)
        float_image_data = self._output_layer_float_data()
        if float_image_data is None:
            image_data = runtime.normalized_image_data(np.asarray(output_layer.data)[..., :3])
        else:
            image_data = np.asarray(float_image_data)[..., :3]

        source_color_space, source_cctf_encoding = self._output_layer_render_settings(
            default_color_space=gui_state.simulation.output_color_space,
            default_cctf_encoding=True,
        )
        saving_color_space = gui_state.simulation.saving_color_space
        saving_cctf_encoding = gui_state.simulation.saving_cctf_encoding
        if source_color_space != saving_color_space:
            image_data = colour.RGB_to_RGB(
                image_data,
                source_color_space,
                saving_color_space,
                apply_cctf_decoding=source_cctf_encoding,
                apply_cctf_encoding=saving_cctf_encoding,
            )
        elif source_cctf_encoding != saving_cctf_encoding:
            image_data = colour.RGB_to_RGB(
                image_data,
                source_color_space,
                saving_color_space,
                apply_cctf_decoding=source_cctf_encoding,
                apply_cctf_encoding=saving_cctf_encoding,
            )

        source_metadata = None

        if self._current_input_path is not None:
            source_metadata = read_image_metadata(self._current_input_path)

        try:
            save_image_oiio(
                filepath,
                image_data,
                color_space=saving_color_space,
                cctf_encoding=saving_cctf_encoding,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save output', f'Failed to save output image.\n\n{exc}')
            return

        metadata_write_error = None
        try:
            write_image_metadata(
                filepath,
                source_metadata,
                saving_color_space=saving_color_space,
                saving_cctf_encoding=saving_cctf_encoding,
                custom_tags=self._custom_metadata_tags(),
            )
        except Exception as exc:
            metadata_write_error = exc

        if metadata_write_error is not None:
            set_status(
                self._viewer,
                f"Saved output image to {filepath}, but failed to copy metadata: {metadata_write_error}",
            )
        else:
            set_status(self._viewer, f"Saved output image to {filepath}")

    # ------------------------------------------------------------------ #
    # Export: full-precision (or fast-LUT) re-render written off the UI thread.
    # Acts as a film scanner -- latitude-preserving, high-bit base for grading
    # elsewhere. Unlike save_output_layer (which writes the already-rendered
    # interactive buffer), export re-renders the FULL-resolution input with the
    # export settings so 16-bit output is genuine high precision from the float
    # pipeline.
    # ------------------------------------------------------------------ #
    def export_output(self) -> None:
        if self._active_export_worker is not None:
            # Second click while exporting acts as Cancel (pre-render / pre-write
            # granularity; a single full-res render is not interrupted mid-pass).
            self.cancel_export()
            return
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'A render is already running; wait before exporting.')
            return

        state = collect_gui_state(widgets=self._widgets)
        ext, bit_depth = export_format_spec(state.simulation.export_format)
        saving_color_space = state.simulation.saving_color_space
        saving_cctf_encoding = state.simulation.saving_cctf_encoding
        full_precision = state.simulation.export_full_precision

        # Source of the export pixels:
        #  - full precision -> re-render the full-resolution INPUT (slow, LUT-off);
        #  - otherwise      -> reuse the buffer from the last Scan/Preview, no recompute.
        image_data = None
        prerendered = None
        source_color_space = saving_color_space
        source_cctf_encoding = saving_cctf_encoding
        if full_precision:
            image_data = self._simulation_input_image(source_layer_name=INPUT_LAYER_NAME)
            if image_data is None:
                QMessageBox.warning(dialog_parent(self._viewer), 'Export', 'Load an input image before exporting.')
                return
        else:
            prerendered = self._output_layer_float_data()
            if prerendered is None:
                QMessageBox.warning(
                    dialog_parent(self._viewer),
                    'Export',
                    'Run a Scan first, or tick "Full-precision re-render" to render from the input.',
                )
                return
            source_color_space, source_cctf_encoding = self._output_layer_render_settings(
                default_color_space=state.simulation.output_color_space,
                default_cctf_encoding=True,
            )

        if self._current_input_path is not None:
            default_name = Path(self._current_input_path).stem + '.' + ext
        else:
            default_name = 'export.' + ext

        file_filter = self._export_file_filter(ext)
        filepath, _ = _DirMemoryDialog('save_output').get_save_file_name(
            dialog_parent(self._viewer),
            'Export image',
            default_name,
            file_filter,
        )
        if not filepath:
            return
        # The export-panel format is the source of truth: force the extension so
        # the on-disk format and bit depth always match the chosen preset.
        filepath = str(Path(filepath).with_suffix('.' + ext))

        params = None
        if full_precision:
            self._sync_white_border(white_padding=state.display.white_padding)
            params = configure_export_params(
                build_params_from_state(state),
                saving_color_space=saving_color_space,
                saving_cctf_encoding=saving_cctf_encoding,
                full_precision=True,
            )

        source_metadata = None
        if self._current_input_path is not None:
            source_metadata = read_image_metadata(self._current_input_path)

        request = ExportRequest(
            image=np.double(image_data) if image_data is not None else None,
            params=params,
            filepath=filepath,
            saving_color_space=saving_color_space,
            saving_cctf_encoding=saving_cctf_encoding,
            bit_depth=bit_depth,
            source_metadata=source_metadata,
            full_precision=full_precision,
            prerendered=(np.asarray(prerendered) if prerendered is not None else None),
            source_color_space=source_color_space,
            source_cctf_encoding=source_cctf_encoding,
            custom_tags=self._custom_metadata_tags(),
        )
        worker = ExportWorker(request, render_fn=self._render_export, write_fn=self._write_export)
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.failed.connect(self._on_export_failed)
        self._active_export_worker = worker
        self._set_export_controls_enabled(False)
        quality = 'full precision' if full_precision else 'current scan'
        set_status(self._viewer, f'Exporting {Path(filepath).name} ({quality})...', timeout_ms=0)
        self._thread_pool.start(worker)

    def export_roll(self) -> None:
        """Batch export: render and write EVERY frame of the roll (full
        precision, each frame with its own stored look) into a chosen folder.
        Runs on a single BatchExportWorker; cancel applies between frames."""
        if self._active_export_worker is not None:
            self.cancel_export()
            return
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'A render is already running; wait before exporting.')
            return
        if not self._roll.files:
            QMessageBox.warning(dialog_parent(self._viewer), 'Export roll', 'The roll is empty. Add files first.')
            return

        from copy import deepcopy

        # Freeze the on-screen look into the current frame before resolving.
        current_state = collect_gui_state(widgets=self._widgets)
        self._roll.store_state_for_current(current_state)

        export_section = getattr(self._widgets, 'export', None)
        sync_settings = True
        sync_metadata = True
        if export_section is not None and callable(getattr(export_section, 'sync_export_settings_value', None)):
            sync_settings = export_section.sync_export_settings_value()
        if export_section is not None and callable(getattr(export_section, 'sync_metadata_value', None)):
            sync_metadata = export_section.sync_metadata_value()

        out_dir = QFileDialog.getExistingDirectory(dialog_parent(self._viewer), 'Export roll to folder')
        if not out_dir:
            return

        # Sync metadata ON: the Metadata panel values stamp every frame (the
        # per-frame star rating is merged per job below, never shared).
        # OFF: frames keep only their own source-file metadata (no panel tags).
        panel_tags = self._custom_metadata_tags() if sync_metadata else None
        if panel_tags is not None:
            panel_tags.pop('Exif.Image.Rating', None)   # per-frame, see below
        jobs = []
        skipped_by_filter = 0
        for roll_file in self._roll.files:
            if not self._passes_rating_filter(roll_file):
                skipped_by_filter += 1
                continue
            frame_state = self._roll.state_for(roll_file) or deepcopy(current_state)
            prefs = current_state.simulation if sync_settings else frame_state.simulation
            ext, bit_depth = export_format_spec(prefs.export_format)
            frame_tags = dict(panel_tags) if panel_tags is not None else None
            rating = self._roll.rating_for(roll_file)
            if rating:
                frame_tags = dict(frame_tags or {})
                frame_tags['Exif.Image.Rating'] = str(rating)
            jobs.append(runtime.BatchFrameJob(
                name=roll_file.name,
                path=roll_file.path,
                kind=roll_file.kind,
                state=frame_state,
                saving_color_space=prefs.saving_color_space,
                saving_cctf_encoding=prefs.saving_cctf_encoding,
                ext=ext,
                bit_depth=bit_depth,
                custom_tags=frame_tags,
            ))
        if not jobs:
            QMessageBox.warning(dialog_parent(self._viewer), 'Export roll',
                                'The star filter excludes every frame of the roll.')
            return
        if skipped_by_filter:
            set_status(self._viewer,
                       f'Star filter >= {self._rating_filter}: exporting '
                       f'{len(jobs)} of {len(self._roll.files)} frames')

        worker = runtime.BatchExportWorker(
            jobs,
            render_fn=self._batch_render_frame,
            write_fn=lambda job, buffer: self._batch_write_frame(job, buffer, out_dir),
        )
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.failed.connect(self._on_export_failed)
        self._active_export_worker = worker
        self._set_export_controls_enabled(False)
        set_status(self._viewer, f'Exporting roll: {len(jobs)} frame(s) to {out_dir}...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _batch_render_frame(self, job) -> np.ndarray:
        """Load + full-precision render of one roll frame with ITS OWN look.
        Mirrors load_raw_image / load_input_image processing and the
        full-precision branch of _render_export; runs on the worker thread
        (all callees are pure / thread-safe, a dedicated simulator per frame)."""
        state = job.state
        if job.kind == 'raw':
            image = load_and_process_raw_file(
                job.path,
                white_balance=state.load_raw.white_balance,
                temperature=state.load_raw.temperature,
                tint=state.load_raw.tint,
                lens_correction=state.load_raw.lens_correction,
                output_colorspace=state.input_image.input_color_space,
                output_cctf_encoding=state.input_image.apply_cctf_decoding,
            )
        else:
            image = load_image_oiio(job.path)[..., :3]
            image = apply_rgb_white_balance(
                image,
                white_balance=state.load_rgb.white_balance,
                temperature=state.load_rgb.temperature,
                tint=state.load_rgb.tint,
                color_space=state.input_image.input_color_space,
                cctf_encoded=state.input_image.apply_cctf_decoding,
            )
        params = configure_export_params(
            build_params_from_state(state),
            saving_color_space=job.saving_color_space,
            saving_cctf_encoding=job.saving_cctf_encoding,
            full_precision=True,
        )
        digested_params = digest_params(params, apply_stocks_specifics=True)
        simulator = runtime_simulator(digested_params)
        return np.asarray(simulator.process(np.double(image)))

    def _batch_write_frame(self, job, buffer: np.ndarray, out_dir: str) -> str:
        target = runtime.unique_export_path(out_dir, Path(job.name).stem, job.ext)
        save_image_oiio(
            str(target),
            np.asarray(buffer)[..., :3],
            bit_depth=job.bit_depth,
            color_space=job.saving_color_space,
            cctf_encoding=job.saving_cctf_encoding,
        )
        try:
            write_image_metadata(
                str(target),
                read_image_metadata(job.path),
                saving_color_space=job.saving_color_space,
                saving_cctf_encoding=job.saving_cctf_encoding,
                custom_tags=job.custom_tags,
            )
        except Exception:  # metadata is non-fatal; pixels + ICC are written
            pass
        return target.name

    def export_lut(self) -> None:
        """Presets 'Export LUT...': bake the current look into a 3D LUT
        (.cube/.3dl/HaldCLUT) and optional OCIO config, off the UI thread."""
        if self._active_export_worker is not None or self._active_simulation_worker is not None:
            set_status(self._viewer, 'Busy; wait before exporting a LUT.')
            return
        from spektrafilm_gui.lut_export_dialog import LutExportDialog

        dialog = LutExportDialog(dialog_parent(self._viewer))
        if not dialog.exec_() if hasattr(dialog, 'exec_') else not dialog.exec():
            return
        req = dialog.result_spec()

        state = collect_gui_state(widgets=self._widgets)
        params = build_params_from_state(state)

        def _run() -> str:
            from spektrafilm_lut_creator.look_export import (
                LookLutSpec, sample_look_lut, write_look_luts, write_ocio_config,
            )
            spec = LookLutSpec(
                size=req.size,
                input_color_space=req.input_color_space,
                title=f'spektrafilm {state.simulation.film_stock} / {state.simulation.print_paper}',
            )
            lut = sample_look_lut(params, spec)
            written, errors = write_look_luts(lut, req.out_path, req.formats)
            ocio_note = ''
            if req.write_ocio and any(p.endswith('.cube') for p in written):
                cube = next(p for p in written if p.endswith('.cube'))
                cfg = write_ocio_config(str(Path(cube).parent), Path(cube).name)
                ocio_note = f'; OCIO config {Path(cfg).name}'
            msg = f'Exported LUT: {len(written)} file(s) to {Path(req.out_path).parent}{ocio_note}'
            if errors:
                msg += f' (skipped: {", ".join(f"{k}: {v}" for k, v in errors.items())})'
            return msg

        worker = runtime.LutExportWorker(_run)
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(self._on_lut_export_finished)
        worker.signals.failed.connect(self._on_lut_export_failed)
        self._active_export_worker = worker
        self._set_export_controls_enabled(False)
        set_status(self._viewer, f'Baking LUT ({req.size}^3)...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _on_lut_export_finished(self, status_message: str) -> None:
        self._active_export_worker = None
        self._set_export_controls_enabled(True)
        set_status(self._viewer, status_message)
        self._replay_pending_auto_preview()

    def _on_lut_export_failed(self, message: str) -> None:
        self._active_export_worker = None
        self._set_export_controls_enabled(True)
        QMessageBox.critical(dialog_parent(self._viewer), 'Export LUT', f'LUT export failed.\n\n{message}')
        set_status(self._viewer, 'LUT export failed')
        self._replay_pending_auto_preview()

    def cancel_export(self) -> None:
        worker = self._active_export_worker
        if worker is None:
            return
        worker.cancel()
        set_status(self._viewer, 'Export cancel requested...', timeout_ms=0)

    def set_gpu_backend(self, enabled: bool) -> None:
        """Force the spectral-pipeline backend GPU (Vulkan) vs CPU at runtime.

        Sets the SPEKTRAFILM_GPU env var that every kernel re-reads per call, so
        it takes effect on the next preview/scan. A diagnostic A/B: if turning
        the GPU off slows scans, the GPU is engaging; if not, it is not.
        """
        os.environ['SPEKTRAFILM_GPU'] = 'auto' if enabled else 'cpu'
        self.refresh_gpu_status()
        set_status(
            self._viewer,
            f"GPU acceleration {'ON (Vulkan)' if enabled else 'OFF (CPU fallback)'} - applies on the next preview/scan",
            timeout_ms=0,
        )

    def refresh_gpu_status(self) -> None:
        gpu_widget = getattr(self._widgets, 'gpu_backend', None)
        set_text = getattr(gpu_widget, 'set_gpu_status_text', None)
        if not callable(set_text):
            return
        env = os.environ.get('SPEKTRAFILM_GPU')
        if isinstance(env, str) and env.strip().lower() in ('0', 'cpu', 'false', 'off', 'no'):
            set_text('CPU fallback (GPU off)')
            return
        try:
            report = import_module('spektrafilm.gpu.backend').backend_report()
        except Exception:
            set_text('GPU status unavailable')
            return
        if report.get('gpu_available'):
            set_text(f"GPU acceleration: {report.get('backend_name') or 'GPU'}")
        else:
            err = report.get('init_error')
            set_text(f'CPU only (no GPU: {err})' if err else 'CPU only (no GPU detected)')

    @staticmethod
    def _export_file_filter(ext: str) -> str:
        if ext in {'jpg', 'jpeg'}:
            return 'JPEG image (*.jpg *.jpeg)'
        if ext in {'tif', 'tiff'}:
            return 'TIFF image (*.tif *.tiff)'
        if ext == 'png':
            return 'PNG image (*.png)'
        return 'Images (*.jpg *.jpeg *.tif *.tiff *.png)'

    def _render_export(self, request: ExportRequest) -> np.ndarray:
        if request.full_precision:
            # A dedicated simulator isolates the export from the interactive
            # self._runtime_simulator (different LUT/precision settings, separate
            # thread). digest_params resolves stock specifics like the live path.
            # configure_export_params already set the output to the saving space
            # + encoding, so the buffer needs no further colour conversion.
            digested_params = digest_params(request.params, apply_stocks_specifics=True)
            simulator = runtime_simulator(digested_params)
            return np.asarray(simulator.process(request.image))

        # Fast path: reuse the buffer from the last Scan/Preview (no recompute);
        # only convert it from its render space/encoding to the saving choice.
        buffer = np.asarray(request.prerendered)[..., :3]
        if (
            request.source_color_space != request.saving_color_space
            or request.source_cctf_encoding != request.saving_cctf_encoding
        ):
            buffer = colour.RGB_to_RGB(
                buffer,
                request.source_color_space,
                request.saving_color_space,
                apply_cctf_decoding=request.source_cctf_encoding,
                apply_cctf_encoding=request.saving_cctf_encoding,
            )
        return np.asarray(buffer)

    def _write_export(self, request: ExportRequest, buffer: np.ndarray) -> str:
        # The render already produced pixels in the saving colour space + encoding
        # (configure_export_params set io.output_color_space / output_cctf_encoding),
        # so there is no colour conversion here -- only bit-depth encoding + ICC.
        image_data = np.asarray(buffer)[..., :3]
        # Paper-texture finish (NAT 2026-07-26): composite + optional padding
        # burn. No-op when inactive, so default exports are bit-identical.
        image_data = self.apply_paper_finish_to_export(image_data)
        save_image_oiio(
            request.filepath,
            image_data,
            bit_depth=request.bit_depth,
            color_space=request.saving_color_space,
            cctf_encoding=request.saving_cctf_encoding,
        )
        metadata_error = None
        try:
            write_image_metadata(
                request.filepath,
                request.source_metadata,
                saving_color_space=request.saving_color_space,
                saving_cctf_encoding=request.saving_cctf_encoding,
                custom_tags=getattr(request, 'custom_tags', None),
            )
        except Exception as exc:  # metadata is non-fatal; pixels + ICC are written
            metadata_error = exc
        if metadata_error is not None:
            return f'Exported {request.filepath} (metadata copy failed: {metadata_error})'
        return f'Exported {request.filepath}'

    def _on_export_progress(self, message: str) -> None:
        set_status(self._viewer, message, timeout_ms=0)

    def _on_export_finished(self, status_message: str) -> None:
        self._active_export_worker = None
        self._set_export_controls_enabled(True)
        set_status(self._viewer, status_message)
        self._replay_pending_auto_preview()

    def _on_export_failed(self, message: str) -> None:
        self._active_export_worker = None
        self._set_export_controls_enabled(True)
        if message.startswith('Export cancelled'):
            set_status(self._viewer, 'Export cancelled')
        else:
            QMessageBox.critical(dialog_parent(self._viewer), 'Export', f'Export failed.\n\n{message}')
            set_status(self._viewer, 'Export failed')
        self._replay_pending_auto_preview()

    def _set_export_controls_enabled(self, enabled: bool) -> None:
        # Block interactive renders while exporting (shared simulator-free, but a
        # single GPU/CPU at a time keeps timing predictable), and reflect the
        # exporting state on the export panel button (Export <-> Cancel).
        self._set_simulation_controls_enabled(enabled)
        export_section = getattr(self._widgets, 'export', None)
        set_exporting = getattr(export_section, 'set_exporting', None)
        if callable(set_exporting):
            set_exporting(not enabled)

    def save_current_as_default(self) -> None:
        persistence_actions.save_current_as_default(
            viewer=self._viewer,
            widgets=self._widgets,
            collect_gui_state_fn=collect_gui_state,
            save_default_gui_state_fn=save_default_gui_state,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def save_current_state_to_file(self) -> None:
        persistence_actions.save_current_state_to_file(
            viewer=self._viewer,
            widgets=self._widgets,
            file_dialog=_DirMemoryDialog('gui_state'),
            collect_gui_state_fn=collect_gui_state,
            save_gui_state_to_path_fn=save_gui_state_to_path,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def load_state_from_file(self) -> None:
        persistence_actions.load_state_from_file(
            viewer=self._viewer,
            widgets=self._widgets,
            file_dialog=_DirMemoryDialog('gui_state'),
            load_gui_state_from_path_fn=load_gui_state_from_path,
            apply_gui_state_fn=apply_gui_state,
            sync_canvas_background_fn=self._sync_canvas_background,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )
        self._warn_if_missing_custom_stock()

    # ------------------------------------------------------------------
    # Phase 14: preset browser (.sfpreset dropdown over the user dir)
    # ------------------------------------------------------------------
    def apply_preset(self, name: str) -> None:
        """Dropdown selection: apply the named preset like Load from file."""
        from spektrafilm_gui.persistence import gui_state_from_dict
        from spektrafilm_gui.preset_store import (
            load_preset_payload, load_preset_sections)
        from spektrafilm_gui.state_bridge import apply_gui_state_sections

        try:
            state = gui_state_from_dict(load_preset_payload(name))
            sections = load_preset_sections(name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(dialog_parent(self._viewer), 'Preset',
                                f'Could not load the preset.\n\n{exc}')
            return
        if sections:
            # A partial preset touches only its saved sections.
            apply_gui_state_sections(
                state, widgets=self._widgets, section_names=tuple(sections))
        else:
            apply_gui_state(state, widgets=self._widgets)
        self._sync_canvas_background()
        self._warn_if_missing_custom_stock()
        set_status(self._viewer, f'Preset applied: {name}')

    def save_current_as_preset(self) -> None:
        from qtpy.QtWidgets import QInputDialog

        from spektrafilm_gui.persistence import gui_state_to_dict
        from spektrafilm_gui.preset_store import (
            join_qualified_name, list_preset_folders, preset_path,
            save_preset, split_qualified_name)
        from spektrafilm_gui.section_picker import pick_settings_sections
        from spektrafilm_gui.settings_groups import group_labels
        from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES

        section = getattr(self._widgets, 'gui_config', None)
        suggested = section.current_preset() if section is not None else ''
        suggested_folder, suggested_leaf = split_qualified_name(suggested)
        name, accepted = QInputDialog.getText(
            dialog_parent(self._viewer), 'Save preset', 'Preset name:',
            text=suggested_leaf)
        if not accepted or not name.strip():
            return
        name = name.strip()
        # Folders (NAT 2026-07-25): pick an existing folder or type a new one;
        # the top choice keeps the preset un-foldered.
        no_folder = '(no folder)'
        choices = [no_folder] + list_preset_folders()
        default_index = (choices.index(suggested_folder)
                         if suggested_folder in choices else 0)
        folder, ok = QInputDialog.getItem(
            dialog_parent(self._viewer), 'Save preset',
            'Folder (type a new name to create one):',
            choices, default_index, True)
        if not ok:
            return
        folder = '' if folder.strip() == no_folder else folder.strip()
        name = join_qualified_name(folder, name)
        # Which settings does this preset carry? Default = everything (a full
        # preset); the user can narrow it to 'just the grain' etc. (NAT).
        chosen = pick_settings_sections(
            dialog_parent(self._viewer),
            title='Save preset — which settings?',
            intro='Tick the settings this preset should carry. Applying it '
                  'later changes only the ticked ones.',
            preselected=group_labels())
        if chosen is None:
            return  # cancelled
        if not chosen:
            set_status(self._viewer, 'Nothing selected — preset not saved')
            return
        # A preset covering every section is a plain full preset (sections=None).
        sections_arg = (None if set(chosen) == set(GUI_STATE_SECTION_NAMES)
                        else tuple(chosen))
        try:
            if preset_path(name).is_file():
                answer = QMessageBox.question(
                    dialog_parent(self._viewer), 'Save preset',
                    f"A preset named '{name}' exists. Overwrite it?")
                if answer != QMessageBox.Yes:
                    return
            save_preset(name, gui_state_to_dict(collect_gui_state(widgets=self._widgets)),
                        sections=sections_arg)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(dialog_parent(self._viewer), 'Save preset',
                                f'Could not save the preset.\n\n{exc}')
            return
        if section is not None:
            section.refresh_presets(select=name)
        set_status(self._viewer, f'Preset saved: {name}')

    def update_preset(self, name: str) -> None:
        """NAT 2026-07-22: overwrite the SELECTED preset with the current
        look — no dialogs, the selection is the intent."""
        from spektrafilm_gui.persistence import gui_state_to_dict
        from spektrafilm_gui.preset_store import save_preset

        try:
            save_preset(name, gui_state_to_dict(collect_gui_state(widgets=self._widgets)))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(dialog_parent(self._viewer), 'Update preset',
                                f'Could not update the preset.\n\n{exc}')
            return
        section = getattr(self._widgets, 'gui_config', None)
        if section is not None:
            section.refresh_presets(select=name)
        set_status(self._viewer, f'Preset updated: {name}')

    def delete_preset(self, name: str) -> None:
        from spektrafilm_gui.preset_store import delete_preset as _delete_preset

        answer = QMessageBox.question(
            dialog_parent(self._viewer), 'Delete preset', f"Delete preset '{name}'?")
        if answer != QMessageBox.Yes:
            return
        _delete_preset(name)
        section = getattr(self._widgets, 'gui_config', None)
        if section is not None:
            section.refresh_presets(select='')
        set_status(self._viewer, f'Preset deleted: {name}')

    def restore_factory_default(self) -> None:
        persistence_actions.restore_factory_default(
            viewer=self._viewer,
            widgets=self._widgets,
            project_default_gui_state=PROJECT_DEFAULT_GUI_STATE,
            clear_saved_default_gui_state_fn=clear_saved_default_gui_state,
            apply_gui_state_fn=apply_gui_state,
            sync_canvas_background_fn=self._sync_canvas_background,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def _preview_input_layer(self) -> Any | None:
        return self._layers.preview_input_layer()

    def _white_border_layer(self) -> Any | None:
        return self._layers.white_border_layer()

    # Phase 6A: optional wgpu canvas mirror (flagged; napari stays authoritative
    # until 6C). The canvas shows whatever the output/preview funnel shows.
    _gpu_canvas = None

    def set_gpu_canvas(self, widget) -> None:
        self._gpu_canvas = widget
        if widget is not None:
            try:  # restore the persisted composition-guide overlay (Ctrl+O / Shift+O)
                settings = QSettings('spektrafilm', 'spektrafilm')
                mode = str(settings.value('canvas_guides', '') or 'off')
                widget.set_guide_mode(mode)
                if hasattr(widget, 'set_guide_orient'):
                    widget.set_guide_orient(int(settings.value('canvas_guide_orient', 0) or 0))
            except Exception:
                pass

    _last_canvas_image = None
    # Ctrl+R before/after (NAT): flip the canvas between the cached input
    # DISPLAY and the last render, without re-running the pipeline.
    _input_display_cached = None
    _output_display_cached = None
    _showing_input = False
    # Input / sorting MODE (NAT 2026-07-25): a persistent cull mode, separate
    # from Ctrl+R. It shows the camera's embedded thumbnail (instant, no decode
    # or render) and stays on across roll navigation so you can sort fast.
    # Default ON (NAT 2026-07-26): the app opens ready to cull; 'edit' renders.
    _input_mode = True
    _input_mode_navigated = False
    _input_mode_button = None

    _last_canvas_is_output = False

    def _mirror_to_gpu_canvas(self, image, *, is_output: bool = False) -> None:
        """Display funnel to the wgpu canvas. The white padding frame is baked
        in HERE at the pixel level (the napari-era world-geometry border layer
        no longer reaches the canvas since 6C); the canvas gets the padding
        fraction as a content inset so crop mapping and the composition guides
        keep working on the photo, not the border.

        For the OUTPUT (is_output), the paper-texture finish is composited here
        too — before padding when padding-burn is off (texture on the image),
        after when it is on (texture bleeds into the frame). NAT 2026-07-26."""
        if self._gpu_canvas is None:
            return
        try:
            data = np.asarray(image)[..., :3]
            self._last_canvas_image = data
            self._last_canvas_is_output = is_output
            paper = self._active_paper_state() if is_output else None
            if paper is not None and not paper.padding_burn:
                data = self._paper_composite(data, paper)
            padding_fraction = float(self._widgets.display.white_padding.value)
            padding = runtime.padding_pixels_for_image(data, padding_fraction)
            if padding > 0:
                # Scan-film view shows the NEGATIVE: frame it in the blank-
                # film base color instead of paper white (NAT 2026-07-11).
                fill = self._blank_film_rgb() if self._scan_film_active() else None
                data = runtime.apply_border_padding(data, padding, fill=fill)
            if paper is not None and paper.padding_burn:
                data = self._paper_composite(data, paper)
            self._gpu_canvas.set_image(data)
            set_inset = getattr(self._gpu_canvas, 'set_content_inset', None)
            if callable(set_inset):
                if padding > 0:
                    set_inset(padding / data.shape[1], padding / data.shape[0])
                else:
                    set_inset(0.0, 0.0)
        except Exception:
            pass  # display mirror must never break the render path

    def toggle_preview_view(self) -> None:
        """Ctrl+R: rolling BEFORE/AFTER. Flip the canvas between the cached
        input DISPLAY and the last render, without re-running the pipeline
        (this is the accurate full-quality input, distinct from the instant
        thumbnail of the sorting mode). First press with nothing rendered kicks
        a preview so there is an 'after' to compare against (NAT 2026-07-18)."""
        if self._output_display_cached is None and self._output_layer() is None:
            self._run_preview(report_status=True)
            self._showing_input = False
            return
        if self._showing_input:
            target = self._output_display_cached
            if target is None:
                layer = self._output_layer()
                target = np.asarray(layer.data)[..., :3] if layer is not None else None
            if target is not None:
                self._mirror_to_gpu_canvas(target, is_output=True)
            self._showing_input = False
            set_status(self._viewer, 'After: showing render')
        else:
            source = self._input_display_cached
            if source is None:
                getter = getattr(self._layers, 'preview_input_layer', None)
                layer = getter() if callable(getter) else None
                source = np.asarray(layer.data)[..., :3] if layer is not None else None
            if source is not None:
                self._mirror_to_gpu_canvas(source)
            self._showing_input = True
            set_status(self._viewer, 'Before: showing input')

    # -- Input / sorting mode (segmented top-bar button) ---------------------
    def toggle_input_mode(self) -> None:
        self.set_input_mode(not self._input_mode)

    def set_input_mode(self, on: bool) -> None:
        """Sorting mode (NAT 2026-07-25): show the camera's embedded thumbnail
        (instant, no decode/render) and keep showing it across roll navigation,
        so a roll can be culled without the ~5s per-frame input load. Leaving
        the mode loads + renders the current frame properly so it can be
        edited."""
        on = bool(on)
        was_on = self._input_mode
        self._input_mode = on
        if on:
            self._input_mode_navigated = False
            self._show_instant_input()
            set_status(self._viewer,
                       'Sorting mode: instant input thumbnails (no edits, no render)')
        else:
            set_status(self._viewer, 'Edit mode')
            # If the user navigated frames while sorting, the loaded image is
            # stale — reload + render the current frame so editing is correct.
            if was_on and self._input_mode_navigated:
                if self._roll.current is not None:
                    self._roll_load_current(restore_state=True)
                elif self._current_input_path:
                    self._reload_current_input()
            else:
                # Didn't navigate: just restore the render/input we already had.
                target = self._output_display_cached
                restore_output = target is not None
                if target is None and self._input_display_cached is not None:
                    target = self._input_display_cached
                if target is not None:
                    self._mirror_to_gpu_canvas(target, is_output=restore_output)
        self._sync_input_mode_button()

    def _reload_current_input(self) -> None:
        """Re-run the real load for the current single (non-roll) image."""
        path = self._current_input_path
        if not path:
            return
        from spektrafilm_gui.roll_session import classify_path
        if classify_path(path) == 'raw':
            self.load_raw_image(path)
        else:
            self.load_input_image(path)

    def _show_instant_input(self):
        """Push the current frame's instant thumbnail to the canvas; returns
        the image shown (or None)."""
        path = self._current_input_path
        if not path:
            if self._input_display_cached is not None:
                self._mirror_to_gpu_canvas(self._input_display_cached)
            return None
        from spektrafilm_gui.instant_preview import load_instant_preview
        image = load_instant_preview(path)
        if image is not None:
            self._mirror_to_gpu_canvas(image)
            return image
        if self._input_display_cached is not None:
            self._mirror_to_gpu_canvas(self._input_display_cached)
        return None

    def _sync_input_mode_button(self) -> None:
        """Keep the segmented top-bar input/edit toggle in sync."""
        button = getattr(self, '_input_mode_button', None)
        if button is None:
            return
        try:
            button.set_input_active(self._input_mode)
        except Exception:
            pass

    # -- Paper-texture finish (NAT 2026-07-26 v1.0.2) ------------------------
    _paper_texture_cache: dict | None = None

    def _active_paper_state(self):
        """The paper state IF the finish is active with at least one loadable
        texture, else None (so default/empty output stays bit-identical)."""
        section = getattr(self._widgets, 'paper', None)
        if section is None:
            return None
        try:
            state = section.get_state()
        except Exception:
            return None
        if not getattr(state, 'active', False) or not getattr(state, 'textures', ()):  # noqa: E501
            return None
        return state

    def _load_paper_texture(self, path: str):
        if self._paper_texture_cache is None:
            self._paper_texture_cache = {}
        if path in self._paper_texture_cache:
            return self._paper_texture_cache[path]
        image = None
        try:
            from PIL import Image as PILImage
            prev = getattr(PILImage, 'MAX_IMAGE_PIXELS', 0)
            if prev is not None:
                PILImage.MAX_IMAGE_PIXELS = None
            with PILImage.open(path) as handle:
                image = np.asarray(handle.convert('RGB'))
        except Exception:
            image = None
        self._paper_texture_cache[path] = image
        return image

    def _paper_texture_list(self, paper):
        textures = []
        for entry in paper.textures:
            arr = self._load_paper_texture(str(entry[0]))
            if arr is None:
                continue
            mode = str(entry[1]) if len(entry) > 1 else 'Multiply'
            opacity = float(entry[2]) if len(entry) > 2 else 1.0
            textures.append((arr, mode, opacity))
        return textures

    def _paper_composite(self, data_uint8, paper) -> np.ndarray:
        """Composite the active textures onto a uint8 RGB display image."""
        from spektrafilm_gui.paper_texture import composite_textures
        textures = self._paper_texture_list(paper)
        if not textures:
            return data_uint8
        base = np.asarray(data_uint8).astype(np.float32) / 255.0
        out = composite_textures(base, textures)
        return np.uint8(np.clip(out, 0.0, 1.0) * 255)

    def apply_paper_finish_to_export(self, buffer: np.ndarray) -> np.ndarray:
        """Export-side paper finish: optional padding-burn border (reuses the
        Display white-padding width) then the texture composite, in the buffer's
        own dtype/space (matches the preview for sRGB/normal encodings; a
        defensible 'texture is part of the scan' for wide-gamut ones). Returns
        the buffer unchanged when the finish is inactive (bit-identical)."""
        paper = self._active_paper_state()
        if paper is None:
            return buffer
        from spektrafilm_gui.paper_texture import composite_textures
        arr = np.asarray(buffer)
        if paper.padding_burn:
            pad = runtime.padding_pixels_for_image(
                arr, float(self._widgets.display.white_padding.value))
            if pad > 0:
                arr = runtime.apply_border_padding(arr, pad, fill=None)  # paper-white
        textures = self._paper_texture_list(paper)
        if not textures:
            return arr
        dtype = arr.dtype
        if np.issubdtype(dtype, np.integer):
            max_value = float(np.iinfo(dtype).max)
            base = arr.astype(np.float32) / max_value
            out = composite_textures(base, textures)
            return (np.clip(out, 0.0, 1.0) * max_value).round().astype(dtype)
        base = arr.astype(np.float32)
        return composite_textures(base, textures).astype(dtype)

    def refresh_paper_finish(self) -> None:
        """Re-composite the paper finish onto the last render, no re-simulation
        (paper is a display finish). Textures may have changed, so drop the
        loaded-texture cache first."""
        self._paper_texture_cache = None
        if self._showing_input or self._input_mode:
            return          # not looking at the render right now
        target = self._output_display_cached
        if target is None:
            layer = self._output_layer()
            target = np.asarray(layer.data)[..., :3] if layer is not None else None
        if target is not None:
            self._mirror_to_gpu_canvas(target, is_output=True)

    def refresh_canvas_padding(self) -> None:
        """Re-push the last displayed image(s) (padding setting changed live).
        The compare quad follows with its own border (white for the input,
        film-base for the negative)."""
        if self._last_canvas_image is not None:
            self._mirror_to_gpu_canvas(self._last_canvas_image,
                                       is_output=self._last_canvas_is_output)
        if self._last_compare is not None and self._gpu_canvas is not None:
            try:
                image, mode = self._last_compare
                self._gpu_canvas.set_compare(self._padded_compare(image, mode))
            except Exception:
                pass

    def _set_or_add_output_layer(
        self,
        image: np.ndarray,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
    ) -> None:
        self._layers.set_or_add_output_layer(
            image,
            float_image=float_image,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            use_display_transform=use_display_transform,
            output_interpolation_mode=self._output_interpolation_mode(),
        )
        self._mirror_to_gpu_canvas(image, is_output=True)

    def _set_or_add_input_stack(
        self,
        image: np.ndarray,
    ) -> None:
        self._update_preview_cache(
            image,
            home_input_stack=True,
            hide_output=True,
        )
        self._update_banner_and_metadata(image)

    def _update_banner_and_metadata(self, image: np.ndarray) -> None:
        banner = getattr(self._widgets, 'banner', None)
        if banner is not None:
            name = os.path.basename(self._current_input_path) if self._current_input_path else ''
            arr = np.asarray(image)
            height = int(arr.shape[0]) if arr.ndim >= 2 else 0
            width = int(arr.shape[1]) if arr.ndim >= 2 else 0
            banner.set_image_info(name, width, height)
        metadata_panel = getattr(self._widgets, 'metadata', None)
        if metadata_panel is not None and self._current_input_path is not None:
            try:
                read_camera_lens = import_module('spektrafilm.utils.io').read_camera_lens_model
                camera, lens = read_camera_lens(read_image_metadata(self._current_input_path))
                metadata_panel.set_camera_lens(camera, lens)
            except Exception:
                pass

    def _update_preview_cache(
        self,
        image: np.ndarray,
        *,
        home_input_stack: bool,
        hide_output: bool,
    ) -> None:
        state = collect_gui_state(widgets=self._widgets)
        preview_image = self._resize_for_preview(image, max_size=state.display.preview_max_size)
        preview_display_image = self._prepare_input_color_preview_image(
            preview_image,
            input_color_space=state.input_image.input_color_space,
            apply_cctf_decoding=state.input_image.apply_cctf_decoding,
        )
        self._current_input_image = image
        self._current_preview_image = preview_image
        # Sorting view (NAT 2026-07-25): cache the input display so 'Show
        # input' can flip to it INSTANTLY (no re-render, no colour transform).
        self._input_display_cached = preview_display_image
        # Phase 16C: a new preview image invalidates the scan-stage cache.
        self._preview_image_version += 1
        self._stage_cache = None
        self._layers.set_or_add_input_preview_layer(
            preview_display_image,
            watermark_source_size=tuple(int(dimension) for dimension in image.shape[:2]),
            white_padding=state.display.white_padding,
            hide_output=hide_output,
            set_active=home_input_stack or self._output_layer() is None,
        )
        if hide_output:
            self._mirror_to_gpu_canvas(preview_display_image)
        if home_input_stack:
            self._home_input_stack()

    def _sync_white_border(self, *, white_padding: float) -> None:
        self._layers.sync_white_border(white_padding=white_padding)

    def _home_input_stack(self) -> None:
        if self._white_border_layer() is None:
            return
        reset_viewer_camera(self._viewer)
        self._set_active_layer(self._white_border_layer())

    def _simulation_input_image(self, *, source_layer_name: str) -> np.ndarray | None:
        if source_layer_name == INPUT_PREVIEW_LAYER_NAME:
            return self._current_preview_image
        if source_layer_name == INPUT_LAYER_NAME:
            return self._current_input_image
        return None

    def _auto_preview_enabled(self) -> bool:
        # Gated on the session's first render: with auto preview on by
        # default, a freshly opened image must stay on the input view until
        # the user explicitly renders once (NAT 2026-07-18).
        if not self._first_render_done:
            return False
        simulation_section = getattr(self._widgets, 'simulation', None)
        auto_preview_value = getattr(simulation_section, 'auto_preview_value', None)
        return bool(auto_preview_value()) if callable(auto_preview_value) else False

    def _run_scheduled_auto_preview(self) -> None:
        self._auto_preview_scheduled = False
        if not self._auto_preview_enabled() or self._current_preview_image is None:
            self._pending_auto_preview = False
            return
        if self._active_simulation_worker is not None or self._active_export_worker is not None:
            self._pending_auto_preview = True
            return
        self._run_preview(report_status=False)

    def _replay_pending_auto_preview(self) -> None:
        if self._pending_live_render:
            # Editor drags that landed while a render was in flight: replay
            # through the live-recipe path (not gated by auto preview).
            self._pending_live_render = False
            self._pending_auto_preview = False
            self._request_live_recipe_render()
            return
        if not self._pending_auto_preview:
            return
        self._pending_auto_preview = False
        self.request_auto_preview()

    def _output_layer(self) -> Any | None:
        return self._layers.output_layer()

    def _set_active_layer(self, layer: Any | None) -> None:
        self._layers.set_active_layer(layer)

    def _output_layer_float_data(self) -> np.ndarray | None:
        output_layer = self._output_layer()
        if output_layer is None:
            return None
        float_data = output_layer.metadata.get(OUTPUT_FLOAT_DATA_KEY)
        if float_data is None:
            return None
        return np.asarray(float_data)

    def _output_layer_render_settings(
        self,
        *,
        default_color_space: str,
        default_cctf_encoding: bool,
    ) -> tuple[str, bool]:
        output_layer = self._output_layer()
        if output_layer is None:
            return default_color_space, default_cctf_encoding
        color_space = output_layer.metadata.get(OUTPUT_COLOR_SPACE_KEY, default_color_space)
        cctf_encoding = output_layer.metadata.get(OUTPUT_CCTF_ENCODING_KEY, default_cctf_encoding)
        return str(color_space), bool(cctf_encoding)

    def _output_interpolation_mode(self) -> str:
        display_section = getattr(self._widgets, 'display', None)
        editor = getattr(display_section, 'output_interpolation', None)
        value = getattr(editor, 'value', None)
        if isinstance(value, str) and value:
            return value
        current_text = getattr(editor, 'currentText', None)
        if callable(current_text):
            text = current_text()
            if isinstance(text, str) and text:
                return text
        return 'spline36'

    @staticmethod
    def _resize_for_preview(image_data: np.ndarray, *, max_size: int) -> np.ndarray:
        return resize_for_preview(image_data, max_size)

    @staticmethod
    def _prepare_input_color_preview_image(
        image_data: np.ndarray,
        *,
        input_color_space: str,
        apply_cctf_decoding: bool,
    ) -> np.ndarray:
        return runtime.prepare_input_color_preview_image(
            image_data,
            input_color_space=input_color_space,
            apply_cctf_decoding=apply_cctf_decoding,
            colour_module=colour,
        )

    @staticmethod
    def _prepare_output_display_image(
        image_data: np.ndarray,
        *,
        output_color_space: str,
        use_display_transform: bool,
        padding_pixels: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        return runtime.prepare_output_display_image(
            image_data,
            output_color_space=output_color_space,
            use_display_transform=use_display_transform,
            padding_pixels=padding_pixels,
            imagecms_module=ImageCms,
            colour_module=colour,
            pil_image_module=PILImage,
        )

    # Phase 16C partial re-render — the two proven-byte-identical cache tiers.
    # SCAN: only scan_finishing changed -> re-run just the scan stage from the
    # cached pre-scan tap (scan_finishing feeds only that stage). DEVELOP: only
    # develop-and-later sections changed -> re-run develop+print+scan from the
    # cached LOG_E_FILM tap (skipping preprocess + the ~54ms spectral expose).
    # halation is EXCLUDED from develop — it is applied in the EXPOSE stage.
    # Each set is a PARTITION-by-exclusion of the gui state, so any unlisted
    # change flips the key and forces a full render (safe by construction).
    _SCAN_ONLY_SECTIONS = ('scan_finishing',)
    _DEVELOP_AND_LATER_SECTIONS = (
        'scan_finishing', 'grain', 'couplers', 'chemistry', 'aging')

    @staticmethod
    def _partial_tiers_eligible(state) -> bool:
        """The partial tiers skip the filming/printing stages that populate
        the scanner black/white-correction reference (a per-render pipeline
        side effect reset on every update_params). When either correction is
        on, force a FULL render so the reference is always computed — else the
        scan stage subtracts None - None (bug 2026-07-22). Corrections default
        off, so the common case keeps the speedup."""
        simulation = getattr(state, 'simulation', None)
        return not (getattr(simulation, 'scan_white_correction', False)
                    or getattr(simulation, 'scan_black_correction', False))

    @classmethod
    def _stage_cache_keys(cls, state, rendered_long_edge: int) -> tuple:
        """Return (scan_key, develop_key). A key changes whenever ANYTHING
        outside its section set does, so a match proves the corresponding tap
        (pre-scan / LOG_E_FILM) is byte-valid to reuse."""
        import json

        from spektrafilm_gui.persistence import gui_state_to_dict

        payload = gui_state_to_dict(state)

        def signature(exclude):
            kept = {k: v for k, v in payload.items() if k not in exclude}
            return (rendered_long_edge,
                    json.dumps(kept, sort_keys=True, default=str))

        return (signature(cls._SCAN_ONLY_SECTIONS),
                signature(cls._DEVELOP_AND_LATER_SECTIONS))

    def _process_image_with_runtime(self, image_data: np.ndarray, params) -> np.ndarray:
        import time as _time

        timing = bool(os.environ.get('SPEKTRAFILM_TIMING'))
        apply_stocks_specifics = (
            self._runtime_simulator is None
            or self._next_runtime_digest_applies_stock_specifics
        )
        t0 = _time.perf_counter()
        digested_params = digest_params(
            params,
            apply_stocks_specifics=apply_stocks_specifics,
        )
        t1 = _time.perf_counter()
        try:
            # Stash the active film's characteristic curves (thread-safe numpy
            # copies) for the Analysis tab; applied on the GUI thread when the
            # render finishes.
            film_data = digested_params.film.data
            self._last_film_curves = (
                np.asarray(film_data.log_exposure).copy(),
                np.asarray(film_data.density_curves).copy(),
            )
        except Exception:
            self._last_film_curves = None
        cache_key = self._pending_stage_cache_keys
        self._pending_stage_cache_keys = None    # consume once
        try:
            built = self._runtime_simulator is None
            if built:
                self._runtime_simulator = runtime_simulator(digested_params)
                self._stage_cache = None        # fresh pipeline: drop stale cache
            else:
                self._runtime_simulator.update_params(digested_params)
            self._next_runtime_digest_applies_stock_specifics = False
            t2 = _time.perf_counter()
            result, mode = self._run_with_stage_cache(image_data, cache_key)
            t3 = _time.perf_counter()
        except Exception:
            self._runtime_simulator = None
            self._stage_cache = None
            raise
        if timing:
            try:
                stage = self._runtime_simulator.format_timings()
            except Exception:
                stage = '(stage timings unavailable)'
            setup = 'build' if built else 'update'
            print(
                f"[spektrafilm-timing] digest={t1 - t0:.2f}s {setup}(LUT)={t2 - t1:.2f}s "
                f"process[{mode}]={t3 - t2:.2f}s\n{stage}",
                flush=True,
            )
        return result

    def _run_with_stage_cache(self, image_data, keys):
        """Phase 16C: return (result, mode). Reuse the latest byte-valid tap:
        the pre-scan tap when only scan_finishing changed, else the LOG_E_FILM
        tap when only develop-and-later sections changed, else a full render
        (split at both taps to refill the cache for free). Every partial path
        is byte-identical to a cold full render (test-pinned)."""
        from spektrafilm.runtime.topology import Tap

        sim = self._runtime_simulator
        if keys is None:
            # Exports / scans / live-recipe: plain end-to-end, no caching.
            return sim.process(image_data), 'full'

        scan_key, develop_key = keys
        pre_tap = sim.pre_scan_tap()
        cache = self._stage_cache
        res = image_data.shape[:2]

        # 1. SCAN tier: only scan_finishing changed -> re-run just the scan
        #    stage from the cached pre-scan densities.
        if (cache is not None and cache.get('scan_key') == scan_key
                and cache.get('pre_tap') == pre_tap
                and cache.get('pre') is not None
                and cache['pre'].shape[:2] == res):
            return sim.process(cache['pre'], inject=pre_tap), 'scan'

        # 2. DEVELOP tier: only develop-and-later sections changed -> re-run
        #    develop+print+scan from the cached LOG_E_FILM, skipping the
        #    expensive preprocess + spectral expose. Restore the one preprocess
        #    side effect (pixel_size_um) the develop stage reads.
        if (cache is not None and cache.get('develop_key') == develop_key
                and cache.get('log_e_film') is not None
                and cache['log_e_film'].shape[:2] == res
                and cache.get('pixel_size_um') is not None):
            sim.restore_pixel_size_um(cache['pixel_size_um'])
            pre = sim.process(cache['log_e_film'], inject=Tap.LOG_E_FILM,
                              collect=pre_tap)
            result = sim.process(pre, inject=pre_tap)
            # refresh the scan tier under the (now-current) scan_key
            self._stage_cache = {**cache, 'pre': pre, 'pre_tap': pre_tap,
                                 'scan_key': scan_key}
            return result, 'develop'

        # 3. FULL render, split at BOTH taps so both tiers refill for free.
        log_e = sim.process(image_data, collect=Tap.LOG_E_FILM)
        pixel_size_um = sim.pixel_size_um()
        pre = sim.process(log_e, inject=Tap.LOG_E_FILM, collect=pre_tap)
        result = sim.process(pre, inject=pre_tap)
        self._stage_cache = {
            'scan_key': scan_key, 'develop_key': develop_key,
            'log_e_film': log_e, 'pixel_size_um': pixel_size_um,
            'pre': pre, 'pre_tap': pre_tap}
        return result, 'full'

    def _set_display_transform_checked(self, enabled: bool) -> None:
        display_section = getattr(self._widgets, 'display', None)
        toggle = getattr(display_section, 'use_display_transform', None)
        if toggle is None:
            return

        block_signals = getattr(toggle, 'blockSignals', None)
        set_checked = getattr(toggle, 'setChecked', None)
        if not callable(set_checked):
            return

        previous_block_state = None
        if callable(block_signals):
            previous_block_state = block_signals(True)
        try:
            set_checked(enabled)
        finally:
            if callable(block_signals):
                block_signals(bool(previous_block_state))

    def _sync_canvas_background(self) -> None:
        # The Display toggle is retired: restore the persisted nav-bar swatch
        # choice, defaulting to the Dark grey surround (NAT 2026-07-12).
        from spektrafilm_gui.napari_layout import (
            DEFAULT_CANVAS_BACKGROUND, set_canvas_background_color)
        color = (QSettings('spektrafilm', 'spektrafilm').value('canvas_background', '')
                 or DEFAULT_CANVAS_BACKGROUND)
        set_canvas_background_color(self._viewer, str(color))

    def _execute_simulation_request(self, request: SimulationRequest) -> SimulationResult:
        return runtime.execute_simulation_request(
            request,
            run_simulation_fn=self._process_image_with_runtime,
            prepare_output_display_image_fn=self._prepare_output_display_image,
        )

    # Grain-in-preview (NAT 2026-07-26 v1.0.2): a session toggle. Off = the
    # LIVE preview skips grain for fast scrubbing; Scan/export always grain.
    _grain_in_preview = True

    def _configure_simulation_params(self, params, *, source_layer_name: str,
                                     rendered_image=None):
        settings = getattr(params, 'settings', None)
        is_preview = source_layer_name == INPUT_PREVIEW_LAYER_NAME
        if settings is not None and hasattr(settings, 'preview_mode'):
            settings.preview_mode = is_preview
        # Preview-only grain skip (never touches Scan/export, which render from
        # the full input layer). params is built fresh per render, so disabling
        # here does not mutate the stored state.
        if is_preview and not self._grain_in_preview:
            try:
                params.film_render.grain.active = False
            except Exception:
                pass
        # NAT 2026-07-22 (macOS-feel): preview renders tell the 'preview'
        # grain tier the full/rendered width ratio so its noise carries the
        # FULL-RESOLUTION strength (exports keep the neutral 1.0). The
        # rendered width is the ACTUAL frame — reduced further while dragging
        # (Phase 16B) — so grain strength stays constant across the ladder.
        if source_layer_name == INPUT_PREVIEW_LAYER_NAME:
            try:
                full = self._simulation_input_image(source_layer_name=INPUT_LAYER_NAME)
                rendered = rendered_image if rendered_image is not None else \
                    self._simulation_input_image(source_layer_name=INPUT_PREVIEW_LAYER_NAME)
                if full is not None and rendered is not None and rendered.shape[1]:
                    params.film_render.grain.preview_scale = max(
                        1.0, float(full.shape[1]) / float(rendered.shape[1]))
            except Exception:
                pass
        return params

    # Phase 16B: drag frames never go below this long-edge (px) — past this
    # the pipeline overhead dominates and the render stops getting faster.
    _DRAG_MIN_LONG_EDGE = 384

    def _on_drag_ended(self) -> None:
        """Phase 16B: the last slider/knob/fader drag ended. If the stream
        rendered reduced-resolution frames, render once at the full preview
        size (coalesced by the scheduler if a frame is still in flight)."""
        if not self._last_preview_reduced:
            return
        self._last_preview_reduced = False
        self.request_auto_preview()

    def _start_simulation(self, *, source_layer_name: str, mode_label: str, report_status: bool = True) -> None:
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'Simulation already running')
            return
        if self._active_export_worker is not None:
            set_status(self._viewer, 'Export running; wait before rendering.')
            return

        image_data = self._simulation_input_image(source_layer_name=source_layer_name)
        if image_data is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Load an input image before running the simulation.')
            return

        # Phase 16B: a live drag streams reduced-resolution frames; the full
        # preview frame follows on release (via _on_drag_ended). Only the
        # preview path scales down — Scan always renders full.
        from spektrafilm_gui.widget_slider import drag_active
        reduced = False
        if source_layer_name == INPUT_PREVIEW_LAYER_NAME and drag_active():
            long_edge = int(max(image_data.shape[:2]))
            target = int(round(long_edge / self._drag_preview_divisor))
            if self._DRAG_MIN_LONG_EDGE <= target < long_edge:
                image_data = self._resize_for_preview(image_data, max_size=target)
                reduced = True
        self._last_preview_reduced = reduced

        state = collect_gui_state(widgets=self._widgets)
        self._sync_white_border(white_padding=state.display.white_padding)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=source_layer_name,
            rendered_image=image_data,
        )
        # Phase 16C: the scan-stage fast path is eligible only for PREVIEW
        # renders with no live-recipe override (the recipe isn't in gui
        # state). The key changes whenever ANYTHING but scan_finishing does,
        # so a match proves the pre-scan tap is byte-valid.
        if (source_layer_name == INPUT_PREVIEW_LAYER_NAME
                and self._live_stock_recipe is None
                and self._partial_tiers_eligible(state)):
            self._pending_stage_cache_keys = self._stage_cache_keys(
                state, int(max(image_data.shape[:2])))
        else:
            self._pending_stage_cache_keys = None
        if self._live_stock_recipe is not None:
            # Phase 10C: the stock editor's UNSAVED recipe replaces the film
            # profile for this render (live preview while handles drag).
            try:
                from spektrafilm.profiles.custom_stocks import bake_recipe_into_profile
                from spektrafilm.profiles.io import load_profile as _load_profile
                params.film = bake_recipe_into_profile(
                    _load_profile(self._live_stock_recipe.base_stock),
                    self._live_stock_recipe,
                )
            except (OSError, ValueError, NotImplementedError):
                set_status(self._viewer, 'Live stock preview failed; rendering the saved stock')

        self._first_render_done = True
        image = np.double(image_data)
        request = SimulationRequest(
            mode_label=mode_label,
            image=image,
            params=params,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
        )

        worker = runtime.SimulationWorker(request, execute_request=self._execute_simulation_request)
        worker.signals.finished.connect(self._on_simulation_finished)
        worker.signals.failed.connect(self._on_simulation_failed)
        self._active_simulation_worker = worker
        self._active_simulation_label = mode_label
        self._active_simulation_reports_status = report_status
        self._set_simulation_controls_enabled(False)
        if report_status:
            set_status(self._viewer, f'Computing {mode_label.lower()}...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        report_status = self._active_simulation_reports_status
        self._active_simulation_worker = None
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        # A completed render always lands on the 'after' side and leaves the
        # sorting mode (the user asked to see a result, so show it).
        self._showing_input = False
        if self._input_mode:
            self._input_mode = False
            self._sync_input_mode_button()
        self._output_display_cached = result.display_image
        self._set_or_add_output_layer(
            result.display_image,
            float_image=result.float_image,
            output_color_space=result.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=result.use_display_transform,
        )
        self._update_analysis(result.display_image)
        if report_status:
            set_status(self._viewer, f'{result.mode_label} completed. {result.status_message}')
        self._replay_pending_auto_preview()

    def _custom_metadata_tags(self) -> dict:
        panel = getattr(self._widgets, 'metadata', None)
        getter = getattr(panel, 'custom_tags', None)
        tags = {}
        if callable(getter):
            try:
                tags = dict(getter())
            except Exception:
                tags = {}
        # The CURRENT frame's star rating (batch export merges per frame).
        rating = self._current_rating()
        if rating:
            tags['Exif.Image.Rating'] = str(rating)
        return tags

    def _update_analysis(self, display_image) -> None:
        analysis = getattr(self._widgets, 'analysis', None)
        if analysis is None:
            return
        try:
            analysis.set_image(display_image)
            if self._last_film_curves is not None:
                log_exposure, density_curves = self._last_film_curves
                analysis.set_curves(log_exposure, density_curves)
        except Exception:
            pass

    def _on_simulation_failed(self, message: str) -> None:
        self._active_simulation_worker = None
        mode_label = self._active_simulation_label or 'Simulation'
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        QMessageBox.critical(dialog_parent(self._viewer), 'Run simulation', f'Simulation failed.\n\n{message}')
        set_status(self._viewer, f'{mode_label} failed')
        self._replay_pending_auto_preview()

    def _set_simulation_controls_enabled(self, enabled: bool) -> None:
        simulation_section = getattr(self._widgets, 'simulation', None)
        if simulation_section is None:
            return
        for button_name in ('preview_button', 'scan_button', 'save_button'):
            button = getattr(simulation_section, button_name, None)
            set_enabled = getattr(button, 'setEnabled', None)
            if callable(set_enabled):
                set_enabled(enabled)

    def _run_simulation(self, *, source_layer_name: str) -> None:
        image_data = self._simulation_input_image(source_layer_name=source_layer_name)
        if image_data is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Load an input image before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        self._sync_white_border(white_padding=state.display.white_padding)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=source_layer_name,
        )

        image = np.double(image_data)
        scan = self._process_image_with_runtime(image, params)
        scan_display, display_status = self._prepare_output_display_image(
            scan,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
        )
        self._set_or_add_output_layer(
            scan_display,
            float_image=scan,
            output_color_space=state.simulation.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=state.display.use_display_transform,
        )
        set_status(self._viewer, display_status)

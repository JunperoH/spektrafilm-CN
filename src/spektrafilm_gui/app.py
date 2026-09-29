from dataclasses import dataclass, fields
from importlib import import_module
from typing import Any, Callable, cast

import os
import sys
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.napari_layout import (
    build_controls_panel,
    build_left_panel,
    build_main_window,
    set_canvas_background_color,
    show_viewer_window,
)
from spektrafilm_gui.persistence import load_default_gui_state
from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES, apply_gui_state
from spektrafilm_gui.theme_palette import GRAY_1, GRAY_2, GRAY_3, TEXT_DIM, TEXT_MAIN, TEXT_SELECTION_BG, WINDOW_BG
from spektrafilm_gui.widgets import WidgetBundle, create_widget_bundle
from spektrafilm.utils.numba_warmup import warmup

QThreadPool = getattr(QtCore, 'QThreadPool')
QRunnable = getattr(QtCore, 'QRunnable')
QTimer = getattr(QtCore, 'QTimer')

_background_warmup_started = False
_background_warmup_scheduled = False
_background_warmup_pool: Any | None = None
WARMUP_IMAGE_SHAPE = (16, 16, 3)

@dataclass(slots=True)
class GuiApp:
    viewer: Any
    widgets: WidgetBundle
    controller: GuiController
    main_window: QtWidgets.QMainWindow


def _warmup_launch_input_path(gui_state: object | None = None) -> None:
    state = load_default_gui_state() if gui_state is None else gui_state
    try:
        colour_module = import_module('colour')
        controller_runtime = import_module('spektrafilm_gui.controller_runtime')
        import_module('spektrafilm.utils.io')
        import_module('spektrafilm.utils.preview')
    except (AttributeError, ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError):
        return

    warmup_image = np.full(WARMUP_IMAGE_SHAPE, 0.18, dtype=np.float64)
    try:
        controller_runtime.prepare_input_color_preview_image(
            warmup_image,
            input_color_space=state.input_image.input_color_space,
            apply_cctf_decoding=state.input_image.apply_cctf_decoding,
            colour_module=colour_module,
        )
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
        return


def _warmup_full_gui() -> None:
    gui_state = load_default_gui_state()
    warmup()

    colour_module = import_module('colour')
    pil_image_module = import_module('PIL.Image')
    imagecms_module = import_module('PIL.ImageCms')
    controller_runtime = import_module('spektrafilm_gui.controller_runtime')
    params_mapper = import_module('spektrafilm_gui.params_mapper')
    runtime_api = import_module('spektrafilm.runtime.api')
    import_module('spektrafilm.utils.io')
    import_module('spektrafilm.utils.preview')
    import_module('spektrafilm.utils.raw_file_processor')
    # Stock editor deps (NAT 2026-07-20 'takes a while to load'): pay the
    # scipy.optimize + dialog import cost here instead of on the first
    # Edit / 'Custom...' click (mainly matters in the frozen build).
    import_module('scipy.optimize')
    import_module('spektrafilm_gui.stock_editor')

    warmup_image = np.full(WARMUP_IMAGE_SHAPE, 0.18, dtype=np.float64)
    controller_runtime.prepare_input_color_preview_image(
        warmup_image,
        input_color_space=gui_state.input_image.input_color_space,
        apply_cctf_decoding=gui_state.input_image.apply_cctf_decoding,
        colour_module=colour_module,
    )

    params = params_mapper.build_params_from_state(gui_state)
    simulator = runtime_api.Simulator(runtime_api.digest_params(params))
    scan = np.asarray(simulator.process(warmup_image), dtype=np.float32)

    # Force the display path once as part of startup so the first preview avoids lazy import/setup cost.
    controller_runtime.prepare_output_display_image(
        scan,
        output_color_space=gui_state.simulation.output_color_space,
        use_display_transform=True,
        imagecms_module=imagecms_module,
        colour_module=colour_module,
        pil_image_module=pil_image_module,
    )


class _WarmupTask(QRunnable):
    def __init__(self, *, warmup_fn: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._warmup_fn = _warmup_full_gui if warmup_fn is None else warmup_fn

    def run(self) -> None:
        try:
            self._warmup_fn()
        except (AttributeError, ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError):
            return


def _build_app_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    # NAT 2026-07-22: WINDOW_BG, not black — scroll-area viewports autofill
    # this role, and it shows through now that plain widgets paint nothing.
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(WINDOW_BG))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(GRAY_1))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(GRAY_2))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(GRAY_1))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(GRAY_1))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Mid, QtGui.QColor(GRAY_3))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(TEXT_SELECTION_BG))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(TEXT_MAIN))
    placeholder_role = getattr(QtGui.QPalette, 'PlaceholderText', None)
    if placeholder_role is not None:
        palette.setColor(placeholder_role, QtGui.QColor(TEXT_DIM))

    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor(TEXT_DIM))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(TEXT_DIM))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(TEXT_DIM))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText, QtGui.QColor(TEXT_DIM))
    return palette


def _apply_app_palette() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.setPalette(_build_app_palette())


def _create_viewer() -> Any:
    # Phase 6C: napari removed. The HeadlessViewer shim is the plain image/
    # layer model; the wgpu canvas (or the CPU fallback) is the display.
    from spektrafilm_gui.viewer_shim import HeadlessViewer
    return HeadlessViewer()


def _create_widgets() -> WidgetBundle:
    return create_widget_bundle()


def _start_background_warmup(
    *,
    thread_pool: Any | None = None,
    thread_pool_factory: Callable[[], Any] | None = None,
    task_factory: Callable[[], QRunnable] = _WarmupTask,
) -> None:
    global _background_warmup_started, _background_warmup_scheduled, _background_warmup_pool
    if _background_warmup_started:
        return
    _background_warmup_started = True
    _background_warmup_scheduled = False
    if thread_pool is None:
        if _background_warmup_pool is None:
            pool_factory = QThreadPool if thread_pool_factory is None else thread_pool_factory
            _background_warmup_pool = pool_factory()
            set_max_thread_count = getattr(_background_warmup_pool, 'setMaxThreadCount', None)
            if callable(set_max_thread_count):
                set_max_thread_count(1)
        pool = _background_warmup_pool
    else:
        pool = thread_pool
    pool.start(task_factory())


def _schedule_background_warmup(
    *,
    single_shot_fn: Callable[[int, Callable[[], None]], None] | None = None,
) -> None:
    global _background_warmup_scheduled
    if _background_warmup_started or _background_warmup_scheduled:
        return
    _background_warmup_scheduled = True
    scheduler = QTimer.singleShot if single_shot_fn is None else single_shot_fn
    scheduler(0, _start_background_warmup)


def _connect_auto_preview_signal(widget: Any, callback: Callable[..., None]) -> None:
    editors = getattr(widget, '_editors', None)
    if editors is not None:
        for editor in editors:
            _connect_auto_preview_signal(editor, callback)
        return

    for signal_name in ('toggled', 'currentTextChanged', 'valueChanged'):
        signal = getattr(widget, signal_name, None)
        if signal is not None and hasattr(signal, 'connect'):
            signal.connect(callback)
            return


def connect_auto_preview_signals(controller: GuiController, widgets: WidgetBundle) -> None:
    for section_name in GUI_STATE_SECTION_NAMES:
        # load_rgb settings apply at import time: the section re-imports the
        # file itself (FilePickerSection.schedule_reprocess) and that reload
        # requests its own preview, so wiring it here would only re-render the
        # unchanged stored image.
        if section_name == 'load_rgb':
            continue
        section = getattr(widgets, section_name, None)
        if section is None:
            continue

        state_cls = getattr(section, '_state_cls', None)
        if state_cls is None:
            continue

        for field_info in fields(state_cls):
            if section_name == 'display' and field_info.name in {'preview_max_size', 'output_interpolation'}:
                continue
            _connect_auto_preview_signal(getattr(section, field_info.name), controller.request_auto_preview)

    widgets.simulation.bottom_auto_preview.toggled.connect(controller.request_auto_preview)
    # Grain-in-preview session toggle (NAT 2026-07-26): off = fast scrub.
    widgets.simulation.bottom_grain_preview.toggled.connect(controller.set_grain_in_preview)
    # Scan-film / scan-for-print are VIEW switches: they re-render even when
    # auto preview is off (see controller.request_view_refresh).
    widgets.simulation.bottom_scan_film.toggled.connect(controller.request_view_refresh)
    widgets.simulation.bottom_scan_for_print.toggled.connect(controller.request_view_refresh)


def connect_controller_signals(controller: GuiController, widgets: WidgetBundle) -> None:
    widgets.filepicker.load_requested.connect(controller.load_input_image)
    widgets.load_raw.load_requested.connect(controller.load_raw_image)
    widgets.roll.files_added.connect(controller.roll_add_files)
    widgets.roll.file_activated.connect(controller.roll_activate)
    widgets.roll.remove_requested.connect(controller.roll_remove)
    widgets.roll.clear_requested.connect(controller.roll_clear)
    widgets.roll.apply_look_requested.connect(controller.roll_apply_look_to_all)
    widgets.roll.analyze_requested.connect(controller.roll_analyze)
    widgets.roll.apply_roll_exposure_requested.connect(controller.roll_apply_exposure)
    widgets.roll.save_roll_requested.connect(controller.roll_save_to_file)
    widgets.roll.open_roll_requested.connect(controller.roll_open_from_file)
    # White padding is baked into the canvas mirror: re-push on edit (live).
    widgets.display.white_padding.valueChanged.connect(
        lambda _value: controller.refresh_canvas_padding())
    # Paper texture is a display finish: re-composite on edit, no re-simulate.
    widgets.paper.changed.connect(controller.refresh_paper_finish)
    widgets.film_strip.file_activated.connect(controller.roll_activate)
    widgets.simulation.film_stock.textActivated.connect(controller.apply_profile_defaults)
    widgets.simulation.print_paper.textActivated.connect(controller.apply_profile_defaults)
    # Phase 10C: 'Custom...' creator item + the Edit button open the editor.
    widgets.simulation.film_stock.create_custom_requested.connect(controller.open_stock_editor)
    widgets.simulation.edit_stock_requested.connect(controller.open_stock_editor)
    # Phase 14: preset browser
    widgets.gui_config.preset_selected.connect(controller.apply_preset)
    widgets.gui_config.save_preset_requested.connect(controller.save_current_as_preset)
    widgets.gui_config.update_preset_requested.connect(controller.update_preset)
    widgets.gui_config.delete_preset_requested.connect(controller.delete_preset)
    widgets.gui_config.save_current_as_default_requested.connect(controller.save_current_as_default)
    widgets.gui_config.save_current_to_file_requested.connect(controller.save_current_state_to_file)
    widgets.gui_config.load_from_file_requested.connect(controller.load_state_from_file)
    widgets.gui_config.restore_factory_default_requested.connect(controller.restore_factory_default)
    widgets.gui_config.export_lut_requested.connect(controller.export_lut)
    widgets.gui_config.open_logs_requested.connect(controller.open_logs)
    widgets.metadata.rating_changed.connect(controller.set_current_rating)
    widgets.scan_frame.tool_changed.connect(controller.set_crop_tool)
    # Phase 9: LR crop extras — reset, orientation swap, ratio re-fit
    widgets.scan_frame.crop_reset_requested.connect(controller.reset_crop)
    widgets.scan_frame.orientation_swap_requested.connect(controller.swap_crop_orientation)
    widgets.scan_frame.ratio_changed.connect(lambda _text: controller.refit_crop_to_ratio())
    widgets.scan_finishing.auto_levels_requested.connect(controller.scan_auto_levels)
    widgets.simulation.preview_requested.connect(controller.run_preview)
    widgets.simulation.scan_requested.connect(controller.run_scan)
    widgets.simulation.save_requested.connect(controller.save_output_layer)
    widgets.export.export_requested.connect(controller.export_output)
    widgets.export.export_roll_requested.connect(controller.export_roll)
    widgets.gpu_backend.gpu_backend_toggled.connect(controller.set_gpu_backend)
    # Populate the GPU-backend label once the event loop is running (after napari
    # is up), so it reports the real device that initializes inside the GUI.
    QTimer.singleShot(0, controller.refresh_gpu_status)
    widgets.display.use_display_transform.toggled.connect(controller.report_display_transform_status)
    widgets.display.output_interpolation.currentTextChanged.connect(controller.set_output_interpolation_mode)
    widgets.display.update_preview_requested.connect(controller.refresh_preview_cache)
    widgets.display.update_preview_requested.connect(controller.request_auto_preview)
    connect_auto_preview_signals(controller, widgets)


def gray_18_canvas_enabled(widgets: WidgetBundle) -> bool:
    # The Display toggle was retired (2026-07-10): 18% gray is the startup
    # default; the nav-bar swatches (QSettings-persisted) override after.
    return True


def initialize_controller(
    *,
    viewer: Any,
    widgets: WidgetBundle,
    controller_cls: type[GuiController] = GuiController,
    connect_signals_fn: Callable[[GuiController, WidgetBundle], None] = connect_controller_signals,
) -> GuiController:
    controller = controller_cls(viewer=viewer, widgets=widgets)
    controller.sync_display_transform_availability(report_status=False)
    show_startup_placeholder = getattr(controller, 'show_startup_placeholder', None)
    if callable(show_startup_placeholder):
        show_startup_placeholder()
    connect_signals_fn(controller, widgets)
    return controller


def _build_top_bar_controls(widgets: WidgetBundle) -> Any:
    """NAT 2026-07-22: viewer top-bar strip, LEFT side — preview max size
    first (arrow stepper, no slider), then white padding (slider), then
    update. The editors are the Display section's own (state round-trips
    unchanged)."""
    strip = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(strip)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    # Input / edit sorting toggle sits to the LEFT of the preview setting
    # (NAT 2026-07-25). Wired to the controller in build_main_window_for_app.
    from spektrafilm_gui.widget_primitives import InputModeToggle
    layout.addWidget(InputModeToggle())
    layout.addSpacing(14)
    for text, editor, width in (
            ('preview max size', widgets.display.preview_max_size, 110),
            ('white padding', widgets.display.white_padding, 190)):
        label = QtWidgets.QLabel(text)
        label.setStyleSheet('color: #8d8d8d;')
        layout.addWidget(label)
        editor.setMaximumWidth(width)
        layout.addWidget(editor)
        layout.addSpacing(14)   # NAT 2026-07-22: 'more space' between groups
    layout.addWidget(widgets.display.update_preview_button)
    return strip


def build_main_window_for_app(
    *,
    viewer: Any,
    widgets: WidgetBundle,
    controller: GuiController | None = None,
    build_controls_panel_fn: Callable[[Any, WidgetBundle], Any] = build_controls_panel,
    build_main_window_fn: Callable[[Any, Any], Any] = build_main_window,
) -> Any:
    controls_panel = build_controls_panel_fn(viewer, widgets)
    left_panel = build_left_panel(widgets)
    film_strip = getattr(widgets, 'film_strip', None)
    banner = getattr(widgets, 'banner', None)
    if controller is None:
        return build_main_window_fn(viewer, controls_panel, left_panel=left_panel, banner=banner, film_strip=film_strip)
    # Phase 6C: the canvas IS the display. wgpu when a device exists, the
    # software QLabel fallback otherwise (no-GPU machines stay usable).
    from spektrafilm_gui.gpu_canvas import create_cpu_canvas_widget, create_gpu_canvas_widget
    gpu_canvas = create_gpu_canvas_widget()
    if gpu_canvas is None:
        gpu_canvas = create_cpu_canvas_widget()
    controller.set_gpu_canvas(gpu_canvas)
    viewer._reset_view_hook = gpu_canvas.reset_view
    main_window = build_main_window_fn(
        viewer,
        controls_panel,
        left_panel=left_panel,
        banner=banner,
        top_bar_controls=_build_top_bar_controls(widgets),
        film_strip=film_strip,
        gpu_canvas=gpu_canvas,
        on_rotate_ccw=controller.rotate_input_image_counterclockwise,
        on_rotate_cw=controller.rotate_input_image_clockwise,
        on_prev_frame=lambda: controller.roll_step(-1),
        on_next_frame=lambda: controller.roll_step(+1),
        on_background_color=controller.set_canvas_background_color,
        on_mirror_horizontal=controller.mirror_input_image_horizontal,
        on_mirror_vertical=controller.mirror_input_image_vertical,
        on_scan=controller.run_scan,
        on_save=controller.save_output_layer,
        on_export=controller.export_output,
        on_fullscreen=controller.show_fullscreen_output,
        on_compare=controller.set_compare_mode,
        on_rating_filter=controller.set_rating_filter,
    )
    # Sorting mode: wire the segmented top-bar input/edit toggle to the
    # controller, and hand the controller a reference so a completed render
    # (or Ctrl+R) can reflect the state (NAT 2026-07-25).
    input_mode_toggle = main_window.findChild(QtWidgets.QWidget, 'inputModeToggle')
    controller._input_mode_button = input_mode_toggle
    if input_mode_toggle is not None:
        input_mode_toggle.toggled.connect(controller.set_input_mode)
    # Restore the persisted navigation-bar background swatch; nothing
    # persisted = the Dark grey default (NAT 2026-07-12; was black).
    from spektrafilm_gui.napari_layout import DEFAULT_CANVAS_BACKGROUND
    persisted_background = QtCore.QSettings('spektrafilm', 'spektrafilm').value('canvas_background', '')
    set_canvas_background_color(
        viewer, str(persisted_background) if persisted_background else DEFAULT_CANVAS_BACKGROUND)
    _install_shortcuts(main_window, controller)
    controller.history_capture_baseline()
    return main_window


def _install_shortcuts(main_window: QtWidgets.QMainWindow, controller: GuiController) -> None:
    """Application keyboard shortcuts. Letters without modifiers are avoided
    so typing in metadata/path fields can never trigger an action."""
    shortcut_cls = getattr(QtGui, 'QShortcut', None) or getattr(QtWidgets, 'QShortcut')

    def bind(sequence: str, callback) -> None:
        shortcut = shortcut_cls(QtGui.QKeySequence(sequence), main_window)
        shortcut.setContext(QtCore.Qt.WindowShortcut)
        shortcut.activated.connect(callback)

    def cycle_compare() -> None:
        combo = main_window.findChild(QtWidgets.QComboBox, 'compareCombo')
        if combo is not None:
            combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

    bind('Ctrl+Z', controller.undo)
    bind('Ctrl+Shift+Z', controller.redo)
    bind('Ctrl+Y', controller.redo)                     # Windows-classic redo
    bind('F5', controller.run_preview)
    bind('F6', controller.run_scan)
    bind('Ctrl+F', controller.toggle_scan_film)         # toggle 'scan film' (negative vs print)
    bind('Ctrl+R', controller.toggle_preview_view)      # before/after: switch input <-> preview
    bind('Ctrl+P', controller.show_fullscreen_output)
    bind('F11', controller.show_fullscreen_output)
    bind('Ctrl+D', cycle_compare)                       # cycles off/input/negative
    bind('Ctrl+E', controller.export_output)
    bind('Ctrl+Right', lambda: controller.roll_step(+1))
    bind('Ctrl+Left', lambda: controller.roll_step(-1))
    bind('Ctrl+Del', controller.roll_remove_current)    # drop the current frame
    bind('Ctrl+Shift+C', controller.copy_current_look)  # copy the look...
    bind('Ctrl+Shift+V', controller.paste_look)         # ...paste onto this frame
    bind('Ctrl+O', controller.cycle_canvas_guides)      # grid/thirds/golden/diagonals/triangle/spiral/center
    bind('Shift+O', controller.flip_canvas_guides)      # flip triangle / golden-spiral orientation
    bind('Ctrl+L', controller.cycle_canvas_background)  # cycle the three swatches
    # Star ratings: Ctrl+1..5 rate the current frame, Ctrl+0 clears.
    for stars in range(6):
        bind(f'Ctrl+{stars}', lambda s=stars: controller.set_current_rating(s))
    # Darkroom filter nudges: print Y/M filter shifts in 0.05 CC steps.
    # (Ctrl+Shift since 2026-07-12: plain Ctrl+digits belong to the ratings.)
    bind('Ctrl+Shift+8', lambda: controller.adjust_print_filter_shift('y', +0.05))
    bind('Ctrl+Shift+5', lambda: controller.adjust_print_filter_shift('y', -0.05))
    bind('Ctrl+Shift+9', lambda: controller.adjust_print_filter_shift('m', +0.05))
    bind('Ctrl+Shift+6', lambda: controller.adjust_print_filter_shift('m', -0.05))
    # Cheat sheet listing all of the above (toggles).
    from spektrafilm_gui.shortcut_help import toggle_shortcut_help
    bind('F1', lambda: toggle_shortcut_help(main_window))
    bind('Ctrl+?', lambda: toggle_shortcut_help(main_window))


def create_app() -> GuiApp:
    # POINT decimals app-wide regardless of the system locale (NAT
    # 2026-07-12): the C locale is the default for every Qt widget built
    # after this (the numeric editors also pin it individually).
    QtCore.QLocale.setDefault(QtCore.QLocale.c())
    # napari used to construct the QApplication implicitly; own it now.
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv if sys.argv else ['spektrafilm'])
    viewer = _create_viewer()
    _apply_app_palette()
    widgets = _create_widgets()
    gui_state = load_default_gui_state()
    apply_gui_state(gui_state, widgets=widgets)
    _warmup_launch_input_path(gui_state)
    controller = initialize_controller(viewer=viewer, widgets=widgets)
    main_window = build_main_window_for_app(viewer=viewer, widgets=widgets, controller=controller)
    _schedule_background_warmup()
    return GuiApp(
        viewer=viewer,
        widgets=widgets,
        controller=controller,
        main_window=main_window,
    )


def _get_splash_path() -> str:
    if getattr(sys, '__compiled__', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'splash.png')


def _show_splash(qapp) -> 'QtWidgets.QSplashScreen | None':
    try:
        if qapp is None:
            return None
        pixmap = QtGui.QPixmap(_get_splash_path())
        if pixmap.isNull():
            return None
        screen = qapp.primaryScreen()
        if screen is not None:
            pixmap.setDevicePixelRatio(screen.devicePixelRatio())
        flags = getattr(QtCore.Qt, 'WindowStaysOnTopHint', None) or getattr(QtCore.Qt.WindowType, 'WindowStaysOnTopHint', 0)
        splash = QtWidgets.QSplashScreen(pixmap, flags)
        splash.show()
        qapp.processEvents()
        return splash
    except Exception:
        return None


class _RenderCanvasNoiseFilter:
    """Suppress rendercanvas's benign call-later teardown race: its scheduling
    thread can fire a queued draw callback after the Qt helper object died
    ('Signal source has been deleted'). The draw is simply dropped and the
    next frame reschedules — nothing to act on, so keep the log clean."""

    def filter(self, record) -> bool:  # noqa: A003 - logging API name
        return 'Signal source has been deleted' not in record.getMessage()


def main():
    from spektrafilm_gui.qt_translation import install_qt_chinese
    from spektrafilm_gui.localization_gpu_zh_cn import retranslate_widget_tree
    # Phase 6C: plain Qt event loop (napari.run removed). Splash, maximize and
    # pyi_splash behavior preserved exactly (protected app.py contract).
    import logging
    try:  # rotating file log + crash capture (the windowed exe has no console)
        from spektrafilm_gui.app_logging import install_qt_message_logging, setup_file_logging
        setup_file_logging()
        install_qt_message_logging()   # Qt warnings -> log file, benign noise demoted
    except Exception:
        pass
    logging.getLogger('rendercanvas').addFilter(_RenderCanvasNoiseFilter())
    qapp = QtWidgets.QApplication.instance()
    if qapp is None:
        qapp = QtWidgets.QApplication(sys.argv)
    install_qt_chinese()
    splash = _show_splash(qapp)
    app = create_app()
    retranslate_widget_tree(app.main_window)
    show_viewer_window(app.viewer)
    try:
        app.main_window.showMaximized()
    except Exception:
        pass
    if splash is not None:
        splash.finish(app.main_window)
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass
    # Phase 8B: .sfroll double-click support, AROUND the protected splash /
    # maximize / pyi_splash sequence above (which is untouched). Frozen runs
    # self-heal the per-user file association (no-op from source); a roll
    # path on argv opens once the event loop is up. No path = normal launch.
    roll_path = None
    try:
        from spektrafilm_gui.file_association import (
            register_sfroll_association, sfroll_path_from_argv)
        register_sfroll_association()
        roll_path = sfroll_path_from_argv(sys.argv[1:])
        if roll_path:
            QtCore.QTimer.singleShot(
                0, lambda: app.controller.open_roll_path(roll_path))
    except Exception:
        pass
    # Roll autosave + crash recovery: a leftover recovery file means the last
    # session crashed — offer to restore it (skipped when a roll came on
    # argv); autosave runs every few minutes; a clean exit removes the file.
    try:
        if not roll_path:
            QtCore.QTimer.singleShot(0, app.controller.offer_roll_recovery)
        app.controller.start_roll_autosave()
        qapp.aboutToQuit.connect(app.controller.clear_roll_recovery)
    except Exception:
        pass
    exec_fn = getattr(qapp, 'exec', None) or getattr(qapp, 'exec_')
    sys.exit(exec_fn())


if __name__ == "__main__":
    main()


from __future__ import annotations
from spektrafilm_gui.localization_gpu_zh_cn import translate_text
from spektrafilm_gui.localization_zh_cn import translate_status

from importlib import import_module
import sys
from typing import TYPE_CHECKING, Callable
from typing import Any, cast

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt


QFrame = QtWidgets.QFrame
QIcon = QtGui.QIcon
QMainWindow = QtWidgets.QMainWindow
QPushButton = QtWidgets.QPushButton
QStatusBar = QtWidgets.QStatusBar
QScrollArea = QtWidgets.QScrollArea
QWidget = QtWidgets.QWidget

from spektrafilm_gui.theme import APP_STYLE_SHEET
from spektrafilm_gui.theme_palette import (
    GRAY_0,
    GRAY_18,
    PANEL_BG,
    RADIUS_CARD,
    RADIUS_CONTROL,
    RAMP_CONTROL,
    RAMP_HOVER,
    SIZE_APP_MARGIN,
    SIZE_FOOTER_BOTTOM_INSET,
    SIZE_FOOTER_MIN_HEIGHT,
    SIZE_FOOTER_ITEM_SPACING,
    SIZE_FOOTER_TOP_SPACING,
    SIZE_PANEL_MARGIN,
    SIZE_SPLITTER_HANDLE_MARGIN_LEFT,
    SIZE_TAB_CONTENT_TOP_MARGIN,
)
from spektrafilm_gui.widgets import WidgetBundle
from spektrafilm_gui.widgets import CollapsibleSection, platform_default_font


# NAT 2026-07-22 (IMG_4993 + follow-up 'a bit bigger'): compact default,
# comfortably above the floors (sidebar 400 / left 360).
DEFAULT_CONTROLS_PANEL_WIDTH = 430
# NAT 2026-07-26: the RIGHT settings panel a touch wider than the left.
DEFAULT_SETTINGS_PANEL_WIDTH = 460
DEFAULT_VIEWER_SPLITTER_WIDTH = 1040
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
class AppMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._viewer_status_bar: QStatusBar | None = None

    def set_viewer_status_bar(self, status_bar: QStatusBar) -> None:
        self._viewer_status_bar = status_bar

    def statusBar(self) -> QStatusBar:  # noqa: N802 - Qt API name
        if self._viewer_status_bar is not None:
            return self._viewer_status_bar
        return super().statusBar()

def _get_current_stylesheet() -> str:
    # Phase 6C: the napari theme borrow is gone; the app's own QSS is the theme.
    return ''


def _request_dark_title_bar(window: QWidget) -> bool:
    if sys.platform != 'win32':
        return False

    try:
        hwnd = int(window.winId())
    except (AttributeError, TypeError, ValueError):
        return False

    if hwnd == 0:
        return False

    try:
        ctypes = import_module('ctypes')
        dwmapi = ctypes.windll.dwmapi
        set_window_attribute = dwmapi.DwmSetWindowAttribute
    except (ImportError, AttributeError):
        return False

    value = ctypes.c_int(1)
    value_size = ctypes.sizeof(value)
    for attribute in (_DWMWA_USE_IMMERSIVE_DARK_MODE, _DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1):
        try:
            result = set_window_attribute(hwnd, attribute, ctypes.byref(value), value_size)
        except (OSError, AttributeError, TypeError, ValueError):
            continue
        if result == 0:
            return True
    return False
def _canvas_background_color(*, gray_18_canvas: bool) -> str:
    return GRAY_18 if gray_18_canvas else GRAY_0


# Canvas background swatches: NegPy's black and dark grey, plus 18% gray as
# the lightest option (the app's photographic reference surround, GRAY_18).
CANVAS_BACKGROUND_OPTIONS: tuple[tuple[str, str], ...] = (
    ('Black', '#050505'),
    ('Dark grey', '#1C1C1C'),
    ('18% grey', GRAY_18),
)

# The app's default canvas surround when nothing is persisted: the Dark grey
# swatch (NAT 2026-07-12; was black).
DEFAULT_CANVAS_BACKGROUND = CANVAS_BACKGROUND_OPTIONS[1][1]


def set_canvas_background(viewer: Any, *, gray_18_canvas: bool) -> None:
    set_canvas_background_color(
        viewer, _canvas_background_color(gray_18_canvas=gray_18_canvas))


def set_canvas_background_color(viewer: Any, background: str) -> None:
    # NAT 2026-07-21 (v1.0.0 polish): the canvas background EXTENDS to the
    # whole viewer panel — the top bar and bottom nav bar sit on the same
    # surround as the image, with no control backgrounds (translucent hover
    # only). This MUST run before the napari-specific part below: with the
    # headless viewer shim there is no _qt_viewer, and returning early there
    # left the bars on WINDOW_BG (NAT's 'still not the same colour').
    host = _host_window(viewer)
    if host is not None:
        panel = host.findChild(QtWidgets.QFrame, 'viewerPanel')
        if panel is not None:
            # NAT 2026-07-22 'bring back the rounded buttons': containers stay
            # transparent on the swatch, but interactive controls keep their
            # rounded control fill (source order beats the QWidget rule at
            # equal specificity).
            panel.setStyleSheet(
                f'QFrame#viewerPanel {{ background: {background}; }}'
                f'QFrame#viewerPanel QWidget {{ background: transparent; }}'
                f'QFrame#viewerPanel QPushButton, '
                f'QFrame#viewerPanel QComboBox, '
                f'QFrame#viewerPanel QAbstractSpinBox '
                f'{{ background: {RAMP_CONTROL}; '
                f'border-radius: {RADIUS_CONTROL}; }}'
                f'QFrame#viewerPanel QPushButton:hover, '
                f'QFrame#viewerPanel QComboBox:hover '
                f'{{ background: {RAMP_HOVER}; }}')

    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        return

    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    if hasattr(qt_viewer, 'setStyleSheet'):
        qt_viewer.setStyleSheet(f'background: {background};')

    canvas = getattr(qt_viewer, 'canvas', None)
    if canvas is not None:
        if hasattr(canvas, 'bgcolor'):
            setattr(canvas, 'bgcolor', background)
        native = getattr(canvas, 'native', None)
        if native is not None and hasattr(native, 'setStyleSheet'):
            native.setStyleSheet(f'background: {background};')


def configure_napari_chrome(viewer: Any, *, gray_18_canvas: bool = True) -> None:
    qt_window = getattr(viewer.window, '_qt_window', None)
    if qt_window is not None:
        menu_bar = qt_window.menuBar()
        if menu_bar is not None:
            menu_bar.hide()

    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    set_canvas_background(viewer, gray_18_canvas=gray_18_canvas)

    set_welcome_visible = getattr(qt_viewer, 'set_welcome_visible', None)
    if callable(set_welcome_visible):
        set_welcome_visible(False)

    layer_controls = getattr(qt_viewer, 'dockLayerControls', None)
    if layer_controls is not None:
        layer_controls.hide()

    layer_list = getattr(qt_viewer, 'dockLayerList', None)
    if layer_list is not None:
        layer_list.hide()


def set_host_window(viewer: Any, host_window: QWidget) -> None:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is not None:
        setattr(viewer_window, '_agx_host_window', host_window)


def _host_window(viewer: Any) -> QWidget | None:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        return None
    host_window = getattr(viewer_window, '_agx_host_window', None)
    if host_window is not None:
        return host_window
    qt_window = getattr(viewer_window, '_qt_window', None)
    if qt_window is not None:
        return qt_window
    return None


def take_viewer_widget(viewer: Any) -> QWidget:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        raise RuntimeError('Napari viewer window is not available')

    qt_window = getattr(viewer_window, '_qt_window', None)
    if qt_window is not None:
        take_central_widget = getattr(qt_window, 'takeCentralWidget', None)
        if callable(take_central_widget):
            central_widget = take_central_widget()
            if central_widget is not None:
                return central_widget

    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is not None:
        return qt_viewer
    raise RuntimeError('Napari Qt viewer widget is not available')


def _wrap_scrollable(widget: QWidget) -> QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setSizeAdjustPolicy(QScrollArea.AdjustIgnored)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _build_sidebar(controls_panel: QWidget) -> QFrame:
    sidebar = QtWidgets.QFrame()
    sidebar.setObjectName('sidebarPanel')
    # NAT 2026-07-22: the splitter bottoms out at the width where every row
    # still fits COMPRESSED (sliders/boxes shrink first, reset arrows never
    # clip). Below this the widest fixed row (the coupler fader packs)
    # would push the reset off the edge.
    sidebar.setMinimumWidth(420)

    layout = QtWidgets.QVBoxLayout(sidebar)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(0)
    layout.addWidget(controls_panel, 1)
    return sidebar


def _build_controls_tab(*widgets: QWidget) -> QWidget:
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    layout.setContentsMargins(0, SIZE_TAB_CONTENT_TOP_MARGIN, 0, 0)
    layout.setAlignment(Qt.AlignTop)
    for widget in widgets:
        layout.addWidget(widget)
    # NAT 2026-07-22: when the tab is SHORTER than the viewport, the spare
    # height goes here — without this it was distributed INTO the section
    # cards (the 'big empty space at the bottom of couplers').
    layout.addStretch(1)
    return tab


def _borrow_layer_list_widget(viewer: Any) -> QWidget | None:
    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    layer_list = getattr(qt_viewer, 'dockLayerList', None) if qt_viewer is not None else None
    if layer_list is None or not hasattr(layer_list, 'widget'):
        return None
    widget = layer_list.widget()
    if isinstance(widget, QWidget):
        widget.setStyleSheet(_get_current_stylesheet())
    return widget


def reset_viewer_camera(viewer: Any) -> None:
    reset_view = getattr(viewer, 'reset_view', None)
    if callable(reset_view):
        reset_view()


def _viewer_device_pixel_ratio(viewer: Any) -> float:
    viewer_window = getattr(viewer, 'window', None)
    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is None:
        return 1.0

    device_pixel_ratio = getattr(qt_viewer, 'devicePixelRatioF', None)
    if callable(device_pixel_ratio):
        try:
            return max(float(device_pixel_ratio()), 0.01)
        except (TypeError, ValueError):
            return 1.0

    if hasattr(qt_viewer, 'devicePixelRatio'):
        try:
            return max(float(qt_viewer.devicePixelRatio()), 0.01)
        except (TypeError, ValueError):
            return 1.0

    return 1.0


def _layer_pixel_world_size(layer: object | None) -> float:
    if layer is None:
        return 1.0

    scale = getattr(layer, 'scale', None)
    if scale is None:
        return 1.0

    if isinstance(scale, (int, float)):
        try:
            return max(abs(float(scale)), 1e-12)
        except (TypeError, ValueError):
            return 1.0

    try:
        scale_values = tuple(scale)
    except TypeError:
        return 1.0

    axis_scales: list[float] = []
    for value in scale_values[:2]:
        try:
            numeric_value = abs(float(value))
        except (TypeError, ValueError):
            continue
        if numeric_value > 0.0:
            axis_scales.append(numeric_value)

    if not axis_scales:
        return 1.0
    return sum(axis_scales) / len(axis_scales)


def _home_view_target_layer(viewer: Any) -> object | None:
    layers = getattr(viewer, 'layers', None)
    if layers is None:
        return None

    selection = getattr(layers, 'selection', None)
    active_layer = getattr(selection, 'active', None)
    if active_layer is not None and getattr(active_layer, 'visible', True):
        return active_layer

    for layer in reversed(list(layers)):
        if getattr(layer, 'visible', True):
            return layer
    return None


def set_viewer_zoom_percent(viewer: Any, percent: float) -> None:
    viewer_window = getattr(viewer, 'window', None)
    camera = getattr(viewer, 'camera', None)
    if camera is None:
        camera = getattr(viewer_window, 'camera', None)
    if camera is None or not hasattr(camera, 'zoom'):
        return

    pixel_ratio = _viewer_device_pixel_ratio(viewer)
    target_layer = _home_view_target_layer(viewer)
    layer_pixel_world_size = _layer_pixel_world_size(target_layer)
    zoom_scale = max(0.0, float(percent)) / 100.0
    camera.zoom = max((pixel_ratio * zoom_scale) / layer_pixel_world_size, pixel_ratio * 0.01)


class _FullscreenImageView(QtWidgets.QDialog):
    """Distraction-free view: the rendered image (padding included) scaled to
    fit, centered on a completely black full screen. Esc, click, Enter or F11
    leave the view."""

    def __init__(self, image: 'QtGui.QImage', parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet('background-color: #000000;')
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._label = QtWidgets.QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet('background-color: #000000;')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def _rescale(self) -> None:
        if self._pixmap.isNull():
            return
        self._label.setPixmap(self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_F11):
            self.close()
        else:
            super().keyPressEvent(event)


def show_fullscreen_image(image_rgb8: 'np.ndarray', parent: QWidget | None = None) -> None:
    """Open the fullscreen black-background view on an (H, W, 3) uint8 array."""
    import numpy as np

    data = np.ascontiguousarray(np.asarray(image_rgb8)[..., :3], dtype=np.uint8)
    height, width = data.shape[:2]
    qimage = QtGui.QImage(data.data, width, height, width * 3,
                          QtGui.QImage.Format_RGB888).copy()
    view = _FullscreenImageView(qimage, parent=parent)
    view.showFullScreen()
    view.exec_() if hasattr(view, 'exec_') else view.exec()


def _nav_separator(height: int) -> QFrame:
    separator = QtWidgets.QFrame()
    separator.setFrameShape(QtWidgets.QFrame.VLine)
    separator.setObjectName('navSeparator')
    separator.setFixedHeight(height - 6)
    return separator


def _build_viewer_panel(
    viewer_widget: QWidget,
    status_bar: QStatusBar,
    *,
    banner: QWidget | None = None,
    top_bar_controls: QWidget | None = None,
    film_strip: QWidget | None = None,
    on_rotate_ccw: Callable[[], None] | None = None,
    on_rotate_cw: Callable[[], None] | None = None,
    on_zoom_100: Callable[[], None] | None = None,
    on_zoom_200: Callable[[], None] | None = None,
    on_zoom_400: Callable[[], None] | None = None,
    on_home_view: Callable[[], None] | None = None,
    on_prev_frame: Callable[[], None] | None = None,
    on_next_frame: Callable[[], None] | None = None,
    on_background_color: Callable[[str], None] | None = None,
    on_mirror_horizontal: Callable[[], None] | None = None,
    on_mirror_vertical: Callable[[], None] | None = None,
    on_scan: Callable[[], None] | None = None,
    on_save: Callable[[], None] | None = None,
    on_export: Callable[[], None] | None = None,
    on_fullscreen: Callable[[], None] | None = None,
    on_compare: Callable[[str], None] | None = None,
    on_rating_filter: Callable[[int], None] | None = None,
) -> QFrame:
    """Viewer column with the NegPy-style bottom navigation bar: the star
    filter at the far left (Lightroom-style), the status text, then prev/next
    | zoom + reset view | background swatches | rotate + mirror | compare |
    scan/save/export/fullscreen."""
    panel = QtWidgets.QFrame()
    panel.setObjectName('viewerPanel')
    divider_gap = int(SIZE_SPLITTER_HANDLE_MARGIN_LEFT.removesuffix('px'))
    row_height = int(SIZE_FOOTER_MIN_HEIGHT.removesuffix('px'))

    status_container = QtWidgets.QWidget()
    status_layout = QtWidgets.QHBoxLayout(status_container)
    # NAT 2026-07-22 round 2: SYMMETRIC insets on both bars — the gap to
    # the left panel matches the right side and the top bar.
    status_layout.setContentsMargins(14, 0, 14, 0)
    status_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
    # -- star filter (far left, LR-style): minimum rating for the roll ------
    from spektrafilm_gui.widget_primitives import StarRating
    rating_filter = StarRating(compact=True, tooltip=(
        'Filter the roll by star rating: frames below this rating are grayed '
        'out, prev/next skips them, and Export roll only writes the rest. '
        'Click the active star again to show everything.'))
    rating_filter.setObjectName('ratingFilter')
    if on_rating_filter is not None:
        rating_filter.valueChanged.connect(on_rating_filter)
    status_layout.addWidget(rating_filter)
    status_layout.addWidget(_nav_separator(row_height))
    status_bar.setContentsMargins(0, 0, 0, 0)
    status_bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    status_layout.addWidget(status_bar, 1)

    def _nav_button(text: str, object_name: str, callback, *, tooltip: str | None = None,
                    width: int | None = None) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        if tooltip:
            button.setToolTip(tooltip)
        if callback is not None:
            button.clicked.connect(callback)
        button.setFixedHeight(row_height)
        if width is not None:
            button.setFixedWidth(width)
        status_layout.addWidget(button)
        return button

    # -- frame navigation (roll) -----------------------------------------
    # NAT 2026-07-22: every arrow button matches the top-bar stepper's
    # native arrows (same widget type, same 20x24 size).
    for arrow, object_name, callback, tip in (
            (Qt.LeftArrow, 'prevFrameButton', on_prev_frame, 'Previous image in the roll'),
            (Qt.RightArrow, 'nextFrameButton', on_next_frame, 'Next image in the roll')):
        button = QtWidgets.QToolButton()
        button.setObjectName(object_name)
        button.setArrowType(arrow)
        button.setToolTip(tip)
        button.setFixedSize(20, 24)
        button.setCursor(Qt.PointingHandCursor)
        if callback is not None:
            button.clicked.connect(callback)
        status_layout.addWidget(button, 0, Qt.AlignVCenter)
    status_layout.addWidget(_nav_separator(row_height))

    # -- zoom (kept as-is) + reset view ------------------------------------
    _nav_button('100%', 'zoom100Button', on_zoom_100, tooltip='Pixel of the screen mapped 1 to 1 to the image pixel')
    _nav_button('200%', 'zoom200Button', on_zoom_200, tooltip='2 screen pixels mapped to 1 image pixel')
    _nav_button('400%', 'zoom400Button', on_zoom_400, tooltip='4 screen pixels mapped to 1 image pixel')
    _nav_button('reset view', 'homeViewButton', on_home_view, tooltip='Fit the image back into the view')
    status_layout.addWidget(_nav_separator(row_height))

    # -- canvas background swatches (NegPy's three) -------------------------
    swatch_group = QtWidgets.QButtonGroup(status_container)
    swatch_group.setExclusive(True)
    for index, (label, color) in enumerate(CANVAS_BACKGROUND_OPTIONS):
        swatch = QtWidgets.QToolButton()
        swatch.setObjectName(f'canvasSwatch{index}')
        swatch.setCheckable(True)
        swatch.setFixedSize(18, 18)
        swatch.setToolTip(f'Canvas background: {label}')
        swatch.setStyleSheet(
            f'QToolButton {{ background-color: {color}; border: 1px solid #666; }}'
            f'QToolButton:checked {{ border: 1px solid #ffffff; }}'
        )
        if on_background_color is not None:
            swatch.clicked.connect(lambda _=False, c=color: on_background_color(c))
        swatch_group.addButton(swatch)
        # NAT 2026-07-22: center the 18px swatches on the 26px button row.
        status_layout.addWidget(swatch, 0, Qt.AlignVCenter)
    status_layout.addWidget(_nav_separator(row_height))

    # -- orientation --------------------------------------------------------
    _nav_button('ccw rotate', 'rotateCcwButton', on_rotate_ccw, tooltip='Rotate the input image 90 deg counter-clockwise')
    _nav_button('cw rotate', 'rotateCwButton', on_rotate_cw, tooltip='Rotate the input image 90 deg clockwise')
    _nav_button('mirror h', 'mirrorHButton', on_mirror_horizontal, tooltip='Mirror the input image horizontally (left-right)')
    _nav_button('mirror v', 'mirrorVButton', on_mirror_vertical, tooltip='Mirror the input image vertically (top-bottom)')
    status_layout.addWidget(_nav_separator(row_height))

    # -- side-by-side compare ------------------------------------------------
    compare_combo = QtWidgets.QComboBox()
    compare_combo.setObjectName('compareCombo')
    for label in ('compare: off', 'compare: input', 'compare: negative'):
        compare_combo.addItem(label)
    compare_combo.setToolTip(
        'Show a second image beside the working one (shared zoom/pan): the base '
        'input or the developed negative.')
    compare_combo.setFixedHeight(row_height)
    if on_compare is not None:
        compare_combo.currentTextChanged.connect(
            lambda text: on_compare(text.split(':', 1)[-1].strip()))
    status_layout.addWidget(compare_combo)
    status_layout.addWidget(_nav_separator(row_height))

    # -- render / output actions (scan/save live in the side panels; NAT
    # trimmed them from the bar 2026-07-10) ---------------------------------
    _nav_button('export', 'navExportButton', on_export, tooltip='Export through the export panel settings')
    _nav_button('full screen', 'fullscreenButton', on_fullscreen,
                tooltip='Show the output image (with its padding) full screen on a black background. Esc or click to leave.')

    status_bar.setFixedHeight(row_height)
    status_container.setFixedHeight(row_height)

    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, divider_gap, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(0)
    if banner is not None or top_bar_controls is not None:
        top_row = QtWidgets.QWidget()
        top_grid = QtWidgets.QGridLayout(top_row)
        # NAT 2026-07-22 round 2: same insets as the bottom bar.
        top_grid.setContentsMargins(14, 0, 14, 0)
        top_grid.setHorizontalSpacing(SIZE_FOOTER_ITEM_SPACING)
        # NAT 2026-07-22 (IMG_4993): the image NAME sits at the CENTER of
        # the viewer; the settings strip stays left, the dimensions right.
        # Equal-stretch side columns keep the middle column screen-centered.
        if top_bar_controls is not None:
            top_grid.addWidget(top_bar_controls, 0, 0,
                               Qt.AlignLeft | Qt.AlignVCenter)
        if banner is not None and hasattr(banner, 'name_label'):
            banner.name_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            top_grid.addWidget(banner.name_label, 0, 1, Qt.AlignVCenter)
            top_grid.addWidget(banner.dims_label, 0, 2,
                               Qt.AlignRight | Qt.AlignVCenter)
        elif banner is not None:
            top_grid.addWidget(banner, 0, 1, Qt.AlignVCenter)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(2, 1)
        layout.addWidget(top_row)
        layout.addSpacing(2)
    layout.addWidget(viewer_widget, 1)
    if film_strip is not None:
        # NAT 2026-07-22: the roll preview sits in a REAL rounded card in
        # the side panels' grey (it read as just an underline before). Own
        # stylesheet so the viewer-panel transparency rule can't strip it.
        strip_card = QtWidgets.QFrame()
        strip_card.setObjectName('filmStripCard')
        strip_card.setStyleSheet(
            f'QFrame#filmStripCard {{ background: {PANEL_BG};'
            f' border-radius: {RADIUS_CARD}; }}'
            'QFrame#filmStripCard QListWidget { background: transparent;'
            ' border: none; }'
            # NAT 2026-07-22: slim scrollbar under the thumbnails
            'QFrame#filmStripCard QScrollBar:horizontal { height: 6px;'
            ' background: transparent; border: none; margin: 0; }'
            'QFrame#filmStripCard QScrollBar::handle:horizontal {'
            ' background: #3a3a3a; border-radius: 3px; min-width: 24px; }'
            'QFrame#filmStripCard QScrollBar::add-line:horizontal,'
            'QFrame#filmStripCard QScrollBar::sub-line:horizontal {'
            ' width: 0; height: 0; }')
        strip_card_layout = QtWidgets.QVBoxLayout(strip_card)
        strip_card_layout.setContentsMargins(8, 6, 8, 6)
        strip_card_layout.addWidget(film_strip)
        layout.addWidget(strip_card)
    layout.addSpacing(SIZE_FOOTER_TOP_SPACING)
    layout.addWidget(status_container)
    return panel


def build_left_panel(widgets: WidgetBundle) -> QWidget:
    """Left sidebar, top to bottom: Analysis (histogram + film curves), GPU
    backend toggle, the import/export sections (scrollable), and the Metadata
    panel pinned at the bottom."""
    frame = QtWidgets.QFrame()
    frame.setObjectName('leftPanel')
    # NAT 2026-07-22: same floor logic as the sidebar — compressed rows
    # still fit, reset arrows never clip.
    frame.setMinimumWidth(360)
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(4)

    analysis_content = QtWidgets.QWidget()
    analysis_content_layout = QtWidgets.QVBoxLayout(analysis_content)
    analysis_content_layout.setContentsMargins(0, 0, 0, 0)
    analysis_content_layout.addWidget(widgets.analysis)
    widgets.analysis.setMinimumHeight(200)
    widgets.analysis.setMaximumHeight(320)
    layout.addWidget(CollapsibleSection('Analysis', analysis_content, expanded=True, content_indent=4))

    layout.addWidget(widgets.gpu_backend)
    layout.addWidget(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.roll,
                widgets.filepicker,
                widgets.load_raw,
                widgets.input_image,
                widgets.export,
            ),
        ),
        1,
    )
    layout.addWidget(widgets.metadata)
    return frame


def build_controls_panel(viewer: Any, widgets: WidgetBundle) -> QWidget:
    panel = QtWidgets.QTabWidget()
    panel.setObjectName('controlsTabWidget')
    panel.setDocumentMode(True)
    panel.setUsesScrollButtons(False)
    panel.tabBar().setDrawBase(False)
    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.gui_config,          # 'Presets', now at the top of MAIN
                widgets.preview_crop,        # Preview and crop, under Presets
                widgets.camera,
                widgets.simulation,
                widgets.exposure_control,
                widgets.enlarger,
                widgets.scanner,
            ),
        ),
        translate_text('MAIN'),
    )
    panel.addTab(
        _wrap_scrollable(_build_controls_tab(widgets.push_pull, widgets.development, widgets.aging, widgets.halation, widgets.couplers, widgets.grain, widgets.grain_synthesis, widgets.bleach_bypass, widgets.camera_diffusion)),
        translate_text('FILM'),
    )
    panel.addTab(
        _wrap_scrollable(_build_controls_tab(widgets.chemistry, widgets.paper_shaping, widgets.glare, widgets.preflashing, widgets.diffusion)),
        translate_text('PRINT'),
    )
    panel.addTab(
        _wrap_scrollable(_build_controls_tab(
            widgets.scan_frame,       # 'Frame' (crop tool + skew + desqueeze)
            widgets.scan_finishing,   # 'Exposure' (active + EV gains)
            widgets.scan_levels,
            widgets.scan_response,
            widgets.scan_curve,   # Phase 10A point curve (engine applies it after split toning)
            widgets.scan_color,
            widgets.scan_toning,
        )),
        translate_text('SCAN'),
    )
    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.spectral_upsampling,
                widgets.input_gamut_compress,
                widgets.output_gamut_compress,
                widgets.tune,
                widgets.special,
                widgets.paper,
            ),
        ),
        translate_text('ADVANCED'),
    )

    # Phase 6C: the borrowed napari layer list is gone (the layer model is the
    # plain shim; the canvas is the display) — CONFIG keeps the Display section.
    panel.addTab(
        _wrap_scrollable(_build_controls_tab(widgets.display, widgets.config_defaults, widgets.locations)),
        translate_text('CONFIG'),
    )

    container = QtWidgets.QWidget()
    container_layout = QtWidgets.QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(4)
    container_layout.addWidget(panel, 1)
    container_layout.addWidget(widgets.simulation.action_bar())

    return container


def build_main_window(
    viewer: Any,
    controls_panel: QWidget,
    *,
    left_panel: QWidget | None = None,
    banner: QWidget | None = None,
    top_bar_controls: QWidget | None = None,
    film_strip: QWidget | None = None,
    on_rotate_ccw: Callable[[], None] | None = None,
    on_rotate_cw: Callable[[], None] | None = None,
    on_prev_frame: Callable[[], None] | None = None,
    on_next_frame: Callable[[], None] | None = None,
    on_background_color: Callable[[str], None] | None = None,
    on_mirror_horizontal: Callable[[], None] | None = None,
    on_mirror_vertical: Callable[[], None] | None = None,
    on_scan: Callable[[], None] | None = None,
    on_save: Callable[[], None] | None = None,
    on_export: Callable[[], None] | None = None,
    on_fullscreen: Callable[[], None] | None = None,
    on_compare: Callable[[str], None] | None = None,
    on_rating_filter: Callable[[int], None] | None = None,
    gpu_canvas: QWidget | None = None,
) -> QMainWindow:
    # Phase 6A: behind SPEKTRAFILM_WGPU_CANVAS the wgpu canvas replaces the
    # embedded napari widget (napari keeps running headless as the model until
    # 6C); reset view then drives both cameras.
    viewer_widget = gpu_canvas if gpu_canvas is not None else take_viewer_widget(viewer)

    def _home_view() -> None:
        reset_viewer_camera(viewer)
        if gpu_canvas is not None:
            gpu_canvas.reset_view()

    def _zoom_percent(percent: float) -> None:
        set_viewer_zoom_percent(viewer, percent)
        if gpu_canvas is not None:
            gpu_canvas.set_zoom_percent(percent)
    status_bar = QtWidgets.QStatusBar()
    status_bar.setSizeGripEnabled(False)

    main_window = AppMainWindow()
    main_window.setWindowTitle('spektrafilm')
    main_window.setWindowIcon(QIcon())
    extra_left = DEFAULT_CONTROLS_PANEL_WIDTH if left_panel is not None else 0
    main_window.resize(DEFAULT_SETTINGS_PANEL_WIDTH + DEFAULT_VIEWER_SPLITTER_WIDTH + extra_left, 980)
    main_window.setFont(platform_default_font())
    main_window.setStyleSheet(APP_STYLE_SHEET)
    main_window.set_viewer_status_bar(status_bar)
    set_host_window(viewer, main_window)

    splitter = QtWidgets.QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    viewer_panel = _build_viewer_panel(
        viewer_widget,
        status_bar,
        banner=banner,
        top_bar_controls=top_bar_controls,
        film_strip=film_strip,
        on_rotate_ccw=on_rotate_ccw,
        on_rotate_cw=on_rotate_cw,
        on_zoom_100=lambda: _zoom_percent(100.0),
        on_zoom_200=lambda: _zoom_percent(200.0),
        on_zoom_400=lambda: _zoom_percent(400.0),
        on_home_view=_home_view,
        on_prev_frame=on_prev_frame,
        on_next_frame=on_next_frame,
        on_background_color=on_background_color,
        on_mirror_horizontal=on_mirror_horizontal,
        on_mirror_vertical=on_mirror_vertical,
        on_scan=on_scan,
        on_save=on_save,
        on_export=on_export,
        on_fullscreen=on_fullscreen,
        on_compare=on_compare,
        on_rating_filter=on_rating_filter,
    )
    if left_panel is not None:
        splitter.addWidget(left_panel)
    splitter.addWidget(viewer_panel)
    splitter.addWidget(_build_sidebar(controls_panel))
    if left_panel is not None:
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([DEFAULT_CONTROLS_PANEL_WIDTH, DEFAULT_VIEWER_SPLITTER_WIDTH, DEFAULT_SETTINGS_PANEL_WIDTH])
    else:
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([DEFAULT_VIEWER_SPLITTER_WIDTH, DEFAULT_SETTINGS_PANEL_WIDTH])

    central = QtWidgets.QWidget()
    central.setObjectName('appCentral')
    central_layout = QtWidgets.QHBoxLayout(central)
    central_layout.setContentsMargins(SIZE_APP_MARGIN, SIZE_APP_MARGIN, SIZE_APP_MARGIN, SIZE_APP_MARGIN)
    central_layout.addWidget(splitter, 1)

    main_window.setCentralWidget(central)
    _request_dark_title_bar(main_window)
    main_window.statusBar().showMessage('ready', 3000)
    return main_window


def dialog_parent(viewer: Any) -> QWidget | None:
    return _host_window(viewer)


def set_status(viewer: Any, message: str, *, timeout_ms: int = 5000) -> None:
    host_window = cast(Any, _host_window(viewer))
    if host_window is None:
        return
    status_bar = host_window.statusBar() if hasattr(host_window, 'statusBar') else None
    if status_bar is not None:
        status_bar.showMessage(translate_text(translate_status(message)), timeout_ms)


def show_viewer_window(viewer: Any) -> None:
    host_window = _host_window(viewer)
    if host_window is not None and hasattr(host_window, 'show'):
        host_window.show()

    app = QtWidgets.QApplication.instance()
    if app is None:
        return

    active_window = app.activeWindow()
    if active_window is not None:
        active_window.showMaximized()
        return

    for window in reversed(app.topLevelWidgets()):
        if window.isVisible():
            window.showMaximized()
            return

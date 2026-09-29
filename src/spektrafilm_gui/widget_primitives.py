from __future__ import annotations

from functools import lru_cache
from spektrafilm_gui.localization_gpu_zh_cn import translate_text

from qtpy import QtCore, QtGui, QtWidgets

QPointF = getattr(QtCore, 'QPointF')
QSize = getattr(QtCore, 'QSize')

from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    CONTROL_BG,
    CONTROL_BG_HOVER,
    HEADER_DIVIDER_LINE,
    RADIUS_CONTROL,
    SIZE_COMPACT_BUTTON_MIN_HEIGHT,
    SIZE_CONTROL_PADDING,
    SIZE_FORM_SPACING,
    SIZE_SECTION_FRAME_INDENT,
    SIZE_SECTION_FRAME_MARGIN,
    SIZE_SECTION_STACK_SPACING,
    TEXT_CONTROL,
)
from spektrafilm_gui.icons import HEADER_ICON_SIZE, section_header_icon
from spektrafilm_gui.theme import resolve_theme_qcolor


def normalize_ui_text(text: str) -> str:
    return translate_text(text)


def platform_default_font() -> QtGui.QFont:
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)


class InputModeToggle(QtWidgets.QWidget):
    """[ input ] [ edit ] switch for the top bar (NAT 2026-07-25). Two buttons
    styled like the app's other buttons, with the ACTIVE one accent-highlighted
    (NAT 2026-07-26). Emits ``toggled(True)`` for input (sorting) mode,
    ``toggled(False)`` for edit. ``set_input_active`` syncs it silently (no
    signal) so the controller can reflect state (e.g. a completed render)."""

    toggled = QtCore.Signal(bool)

    # Standard-button look (matches the theme) + accent highlight when checked.
    # NAT 2026-07-25: the metrics are the top bar's OWN control metrics
    # (compact-button min-height + the shared control padding/radius), so the
    # toggle is exactly as tall as the 'update' button and the preview-size
    # box beside it. It used to be a 24px fixed-height pill and read taller
    # than everything else on the line.
    _STYLE = (
        'QPushButton { background: %s; color: %s; border: none; '
        'border-radius: %s; padding: %s; min-height: %s; }'
        'QPushButton:hover { background: %s; }'
        'QPushButton:checked { background: %s; color: #101010; font-weight: 600; }'
        % (CONTROL_BG, TEXT_CONTROL, RADIUS_CONTROL, SIZE_CONTROL_PADDING,
           SIZE_COMPACT_BUTTON_MIN_HEIGHT, CONTROL_BG_HOVER, ACCENT_COLOR_TEXT)
    )

    def __init__(self):
        super().__init__()
        self.setObjectName('inputModeToggle')
        # Default to INPUT / sorting mode (NAT 2026-07-26): the app opens ready
        # to cull; switch to 'edit' to render.
        self._input_active = True

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._input_btn = QtWidgets.QPushButton(translate_text('input'))
        self._edit_btn = QtWidgets.QPushButton(translate_text('edit'))
        for button in (self._input_btn, self._edit_btn):
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            # No fixed height: the stylesheet's min-height + padding size the
            # button exactly like the row's other controls.
            button.setSizePolicy(QtWidgets.QSizePolicy.Maximum,
                                 QtWidgets.QSizePolicy.Fixed)
            button.setMinimumWidth(52)
            button.setStyleSheet(self._STYLE)
        self._input_btn.setObjectName('segInput')
        self._edit_btn.setObjectName('segEdit')
        self._input_btn.setToolTip(
            'Sorting mode: show the camera thumbnail instantly, no edits, no '
            'render — cull a roll without waiting for each input to load.')
        self._edit_btn.setToolTip(
            'Edit mode: load and render the frame so you can work on it.')
        self._input_btn.clicked.connect(lambda: self._select(True))
        self._edit_btn.clicked.connect(lambda: self._select(False))
        layout.addWidget(self._input_btn)
        layout.addWidget(self._edit_btn)
        self._restyle()

    def _select(self, input_active: bool) -> None:
        if bool(input_active) == self._input_active:
            self._restyle()          # keep the checked state coherent
            return
        self._input_active = bool(input_active)
        self._restyle()
        self.toggled.emit(self._input_active)

    def set_input_active(self, active: bool) -> None:
        self._input_active = bool(active)
        self._restyle()

    def is_input_active(self) -> bool:
        return self._input_active

    def _restyle(self) -> None:
        self._input_btn.setChecked(self._input_active)
        self._edit_btn.setChecked(not self._input_active)


class FolderMenuComboBox(QtWidgets.QComboBox):
    """A combo box whose popup is a CASCADING menu (NAT 2026-07-25).

    Items named ``'Folder / Name'`` are grouped under a submenu that flies out
    to the side on hover, Explorer-style, instead of every folder's presets
    sitting in one long flat list. Items with no separator stay at the top
    level, in model order.

    The MODEL stays flat: every item keeps its qualified text and its index, so
    ``findText`` / ``currentText`` / ``setCurrentIndex`` / ``activated`` behave
    exactly as before; only the popup's presentation changes.
    """

    #: matches preset_store.FOLDER_SEPARATOR (kept local: this widget is
    #: generic and must not depend on the preset store).
    FOLDER_SEPARATOR = ' / '

    def __init__(self, parent: QtWidgets.QWidget | None = None,
                 *, separator: str = FOLDER_SEPARATOR):
        super().__init__(parent)
        self._separator = separator

    def _is_separator_item(self, index: int) -> bool:
        role = getattr(QtCore.Qt, 'AccessibleDescriptionRole')
        return str(self.itemData(index, role) or '') == 'separator'

    def build_popup_menu(self) -> QtWidgets.QMenu:
        """The cascading popup. Exposed (not private) so it can be tested
        without showing a window."""
        menu = QtWidgets.QMenu(self)
        submenus: dict[str, QtWidgets.QMenu] = {}
        current = self.currentIndex()
        for index in range(self.count()):
            if self._is_separator_item(index):
                menu.addSeparator()
                continue
            text = self.itemText(index)
            folder, _, leaf = text.partition(self._separator)
            target, label = menu, text
            if leaf:
                submenu = submenus.get(folder)
                if submenu is None:
                    # Create with an EXPLICIT C++ parent, then add it. Letting
                    # menu.addMenu(str) build the submenu returns a Python-owned
                    # QMenu that shiboken deletes once its wrapper is dropped,
                    # even though the parent still lists it (PySide6 gotcha).
                    submenu = QtWidgets.QMenu(folder, menu)
                    submenu.setIcon(_folder_menu_icon())
                    menu.addMenu(submenu)
                    submenus[folder] = submenu
                target, label = submenu, leaf
            action = target.addAction(label)
            action.setCheckable(True)
            action.setChecked(index == current)
            icon = self.itemIcon(index)
            if not icon.isNull():
                action.setIcon(icon)
            tooltip = self.itemData(index, QtCore.Qt.ToolTipRole)
            if tooltip:
                action.setToolTip(str(tooltip))
            action.setData(index)
            action.triggered.connect(
                lambda _checked=False, chosen=index: self._choose(chosen))
        # Keep the built menu AND its submenus alive. PySide6's addMenu(str)
        # returns a Python-OWNED QMenu, so dropping the only Python reference
        # deletes the C++ object even though the parent menu still lists it —
        # a caller holding just a submenu (or the tests, which call this
        # directly) would then hit 'already deleted'.
        self._popup_menu = menu
        self._popup_submenus = tuple(submenus.values())
        return menu

    def _choose(self, index: int) -> None:
        if index < 0 or index >= self.count():
            return
        self.setCurrentIndex(index)
        # Same contract as a normal popup pick: activated fires on every
        # selection, unchanged or not.
        self.activated.emit(index)
        self.textActivated.emit(self.itemText(index))

    def showPopup(self) -> None:  # noqa: N802 - Qt API name
        menu = self.build_popup_menu()
        if menu.isEmpty():
            super().showPopup()
            return
        menu.setMinimumWidth(self.width())
        position = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        exec_menu = getattr(menu, 'exec', None) or getattr(menu, 'exec_')
        try:
            exec_menu(position)
        finally:
            # One menu per popup: drop it once it closes (the triggered
            # handler has already run by then).
            menu.deleteLater()


@lru_cache(maxsize=1)
def _folder_menu_icon() -> QtGui.QIcon:
    """Small folder glyph for the submenu rows (empty icon when the icon
    service is unavailable; never a hard dependency)."""
    from spektrafilm_gui.icons import icon_by_name

    return icon_by_name('tabler:folder')


class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title: str, content: QtWidgets.QWidget, *, expanded: bool = True,
                 content_indent: int | None = None):
        super().__init__()
        self._content = content

        self._toggle = QtWidgets.QToolButton()
        self._toggle.setProperty('role', 'sectionToggle')
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._toggle.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.toggled.connect(self._set_expanded)

        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self._icon_label.setFixedSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE)
        self._apply_header_icon(title)

        self._title_button = QtWidgets.QToolButton()
        self._title_button.setProperty('role', 'sectionToggle')
        self._title_button.setText(normalize_ui_text(title))
        self._title_button.setAutoRaise(True)
        self._title_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._title_button.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self._title_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._title_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self._title_button.clicked.connect(self._toggle.toggle)

        self._header_line = HeaderDivider()
        self._header_line.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._header_line.setFixedHeight(max(self._toggle.sizeHint().height(), self._title_button.sizeHint().height()))

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(int(SIZE_FORM_SPACING.removesuffix('px')))
        header_layout.setAlignment(QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._toggle, 0, QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._icon_label, 0, QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._title_button, 0, QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._header_line, 1, QtCore.Qt.AlignVCenter)
        # Phase 15: per-section reset (hidden until a callback is attached)
        self._reset_button = QtWidgets.QToolButton()
        self._reset_button.setText('↺')
        self._reset_button.setAutoRaise(True)
        self._reset_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._reset_button.setToolTip(
            'Reset every setting in this group to its default')
        self._reset_button.setVisible(False)
        header_layout.addWidget(self._reset_button, 0, QtCore.Qt.AlignVCenter)

        self._frame = QtWidgets.QFrame()
        self._frame.setObjectName('sectionCard')   # Phase 15C: rounded card
        # NAT 2026-07-21: NoFrame — the StyledPanel outline doubled the card
        self._frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        frame_layout = QtWidgets.QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(
            SIZE_SECTION_FRAME_INDENT if content_indent is None else content_indent,
            SIZE_SECTION_FRAME_MARGIN,
            SIZE_SECTION_FRAME_MARGIN,
            SIZE_SECTION_FRAME_MARGIN,
        )
        frame_layout.setSpacing(0)
        frame_layout.setAlignment(QtCore.Qt.AlignTop)
        frame_layout.addWidget(self._content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(int(SIZE_SECTION_STACK_SPACING.removesuffix('px')))
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(header)
        layout.addWidget(self._frame)

        self._set_expanded(expanded)

    def set_reset_callback(self, callback) -> None:
        """Phase 15: attach the group's reset-all action; shows the button."""
        self._reset_button.clicked.connect(callback)
        self._reset_button.setVisible(True)

    def _apply_header_icon(self, title: str) -> None:
        icon = section_header_icon(title)
        if icon.isNull():
            self._icon_label.hide()
            return

        pixmap = icon.pixmap(HEADER_ICON_SIZE, HEADER_ICON_SIZE)
        if pixmap.isNull():
            self._icon_label.hide()
            return

        self._icon_label.setPixmap(pixmap)
        self._icon_label.show()

    def _set_expanded(self, expanded: bool) -> None:
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._frame.setVisible(expanded)

    def has_header_icon(self) -> bool:
        pixmap = self._icon_label.pixmap()
        return not self._icon_label.isHidden() and pixmap is not None and not pixmap.isNull()


class HeaderDivider(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

    def sizeHint(self):  # noqa: N802 - Qt API name
        return QSize(48, max(12, self.fontMetrics().height()))

    def minimumSizeHint(self):  # noqa: N802 - Qt API name
        return QSize(12, max(12, self.fontMetrics().height()))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        # Phase 15C follow-up (NAT): with the rounded section cards the old
        # header underline read as a double border - the divider now only
        # spaces the header (paints nothing).
        del event

class StarRating(QtWidgets.QWidget):
    """A 0-5 star row (Lightroom-style). Clicking star N sets the value to N;
    clicking the current value again clears to 0. Used both as the per-frame
    rating editor (Metadata panel) and as the minimum-rating filter (nav bar).
    """

    valueChanged = QtCore.Signal(int)

    def __init__(self, *, tooltip: str = '', compact: bool = False):
        super().__init__()
        self._value = 0
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._stars: list[QtWidgets.QToolButton] = []
        size = 20 if compact else 24   # NAT 2026-07-21: bigger
        for index in range(1, 6):
            star = QtWidgets.QToolButton()
            star.setObjectName(f'starRating{index}')
            star.setAutoRaise(True)
            star.setFixedSize(size, size)
            star.setCursor(QtCore.Qt.PointingHandCursor)
            if tooltip:
                star.setToolTip(tooltip)
            star.clicked.connect(lambda _=False, n=index: self._clicked(n))
            layout.addWidget(star)
            self._stars.append(star)
        # NAT 2026-07-22 'stars closer to each other': pack left — without the
        # stretch a stretched form row spreads the fixed-size stars apart.
        layout.addStretch(1)
        self._repaint()

    def _clicked(self, n: int) -> None:
        self.set_value(0 if n == self._value else n, emit=True)

    def value(self) -> int:
        return self._value

    def set_value(self, value: int, *, emit: bool = False) -> None:
        value = min(max(int(value), 0), 5)
        changed = value != self._value
        self._value = value
        self._repaint()
        if emit and changed:
            self.valueChanged.emit(self._value)

    def _repaint(self) -> None:
        # NAT 2026-07-21: always the FULL star glyph — lit when selected,
        # GHOSTED (dim, translucent) when not. The hollow outline read badly.
        # NAT 2026-07-22: lit = WHITE, not yellow.
        for index, star in enumerate(self._stars, start=1):
            filled = index <= self._value
            star.setText('★')
            star.setStyleSheet(
                'QToolButton { border: none; background: transparent; '
                'color: %s; font-size: %dpx; }'
                % ('#f2f2f2' if filled else 'rgba(160, 160, 160, 90)',
                   star.height() - 4))

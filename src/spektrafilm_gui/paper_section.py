"""Paper-texture section UI (ADVANCED tab, NAT 2026-07-26 v1.0.2).

Active toggle · a stackable list of texture rows (file + fusion mode + remove) ·
Add texture · padding-burn toggle. Implements the house section contract
(get_state / set_state over PaperState) plus a ``changed`` signal the
controller connects to re-composite the paper finish.
"""
from __future__ import annotations

import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt, Signal

from spektrafilm_gui.paper_texture import DEFAULT_FUSION_MODE, FUSION_MODES
from spektrafilm_gui.state import PaperState
from spektrafilm_gui.widget_editors import BoolEditor
from spektrafilm_gui.widget_primitives import CollapsibleSection

_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"


class PaperTextureSection(QtWidgets.QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self._rows: list[dict] = []
        self._loading = False

        content = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(content)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        self.active = BoolEditor()
        self.active.setChecked(False)
        self.active.setToolTip(
            'Blend paper-scan textures onto the rendered image. A display '
            'finish — it never changes the film science.')
        col.addLayout(self._labeled_row('Active', self.active))

        self._rows_container = QtWidgets.QWidget()
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        col.addWidget(self._rows_container)

        self.add_button = QtWidgets.QPushButton('Add texture')
        self.add_button.setToolTip('Pick a paper-texture image to add to the stack.')
        self.add_button.clicked.connect(self._on_add)
        col.addWidget(self.add_button)

        self.padding_burn = BoolEditor()
        self.padding_burn.setChecked(False)
        self.padding_burn.setToolTip(
            'On export, add a border (the Display white-padding width) around '
            'the image and let the paper texture bleed into it. Shown live in '
            'the preview too.')
        col.addLayout(self._labeled_row('Padding burn', self.padding_burn))

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(CollapsibleSection('Paper', content, expanded=False))

        self.active.toggled.connect(self._emit_changed)
        self.padding_burn.toggled.connect(self._emit_changed)

    # -- house contract ------------------------------------------------------
    def get_state(self) -> PaperState:
        textures = tuple(
            (row['path'], row['combo'].currentText(), row['opacity'].value() / 100.0)
            for row in self._rows)
        return PaperState(active=self.active.isChecked(),
                          padding_burn=self.padding_burn.isChecked(),
                          textures=textures)

    def set_state(self, state) -> None:
        self._loading = True
        try:
            for row in list(self._rows):
                self._drop_row(row)
            self.active.setChecked(bool(getattr(state, 'active', False)))
            self.padding_burn.setChecked(bool(getattr(state, 'padding_burn', False)))
            for entry in (getattr(state, 'textures', ()) or ()):
                path = str(entry[0])
                mode = str(entry[1]) if len(entry) > 1 else DEFAULT_FUSION_MODE
                opacity = float(entry[2]) if len(entry) > 2 else 1.0
                self._add_row(path, mode, opacity)
        finally:
            self._loading = False

    # -- helpers -------------------------------------------------------------
    def _labeled_row(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(label)
        lab.setToolTip(widget.toolTip())
        row.addWidget(lab)
        row.addWidget(widget)
        row.addStretch(1)
        return row

    def _emit_changed(self, *_a) -> None:
        if not self._loading:
            self.changed.emit()

    def _on_add(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Add paper texture', '', _IMAGE_FILTER)
        if path:
            self._add_row(path, DEFAULT_FUSION_MODE)
            self._emit_changed()

    def _add_row(self, path: str, mode: str, opacity: float = 1.0) -> None:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        name = QtWidgets.QLabel(os.path.basename(path))
        name.setToolTip(path)
        name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        combo = QtWidgets.QComboBox()
        combo.addItems(FUSION_MODES)
        combo.setCurrentText(mode if mode in FUSION_MODES else DEFAULT_FUSION_MODE)
        combo.currentTextChanged.connect(self._emit_changed)
        opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(int(round(max(0.0, min(1.0, opacity)) * 100)))
        opacity_slider.setFixedWidth(72)
        opacity_slider.setToolTip('Texture opacity (0-100%).')
        pct = QtWidgets.QLabel()
        pct.setFixedWidth(34)
        pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pct.setText(f'{opacity_slider.value()}%')
        opacity_slider.valueChanged.connect(lambda v: pct.setText(f'{v}%'))
        opacity_slider.valueChanged.connect(self._emit_changed)
        remove = QtWidgets.QToolButton()
        remove.setText('✕')
        remove.setToolTip('Remove this texture')
        remove.setCursor(Qt.PointingHandCursor)
        h.addWidget(name, 1)
        h.addWidget(combo)
        h.addWidget(opacity_slider)
        h.addWidget(pct)
        h.addWidget(remove)
        entry = {'row': row, 'path': path, 'combo': combo, 'opacity': opacity_slider}
        remove.clicked.connect(lambda: (self._drop_row(entry), self._emit_changed()))
        self._rows.append(entry)
        self._rows_layout.addWidget(row)

    def _drop_row(self, entry: dict) -> None:
        entry['row'].setParent(None)
        if entry in self._rows:
            self._rows.remove(entry)

"""Modal dialog for the Presets 'Export LUT...' action.

Collects LUT bake options (resolution, formats, input colour space, OCIO) and
an output path. Pure view: the controller reads ``result_spec()`` after exec().
"""

from __future__ import annotations

from dataclasses import dataclass

from qtpy import QtWidgets

from spektrafilm_gui.options import RGBColorSpaces

# Colour-space choices mirror the export panel's colour-space list.
_COLOR_SPACES = [e.value for e in RGBColorSpaces]


@dataclass
class LutExportRequest:
    size: int
    input_color_space: str
    formats: list  # subset of ('cube', '3dl', 'hald_png')
    write_ocio: bool
    out_path: str


class LutExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, default_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Export LUT")
        self.setModal(True)
        self._out_path = ""
        self._default_dir = default_dir

        form = QtWidgets.QFormLayout()

        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(2, 129)
        self.size_spin.setValue(33)
        self.size_spin.setToolTip(
            "Cube resolution (N^3 samples). Standard high-quality sizes are the "
            "power-of-two-plus-one grids 17 / 33 / 65 / 129 (they sit exactly on "
            "the endpoints and midpoints). 33 is the common default, 65 high, 129 "
            "very high (all bake in well under a second on GPU). Note: some tools "
            "(e.g. DaVinci Resolve) read .cube up to 65; HaldCLUT needs a perfect "
            "square (16, 25, 36, 49, 64, 81, 100, 121)."
        )
        form.addRow("Resolution", self.size_spin)

        self.color_space = QtWidgets.QComboBox()
        self.color_space.addItems(_COLOR_SPACES)
        self.color_space.setToolTip("Input colour space the LUT expects (scene-linear). "
                                    "The film look is applied to input in this space.")
        form.addRow("Input colour space", self.color_space)

        self.cb_cube = QtWidgets.QCheckBox(".cube (Adobe/Resolve)")
        self.cb_cube.setChecked(True)
        self.cb_3dl = QtWidgets.QCheckBox(".3dl (Autodesk/Nuke)")
        self.cb_hald = QtWidgets.QCheckBox("HaldCLUT .png")
        fmt_box = QtWidgets.QVBoxLayout()
        for cb in (self.cb_cube, self.cb_3dl, self.cb_hald):
            fmt_box.addWidget(cb)
        fmt_holder = QtWidgets.QWidget()
        fmt_holder.setLayout(fmt_box)
        form.addRow("Formats", fmt_holder)

        self.cb_ocio = QtWidgets.QCheckBox("Also write an OCIO config referencing the .cube")
        self.cb_ocio.setChecked(True)
        form.addRow("OCIO", self.cb_ocio)

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Choose an output name...")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._choose_path)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_holder = QtWidgets.QWidget()
        path_holder.setLayout(path_row)
        form.addRow("Output", path_holder)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._ok_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        self._warn = QtWidgets.QLabel("")
        self._warn.setStyleSheet("color: #d08a2a;")
        self._warn.setWordWrap(True)
        layout.addWidget(self._warn)
        layout.addWidget(buttons)

    def _choose_path(self) -> None:
        start = self._out_path or self._default_dir
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "LUT output name", start, "LUT files (*.cube *.3dl *.png);;All files (*)"
        )
        if path:
            self._out_path = path
            self.path_edit.setText(path)

    def _selected_formats(self) -> list:
        out = []
        if self.cb_cube.isChecked():
            out.append("cube")
        if self.cb_3dl.isChecked():
            out.append("3dl")
        if self.cb_hald.isChecked():
            out.append("hald_png")
        return out

    def _on_accept(self) -> None:
        if not self._out_path:
            self._warn.setText("Choose an output name first.")
            return
        if not self._selected_formats():
            self._warn.setText("Select at least one format.")
            return
        self.accept()

    def result_spec(self) -> LutExportRequest:
        return LutExportRequest(
            size=int(self.size_spin.value()),
            input_color_space=self.color_space.currentText(),
            formats=self._selected_formats(),
            write_ocio=self.cb_ocio.isChecked(),
            out_path=self._out_path,
        )

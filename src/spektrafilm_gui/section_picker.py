"""A small reusable 'which settings?' picker (NAT 2026-07-25).

Shared by copy/paste, apply-to-all and preset-save so the user can move a
subset of the edit instead of the whole thing. Returns the flattened
gui_state section names, or None when cancelled.
"""
from __future__ import annotations

from qtpy import QtWidgets

from spektrafilm_gui.settings_groups import (
    SETTINGS_GROUPS,
    default_look_labels,
    sections_for_groups,
)


class SectionSelectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title="Choose settings",
                 intro="", preselected=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        preselected = set(default_look_labels() if preselected is None
                          else preselected)

        layout = QtWidgets.QVBoxLayout(self)
        if intro:
            label = QtWidgets.QLabel(intro)
            label.setWordWrap(True)
            layout.addWidget(label)

        # All / none convenience row.
        row = QtWidgets.QHBoxLayout()
        all_btn = QtWidgets.QPushButton("All")
        none_btn = QtWidgets.QPushButton("None")
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        for group_label, _sections in SETTINGS_GROUPS:
            box = QtWidgets.QCheckBox(group_label)
            box.setChecked(group_label in preselected)
            self._checks[group_label] = box
            layout.addWidget(box)

        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn.clicked.connect(lambda: self._set_all(False))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, checked: bool) -> None:
        for box in self._checks.values():
            box.setChecked(checked)

    def selected_labels(self) -> tuple[str, ...]:
        return tuple(label for label, box in self._checks.items()
                     if box.isChecked())

    def selected_sections(self) -> tuple[str, ...]:
        return sections_for_groups(self.selected_labels())


def pick_settings_sections(parent, *, title, intro="", preselected=None):
    """Show the picker modally. Returns the chosen gui_state section names
    (possibly empty) or None if the user cancelled."""
    dialog = SectionSelectDialog(
        parent, title=title, intro=intro, preselected=preselected)
    if dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec():
        return dialog.selected_sections()
    return None

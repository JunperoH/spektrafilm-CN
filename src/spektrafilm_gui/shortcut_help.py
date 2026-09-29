"""Keyboard-shortcut cheat sheet (F1 / Ctrl+?).

``SHORTCUT_GROUPS`` is the single human-readable inventory of every binding
installed by ``app._install_shortcuts`` plus the modal keys handled elsewhere
(crop frame, fullscreen). A test cross-checks the bound sequences against
this table so the sheet cannot silently drift from the real bindings.

Each entry is ``(display, description, sequences)`` where ``sequences`` lists
the exact QKeySequence strings the entry documents ('' for modal keys that
are not global QShortcuts).
"""
from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

SHORTCUT_GROUPS: list[tuple[str, list[tuple[str, str, list[str]]]]] = [
    ('Render', [
        ('F5', 'Render preview', ['F5']),
        ('F6', 'Render full scan', ['F6']),
        ('Ctrl+E', 'Export the current output', ['Ctrl+E']),
    ]),
    ('View', [
        ('Ctrl+F', 'Toggle scan film (negative ↔ print)', ['Ctrl+F']),
        ('Ctrl+R', 'Before / after (input ↔ preview)', ['Ctrl+R']),
        ('Ctrl+D', 'Cycle compare: off / input / negative', ['Ctrl+D']),
        ('Ctrl+P / F11', 'Fullscreen output', ['Ctrl+P', 'F11']),
        ('Ctrl+O', 'Cycle composition guides', ['Ctrl+O']),
        ('Shift+O', 'Flip triangle / spiral orientation', ['Shift+O']),
        ('Ctrl+L', 'Cycle canvas background swatch', ['Ctrl+L']),
    ]),
    ('Edit', [
        ('Ctrl+Z', 'Undo', ['Ctrl+Z']),
        ('Ctrl+Shift+Z / Ctrl+Y', 'Redo', ['Ctrl+Shift+Z', 'Ctrl+Y']),
        ('Ctrl+Shift+C', 'Copy the current look', ['Ctrl+Shift+C']),
        ('Ctrl+Shift+V', 'Paste the look onto this frame', ['Ctrl+Shift+V']),
    ]),
    ('Roll', [
        ('Ctrl+← / Ctrl+→', 'Previous / next frame (skips filtered)',
         ['Ctrl+Left', 'Ctrl+Right']),
        ('Ctrl+Del', 'Remove the current frame from the roll', ['Ctrl+Del']),
    ]),
    ('Ratings', [
        ('Ctrl+1 … Ctrl+5', 'Rate the current frame',
         ['Ctrl+1', 'Ctrl+2', 'Ctrl+3', 'Ctrl+4', 'Ctrl+5']),
        ('Ctrl+0', 'Clear the rating', ['Ctrl+0']),
    ]),
    ('Darkroom · print filters (0.05 CC steps)', [
        ('Ctrl+Shift+8 / 5', 'Yellow filter + / −',
         ['Ctrl+Shift+8', 'Ctrl+Shift+5']),
        ('Ctrl+Shift+9 / 6', 'Magenta filter + / −',
         ['Ctrl+Shift+9', 'Ctrl+Shift+6']),
    ]),
    ('Crop frame (while cropping)', [
        ('Enter / double-click', 'Commit the crop', []),
        ('Esc', 'Cancel the crop', []),
    ]),
    ('Help', [
        ('F1 / Ctrl+?', 'Show / hide this cheat sheet', ['F1', 'Ctrl+?']),
    ]),
]


def bound_sequences() -> set[str]:
    """Every global QShortcut sequence the sheet documents."""
    return {
        sequence
        for _title, entries in SHORTCUT_GROUPS
        for _display, _description, sequences in entries
        for sequence in sequences
    }


_KEY_CHIP_STYLE = (
    'QLabel { background: rgba(128, 128, 128, 48);'
    ' border: 1px solid rgba(128, 128, 128, 96); border-radius: 4px;'
    ' padding: 2px 8px; font-weight: 600; }'
)
_GROUP_STYLE = 'QLabel { font-weight: 700; padding-top: 8px; }'
_DIALOG_OBJECT_NAME = 'shortcutHelpDialog'


def _build_dialog(parent: QtWidgets.QWidget | None) -> QtWidgets.QDialog:
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName(_DIALOG_OBJECT_NAME)
    dialog.setWindowTitle('Keyboard shortcuts')
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    dialog.setWindowFlag(QtCore.Qt.Tool, True)   # floats above, no taskbar entry

    # The global F1/Ctrl+? shortcuts live on the main window and stop firing
    # once this (separate) window has focus — mirror them here so the same
    # keys close the sheet no matter what is focused.
    shortcut_cls = getattr(QtGui, 'QShortcut', None) or getattr(QtWidgets, 'QShortcut')
    for sequence in ('F1', 'Ctrl+?'):
        shortcut_cls(QtGui.QKeySequence(sequence), dialog).activated.connect(dialog.close)

    columns = QtWidgets.QHBoxLayout(dialog)
    columns.setContentsMargins(18, 12, 18, 14)
    columns.setSpacing(28)

    # Balance the groups over two columns by entry count.
    total = sum(len(entries) + 1 for _t, entries in SHORTCUT_GROUPS)
    grids, used = [], 0
    for _ in range(2):
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setAlignment(QtCore.Qt.AlignTop)
        columns.addLayout(grid)
        grids.append(grid)
    column, row = 0, 0
    for title, entries in SHORTCUT_GROUPS:
        if column == 0 and used >= total / 2:
            column, row = 1, 0
        grid = grids[column]
        header = QtWidgets.QLabel(title)
        header.setStyleSheet(_GROUP_STYLE)
        grid.addWidget(header, row, 0, 1, 2)
        row += 1
        used += 1
        for display, description, _sequences in entries:
            chip = QtWidgets.QLabel(display)
            chip.setStyleSheet(_KEY_CHIP_STYLE)
            chip.setAlignment(QtCore.Qt.AlignCenter)
            chip.setSizePolicy(QtWidgets.QSizePolicy.Maximum,
                               QtWidgets.QSizePolicy.Preferred)
            grid.addWidget(chip, row, 0, QtCore.Qt.AlignRight)
            grid.addWidget(QtWidgets.QLabel(description), row, 1)
            grid.setColumnStretch(1, 1)
            row += 1
            used += 1
    return dialog


def toggle_shortcut_help(parent: QtWidgets.QWidget | None) -> QtWidgets.QDialog | None:
    """Show the cheat sheet, or close it if it is already open (F1 toggles).

    Returns the dialog when it was opened, ``None`` when it was closed.
    """
    if parent is not None:
        existing = parent.findChild(QtWidgets.QDialog, _DIALOG_OBJECT_NAME)
        if existing is not None:
            existing.close()
            return None
    dialog = _build_dialog(parent)
    dialog.show()   # modeless: keep it open while working; Esc closes
    return dialog

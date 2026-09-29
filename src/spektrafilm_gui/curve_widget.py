"""Phase 10A: SCAN-tab finishing point curve editor (Lightroom-style).

Two pieces, matching the DataclassSection architecture:

- ``CurvePointsEditor``: an invisible per-field holder exposing the ``.value``
  property and ``valueChanged`` signal every section editor exposes, so state
  round-trips, undo snapshots, roll persistence and the auto-preview wiring
  all work unchanged (the four curve fields are hidden_fields on the SCAN
  finishing section, exactly like the crop group).
- ``CurveEditorView``: the visible QPainter plot editing one channel at a
  time (Luma / R / G / B selector), click-to-add, drag, double-click-remove,
  per-channel reset. QPainter only — no plotting dependency (Phase 10
  constraint). The curve drawn is the SAME PCHIP the engine samples into its
  1D LUT, so what you see is what renders.
"""
from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm.model.scan_finishing import CURVE_IDENTITY, _curve_evaluator

Signal = QtCore.Signal

_MIN_POINTS = 2
_MAX_POINTS = 16
_HIT_RADIUS_PX = 9.0
_MIN_X_GAP = 0.01

_CHANNEL_ORDER = ('luma', 'r', 'g', 'b')
_CHANNEL_LABELS = {'luma': 'Luma', 'r': 'R', 'g': 'G', 'b': 'B'}
_CHANNEL_COLORS = {
    'luma': QtGui.QColor('#d8d8d8'),
    'r': QtGui.QColor('#e0655a'),
    'g': QtGui.QColor('#61bf6e'),
    'b': QtGui.QColor('#5f8fe0'),
}


def _normalize_points(points) -> tuple:
    """Clamp into the unit square, sort by x, drop duplicate x (keep last)."""
    if points is None:
        return CURVE_IDENTITY
    cleaned = [
        (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0))
        for x, y in points
    ]
    cleaned.sort(key=lambda pt: pt[0])
    deduped: list = []
    for pt in cleaned:
        if deduped and deduped[-1][0] == pt[0]:
            deduped[-1] = pt
        else:
            deduped.append(pt)
    if len(deduped) < _MIN_POINTS:
        return CURVE_IDENTITY
    return tuple(deduped)


class CurvePointsEditor(QtWidgets.QWidget):
    """Invisible value holder for ONE curve field (tuple of (x, y) points)."""

    valueChanged = Signal(object)  # noqa: N815 - Qt signal naming

    def __init__(self):
        super().__init__()
        self._points: tuple = CURVE_IDENTITY
        self.setVisible(False)

    @property
    def value(self) -> tuple:
        return self._points

    @value.setter
    def value(self, points) -> None:
        normalized = _normalize_points(points)
        if normalized == self._points:
            return
        self._points = normalized
        self.valueChanged.emit(normalized)


class _CurvePlot(QtWidgets.QWidget):
    """The QPainter unit-square plot for the active channel."""

    _MARGIN = 8

    def __init__(self, view: 'CurveEditorView'):
        super().__init__()
        self._view = view
        self._drag_index: int | None = None
        self.setMinimumHeight(170)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMouseTracking(False)
        self.setCursor(QtCore.Qt.CrossCursor)

    # ------------------------------------------------------------ transforms
    def _plot_rect(self) -> QtCore.QRectF:
        m = float(self._MARGIN)
        return QtCore.QRectF(m, m, max(self.width() - 2 * m, 1.0), max(self.height() - 2 * m, 1.0))

    def _to_widget(self, x: float, y: float) -> QtCore.QPointF:
        rect = self._plot_rect()
        return QtCore.QPointF(rect.left() + x * rect.width(), rect.bottom() - y * rect.height())

    def _to_curve(self, pos: QtCore.QPointF) -> tuple[float, float]:
        rect = self._plot_rect()
        x = (pos.x() - rect.left()) / rect.width()
        y = (rect.bottom() - pos.y()) / rect.height()
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    # -------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            rect = self._plot_rect()
            palette = self.palette()

            painter.fillRect(rect, palette.color(QtGui.QPalette.Base).darker(105))
            grid_pen = QtGui.QPen(palette.color(QtGui.QPalette.Mid), 1, QtCore.Qt.DotLine)
            painter.setPen(grid_pen)
            for i in (1, 2, 3):
                frac = i / 4.0
                painter.drawLine(self._to_widget(frac, 0.0), self._to_widget(frac, 1.0))
                painter.drawLine(self._to_widget(0.0, frac), self._to_widget(1.0, frac))
            painter.drawLine(self._to_widget(0.0, 0.0), self._to_widget(1.0, 1.0))
            painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.Mid), 1))
            painter.drawRect(rect)

            xs = np.linspace(0.0, 1.0, 129)
            # ghost the OTHER channels' non-identity curves (dimmed) so the
            # combined shape stays visible while editing one channel
            # (NAT 2026-07-20); identity ghosts would just retrace the
            # diagonal gridline, so they are skipped.
            active = self._view.current_channel()
            for key in _CHANNEL_ORDER:
                if key == active:
                    continue
                ghost_eval = _curve_evaluator(self._view.points_for(key))
                if ghost_eval is None:
                    continue
                ghost = QtGui.QColor(_CHANNEL_COLORS[key])
                ghost.setAlpha(80)
                self._draw_curve(
                    painter, xs, np.clip(ghost_eval(xs), 0.0, 1.0), ghost, 1)

            color = _CHANNEL_COLORS[active]
            points = self._view.current_points()
            evaluate = _curve_evaluator(points)
            ys = xs if evaluate is None else np.clip(evaluate(xs), 0.0, 1.0)
            self._draw_curve(painter, xs, ys, color, 2)

            painter.setPen(QtGui.QPen(color.darker(140), 1))
            painter.setBrush(color)
            for x, y in points:
                painter.drawEllipse(self._to_widget(x, y), 4.0, 4.0)
        finally:
            painter.end()

    def _draw_curve(self, painter, xs, ys, color, width) -> None:
        path = QtGui.QPainterPath(self._to_widget(float(xs[0]), float(ys[0])))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self._to_widget(float(x), float(y)))
        painter.setPen(QtGui.QPen(color, width))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)

    # ------------------------------------------------------------------ mouse
    def _hit_index(self, pos: QtCore.QPointF) -> int | None:
        for index, (x, y) in enumerate(self._view.current_points()):
            widget_pt = self._to_widget(x, y)
            if (widget_pt - pos).manhattanLength() <= _HIT_RADIUS_PX * 1.6:
                dx = widget_pt.x() - pos.x()
                dy = widget_pt.y() - pos.y()
                if (dx * dx + dy * dy) ** 0.5 <= _HIT_RADIUS_PX:
                    return index
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos = QtCore.QPointF(event.position() if hasattr(event, 'position') else event.localPos())
        index = self._hit_index(pos)
        if index is None:
            points = list(self._view.current_points())
            if len(points) >= _MAX_POINTS:
                return
            x, y = self._to_curve(pos)
            points.append((x, y))
            points.sort(key=lambda pt: pt[0])
            index = points.index((x, y))
            self._view.set_current_points(points)
        self._drag_index = index
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._drag_index is None:
            return
        pos = QtCore.QPointF(event.position() if hasattr(event, 'position') else event.localPos())
        points = list(self._view.current_points())
        if self._drag_index >= len(points):
            # An exact-x collision was deduped by the holder's normalization.
            self._drag_index = None
            return
        x, y = self._to_curve(pos)
        lo = points[self._drag_index - 1][0] + _MIN_X_GAP if self._drag_index > 0 else 0.0
        hi = (points[self._drag_index + 1][0] - _MIN_X_GAP
              if self._drag_index < len(points) - 1 else 1.0)
        points[self._drag_index] = (min(max(x, lo), max(hi, lo)), y)
        self._view.set_current_points(points)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        self._drag_index = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API name
        pos = QtCore.QPointF(event.position() if hasattr(event, 'position') else event.localPos())
        index = self._hit_index(pos)
        points = list(self._view.current_points())
        if index is None or len(points) <= _MIN_POINTS:
            return
        del points[index]
        self._drag_index = None
        self._view.set_current_points(points)
        self.update()


class CurveEditorView(QtWidgets.QWidget):
    """Channel selector + plot, bound to the four CurvePointsEditor holders."""

    def __init__(self, editors: dict[str, CurvePointsEditor]):
        super().__init__()
        self._editors = dict(editors)
        self._current = 'luma'

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(4)
        self._channel_buttons: dict[str, QtWidgets.QToolButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for key in _CHANNEL_ORDER:
            button = QtWidgets.QToolButton()
            button.setText(_CHANNEL_LABELS[key])
            button.setCheckable(True)
            button.setChecked(key == self._current)
            button.setToolTip(f'Edit the {_CHANNEL_LABELS[key]} curve')
            button.clicked.connect(lambda _checked=False, k=key: self._set_channel(k))
            group.addButton(button)
            selector_row.addWidget(button)
            self._channel_buttons[key] = button
        selector_row.addStretch(1)
        reset_button = QtWidgets.QToolButton()
        reset_button.setText('Reset')
        reset_button.setToolTip('Reset the selected channel curve to identity')
        reset_button.clicked.connect(self._reset_current)
        selector_row.addWidget(reset_button)

        self._plot = _CurvePlot(self)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(selector_row)
        layout.addWidget(self._plot)
        self.setLayout(layout)

        # External changes (state loads, undo, pasted looks) must repaint.
        for editor in self._editors.values():
            editor.valueChanged.connect(lambda _points: self._plot.update())

    def current_channel(self) -> str:
        return self._current

    def current_points(self) -> tuple:
        return self._editors[self._current].value

    def points_for(self, key: str) -> tuple:
        return self._editors[key].value

    def set_current_points(self, points) -> None:
        self._editors[self._current].value = tuple(points)

    def _set_channel(self, key: str) -> None:
        self._current = key
        self._plot.update()

    def _reset_current(self) -> None:
        self._editors[self._current].value = CURVE_IDENTITY
        self._plot.update()

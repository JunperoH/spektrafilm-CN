"""Bottom-left ANALYSIS tab: an RGB histogram of the current render and the
film density characteristic curves, painted with QPainter (no matplotlib, so
nothing new is bundled). NegPy's charts are the visual reference.

Both widgets are pure views. The controller pushes:
  - HistogramWidget.set_image(display_uint8_or_float): the shown render.
  - FilmCurveWidget.set_curves(log_exposure, density_curves): the active film
    profile's characteristic curves (K x 3), on profile change.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt

_BG = QtGui.QColor('#0a0a0a')
_BORDER = QtGui.QColor('#262626')
_GRID = QtGui.QColor('#1a1a1a')
_AXIS_TEXT = QtGui.QColor('#8d8d8d')
_CHANNEL_HEX = ('#e0655f', '#5fc07f', '#5f8fe0')  # r, g, b


class HistogramWidget(QtWidgets.QWidget):
    """256-bin per-channel histogram of the displayed render."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # NAT 2026-07-22 round 4: FLEXIBLE — hard minimums overflowed the
        # capped Analysis area and the plots clipped. The 2:3 histogram/
        # curves ratio lives in the AnalysisPanel stretch factors now.
        self.setMinimumHeight(60)
        self._hists: list[np.ndarray] = []
        self.setToolTip('RGB histogram of the current preview/scan')

    def set_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self._hists = []
            self.update()
            return
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            self._hists = []
            self.update()
            return
        if np.issubdtype(arr.dtype, np.integer):
            data = arr[..., :3].astype(np.float32) / float(np.iinfo(arr.dtype).max or 1)
        else:
            data = np.clip(arr[..., :3].astype(np.float32), 0.0, 1.0)
        # decimate large frames -- the histogram shape is unchanged
        step = max(1, int(np.ceil(max(data.shape[0], data.shape[1]) / 512)))
        data = data[::step, ::step]
        hists = []
        for ch in range(3):
            h, _ = np.histogram(data[..., ch], bins=256, range=(0.0, 1.0))
            m = float(h.max())
            hists.append((h / m) if m > 0 else h.astype(np.float64))
        self._hists = hists
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, _BG)
        painter.setPen(QtGui.QPen(_GRID, 1))
        for i in range(1, 4):
            x = int(w * i / 4)
            painter.drawLine(x, 0, x, h)
        painter.setPen(QtGui.QPen(_BORDER, 1))
        painter.drawRect(rect)
        if not self._hists:
            painter.setPen(QtGui.QPen(_AXIS_TEXT, 1))
            painter.drawText(rect, Qt.AlignCenter, 'no render yet')
            return
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        for ch, series in enumerate(self._hists):
            self._draw_channel(painter, series, _CHANNEL_HEX[ch], w, h)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

    def _draw_channel(self, painter, data, color_hex, w, h) -> None:
        if len(data) < 2:
            return
        path = QtGui.QPainterPath()
        path.moveTo(0, h)
        step = w / (len(data) - 1)
        for i, val in enumerate(data):
            path.lineTo(i * step, h - float(val) * (h - 2))
        path.lineTo(w, h)
        path.closeSubpath()
        fill = QtGui.QColor(color_hex)
        fill.setAlpha(90)
        painter.setBrush(QtGui.QBrush(fill))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)


class FilmCurveWidget(QtWidgets.QWidget):
    """Characteristic (log-exposure -> density) curves of the active film."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # NAT 2026-07-22 round 4: flexible (see HistogramWidget note) — the
        # 2:3 ratio comes from the AnalysisPanel stretch factors.
        self.setMinimumHeight(90)
        self._x: Optional[np.ndarray] = None
        self._curves: Optional[np.ndarray] = None
        self.setToolTip('Film density characteristic curves (log exposure -> density) of the active stock')

    def set_curves(self, log_exposure, density_curves) -> None:
        try:
            x = np.asarray(log_exposure, dtype=np.float64).reshape(-1)
            c = np.asarray(density_curves, dtype=np.float64)
        except (TypeError, ValueError):
            self._x = self._curves = None
            self.update()
            return
        if c.ndim != 2 or c.shape[0] != x.shape[0] or c.shape[1] < 3:
            self._x = self._curves = None
            self.update()
            return
        self._x, self._curves = x, c[:, :3]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, _BG)
        painter.setPen(QtGui.QPen(_GRID, 1))
        for i in range(1, 4):
            painter.drawLine(int(w * i / 4), 0, int(w * i / 4), h)
            painter.drawLine(0, int(h * i / 4), w, int(h * i / 4))
        painter.setPen(QtGui.QPen(_BORDER, 1))
        painter.drawRect(rect)
        if self._x is None or self._curves is None:
            painter.setPen(QtGui.QPen(_AXIS_TEXT, 1))
            painter.drawText(rect, Qt.AlignCenter, 'no film profile')
            return
        x = self._x
        finite = np.isfinite(self._curves)
        if not finite.any():
            return
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        d_max = float(np.nanmax(self._curves[finite]))
        d_max = d_max if d_max > 1e-6 else 1.0
        x_span = (x_max - x_min) or 1.0
        pad = 3
        for ch in range(3):
            col = self._curves[:, ch]
            path = QtGui.QPainterPath()
            started = False
            for i in range(len(x)):
                if not np.isfinite(col[i]):
                    started = False
                    continue
                px = pad + (x[i] - x_min) / x_span * (w - 2 * pad)
                py = (h - pad) - (col[i] / d_max) * (h - 2 * pad)
                if not started:
                    path.moveTo(px, py)
                    started = True
                else:
                    path.lineTo(px, py)
            painter.setPen(QtGui.QPen(QtGui.QColor(_CHANNEL_HEX[ch]), 1.6))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)


class AnalysisPanel(QtWidgets.QWidget):
    """The ANALYSIS tab: histogram over the film curves."""

    def __init__(self):
        super().__init__()
        self.histogram = HistogramWidget()
        self.film_curve = FilmCurveWidget()

        def _titled(title, widget):
            box = QtWidgets.QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)
            label = QtWidgets.QLabel(title)
            label.setStyleSheet('color: #8d8d8d;')
            box.addWidget(label)
            box.addWidget(widget, 1)
            holder = QtWidgets.QWidget()
            holder.setLayout(box)
            return holder

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        # NAT 2026-07-22: 2 : 3 — the histogram at 2/3 of the film curves,
        # both FULLY inside the Analysis area whatever its height.
        layout.addWidget(_titled('Histogram', self.histogram), 2)
        layout.addWidget(_titled('Film curves', self.film_curve), 3)
        self.setLayout(layout)

    def set_image(self, image) -> None:
        self.histogram.set_image(image)

    def set_curves(self, log_exposure, density_curves) -> None:
        self.film_curve.set_curves(log_exposure, density_curves)

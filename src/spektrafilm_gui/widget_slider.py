"""Phase 15 (v1.0.0 UI rework, step 1): slider rows, knobs, per-row reset.

Three composites, all implementing the house editor contract — a ``.value``
property plus a ``valueChanged`` signal — so state persistence, profile
sync, auto-preview wiring, undo, rolls, shortcuts and every existing test
work unchanged (the CurvePointsEditor precedent):

- ``SliderFieldEditor``: [slider][spinbox][reset] for scalar rows. The
  spinbox is authoritative; the slider maps over the spec's (min, max,
  step) and appears only once the spec provides a FINITE range. Slider
  drags update the number live but the render-triggering valueChanged is
  DEFERRED to release (the 2026-07-20 stock-editor lesson: never stream
  full renders during a drag).
- ``KnobEditor``: a minimalist ~26px rotary knob (QPainter — QDial is not
  acceptable) whose indicator line and arc can be TINTED (per-channel
  colors on RGB triplets). Interaction is the audio-plugin standard:
  click + VERTICAL drag (the knob only displays rotation), scroll-wheel
  steps, double-click resets to baseline. Drags defer to release.
- ``KnobTripletEditor``: three channel-tinted knobs + value boxes on ONE
  line — the RGB-triplet answer (three sliders never fit; three knobs do).

RESET SEMANTICS: every composite carries a baseline (``set_baseline``).
DataclassSection seeds it from the factory default state and refreshes it
with each profile sync for profile-synced fields — so reset means "what
this stock gave you", not zero.
"""
from __future__ import annotations

import math

from qtpy import QtCore, QtGui, QtWidgets

Signal = QtCore.Signal

CHANNEL_TINTS = (QtGui.QColor('#e0655a'), QtGui.QColor('#61bf6e'),
                 QtGui.QColor('#5f8fe0'))
# NAT 2026-07-22: per-hue accents for hue-family knob rows (hue saturation,
# CMY filter shifts). Keyed by the hue letter.
HUE_TINTS = {
    'r': QtGui.QColor('#e0655a'), 'g': QtGui.QColor('#61bf6e'),
    'b': QtGui.QColor('#5f8fe0'), 'c': QtGui.QColor('#50b8d0'),
    'm': QtGui.QColor('#d060c8'), 'y': QtGui.QColor('#d8c04a'),
}
_NEUTRAL_TINT = QtGui.QColor('#d8d8d8')
_MAX_SLIDER_STEPS = 1000
_FINITE_BOUND = 999_999.0     # editors default to +-1e6 = "unbounded"
_KNOB_DRAG_RANGE_PX = 150.0   # full-range vertical drag distance

# --------------------------------------------------------------------------
# Phase 16A/16B (NAT 2026-07-22): LIVE-DRAG STREAMING. Drags emit
# valueChanged on every move now — safe because the controller's supersede
# scheduler coalesces renders (at most one pending; the 10C flooding lesson
# moved from the widgets to the scheduler). This module-level tracker tells
# the controller a drag is in progress (16B renders drag frames at half
# resolution) and fires one callback when the LAST drag ends (the full-res
# release render).
# --------------------------------------------------------------------------
_DRAG_DEPTH = 0
_DRAG_END_CALLBACK = None


def set_drag_end_callback(callback) -> None:
    global _DRAG_END_CALLBACK
    _DRAG_END_CALLBACK = callback


def drag_active() -> bool:
    return _DRAG_DEPTH > 0


def _begin_drag() -> None:
    global _DRAG_DEPTH
    _DRAG_DEPTH += 1


def _end_drag() -> None:
    global _DRAG_DEPTH
    _DRAG_DEPTH = max(0, _DRAG_DEPTH - 1)
    if _DRAG_DEPTH == 0 and _DRAG_END_CALLBACK is not None:
        _DRAG_END_CALLBACK()


def _finite_range(minimum: float, maximum: float) -> bool:
    return (-_FINITE_BOUND < minimum and maximum < _FINITE_BOUND
            and maximum > minimum)


class _ResetButton(QtWidgets.QToolButton):
    """Doubles as the Phase 15C MODIFIED indicator: dim at baseline, accent
    (the house yellow) when the row differs from it."""

    def __init__(self) -> None:
        super().__init__()
        from spektrafilm_gui.theme_palette import ACCENT_COLOR_TEXT, TEXT_DIM

        self._accent = ACCENT_COLOR_TEXT
        self._dim = TEXT_DIM
        self.setText('↺')
        self.setToolTip('Reset to default (the stock default for '
                        'profile-synced settings)')
        self.setFixedSize(18, 24)
        self.setAutoRaise(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.set_modified(False)

    def set_modified(self, modified: bool) -> None:
        self.setStyleSheet(
            f'color: {self._accent if modified else self._dim};')


class SliderFieldEditor(QtWidgets.QWidget):
    """[QSlider][spinbox][reset] wrapping a FloatEditor/IntEditor."""

    valueChanged = Signal(object)   # noqa: N815 - editor contract

    def __init__(self, inner: QtWidgets.QAbstractSpinBox):
        super().__init__()
        self._inner = inner
        self._baseline = None
        self._syncing = False
        self._drag_pending = False

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setVisible(False)      # appears once the range is finite
        self._slider.setFixedHeight(24)
        # NAT 2026-07-22: on a narrow panel the SLIDER shrinks first — the
        # reset arrow must never clip off the right edge.
        self._slider.setMinimumWidth(36)
        self._inner.setFixedWidth(74)
        self._reset = _ResetButton()
        self._reset.setEnabled(False)
        self._reset.clicked.connect(self._on_reset)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._inner)
        layout.addWidget(self._reset)

        self._inner.valueChanged.connect(self._on_spin_changed)
        self._slider.valueChanged.connect(self._on_slider_moved)
        self._slider.sliderPressed.connect(_begin_drag)
        self._slider.sliderReleased.connect(_end_drag)
        self._rebuild_scale()

    # ------------------------------------------------------------- contract
    @property
    def value(self):
        return self._inner.value

    @value.setter
    def value(self, new_value) -> None:
        self._inner.value = new_value      # inner emits -> composite emits

    # forwarded spec plumbing (DataclassSection._apply_specs calls these)
    def setMinimum(self, value) -> None:  # noqa: N802 - Qt-style forward
        self._inner.setMinimum(value)
        self._rebuild_scale()

    def setMaximum(self, value) -> None:  # noqa: N802
        self._inner.setMaximum(value)
        self._rebuild_scale()

    def setSingleStep(self, value) -> None:  # noqa: N802
        self._inner.setSingleStep(value)
        self._rebuild_scale()

    def setToolTip(self, text: str) -> None:  # noqa: N802
        super().setToolTip(text)
        self._inner.setToolTip(text)
        self._slider.setToolTip(text)

    # ---------------------------------------------------------------- reset
    def set_baseline(self, value) -> None:
        self._baseline = value
        self._reset.setEnabled(True)
        self._sync_modified()

    def baseline(self):
        return self._baseline

    def _sync_modified(self) -> None:
        self._reset.set_modified(
            self._baseline is not None and self.value != self._baseline)

    def _on_reset(self) -> None:
        if self._baseline is not None:
            self.value = self._baseline

    def reset_to_baseline(self) -> None:
        self._on_reset()

    # --------------------------------------------------------------- wiring
    def has_slider(self) -> bool:
        return self._slider.isVisible() or self._slider_steps() > 0

    def _slider_steps(self) -> int:
        minimum, maximum = self._inner.minimum(), self._inner.maximum()
        if not _finite_range(minimum, maximum):
            return 0
        step = self._inner.singleStep() or 1
        # floor of 200 positions so sparse-step specs still drag smoothly
        # (fine values remain typable / arrow-nudgeable in the box)
        return min(_MAX_SLIDER_STEPS,
                   max(200, int(round((maximum - minimum) / step))))

    def _rebuild_scale(self) -> None:
        steps = self._slider_steps()
        self._slider.setVisible(steps > 0)
        if steps > 0:
            self._syncing = True
            try:
                self._slider.setRange(0, steps)
                self._sync_slider_from_spin()
            finally:
                self._syncing = False

    def _sync_slider_from_spin(self) -> None:
        steps = self._slider.maximum()
        if steps <= 0:
            return
        minimum, maximum = self._inner.minimum(), self._inner.maximum()
        span = maximum - minimum
        if span <= 0:
            return
        fraction = (float(self._inner.value) - minimum) / span
        self._slider.setValue(int(round(fraction * steps)))

    def _slider_to_value(self, position: int):
        minimum, maximum = self._inner.minimum(), self._inner.maximum()
        fraction = position / max(self._slider.maximum(), 1)
        raw = minimum + fraction * (maximum - minimum)
        if isinstance(self._inner, QtWidgets.QSpinBox):
            return int(round(raw))
        return raw

    def _on_spin_changed(self, new_value) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._sync_slider_from_spin()
        finally:
            self._syncing = False
        self._sync_modified()
        self.valueChanged.emit(new_value)

    def _on_slider_moved(self, position: int) -> None:
        if self._syncing:
            return
        new_value = self._slider_to_value(position)
        self._syncing = True
        try:
            self._inner.setValue(new_value)   # display updates live
        finally:
            self._syncing = False
        self._sync_modified()
        # Phase 16A: STREAM every move — the controller's supersede scheduler
        # coalesces renders (drag frames at half res, one full-res render on
        # release via the drag-end callback). Keyboard/programmatic moves
        # (no drag) emit the same way and render at full res immediately.
        self.valueChanged.emit(self._inner.value)


class KnobEditor(QtWidgets.QWidget):
    """Minimalist rotary knob. Vertical drag, wheel steps, double-click
    resets to baseline; the indicator line/arc carries the given tint."""

    valueChanged = Signal(object)   # noqa: N815 - editor contract
    # Display-only stream during a drag (NAT 2026-07-22: the number must
    # follow the knob live, like the slider) — carries NO render trigger;
    # valueChanged still fires once on release.
    dragged = Signal(object)

    _ARC_START = 225.0              # degrees, lower-left
    _ARC_SPAN = -270.0              # clockwise 3/4 turn

    def __init__(self, *, minimum: float = 0.0, maximum: float = 1.0,
                 step: float = 0.01,
                 tint: QtGui.QColor | None = None):
        super().__init__()
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._value = float(minimum)
        self._baseline: float | None = None
        self._tint = tint or _NEUTRAL_TINT
        self._drag: tuple[float, float] | None = None   # (start_y, start_value)
        self._drag_pending = False
        self.setFixedSize(26, 26)
        self.setCursor(QtCore.Qt.SizeVerCursor)

    # ------------------------------------------------------------- contract
    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value) -> None:
        self._set_value(float(new_value), emit=True)

    def setMinimum(self, value) -> None:  # noqa: N802
        self._minimum = float(value)
        self.update()

    def setMaximum(self, value) -> None:  # noqa: N802
        self._maximum = float(value)
        self.update()

    def setSingleStep(self, value) -> None:  # noqa: N802
        self._step = float(value)

    def set_baseline(self, value) -> None:
        self._baseline = float(value)

    def _set_value(self, new_value: float, *, emit: bool) -> None:
        clamped = min(max(new_value, self._minimum), self._maximum)
        if clamped == self._value:
            return
        self._value = clamped
        self.update()
        if emit:
            self.valueChanged.emit(clamped)

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            palette = self.palette()
            rect = QtCore.QRectF(3.0, 3.0, 20.0, 20.0)
            span = self._maximum - self._minimum
            fraction = 0.0 if span <= 0 else (self._value - self._minimum) / span

            painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.Mid), 2.0))
            painter.setBrush(palette.color(QtGui.QPalette.Base).darker(105))
            painter.drawEllipse(rect)

            arc = QtGui.QColor(self._tint)
            arc.setAlpha(120)
            painter.setPen(QtGui.QPen(arc, 2.4, QtCore.Qt.SolidLine,
                                      QtCore.Qt.RoundCap))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawArc(rect, int(self._ARC_START * 16),
                            int(self._ARC_SPAN * fraction * 16))

            angle = math.radians(self._ARC_START + self._ARC_SPAN * fraction)
            center = rect.center()
            tip = QtCore.QPointF(center.x() + 8.0 * math.cos(angle),
                                 center.y() - 8.0 * math.sin(angle))
            hub = QtCore.QPointF(center.x() + 3.0 * math.cos(angle),
                                 center.y() - 3.0 * math.sin(angle))
            painter.setPen(QtGui.QPen(self._tint, 2.0, QtCore.Qt.SolidLine,
                                      QtCore.Qt.RoundCap))
            painter.drawLine(hub, tip)
        finally:
            painter.end()

    # ---------------------------------------------------------- interaction
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.position() if hasattr(event, 'position') else event.localPos()
            self._drag = (pos.y(), self._value)
            _begin_drag()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            return
        pos = event.position() if hasattr(event, 'position') else event.localPos()
        start_y, start_value = self._drag
        delta = (start_y - pos.y()) / _KNOB_DRAG_RANGE_PX
        # Phase 16A: STREAM every move (scheduler coalesces: half-res drag
        # frames, one full-res render on release via the drag-end callback).
        self._set_value(start_value + delta * (self._maximum - self._minimum),
                        emit=True)
        self.dragged.emit(self._value)      # keep the box-follow signal live

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        del event
        if self._drag is not None:
            self._drag = None
            _end_drag()                     # depth 0 -> controller full render

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        del event
        if self._drag is not None:
            self._drag = None
            _end_drag()
        if self._baseline is not None:
            self._set_value(self._baseline, emit=True)

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self._set_value(self._value + steps * self._step, emit=True)
        event.accept()


class KnobTripletEditor(QtWidgets.QWidget):
    """Three channel-tinted knobs + value boxes on one line (RGB triplets)."""

    valueChanged = Signal(object)   # noqa: N815 - editor contract

    def __init__(self, count: int = 3, *, decimals: int = 2,
                 tints: tuple | list | None = None):
        super().__init__()
        from spektrafilm_gui.widget_editors import FloatEditor

        palette = tuple(tints) if tints else CHANNEL_TINTS
        self._syncing = False
        self._baseline: tuple | None = None
        self._knobs: list[KnobEditor] = []
        self._spins: list[QtWidgets.QDoubleSpinBox] = []
        self._reset = _ResetButton()
        self._reset.setEnabled(False)
        self._reset.clicked.connect(self._on_reset)

        layout = QtWidgets.QHBoxLayout(self)
        # NAT 2026-07-22: small gap between the row text and the first knob.
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(3)
        for index in range(count):
            knob = KnobEditor(tint=palette[index % len(palette)])
            spin = FloatEditor(decimals=decimals)
            # NAT 2026-07-22: boxes COMPRESS on narrow panels (44-60px) so
            # the reset arrow never clips off the row.
            spin.setMinimumWidth(44)
            spin.setMaximumWidth(60)
            knob.valueChanged.connect(
                lambda value, i=index: self._on_knob(i, value))
            knob.dragged.connect(
                lambda value, i=index: self._on_knob_dragged(i, value))
            spin.valueChanged.connect(
                lambda value, i=index: self._on_spin(i, value))
            layout.addWidget(knob)
            layout.addWidget(spin)
            self._knobs.append(knob)
            self._spins.append(spin)
        layout.addStretch(1)
        layout.addWidget(self._reset)

    # ------------------------------------------------------------- contract
    @property
    def value(self) -> tuple:
        return tuple(spin.value for spin in self._spins)

    @value.setter
    def value(self, values) -> None:
        values = tuple(values)
        if values == self.value:
            return
        self._syncing = True
        try:
            for spin, knob, item in zip(self._spins, self._knobs, values):
                spin.value = float(item)
                knob._set_value(float(item), emit=False)
        finally:
            self._syncing = False
        self._sync_modified()
        self.valueChanged.emit(values)

    def setMinimum(self, value) -> None:  # noqa: N802
        for widget in (*self._spins, *self._knobs):
            widget.setMinimum(value)

    def setMaximum(self, value) -> None:  # noqa: N802
        for widget in (*self._spins, *self._knobs):
            widget.setMaximum(value)

    def setSingleStep(self, value) -> None:  # noqa: N802
        for widget in (*self._spins, *self._knobs):
            widget.setSingleStep(value)

    def setToolTip(self, text: str) -> None:  # noqa: N802
        super().setToolTip(text)
        for widget in (*self._spins, *self._knobs):
            widget.setToolTip(text)

    def set_baseline(self, values) -> None:
        self._baseline = tuple(values)
        self._reset.setEnabled(True)
        for knob, item in zip(self._knobs, self._baseline):
            knob.set_baseline(float(item))
        self._sync_modified()

    def _sync_modified(self) -> None:
        self._reset.set_modified(
            self._baseline is not None
            and tuple(float(v) for v in self.value)
            != tuple(float(v) for v in self._baseline))

    def _on_reset(self) -> None:
        if self._baseline is not None:
            self.value = self._baseline

    def reset_to_baseline(self) -> None:
        self._on_reset()

    def _on_knob(self, index: int, new_value) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._spins[index].value = float(new_value)
        finally:
            self._syncing = False
        self._sync_modified()
        self.valueChanged.emit(self.value)

    def _on_knob_dragged(self, index: int, new_value) -> None:
        """Mid-drag: the number follows the knob live; NO valueChanged (the
        render still waits for release, exactly like the slider rows)."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self._spins[index].value = float(new_value)
        finally:
            self._syncing = False
        self._sync_modified()

    def _on_spin(self, index: int, new_value) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._knobs[index]._set_value(float(new_value), emit=False)
        finally:
            self._syncing = False
        self._sync_modified()
        self.valueChanged.emit(self.value)


class LinkedKnobRow(QtWidgets.QWidget):
    """One row of tinted [text?][knob][box] units writing through to
    EXISTING scalar editors that stay hidden in their own section (NAT
    2026-07-22: enlarger YMC shifts, chemistry RGB gammas, preflash YM,
    scan channel gains). Presentation-only — state, bridge, profile sync
    and the hidden editors' baselines are untouched by construction.

    units: iterable of (target_editor, tint, text_or_None,
                        minimum, maximum, step, decimals)."""

    def __init__(self, units):
        super().__init__()
        from spektrafilm_gui.widget_editors import FloatEditor

        self._targets = []
        self._knobs: list[KnobEditor] = []
        self._spins: list[QtWidgets.QDoubleSpinBox] = []
        self._syncing = False
        self._reset = _ResetButton()
        self._reset.clicked.connect(self._on_reset)

        layout = QtWidgets.QHBoxLayout(self)
        # NAT 2026-07-22: small gap between the row text and the first knob.
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(3)
        for index, (target, tint, text, minimum, maximum, step,
                    decimals) in enumerate(units):
            if text:
                label = QtWidgets.QLabel(text)
                label.setStyleSheet('color: #8d8d8d;')
                layout.addWidget(label)
            knob = KnobEditor(minimum=minimum, maximum=maximum, step=step,
                              tint=tint)
            spin = FloatEditor(decimals=decimals, minimum=minimum,
                               maximum=maximum)
            spin.setSingleStep(step)
            # NAT 2026-07-22: compressible (44-60px) — reset never clips
            spin.setMinimumWidth(44)
            spin.setMaximumWidth(60)
            knob.valueChanged.connect(
                lambda value, i=index: self._commit(i, value))
            knob.dragged.connect(
                lambda value, i=index: self._display(i, value))
            spin.valueChanged.connect(
                lambda value, i=index: self._commit(i, value))
            target.valueChanged.connect(lambda _value, i=index: self._pull(i))
            layout.addWidget(knob)
            layout.addWidget(spin)
            self._targets.append(target)
            self._knobs.append(knob)
            self._spins.append(spin)
        layout.addStretch(1)
        layout.addWidget(self._reset)
        for index in range(len(self._targets)):
            self._pull(index)

    def _commit(self, index: int, value) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._targets[index].value = float(value)   # emits -> render
        finally:
            self._syncing = False
        self._pull(index)     # reflect any clamping by the target

    def _display(self, index: int, value) -> None:
        """Mid-drag: box follows live, commit waits for release."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self._spins[index].value = float(value)
        finally:
            self._syncing = False

    def _pull(self, index: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            target = self._targets[index]
            current = float(target.value)
            self._knobs[index]._set_value(current, emit=False)
            self._spins[index].blockSignals(True)
            self._spins[index].setValue(current)
            self._spins[index].blockSignals(False)
            baseline = getattr(target, 'baseline', None)
            if callable(baseline) and baseline() is not None:
                self._knobs[index].set_baseline(float(baseline()))
        finally:
            self._syncing = False
        self._sync_modified()

    def _baseline_of(self, target):
        baseline = getattr(target, 'baseline', None)
        return baseline() if callable(baseline) else None

    def _sync_modified(self) -> None:
        modified = any(
            self._baseline_of(target) is not None
            and float(target.value) != float(self._baseline_of(target))
            for target in self._targets)
        self._reset.set_modified(modified)

    def _on_reset(self) -> None:
        for target in self._targets:
            reset = getattr(target, 'reset_to_baseline', None)
            if callable(reset):
                reset()

"""Phase 10C: the custom film stock editor — a MODELESS tool dialog.

The H&D plot is the centerpiece: x = log exposure labeled in stops around
the base half-density anchor, y = density. The R/G/B parametric curves are
drawn over the GHOSTED base stock curves; six handle types write parametric
recipe fields and the curve redraws from the model:

    speed     horizontal drag of the curve body (third-stop snap)
    gamma     vertical drag of the straight-line handle (slope)
    toe       drag of the toe-region handle -> toe_size
    shoulder  drag of the shoulder handle -> shoulder_size
    ceiling   vertical drag of the D_max line
    floor     vertical drag of the fog line (per channel when unlinked
              = colored fog)

Channels are LINKED by default — crossover is an explicit act (unlink, then
pick a channel). Numeric readouts beside the plot are editable and stay in
sync. Look overrides (grain / halation / DIR couplers — each header names
the pipeline stage it affects) are collapsed groups reusing the existing
editor primitives; only enabled groups are stored in the recipe.

The dialog is NEVER modal: it emits ``recipeChanged`` while handles drag and
the controller renders the live recipe through the normal preview worker
(debounced). Save / Save As / Duplicate / Revert / Delete with an
unsaved-changes guard; deleting a stock that is in use falls back to its
base stock (controller side).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm.profiles.custom_stocks import (
    ParametricCurves,
    StockRecipe,
    _apply_fog,
    _base_half_density_anchor,
    _parametric_curves,
    _spline_curves,
    delete_custom_stock,
    fit_parametric_curves,
    is_custom_slug,
    list_custom_stocks,
    load_custom_stock,
    make_slug,
    save_custom_stock,
    spline_fit_residual,
)
from spektrafilm.profiles.io import load_profile

Signal = QtCore.Signal

_CHANNEL_COLORS = (QtGui.QColor('#e0655a'), QtGui.QColor('#61bf6e'), QtGui.QColor('#5f8fe0'))
_LINKED_COLOR = QtGui.QColor('#d8d8d8')
_THIRD_STOP = 1.0 / 3.0

_PARAM_BOUNDS = {
    'gamma': (0.05, 5.0),
    'speed_offset_ev': (-6.0, 6.0),
    'density_max': (0.1, 6.0),
    'toe_size': (0.05, 3.0),
    'shoulder_size': (0.05, 3.0),
    'fog': (0.0, 1.5),
}

# Look override fields per target: (field, label, decimals, step)
_LOOK_FIELDS = {
    'grain': ('develop stage - silver grain', (
        ('particle_area_um2', 'Particle area (um^2)', 3, 0.05),
        ('blur', 'Grain blur', 2, 0.05),
    )),
    'halation': ('exposure stage - scatter and halation', (
        ('halation_amount', 'Halation amount', 2, 0.1),
        ('scatter_amount', 'Scatter amount', 2, 0.1),
    )),
    'dir_couplers': ('develop stage - DIR couplers', (
        ('amount', 'Coupler amount', 2, 0.1),
    )),
}


def _clamp(name: str, value: float) -> float:
    low, high = _PARAM_BOUNDS[name]
    return min(max(float(value), low), high)


class _HDPlot(QtWidgets.QWidget):
    """QPainter H&D plot with the six handle types."""

    _MARGIN_L, _MARGIN_R, _MARGIN_T, _MARGIN_B = 34, 10, 8, 22

    def __init__(self, editor: 'StockEditorDialog'):
        super().__init__()
        self._editor = editor
        self._drag: tuple[str, float, float, tuple] | None = None
        self.setMinimumSize(360, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCursor(QtCore.Qt.CrossCursor)

    # ------------------------------------------------------------ transforms
    def _domain(self):
        profile = self._editor.base_profile
        x = np.asarray(profile.data.log_exposure, dtype=float)
        y_top = max(3.2, float(np.nanmax(self._editor.current_curves())) * 1.15)
        return float(x.min()), float(x.max()), 0.0, y_top

    def _rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self._MARGIN_L, self._MARGIN_T,
            max(self.width() - self._MARGIN_L - self._MARGIN_R, 1),
            max(self.height() - self._MARGIN_T - self._MARGIN_B, 1),
        )

    def _to_widget(self, x: float, y: float) -> QtCore.QPointF:
        x0, x1, y0, y1 = self._domain()
        rect = self._rect()
        fx = (x - x0) / max(x1 - x0, 1e-9)
        fy = (y - y0) / max(y1 - y0, 1e-9)
        return QtCore.QPointF(rect.left() + fx * rect.width(), rect.bottom() - fy * rect.height())

    def _from_widget(self, pos: QtCore.QPointF) -> tuple[float, float]:
        x0, x1, y0, y1 = self._domain()
        rect = self._rect()
        fx = (pos.x() - rect.left()) / rect.width()
        fy = (rect.bottom() - pos.y()) / rect.height()
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)

    # -------------------------------------------------------------- handles
    def _handle_positions(self) -> dict[str, QtCore.QPointF]:
        """Handle anchor points for the ACTIVE channel, in curve space."""
        editor = self._editor
        channel = editor.active_channel()
        p = editor.recipe.parametric
        anchor = float(editor.anchors[channel])
        gamma = float(p.gamma[channel])
        d_max = float(p.density_max[channel])
        fog = float(p.fog[channel])
        speed_shift = -float(p.speed_offset_ev[channel]) * math.log10(2.0)
        loge0 = anchor - d_max / (2.0 * gamma) + speed_shift
        half_x = anchor + speed_shift
        curves = editor.current_curves()
        x = np.asarray(editor.base_profile.data.log_exposure, dtype=float)

        def curve_y(at_x: float) -> float:
            return float(np.interp(at_x, x, curves[:, channel]))

        x0, x1, _y0, y_top = self._domain()
        return {
            'speed': QtCore.QPointF(half_x, curve_y(half_x)),
            'gamma': QtCore.QPointF(half_x + 0.45, curve_y(half_x + 0.45)),
            'toe': QtCore.QPointF(loge0, curve_y(loge0)),
            'shoulder': QtCore.QPointF(loge0 + d_max / gamma, curve_y(loge0 + d_max / gamma)),
            'ceiling': QtCore.QPointF(x1 - 0.35, d_max),
            'floor': QtCore.QPointF(x0 + 0.35, fog),
        }

    # -------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            palette = self.palette()
            rect = self._rect()
            painter.fillRect(rect, palette.color(QtGui.QPalette.Base).darker(105))

            x0, x1, y0, y1 = self._domain()
            editor = self._editor
            anchor_mid = float(np.mean(editor.anchors))
            grid_pen = QtGui.QPen(palette.color(QtGui.QPalette.Mid), 1, QtCore.Qt.DotLine)
            painter.setPen(grid_pen)
            stop = math.log10(2.0)
            # vertical stop lines around the base anchor, labeled in stops
            n_lo = int(math.floor((anchor_mid - x0) / stop))
            for k in range(-n_lo, int(math.floor((x1 - anchor_mid) / stop)) + 1):
                gx = anchor_mid + k * stop
                painter.setPen(grid_pen)
                painter.drawLine(self._to_widget(gx, y0), self._to_widget(gx, y1))
                if k % 2 == 0:
                    painter.setPen(palette.color(QtGui.QPalette.Text))
                    painter.drawText(
                        QtCore.QRectF(self._to_widget(gx, y0).x() - 16, rect.bottom() + 2, 32, 16),
                        QtCore.Qt.AlignCenter, f'{k:+d}' if k else '0')
            for d in np.arange(0.0, y1, 0.5):
                painter.setPen(grid_pen)
                painter.drawLine(self._to_widget(x0, d), self._to_widget(x1, d))
                painter.setPen(palette.color(QtGui.QPalette.Text))
                painter.drawText(
                    QtCore.QRectF(0, self._to_widget(x0, d).y() - 8, self._MARGIN_L - 4, 16),
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f'{d:.1f}')
            painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.Mid), 1))
            painter.drawRect(rect)

            x = np.asarray(editor.base_profile.data.log_exposure, dtype=float)
            # ghost base curves
            base = np.asarray(editor.base_profile.data.density_curves, dtype=float)
            for channel in range(3):
                ghost = QtGui.QColor(_CHANNEL_COLORS[channel])
                ghost.setAlpha(70)
                self._draw_curve(painter, x, base[:, channel], ghost, 1)
            # current recipe curves
            curves = editor.current_curves()
            for channel in range(3):
                color = QtGui.QColor(_CHANNEL_COLORS[channel])
                width = 2
                if not editor.linked() and channel != editor.active_channel():
                    color.setAlpha(110)
                    width = 1
                self._draw_curve(painter, x, curves[:, channel], color, width)

            # Phase 10D free mode: fitted-model overlay (dashed), control
            # points of the active channel — and NO parametric handles.
            if editor.free_mode():
                if editor._fit_curves is not None:
                    fit = np.asarray(editor._fit_curves, dtype=float)
                    overlay = QtGui.QColor('#d8b45a')
                    pen = QtGui.QPen(overlay, 1, QtCore.Qt.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    ch = editor.active_channel()
                    finite = np.isfinite(fit[:, ch])
                    path = None
                    for px, py in zip(x[finite], fit[finite, ch]):
                        widget_pt = self._to_widget(float(px), float(py))
                        if path is None:
                            path = QtGui.QPainterPath(widget_pt)
                        else:
                            path.lineTo(widget_pt)
                    if path is not None:
                        painter.drawPath(path)
                color = (_LINKED_COLOR if editor.linked()
                         else _CHANNEL_COLORS[editor.active_channel()])
                painter.setPen(QtGui.QPen(color.darker(150), 1))
                painter.setBrush(color)
                for px, py in editor.free_points(editor.active_channel()):
                    painter.drawEllipse(self._to_widget(px, py), 4.5, 4.5)
                return

            # handles for the active channel
            color = (_LINKED_COLOR if editor.linked()
                     else _CHANNEL_COLORS[editor.active_channel()])
            for name, point in self._handle_positions().items():
                widget_pt = self._to_widget(point.x(), point.y())
                painter.setPen(QtGui.QPen(color.darker(150), 1))
                painter.setBrush(color)
                if name in ('ceiling', 'floor'):
                    painter.drawRect(QtCore.QRectF(widget_pt.x() - 5, widget_pt.y() - 3, 10, 6))
                elif name == 'speed':
                    diamond = QtGui.QPolygonF([
                        widget_pt + QtCore.QPointF(0, -6), widget_pt + QtCore.QPointF(6, 0),
                        widget_pt + QtCore.QPointF(0, 6), widget_pt + QtCore.QPointF(-6, 0)])
                    painter.drawPolygon(diamond)
                elif name == 'gamma':
                    painter.drawRect(QtCore.QRectF(widget_pt.x() - 4.5, widget_pt.y() - 4.5, 9, 9))
                else:
                    painter.drawEllipse(widget_pt, 5.0, 5.0)
        finally:
            painter.end()

    def _draw_curve(self, painter, x, y, color, width) -> None:
        finite = np.isfinite(y)
        xs, ys = x[finite], y[finite]
        if xs.size < 2:
            return
        path = QtGui.QPainterPath(self._to_widget(float(xs[0]), float(ys[0])))
        for px, py in zip(xs[1:], ys[1:]):
            path.lineTo(self._to_widget(float(px), float(py)))
        painter.setPen(QtGui.QPen(color, width))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)

    # ------------------------------------------------------------------ mouse
    def _hit_handle(self, pos: QtCore.QPointF) -> str | None:
        for name, point in self._handle_positions().items():
            widget_pt = self._to_widget(point.x(), point.y())
            if (widget_pt - pos).manhattanLength() <= 18:
                dx, dy = widget_pt.x() - pos.x(), widget_pt.y() - pos.y()
                if (dx * dx + dy * dy) ** 0.5 <= 10:
                    return name
        return None

    def _event_pos(self, event) -> QtCore.QPointF:
        return QtCore.QPointF(event.position() if hasattr(event, 'position') else event.localPos())

    def _hit_free_point(self, pos: QtCore.QPointF) -> int | None:
        for index, (px, py) in enumerate(
                self._editor.free_points(self._editor.active_channel())):
            widget_pt = self._to_widget(px, py)
            dx, dy = widget_pt.x() - pos.x(), widget_pt.y() - pos.y()
            if (dx * dx + dy * dy) ** 0.5 <= 9.0:
                return index
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos = self._event_pos(event)
        if self._editor.free_mode():
            index = self._hit_free_point(pos)
            if index is None:
                index = self._editor.free_add_point(*self._from_widget(pos))
            self._drag = ('freepoint', index, 0.0, ())
            return
        handle = self._hit_handle(pos)
        if handle is None:
            # dragging the curve body = speed (horizontal)
            handle = 'speed'
        channel = self._editor.active_channel()
        p = self._editor.recipe.parametric
        start_values = {
            'speed': p.speed_offset_ev[channel],
            'gamma': p.gamma[channel],
            'toe': p.toe_size[channel],
            'shoulder': p.shoulder_size[channel],
            'ceiling': p.density_max[channel],
            'floor': p.fog[channel],
        }
        self._drag = (handle, pos.x(), pos.y(), (start_values[handle],))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._drag is None:
            return
        if self._drag[0] == 'freepoint':
            x, y = self._from_widget(self._event_pos(event))
            self._editor.free_move_point(self._drag[1], x, y)
            self.update()
            return
        handle, start_x, start_y, (start_value,) = self._drag
        pos = self._event_pos(event)
        rect = self._rect()
        x0, x1, y0, y1 = self._domain()
        dx_curve = (pos.x() - start_x) / rect.width() * (x1 - x0)
        dy_curve = (start_y - pos.y()) / rect.height() * (y1 - y0)
        editor = self._editor

        # defer_emit: the preview render waits for mouse release
        if handle == 'speed':
            # positive EV = faster film = curve moves LEFT
            ev = start_value - dx_curve / math.log10(2.0)
            ev = round(ev / _THIRD_STOP) * _THIRD_STOP      # third-stop snap
            editor.set_param('speed_offset_ev', ev, defer_emit=True)
        elif handle == 'gamma':
            editor.set_param('gamma', start_value * math.pow(2.0, dy_curve), defer_emit=True)
        elif handle == 'toe':
            editor.set_param('toe_size', start_value * math.pow(2.0, dx_curve), defer_emit=True)
        elif handle == 'shoulder':
            editor.set_param('shoulder_size', start_value * math.pow(2.0, -dx_curve), defer_emit=True)
        elif handle == 'ceiling':
            editor.set_param('density_max', start_value + dy_curve, defer_emit=True)
        elif handle == 'floor':
            editor.set_param('fog', start_value + dy_curve, defer_emit=True)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        self._drag = None
        self._editor.commit_deferred_edits()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if not self._editor.free_mode():
            return
        index = self._hit_free_point(self._event_pos(event))
        if index is not None:
            self._drag = None
            self._editor.free_remove_point(index)


class StockEditorDialog(QtWidgets.QDialog):
    """Modeless custom-stock editor. The controller connects the signals."""

    recipeChanged = Signal(object)   # noqa: N815 - live StockRecipe
    stockSaved = Signal(str)         # noqa: N815 - saved slug
    stockDeleted = Signal(str)       # noqa: N815 - deleted slug

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Tool, True)
        self.setWindowTitle('Custom film stock')
        self.setModal(False)
        self.recipe: StockRecipe | None = None
        self.base_profile = None
        self.anchors = np.zeros(3)
        self._linked = True
        self._active_channel = 1
        self._dirty = False
        self._curves_touched = False
        self._loading = False
        self._deferred_edit_pending = False
        # Phase 10D free tier: working per-channel point lists (log_exposure,
        # density) + the last fit-back residual for the label / overlay.
        self._free_mode = False
        self._spline_working: list[list[tuple]] = [[], [], []]
        self._fit_curves = None
        self._emit_timer = QtCore.QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(120)     # drag-stream debounce
        self._emit_timer.timeout.connect(self._emit_recipe_changed)
        self._build_ui()

    # ------------------------------------------------------------------ state
    def load_seed(self, slug: str) -> None:
        """Seed from the given stock: a custom slug reopens its recipe, a
        bundled slug starts a fresh recipe on that base."""
        self._loading = True
        try:
            if is_custom_slug(slug):
                self.recipe = load_custom_stock(slug)
                self.base_profile = load_profile(self.recipe.base_stock)
                if self.recipe.curve_mode == 'base':
                    self.recipe.parametric = fit_parametric_curves(self.base_profile)
            else:
                self.base_profile = load_profile(slug)
                self.recipe = StockRecipe(
                    name=f'{slug} custom', base_stock=slug,
                    curve_mode='base',
                    parametric=fit_parametric_curves(self.base_profile),
                )
            self.anchors = _base_half_density_anchor(self.base_profile)
            self._dirty = False
            self._curves_touched = self.recipe.curve_mode == 'parametric'
            # Phase 10D: a spline recipe reopens in Free mode with its points
            self._fit_curves = None
            if self.recipe.curve_mode == 'spline' and self.recipe.spline_points:
                self._spline_working = [
                    [tuple(p) for p in channel]
                    for channel in self.recipe.spline_points]
                self._set_free_mode(True)
            else:
                self._spline_working = [[], [], []]
                self._set_free_mode(False)
            self._name_edit.setText(self.recipe.name)
            self._base_label.setText(f'base: {self.recipe.base_stock}')
            self._sync_look_widgets()
            self._sync_readouts()
            self._plot.update()
        finally:
            self._loading = False

    def linked(self) -> bool:
        return self._linked

    def active_channel(self) -> int:
        return self._active_channel

    # -------------------------------------------------------- Phase 10D free
    def free_mode(self) -> bool:
        return self._free_mode

    def free_points(self, channel: int) -> list:
        return self._spline_working[channel]

    def _set_free_mode(self, free: bool) -> None:
        self._free_mode = bool(free)
        for name, button in getattr(self, '_mode_buttons', {}).items():
            button.blockSignals(True)
            button.setChecked((name == 'free') == self._free_mode)
            button.blockSignals(False)
        if hasattr(self, '_csv_button'):
            self._csv_button.setVisible(self._free_mode)
        if hasattr(self, '_residual_label'):
            self._residual_label.setVisible(self._free_mode)

    def _on_mode_selected(self, free: bool) -> None:
        if free and not any(self._spline_working):
            self._seed_spline_from_current()
        self._set_free_mode(free)
        self._plot.update()

    def _seed_spline_from_current(self) -> None:
        """Entering Free mode starts ON the current curve: sample it at 9
        control points across the finite exposure range, per channel."""
        curves = self.current_curves()
        x = np.asarray(self.base_profile.data.log_exposure, dtype=float)
        finite = np.isfinite(curves).all(axis=1) & np.isfinite(x)
        xs = np.linspace(x[finite].min(), x[finite].max(), 9)
        self._spline_working = [
            [(float(px), float(np.interp(px, x[finite], curves[finite, ch])))
             for px in xs]
            for ch in range(3)]

    def _sync_spline_into_recipe(self) -> None:
        self.recipe.spline_points = tuple(
            tuple(tuple(point) for point in channel)
            for channel in self._spline_working)

    def _mark_spline_dirty(self) -> None:
        self._dirty = True
        self.recipe.curve_mode = 'spline'
        self._fit_curves = None
        self._sync_spline_into_recipe()

    def free_add_point(self, x: float, y: float) -> int:
        channels = range(3) if self._linked else (self._active_channel,)
        for ch in channels:
            self._spline_working[ch].append((float(x), max(float(y), 0.0)))
            self._spline_working[ch].sort(key=lambda p: p[0])
        self._mark_spline_dirty()
        self._plot.update()
        self._schedule_emit()
        self._update_residual()
        return self._spline_working[self._active_channel].index(
            (float(x), max(float(y), 0.0)))

    def free_move_point(self, index: int, x: float, y: float) -> None:
        points = self._spline_working[self._active_channel]
        if not (0 <= index < len(points)):
            return
        lo = points[index - 1][0] + 1e-3 if index > 0 else -np.inf
        hi = points[index + 1][0] - 1e-3 if index < len(points) - 1 else np.inf
        new_point = (float(np.clip(x, lo, hi)), max(float(y), 0.0))
        channels = range(3) if self._linked else (self._active_channel,)
        for ch in channels:
            if 0 <= index < len(self._spline_working[ch]):
                self._spline_working[ch][index] = new_point
        self._mark_spline_dirty()
        self._deferred_edit_pending = True      # render once on release
        self._plot.update()

    def free_remove_point(self, index: int) -> None:
        channels = range(3) if self._linked else (self._active_channel,)
        removed = False
        for ch in channels:
            if len(self._spline_working[ch]) > 2 and 0 <= index < len(self._spline_working[ch]):
                del self._spline_working[ch][index]
                removed = True
        if removed:
            self._mark_spline_dirty()
            self._plot.update()
            self._schedule_emit()
            self._update_residual()

    def _update_residual(self) -> None:
        """Fit-back RMS: 'how film-like is this drawing'. Computed on commit
        (not per drag frame — the norm_cdfs fit is a least-squares solve)."""
        if not self._free_mode or self.recipe.curve_mode != 'spline':
            return
        from spektrafilm.model.curve_fit import fit_density_curves_model
        from spektrafilm.utils.morph_curves import apply_print_curves_morph_with_layers
        from types import SimpleNamespace

        try:
            drawn = self.current_curves()
            log_exposure = self.base_profile.data.log_exposure
            profile_type = self.recipe.type or self.base_profile.info.type
            model, _ = fit_density_curves_model(
                log_exposure, drawn, n_layers=3, profile_type=profile_type)
            rms = spline_fit_residual(log_exposure, drawn, model, profile_type)
            self._fit_curves, _ = apply_print_curves_morph_with_layers(
                log_exposure, model, SimpleNamespace(active=False),
                profile_type=profile_type)
        except Exception:
            self._residual_label.setText('fit: —')
            return
        self._residual_label.setText(
            'fit RMS: ' + ' / '.join(f'{value:.3f}' for value in rms))
        self._plot.update()

    def import_csv_points(self, path: str) -> None:
        """Datasheet transcription: rows of log_exposure,density (linked) or
        log_exposure,d_r,d_g,d_b (per channel). '#' comments allowed."""
        data = np.genfromtxt(path, delimiter=',', comments='#')
        data = np.atleast_2d(data[np.isfinite(data).all(axis=-1)]
                             if data.ndim > 1 else data[None, :])
        if data.shape[0] < 2 or data.shape[1] not in (2, 4):
            raise ValueError(
                'CSV needs at least 2 rows of log_exposure,density '
                '(or log_exposure,d_r,d_g,d_b).')
        xs = data[:, 0]
        if data.shape[1] == 2:
            columns = [data[:, 1]] * 3
        else:
            columns = [data[:, 1], data[:, 2], data[:, 3]]
        self._spline_working = [
            sorted(((float(x), max(float(y), 0.0)) for x, y in zip(xs, col)),
                   key=lambda p: p[0])
            for col in columns]
        self._set_free_mode(True)
        self._mark_spline_dirty()
        self._plot.update()
        self._schedule_emit()
        self._update_residual()

    def _on_import_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Import curve CSV', '', 'CSV files (*.csv);;All files (*)')
        if not path:
            return
        try:
            self.import_csv_points(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Import curve CSV',
                                          f'Could not import the CSV.\n\n{exc}')

    def current_curves(self) -> np.ndarray:
        if self.recipe.curve_mode == 'spline' and any(self._spline_working):
            self._sync_spline_into_recipe()
            return _spline_curves(self.recipe, self.base_profile)
        if self._free_mode and any(self._spline_working):
            # seeded but not yet committed: preview the drawing
            preview = dataclasses.replace(
                self.recipe, curve_mode='spline',
                spline_points=tuple(
                    tuple(tuple(p) for p in channel)
                    for channel in self._spline_working))
            return _spline_curves(preview, self.base_profile)
        recipe = dataclasses.replace(self.recipe, curve_mode='parametric')
        curves = _parametric_curves(recipe, self.base_profile)
        return _apply_fog(curves, self.recipe.parametric.fog)

    # ----------------------------------------------------------- param edits
    def set_param(self, name: str, value: float, *, from_readout: bool = False,
                  defer_emit: bool = False) -> None:
        """``defer_emit=True`` is the handle-DRAG path (NAT 2026-07-20):
        the plot and readouts update live, but the full-pipeline preview
        render waits for ``commit_deferred_edits`` on mouse release —
        streaming full renders during the drag made editing a nightmare."""
        p = self.recipe.parametric
        current = tuple(getattr(p, name))
        if self._linked:
            if name in ('speed_offset_ev', 'fog'):
                new = tuple(_clamp(name, value) for _ in range(3))
            else:
                # keep the channels' relative ratios on linked edits
                reference = current[self._active_channel] or 1e-6
                factor = _clamp(name, value) / reference
                new = tuple(_clamp(name, v * factor) for v in current)
        else:
            new = tuple(
                _clamp(name, value) if i == self._active_channel else v
                for i, v in enumerate(current))
        if new == current:
            return
        setattr(p, name, new)
        self._mark_curves_dirty()
        if not from_readout:
            self._sync_readouts()
        self._plot.update()
        if defer_emit:
            self._deferred_edit_pending = True
        else:
            self._schedule_emit()

    def commit_deferred_edits(self) -> None:
        """Mouse released after a drag: render ONCE, and only if the drag
        actually changed something (a plain click commits nothing). In Free
        mode the release also refreshes the fit-back residual (the
        least-squares fit is too heavy to run per drag frame)."""
        if self._deferred_edit_pending:
            self._deferred_edit_pending = False
            self._schedule_emit()
            if self._free_mode:
                self._update_residual()

    def set_param_triplet(self, name: str, triplet) -> None:
        p = self.recipe.parametric
        new = tuple(_clamp(name, v) for v in triplet)
        if new == tuple(getattr(p, name)):
            return
        setattr(p, name, new)
        self._mark_curves_dirty()
        self._plot.update()
        self._schedule_emit()

    def _mark_curves_dirty(self) -> None:
        self._dirty = True
        if not self._curves_touched:
            self._curves_touched = True
            self.recipe.curve_mode = 'parametric'

    def _schedule_emit(self) -> None:
        if not self._loading:
            self._emit_timer.start()

    def _emit_recipe_changed(self) -> None:
        self.recipeChanged.emit(self.live_recipe())

    def live_recipe(self) -> StockRecipe:
        recipe = dataclasses.replace(self.recipe)
        recipe.look = self._collect_look()
        return recipe

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        from spektrafilm_gui.widget_editors import BoolEditor, FloatEditor, FloatTupleEditor

        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel('Name'))
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.textEdited.connect(self._on_name_edited)
        header.addWidget(self._name_edit, 1)
        self._base_label = QtWidgets.QLabel('')
        header.addWidget(self._base_label)
        root.addLayout(header)

        channel_row = QtWidgets.QHBoxLayout()
        # Phase 10D: Handles | Free tier toggle on the SAME plot
        self._mode_buttons = {}
        mode_group = QtWidgets.QButtonGroup(self)
        for key, label, tip in (
            ('handles', 'Handles',
             'Parametric tier: physical handles drive the curve family.'),
            ('free', 'Free',
             'Free tier: draw the curve point by point (physics-constrained: '
             'x-sorted, monotone density, non-negative). Saving fits the '
             'parametric model back so push/pull and aging keep composing - '
             'the fit RMS shows how film-like the drawing is.'),
        ):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setChecked(key == 'handles')
            button.clicked.connect(
                lambda _c=False, k=key: self._on_mode_selected(k == 'free'))
            mode_group.addButton(button)
            channel_row.addWidget(button)
            self._mode_buttons[key] = button
        channel_row.addSpacing(12)
        channel_row.addWidget(QtWidgets.QLabel('Link channels'))
        self._link_toggle = BoolEditor()
        self._link_toggle.setChecked(True)
        self._link_toggle.setToolTip(
            'Linked (default): handle drags move all three channels together, '
            'keeping their offsets - crossover is an explicit act (unlink).')
        self._link_toggle.toggled.connect(self._on_link_toggled)
        channel_row.addWidget(self._link_toggle)
        channel_row.addSpacing(12)
        self._channel_buttons = []
        group = QtWidgets.QButtonGroup(self)
        for index, label in enumerate(('R', 'G', 'B')):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setChecked(index == self._active_channel)
            button.setEnabled(False)
            button.clicked.connect(lambda _c=False, i=index: self._on_channel_selected(i))
            group.addButton(button)
            channel_row.addWidget(button)
            self._channel_buttons.append(button)
        channel_row.addStretch(1)
        self._residual_label = QtWidgets.QLabel('fit: —')
        self._residual_label.setToolTip(
            'RMS between the drawn curve and its fitted parametric model - '
            'the smaller, the more film-like the drawing (the morph engines '
            'act through the fit).')
        self._residual_label.setVisible(False)
        channel_row.addWidget(self._residual_label)
        channel_row.addSpacing(12)
        self._speed_label = QtWidgets.QLabel('')
        channel_row.addWidget(self._speed_label)
        root.addLayout(channel_row)

        body = QtWidgets.QHBoxLayout()
        self._plot = _HDPlot(self)
        body.addWidget(self._plot, 1)

        readouts = QtWidgets.QFormLayout()
        self._readouts: dict[str, FloatTupleEditor] = {}
        for name, label, decimals in (
            ('gamma', 'Gamma', 3),
            ('speed_offset_ev', 'Speed (EV)', 2),
            ('density_max', 'D max', 3),
            ('toe_size', 'Toe', 3),
            ('shoulder_size', 'Shoulder', 3),
            ('fog', 'Fog', 3),
        ):
            editor = FloatTupleEditor(3, decimals=decimals)
            for sub_editor in editor.editors:   # TupleEditor has no own signal
                sub_editor.valueChanged.connect(
                    lambda _v=None, n=name, e=editor: self._on_readout_changed(n, e))
            readouts.addRow(label, editor)
            self._readouts[name] = editor
        readout_panel = QtWidgets.QWidget()
        readout_panel.setLayout(readouts)
        readout_panel.setMaximumWidth(300)
        body.addWidget(readout_panel)
        root.addLayout(body, 1)

        # look overrides, collapsed; header names the pipeline stage
        self._look_groups: dict[str, tuple[QtWidgets.QCheckBox, dict]] = {}
        for target, (stage, fields) in _LOOK_FIELDS.items():
            box = QtWidgets.QGroupBox(f'{target} override  ({stage})')
            box.setCheckable(True)
            box.setChecked(False)
            form = QtWidgets.QFormLayout(box)
            # The app theme flattens QGroupBox (border: none), which drops
            # the reserved title band — without an explicit top margin the
            # rows paint OVER the checkable title (NAT 2026-07-20).
            form.setContentsMargins(10, 26, 10, 8)
            editors = {}
            for field_name, label, decimals, step in fields:
                editor = FloatEditor(decimals=decimals)
                editor.setSingleStep(step)
                editor.valueChanged.connect(self._on_look_changed)
                form.addRow(label, editor)
                editors[field_name] = editor
            box.toggled.connect(self._on_look_changed)
            root.addWidget(box)
            self._look_groups[target] = (box, editors)

        buttons = QtWidgets.QHBoxLayout()
        self._csv_button = QtWidgets.QPushButton('Import CSV...')
        self._csv_button.setToolTip(
            'Load characteristic-curve points from a CSV (datasheet '
            'transcription): rows of log_exposure,density - or '
            'log_exposure,d_r,d_g,d_b for per-channel curves.')
        self._csv_button.clicked.connect(self._on_import_csv)
        self._csv_button.setVisible(False)
        buttons.addWidget(self._csv_button)
        for label, slot in (
            ('Save', self._on_save),
            ('Save As', self._on_save_as),
            ('Duplicate', self._on_duplicate),
            ('Revert', self._on_revert),
            ('Delete', self._on_delete),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        close_button = QtWidgets.QPushButton('Close')
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        root.addLayout(buttons)

    # ------------------------------------------------------------- UI slots
    def _on_link_toggled(self, linked: bool) -> None:
        self._linked = bool(linked)
        for button in self._channel_buttons:
            button.setEnabled(not self._linked)
        self._plot.update()

    def _on_channel_selected(self, index: int) -> None:
        self._active_channel = index
        self._plot.update()

    def _on_name_edited(self, _text: str) -> None:
        self._dirty = True

    def _on_readout_changed(self, name: str, editor) -> None:
        if self._loading:
            return
        self.set_param_triplet(name, editor.value)
        self._update_speed_label()

    def _on_look_changed(self, *_args) -> None:
        if self._loading:
            return
        self._dirty = True
        self._schedule_emit()

    def _sync_readouts(self) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            for name, editor in self._readouts.items():
                editor.value = tuple(getattr(self.recipe.parametric, name))
        finally:
            self._loading = was_loading
        self._update_speed_label()

    def _update_speed_label(self) -> None:
        ev = float(np.mean(self.recipe.parametric.speed_offset_ev))
        self._speed_label.setText(f'{ev:+.2f} EV  (~ ISO x{2.0 ** ev:.2f} of base)')

    def _sync_look_widgets(self) -> None:
        look = self.recipe.look or {}
        for target, (box, editors) in self._look_groups.items():
            overrides = look.get(target) or {}
            box.blockSignals(True)
            box.setChecked(bool(overrides))
            box.blockSignals(False)
            for field_name, editor in editors.items():
                editor.blockSignals(True)
                if field_name in overrides:
                    editor.value = float(overrides[field_name])
                else:
                    editor.value = self._look_default(target, field_name)
                editor.blockSignals(False)

    @staticmethod
    def _look_default(target: str, field_name: str) -> float:
        from spektrafilm.runtime.params_schema import (
            DirCouplersParams, GrainParams, HalationParams)

        defaults = {'grain': GrainParams(), 'halation': HalationParams(),
                    'dir_couplers': DirCouplersParams()}[target]
        return float(getattr(defaults, field_name))

    def _collect_look(self) -> dict:
        look: dict = {}
        for target, (box, editors) in self._look_groups.items():
            if box.isChecked():
                look[target] = {name: float(editor.value)
                                for name, editor in editors.items()}
        return look

    # ---------------------------------------------------------------- actions
    def _finalize_recipe(self) -> StockRecipe:
        self.recipe.name = self._name_edit.text().strip() or self.recipe.name
        self.recipe.look = self._collect_look()
        return self.recipe

    def _on_save(self) -> None:
        recipe = self._finalize_recipe()
        if not is_custom_slug(recipe.slug) or recipe.slug not in list_custom_stocks():
            recipe.slug = make_slug(recipe.name)
        self._save(recipe)

    def _on_save_as(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(
            self, 'Save custom stock as', 'Name:', text=self._name_edit.text())
        if not accepted or not name.strip():
            return
        recipe = self._finalize_recipe()
        recipe.name = name.strip()
        recipe.slug = make_slug(recipe.name)
        self._name_edit.setText(recipe.name)
        self._save(recipe)

    def _on_duplicate(self) -> None:
        recipe = dataclasses.replace(self._finalize_recipe())
        recipe.name = f'{recipe.name} copy'
        recipe.slug = make_slug(recipe.name)
        self._name_edit.setText(recipe.name)
        self.recipe = recipe
        self._save(recipe)

    def _save(self, recipe: StockRecipe) -> None:
        try:
            save_custom_stock(recipe)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, 'Save custom stock',
                                           f'Could not save the stock.\n\n{exc}')
            return
        self._dirty = False
        self.stockSaved.emit(recipe.slug)

    def _on_revert(self) -> None:
        if self.recipe is None:
            return
        slug = (self.recipe.slug if self.recipe.slug in list_custom_stocks()
                else self.recipe.base_stock)
        self.load_seed(slug)
        self._dirty = False
        self._schedule_emit()

    def _on_delete(self) -> None:
        recipe = self.recipe
        if recipe is None or recipe.slug not in list_custom_stocks():
            QtWidgets.QMessageBox.information(
                self, 'Delete custom stock', 'This stock has not been saved yet.')
            return
        answer = QtWidgets.QMessageBox.question(
            self, 'Delete custom stock',
            f"Delete '{recipe.name}' ({recipe.slug})?")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        delete_custom_stock(recipe.slug)
        self._dirty = False
        self.stockDeleted.emit(recipe.slug)
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._dirty:
            answer = QtWidgets.QMessageBox.question(
                self, 'Unsaved custom stock',
                'Save the changes to this custom stock before closing?',
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
                | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QtWidgets.QMessageBox.Save:
                self._on_save()
        super().closeEvent(event)

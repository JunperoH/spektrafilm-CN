from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from qtpy import QtCore, QtGui, QtWidgets

QComboBox = QtWidgets.QComboBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QPushButton = QtWidgets.QPushButton
QSizePolicy = QtWidgets.QSizePolicy
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget
Qt = QtCore.Qt
Signal = QtCore.Signal

from spektrafilm_gui.state import (
    AgingState,
    ChemistryState,
    CouplersState,
    DisplayState,
    GlareState,
    GrainState,
    HalationState,
    InputGamutCompressState,
    InputImageState,
    LoadRawState,
    LoadRGBState,
    OutputGamutCompressState,
    PreflashingState,
    PushPullState,
    ScanFinishingState,
    SimulationState,
    SpecialState,
)
from spektrafilm_gui.options import (
    WIDE_GAMUT_COLOR_SPACES as _WIDE_GAMUT_COLOR_SPACES,
    export_format_spec as _export_format_spec,
)
from spektrafilm_gui.persistence import load_dialog_dir, save_dialog_dir
from spektrafilm_gui.preset_store import FOLDER_SEPARATOR as _PRESET_FOLDER_SEPARATOR
from spektrafilm_gui.theme_palette import SIZE_FOOTER_ITEM_SPACING
from spektrafilm_gui.widget_editors import BoolEditor, EnumEditor, FloatEditor, FloatTupleEditor, IntEditor, IntTupleEditor, ProfileEnumEditor
from spektrafilm_gui.widget_primitives import CollapsibleSection, FolderMenuComboBox, normalize_ui_text as _normalize_ui_text
from spektrafilm_gui.widget_specs import GUI_SECTION_ENUMS, get_auxiliary_spec, get_button_spec, get_widget_spec


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


def _build_collapsible_form_section(
    title: str,
    form: QFormLayout,
    *,
    expanded: bool,
) -> QVBoxLayout:
    content = QWidget()
    content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    content.setLayout(form)

    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(CollapsibleSection(title, content, expanded=expanded))
    return root


def _new_form_layout() -> QFormLayout:
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
    # NAT 2026-07-22: an over-wide row WRAPS under its label instead of
    # widening the whole tab past the panel (the SCAN tab was clipping —
    # the scroll area honors the content's minimum width).
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    return form


def _add_form_rows(form: QFormLayout, rows: list[tuple[str | QLabel, QWidget]]) -> None:
    for label, widget in rows:
        if isinstance(label, str):
            label = _normalize_ui_text(label)
        form.addRow(label, widget)


def _build_linked_form_section(
    title: str,
    rows: list[tuple[str | QLabel, QWidget]],
    *,
    expanded: bool,
    extra_top_widget: QWidget | None = None,
) -> QVBoxLayout:
    form = _new_form_layout()
    if extra_top_widget is not None:
        form.addRow(extra_top_widget)
    _add_form_rows(form, rows)
    return _build_collapsible_form_section(title, form, expanded=expanded)


def _build_button(
    text: str,
    callback: Any,
    *,
    tooltip: str | None = None,
    preserve_case: bool = False,
    role: str | None = None,
) -> QPushButton:
    button = QPushButton(_normalize_ui_text(text))
    if role is not None:
        button.setProperty('role', role)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(callback)
    return button


def _build_widget_label(section_name: str, field_name: str) -> QLabel:
    spec = get_widget_spec(section_name, field_name)
    label_text = spec.label or _format_label(field_name)
    label = QLabel(_normalize_ui_text(label_text))
    if spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _build_auxiliary_label(name: str) -> QLabel:
    spec = get_auxiliary_spec(name)
    label_text = spec.label or name.replace("_", " ")
    label = QLabel(_normalize_ui_text(label_text))
    if spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _spec_row(section_name: str, field_name: str, widget: QWidget) -> tuple[QLabel, QWidget]:
    return _build_widget_label(section_name, field_name), widget


def _compound_spec_row(section_name: str, label_field_name: str, *widgets: QWidget) -> tuple[QLabel, QWidget]:
    return _build_widget_label(section_name, label_field_name), _build_inline_container(*widgets, stretch_last=True)


def _build_button_row(*widgets: QWidget, stretch: int | None = None, spacing: int = 6) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        if stretch is None:
            row.addWidget(widget)
        else:
            row.addWidget(widget, stretch)
    return row


def _build_inline_container(
    *widgets: QWidget,
    spacing: int = 6,
    add_stretch: bool = False,
    stretch_last: bool = False,
) -> QWidget:
    container = QWidget()
    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    last_widget_index = len(widgets) - 1
    for index, widget in enumerate(widgets):
        if stretch_last and index == last_widget_index:
            size_policy = widget.sizePolicy()
            size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
            widget.setSizePolicy(size_policy)
            layout.addWidget(widget, 1)
        else:
            layout.addWidget(widget)
    if add_stretch and not stretch_last:
        layout.addStretch(1)
    return container


def _build_vertical_container(*items: QHBoxLayout | QFormLayout | QWidget, spacing: int = 6) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    layout.setAlignment(Qt.AlignTop)
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        else:
            layout.addLayout(item)
    return container


def _set_single_collapsible_layout(widget: QWidget, title: str, content: QWidget, *, expanded: bool = True) -> None:
    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(CollapsibleSection(_normalize_ui_text(title), content, expanded=expanded))
    widget.setLayout(root)


def _format_label(field_name: str) -> str:
    return _normalize_ui_text(field_name.replace("_", " "))


class DataclassSection(QWidget):
    def __init__(
        self,
        *,
        state_cls: type[Any],
        section_name: str,
        title: str,
        enum_fields: dict[str, type[Any]] | None = None,
        hidden_fields: set[str] | None = None,
        collapsed_by_default: bool = False,
    ):
        super().__init__()
        self._state_cls = state_cls
        self._section_name = section_name
        self._title = title
        self._enum_fields = enum_fields or {}
        self._hidden_fields = hidden_fields or set()
        self._collapsed_by_default = collapsed_by_default
        self._type_hints = get_type_hints(state_cls)
        self._init_extra_widgets()
        self._build_ui()
        self._apply_specs()

    def _init_extra_widgets(self) -> None:
        return

    def _build_ui(self) -> None:
        form = _new_form_layout()
        self._form = form            # NAT 2026-07-21: conditional row visibility
        self._row_labels: dict[str, QWidget] = {}
        self._row_widgets: dict[str, QWidget] = {}
        self._add_extra_rows_before(form)
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            annotation = self._type_hints[field_name]
            widget = self._build_editor(field_name, annotation)
            setattr(self, field_name, widget)
            if field_name not in self._hidden_fields:
                label = _build_widget_label(self._section_name, field_name)
                row = self._row_widget(field_name, widget)
                self._row_labels[field_name] = label
                self._row_widgets[field_name] = row
                form.addRow(label, row)
        self._add_extra_rows_after(form)

        content = QWidget()
        content.setLayout(form)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        section = CollapsibleSection(self._title, content, expanded=not self._collapsed_by_default)
        # Phase 15: per-group reset in the header (visible rows only —
        # hidden fields belong to other groups' wrappers).
        section.set_reset_callback(self.reset_all)
        root.addWidget(section)
        self.setLayout(root)

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        del form
        return

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        del form
        return

    def _row_widget(self, field_name: str, widget: QWidget) -> QWidget:
        del field_name
        return widget

    def _build_editor(self, field_name: str, annotation: Any) -> QWidget:
        spec = get_widget_spec(self._section_name, field_name)
        enum_cls = self._enum_fields.get(field_name)
        if enum_cls is not None:
            if self._section_name == 'simulation' and field_name == 'film_stock':
                # Phase 10C: bundled stocks + saved custom stocks + the
                # pinned 'Custom...' creator item.
                from spektrafilm_gui.widget_editors import FilmStockSelector

                return FilmStockSelector(_enum_values(enum_cls))
            if self._section_name == 'simulation' and field_name in {'film_stock', 'print_paper'}:
                return ProfileEnumEditor(_enum_values(enum_cls))
            return EnumEditor(_enum_values(enum_cls))
        if annotation is bool:
            return BoolEditor()
        # Phase 15: scalar rows become slider+box+reset composites (the
        # slider itself appears only once the spec provides a finite range;
        # the composite honors the .value/valueChanged editor contract, so
        # everything downstream is untouched).
        if annotation is int:
            from spektrafilm_gui.widget_slider import SliderFieldEditor

            return SliderFieldEditor(IntEditor())
        if annotation is float:
            from spektrafilm_gui.widget_slider import SliderFieldEditor

            return SliderFieldEditor(
                FloatEditor(decimals=2 if spec.decimals is None else spec.decimals))
        if get_origin(annotation) is tuple:
            element_types = get_args(annotation)
            if element_types and all(element_type is int for element_type in element_types):
                return IntTupleEditor(len(element_types))
            # Phase 15: RGB-marked 3-float triplets = three channel-tinted
            # knobs + boxes on one line (NAT's knob design).
            if spec.rgb and len(element_types) == 3:
                from spektrafilm_gui.widget_slider import KnobTripletEditor

                return KnobTripletEditor(
                    3, decimals=2 if spec.decimals is None else spec.decimals)
            return FloatTupleEditor(len(element_types), decimals=2 if spec.decimals is None else spec.decimals)
        raise TypeError(f"Unsupported field type for {self._state_cls.__name__}.{field_name}: {annotation!r}")

    def _apply_specs(self) -> None:
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            spec = get_widget_spec(self._section_name, field_name)
            widget = getattr(self, field_name)
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            if spec.min_value is not None:
                self._apply_numeric_attr(widget, 'setMinimum', spec.min_value)
            if spec.max_value is not None:
                self._apply_numeric_attr(widget, 'setMaximum', spec.max_value)
            if spec.step is not None:
                self._apply_numeric_attr(widget, 'setSingleStep', spec.step)

    @staticmethod
    def _apply_numeric_attr(widget: QWidget, method_name: str, value: float | int) -> None:
        method = getattr(widget, method_name, None)
        if callable(method):
            method(value)
            return
        editors = getattr(widget, '_editors', None)
        if editors is not None:
            for editor in editors:
                getattr(editor, method_name)(value)

    def set_state(self, state: Any) -> None:
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            getattr(self, field_name).value = getattr(state, field_name)

    def get_state(self) -> Any:
        values = {field_info.name: getattr(self, field_info.name).value for field_info in fields(self._state_cls)}
        return self._state_cls(**values)

    # ------------------------------------------------------ Phase 15 resets
    def seed_baselines(self) -> None:
        """Factory-default baselines for every reset-capable editor (called
        once at bundle creation; profile sync refreshes the synced fields)."""
        from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE

        defaults = getattr(PROJECT_DEFAULT_GUI_STATE, self._section_name, None)
        if defaults is None:
            return
        self.update_baselines(defaults)

    def set_row_visible(self, field_name: str, visible: bool) -> None:
        """NAT 2026-07-21: show only the settings that matter for the
        selected options (grain model, scan-film, ...)."""
        for widget in (self._row_labels.get(field_name),
                       self._row_widgets.get(field_name)):
            if widget is not None:
                widget.setVisible(visible)

    def reset_all(self) -> None:
        """Header reset: every VISIBLE reset-capable row back to baseline
        (hidden fields are surfaced — and reset — by their wrapper groups).
        Each reset emits valueChanged; the auto-preview debounce coalesces."""
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            if field_name in self._hidden_fields:
                continue
            editor = getattr(self, field_name, None)
            reset = getattr(editor, 'reset_to_baseline', None)
            if callable(reset):
                reset()

    def update_baselines(self, section_state: Any, field_names=None) -> None:
        """Set the reset targets. With field_names (the profile-sync list),
        only those fields move — reset then means 'what this stock gave
        you'; everything else keeps its factory-default baseline."""
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            if field_names is not None and field_name not in field_names:
                continue
            editor = getattr(self, field_name, None)
            set_baseline = getattr(editor, 'set_baseline', None)
            if callable(set_baseline) and hasattr(section_state, field_name):
                set_baseline(getattr(section_state, field_name))


class SimpleDataclassSection(DataclassSection):
    STATE_CLS: type[Any]
    SECTION_NAME: str
    TITLE: str
    COLLAPSED_BY_DEFAULT = True
    ENUM_FIELDS_KEY: str | None = None
    HIDDEN_FIELDS: set[str] = set()

    def __init__(self):
        super().__init__(
            state_cls=self.STATE_CLS,
            section_name=self.SECTION_NAME,
            title=self.TITLE,
            enum_fields=GUI_SECTION_ENUMS[self.ENUM_FIELDS_KEY] if self.ENUM_FIELDS_KEY is not None else None,
            hidden_fields=self.HIDDEN_FIELDS,
            collapsed_by_default=self.COLLAPSED_BY_DEFAULT,
        )


class InputImageSection(SimpleDataclassSection):
    STATE_CLS = InputImageState
    SECTION_NAME = 'input_image'
    TITLE = 'Input'
    ENUM_FIELDS_KEY = 'input_image'
    HIDDEN_FIELDS = {
        'upscale_factor',
        'crop',
        'crop_center',
        'crop_size',
        'spectral_upsampling_method',
        'apply_hanatos2025_adaptation_window',
        'apply_hanatos2025_adaptation_surface',
        'spectral_gaussian_blur',
        'filter_uv',
        'filter_ir',
        'color_filter',
    }

    def __init__(self, filepicker_section: 'FilePickerSection'):
        self._filepicker_section = filepicker_section
        super().__init__()


class LoadRawSection(DataclassSection):
    load_requested = Signal(str)

    def __init__(self):
        super().__init__(
            state_cls=LoadRawState,
            section_name='load_raw',
            title='Import Raw',
            enum_fields=GUI_SECTION_ENUMS['load_raw'],
            collapsed_by_default=False,   # RAW is the primary import (NAT 2026-07-11)
        )

    def _init_extra_widgets(self) -> None:
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text('No raw selected'))
        self.reprocess_button = _build_button('reprocess raw', self._reprocess_raw, role='compactAction')
        self.reprocess_button.setEnabled(False)

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        browse_button = _build_button(
            'Select file',
            self._choose_file,
            tooltip='Load and process a raw file using rawpy, output colorspace and cctf as defined in current input widget state',
            role='compactAction',
        )
        form.addRow(_build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=0))

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        form.addRow(self.reprocess_button)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text('Select input raw'), load_dialog_dir('raw_input'))
        if not path:
            return
        save_dialog_dir('raw_input', str(Path(path).parent))
        self.set_path(path)
        self.load_requested.emit(path)

    def _reprocess_raw(self) -> None:
        path = self.file_path.text().strip()
        if not path:
            return
        self.load_requested.emit(path)

    def set_path(self, path: str) -> None:
        self.file_path.setText(path)
        self.reprocess_button.setEnabled(bool(path.strip()))


class PreviewCropSection(QWidget):
    """MAIN tab camera crop, back to the simple numeric form: this crop
    re-frames the film gate (grain/halation rescale). The complete scan-side
    tool (manual draw, aspect ratios, skew, desqueeze — a cutout of the
    rendered film) is the SCAN tab's Frame section."""

    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(_build_linked_form_section(
            'Crop and upscale',
            [
                _spec_row('input_image', 'upscale_factor', input_image_section.upscale_factor),
                _spec_row('input_image', 'crop', input_image_section.crop),
                _spec_row('input_image', 'crop_center', input_image_section.crop_center),
                _spec_row('input_image', 'crop_size', input_image_section.crop_size),
            ],
            expanded=False,
        ))


class ScanFrameSection(QWidget):
    """SCAN tab 'Frame': the complete crop tool on the RENDERED film —
    aspect-ratio selector, Manual (draw on the image) and Move tools, plus
    skew and anamorphic desqueeze. All editors are borrowed from the
    scan_finishing master section; the controller owns the on-canvas
    interaction and listens to ``tool_changed``."""

    tool_changed = Signal(str)                   # 'off' | 'frame' | 'line'
    crop_reset_requested = Signal()              # back to the original photo format
    orientation_swap_requested = Signal()        # landscape <-> portrait (LR's X)
    ratio_changed = Signal(str)                  # preset picked (re-fit armed frame)

    # Phase 9 presets: the historical set + Original + Cinema + Vertical/social
    # (defined once in crop_frame.RATIO_PRESETS).
    ASPECT_RATIOS = None  # set in __init__ from crop_frame

    def __init__(self, scan_finishing: 'ScanFinishingSection'):
        super().__init__()
        from spektrafilm_gui.crop_frame import RATIO_PRESETS

        if type(self).ASPECT_RATIOS is None:
            type(self).ASPECT_RATIOS = RATIO_PRESETS

        self.ratio_combo = QtWidgets.QComboBox()
        # NAT 2026-07-22: let the combo COMPRESS — its content-width minimum
        # was pushing the Frame tool row (a spanning form row, no wrap)
        # past the panel, clipping the whole SCAN tab.
        self.ratio_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ratio_combo.setMinimumContentsLength(6)
        for ratio in RATIO_PRESETS:
            self.ratio_combo.addItem(_normalize_ui_text(ratio))
        self.ratio_combo.setToolTip(_normalize_ui_text(
            'Aspect preset. Anything but Free LOCKS the crop frame to the ratio '
            '(handle drags stay constrained); picking a preset re-fits the frame. '
            'Original = the image\'s own aspect.'))
        self.ratio_combo.currentTextChanged.connect(self.ratio_changed.emit)

        # Phase 9: ONE persistent Lightroom-style crop tool (replaces the old
        # draw-then-move pair) + a straighten-line tool + a crop reset.
        self.crop_button = QPushButton(_normalize_ui_text('Crop'))
        self.crop_button.setProperty('role', 'compactAction')
        self.crop_button.setToolTip(_normalize_ui_text(
            'Lightroom-style crop: a persistent frame with 8 handles over the full scan. '
            'Drag handles to resize (the aspect preset locks the ratio), drag inside to '
            'move the image under the frame (Shift = fine). Enter or double-click '
            'applies, Esc cancels, toggling the tool off applies.'))
        self.crop_button.setCheckable(True)
        self.straighten_button = QPushButton(_normalize_ui_text('Straighten'))
        self.straighten_button.setProperty('role', 'compactAction')
        self.straighten_button.setToolTip(_normalize_ui_text(
            'Drag a line along a horizon or vertical on the image; the skew angle is set '
            'so the line comes out level, and the crop auto-shrinks to stay inside the '
            'rotated image.'))
        self.straighten_button.setCheckable(True)
        self.reset_button = QPushButton(_normalize_ui_text('Reset'))
        self.reset_button.setProperty('role', 'compactAction')
        self.reset_button.setToolTip(_normalize_ui_text(
            'Reset the crop to the original photo format (full frame, ratio Free).'))
        self.reset_button.clicked.connect(self.crop_reset_requested.emit)
        self.swap_button = QPushButton(_normalize_ui_text('Swap'))
        self.swap_button.setProperty('role', 'compactAction')
        self.swap_button.setToolTip(_normalize_ui_text(
            'Swap the crop landscape <-> portrait (like Lightroom\'s X): the frame\'s '
            'dimensions exchange, shrinking only if the swapped shape no longer fits; '
            'a locked ratio preset flips with it (16:9 <-> 9:16).'))
        self.swap_button.clicked.connect(self.orientation_swap_requested.emit)
        self.crop_button.toggled.connect(lambda on: self._on_tool_toggled('frame', on))
        self.straighten_button.toggled.connect(lambda on: self._on_tool_toggled('line', on))

        tool_row = QHBoxLayout()
        tool_row.setContentsMargins(0, 0, 0, 0)
        tool_row.setSpacing(4)
        tool_row.addWidget(self.ratio_combo, 1)
        tool_row.addWidget(self.swap_button)
        tool_row.addWidget(self.reset_button)
        tool_row.addWidget(self.crop_button)
        tool_row.addWidget(self.straighten_button)
        tool_container = QWidget()
        tool_container.setLayout(tool_row)
        tool_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.setLayout(_build_linked_form_section(
            'Frame',
            [
                _spec_row('scan_finishing', 'crop', scan_finishing.crop),
                _spec_row('scan_finishing', 'crop_center', scan_finishing.crop_center),
                _spec_row('scan_finishing', 'crop_size', scan_finishing.crop_size),
                _spec_row('scan_finishing', 'skew_deg', scan_finishing.skew_deg),
                _spec_row('scan_finishing', 'desqueeze', scan_finishing.desqueeze),
            ],
            expanded=True,
            extra_top_widget=tool_container,
        ))

    def _on_tool_toggled(self, tool: str, on: bool) -> None:
        if on:
            other = self.straighten_button if tool == 'frame' else self.crop_button
            if other.isChecked():
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
            self.tool_changed.emit(tool)
        elif not (self.crop_button.isChecked() or self.straighten_button.isChecked()):
            self.tool_changed.emit('off')

    def release_tools(self) -> None:
        for button in (self.crop_button, self.straighten_button):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)

    def aspect_ratio(self) -> float | None:
        """Numeric ratio of the selected preset (None for Free/Original —
        Original needs the image size, resolved controller-side)."""
        from spektrafilm_gui.crop_frame import parse_ratio
        return parse_ratio(self.ratio_combo.currentText())


class GrainSection(SimpleDataclassSection):
    STATE_CLS = GrainState
    SECTION_NAME = 'grain'
    TITLE = 'Grain'
    COLLAPSED_BY_DEFAULT = False
    ENUM_FIELDS_KEY = 'grain'   # Phase 11A: the 'model' dropdown

    # NAT 2026-07-21: only the selected model's settings are shown. Derived
    # from the engine (model/grain.py): which fields each model actually
    # reads. 'active' / 'model' rows are always visible.
    _MODEL_FIELDS = {
        'production': {'sublayers_active', 'particle_area_um2', 'particle_scale',
                       'particle_scale_layers', 'density_min', 'uniformity',
                       'blur', 'blur_dye_clouds_um', 'micro_structure'},
        'preview': {'particle_area_um2', 'particle_scale', 'density_min',
                    'uniformity', 'blur'},
        'rms': {'rms_granularity', 'rms_strength', 'sublayers_active',
                'particle_scale_layers', 'density_min', 'uniformity',
                'blur', 'blur_dye_clouds_um', 'micro_structure'},
        'synthesis': {'density_min', 'synthesis_size', 'synthesis_amount',
                      'synthesis_sharpness', 'synthesis_quality'},
    }
    _MODEL_ROWS = ({'sublayers_active', 'particle_area_um2', 'particle_scale',
                    'particle_scale_layers', 'density_min', 'uniformity',
                    'blur', 'blur_dye_clouds_um', 'micro_structure',
                    'rms_granularity', 'rms_strength', 'synthesis_size',
                    'synthesis_amount', 'synthesis_sharpness',
                    'synthesis_quality'})

    def __init__(self):
        super().__init__()
        self.model.currentTextChanged.connect(self._sync_model_rows)
        self._sync_model_rows(self.model.value)

    def _sync_model_rows(self, model: str) -> None:
        wanted = self._MODEL_FIELDS.get(model, self._MODEL_FIELDS['production'])
        for field_name in self._MODEL_ROWS:
            self.set_row_visible(field_name, field_name in wanted)
    # Phase 11C: the advanced Boolean-model controls live in the collapsed
    # Grain Synthesis section below (editors still built + state-wired here).
    HIDDEN_FIELDS = {
        'synthesis_samples',
        'synthesis_mean_radius_um',
        'synthesis_radius_stddev_ratio',
        'synthesis_aperture_sigma_um',
        'synthesis_cell_size_ratio',
        'synthesis_max_radius_quantile',
        'synthesis_coverage_epsilon',
        'synthesis_max_grains_per_cell',
        'synthesis_radius_scale',
        'synthesis_layered',
        'synthesis_layer_scale',
    }


class GrainSynthesisSection(QWidget):
    """Phase 11C: the collapsed research group for the Boolean grain model,
    borrowing the GrainSection's hidden editors (EnlargerSection pattern)."""

    def __init__(self, grain_section: GrainSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Grain synthesis',
                [
                    _spec_row('grain', name, getattr(grain_section, name))
                    for name in (
                        'synthesis_samples',
                        'synthesis_mean_radius_um',
                        'synthesis_radius_stddev_ratio',
                        'synthesis_aperture_sigma_um',
                        'synthesis_cell_size_ratio',
                        'synthesis_max_radius_quantile',
                        'synthesis_coverage_epsilon',
                        'synthesis_max_grains_per_cell',
                        'synthesis_radius_scale',
                        'synthesis_layered',
                        'synthesis_layer_scale',
                    )
                ],
                expanded=False,
            ),
        )


def _knob_row(section_name: str, entries) -> 'QWidget':
    """Build a LinkedKnobRow from (target_editor, field_name, tint_key,
    text) entries, ranges pulled from the widget specs. NAT 2026-07-22:
    parameter families stored as separate scalar fields collapse to ONE
    row of tinted knobs (enlarger YMC, chemistry RGB, preflash YM, scan
    channel gains)."""
    from spektrafilm_gui.widget_slider import HUE_TINTS, LinkedKnobRow

    units = []
    for target, field_name, tint_key, text in entries:
        spec = get_widget_spec(section_name, field_name)
        step = spec.step or 0.01
        decimals = 0 if float(step) >= 1 else 2
        units.append((target, HUE_TINTS[tint_key], text,
                      float(spec.min_value), float(spec.max_value),
                      float(step), decimals))
    return LinkedKnobRow(units)


class PreflashingSection(SimpleDataclassSection):
    STATE_CLS = PreflashingState
    SECTION_NAME = 'preflashing'
    TITLE = 'Preflash'
    COLLAPSED_BY_DEFAULT = False   # NAT 2026-07-22: open by default
    # NAT 2026-07-22: the filter shifts collapse to one row of C/M/Y knobs
    # (C added same day — same construction as the print C trim).
    HIDDEN_FIELDS = {'y_filter_shift', 'm_filter_shift', 'c_filter_shift'}

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        self.filter_shift_row = _knob_row('preflashing', (
            (self.c_filter_shift, 'c_filter_shift', 'c', 'c'),
            (self.m_filter_shift, 'm_filter_shift', 'm', 'm'),
            (self.y_filter_shift, 'y_filter_shift', 'y', 'y'),
        ))
        form.addRow(QLabel(_normalize_ui_text('Filter shifts')),
                    self.filter_shift_row)


class HalationSection(SimpleDataclassSection):
    STATE_CLS = HalationState
    SECTION_NAME = 'halation'
    TITLE = 'Halation'


class _FaderCell(QtCore.QObject):
    """Console-fader matrix cell (NAT 2026-07-22, WIDGETS-2 'console
    faders' made minimal): a thin vertical fader plus a value box, held as
    SEPARATE widgets so the group can stagger the boxes (the inverted
    pyramid) while keeping one signal contract. Fader drags update the box
    live and COMMIT once on release; the box types/wheels as usual. The
    SELF (diagonal) fader carries the channel tint in its fill."""

    valueChanged = Signal(float)   # noqa: N815 - editor contract
    _FADER_MAX = 1.0               # fader travel (typical gammas); the box
    _STEPS = 200                   # types up to 8, the fader clamps at top

    def __init__(self, tint_name: str | None):
        super().__init__()
        self._syncing = False
        self._drag_pending = False

        self._slider = QtWidgets.QSlider(Qt.Vertical)
        self._slider.setRange(0, self._STEPS)
        self._slider.setFixedHeight(76)   # NAT 2026-07-22: a bit taller
        fill = tint_name or '#8d8d8d'
        self._slider.setStyleSheet(
            'QSlider::groove:vertical { width: 4px; border-radius: 2px;'
            ' background: #262626; }'
            f'QSlider::sub-page:vertical {{ background: #262626; }}'
            f'QSlider::add-page:vertical {{ background: {fill};'
            ' border-radius: 2px; }'
            'QSlider::handle:vertical { height: 8px; margin: 0 -4px;'
            ' border-radius: 4px; background: #e8e8e8; }')

        self._box = FloatEditor(decimals=2, minimum=0.0, maximum=8.0)
        self._box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self._box.setSingleStep(0.02)
        # NAT 2026-07-22: wide enough to read the number properly — the
        # pyramid stagger is what makes this width possible.
        self._box.setFixedWidth(54)
        self._box.setAlignment(Qt.AlignCenter)

        self._box.valueChanged.connect(self._on_box)
        self._slider.valueChanged.connect(self._on_slider)
        from spektrafilm_gui.widget_slider import _begin_drag, _end_drag
        self._slider.sliderPressed.connect(_begin_drag)
        self._slider.sliderReleased.connect(self._on_release)

    def setToolTip(self, text: str) -> None:  # noqa: N802 - widget-style API
        self._slider.setToolTip(text)
        self._box.setToolTip(text)

    @property
    def value(self) -> float:
        return float(self._box.value)

    def setValue(self, value) -> None:  # noqa: N802 - spinbox-style API
        self._box.setValue(float(value))

    def set_silent(self, value) -> None:
        """Display sync from the hidden editors — no commit, no loop."""
        self._box.blockSignals(True)
        self._box.setValue(float(value))
        self._box.blockSignals(False)
        self._sync_slider()

    def _sync_slider(self) -> None:
        fraction = min(max(float(self._box.value) / self._FADER_MAX, 0.0), 1.0)
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(fraction * self._STEPS)))
        self._slider.blockSignals(False)

    def _on_box(self, value) -> None:
        self._sync_slider()
        if not self._syncing:
            self.valueChanged.emit(float(value))

    def _on_slider(self, position: int) -> None:
        new_value = position / self._STEPS * self._FADER_MAX
        self._syncing = True
        try:
            self._box.blockSignals(True)
            self._box.setValue(new_value)     # box follows the fader live
            self._box.blockSignals(False)
        finally:
            self._syncing = False
        # Phase 16A: STREAM every move (the controller's supersede scheduler
        # coalesces; the drag-end callback fires the full-res release frame).
        self.valueChanged.emit(float(self._box.value))

    def _on_release(self) -> None:
        from spektrafilm_gui.widget_slider import _end_drag
        _end_drag()


class CouplerMatrixEditor(QWidget):
    """The DIR inhibition matrix as three CONSOLE-FADER strips, one per
    SOURCE layer (NAT 2026-07-22: WIDGETS-2 'console faders', made
    minimal): each group holds three thin vertical faders — how much that
    layer inhibits r / g / b — with the value box under each. The SELF
    (diagonal, gamma_samelayer_rgb) fader carries the channel tint in its
    fill; your eye compares heights, not digits. Presentation-only: cells
    write through to the section's HIDDEN field editors, so state/bridge/
    profile-sync/presets are untouched by construction."""

    _CHANNELS = ('R', 'G', 'B')
    _FIELDS = ('gamma_samelayer_rgb', 'gamma_interlayer_r_to_gb',
               'gamma_interlayer_g_to_rb', 'gamma_interlayer_b_to_rg')

    def __init__(self, samelayer, r_to_gb, g_to_rb, b_to_rg):
        super().__init__()
        from spektrafilm_gui.widget_slider import CHANNEL_TINTS, _ResetButton

        self._samelayer = samelayer
        self._editors_by_field = {'gamma_samelayer_rgb': samelayer,
                                  'gamma_interlayer_r_to_gb': r_to_gb,
                                  'gamma_interlayer_g_to_rb': g_to_rb,
                                  'gamma_interlayer_b_to_rg': b_to_rg}
        self._offdiag = {(0, 1): (r_to_gb, 0), (0, 2): (r_to_gb, 1),
                         (1, 0): (g_to_rb, 0), (1, 2): (g_to_rb, 1),
                         (2, 0): (b_to_rg, 0), (2, 1): (b_to_rg, 1)}
        self._syncing = False
        self._baselines: dict[str, tuple] = {}

        def _channel_label(prefix: str, name: str, tint) -> QLabel:
            # words WHITE, only the channel letter carries its color
            label = QLabel(f'{prefix} <span style="color:{tint.name()};">'
                           f'{name.lower()}</span>')
            label.setTextFormat(Qt.RichText)
            return label

        self._reset = _ResetButton()
        self._reset.setEnabled(False)
        self._reset.setToolTip('Reset the DIR matrix to the stock defaults')
        self._reset.clicked.connect(self.reset_to_baseline)

        # NAT 2026-07-22 round 2 (IMG_4983): the 'from x' line and the reset
        # sit UNDER the widget; boxes stagger as an INVERTED PYRAMID (r + b
        # on the first line aligned with their faders, g centered on the
        # second, no spacing between the lines) so they get real width; a
        # small spacer separates the widget from the parameter row above.
        # NAT 2026-07-22 (IMG_4990): 16px above the widget (yellow: double),
        # 28px between the packs (green: bigger), 28px fader columns inside
        # a pack (red: smaller). Pack gaps are COMPRESSIBLE spacers (28px
        # preferred, shrink first) so the reset arrow survives a narrow
        # panel.
        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(0, 16, 0, 2)
        row_layout.setSpacing(0)
        self._cells: dict[tuple[int, int], _FaderCell] = {}
        for row, name in enumerate(self._CHANNELS):
            group = QWidget()
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(2)

            labels = QHBoxLayout()
            labels.setContentsMargins(14, 0, 14, 0)
            labels.setSpacing(0)
            faders = QHBoxLayout()
            faders.setContentsMargins(14, 0, 14, 0)
            faders.setSpacing(0)
            group_cells = []
            for col, target in enumerate(self._CHANNELS):
                is_self = row == col
                tint = CHANNEL_TINTS[row].name() if is_self else None
                cell = _FaderCell(tint)
                if is_self:
                    cell.setToolTip(f'Same-layer DIR gamma ({name}): contrast '
                                    f'reduction inside the {name} layer '
                                    '(gamma samelayer rgb)')
                else:
                    cell.setToolTip(f'DIR inhibition from the {name} layer '
                                    f'onto the {target} layer: crossover / '
                                    'saturation axis')
                target_label = _channel_label('→', target,
                                              CHANNEL_TINTS[col])
                target_label.setAlignment(Qt.AlignHCenter)
                target_label.setFixedWidth(28)
                labels.addWidget(target_label)
                holder = QWidget()
                holder.setFixedWidth(28)
                holder_layout = QVBoxLayout(holder)
                holder_layout.setContentsMargins(0, 0, 0, 0)
                holder_layout.addWidget(cell._slider, 0, Qt.AlignHCenter)
                faders.addWidget(holder)
                cell.valueChanged.connect(
                    lambda value, r=row, c=col: self._push(r, c, value))
                self._cells[(row, col)] = cell
                group_cells.append(cell)
            group_layout.addLayout(labels)
            group_layout.addLayout(faders)

            # NAT 2026-07-22 (IMG_4990, pink): a COMPACT pyramid — the two
            # top boxes touch in the middle, the third centered under the
            # seam; numbers stay centered inside.
            boxes = QVBoxLayout()
            boxes.setContentsMargins(0, 0, 0, 0)
            boxes.setSpacing(3)                # NAT 2026-07-22: a very small
            boxes_top = QHBoxLayout()          # separator between the boxes
            boxes_top.setContentsMargins(0, 0, 0, 0)
            boxes_top.setSpacing(3)
            boxes_top.addStretch(1)
            boxes_top.addWidget(group_cells[0]._box)
            boxes_top.addWidget(group_cells[2]._box)
            boxes_top.addStretch(1)
            boxes.addLayout(boxes_top)
            boxes_bottom = QHBoxLayout()       # g, centered under the seam
            boxes_bottom.setContentsMargins(0, 0, 0, 0)
            boxes_bottom.addWidget(group_cells[1]._box, 0, Qt.AlignHCenter)
            boxes.addLayout(boxes_bottom)
            group_layout.addLayout(boxes)
            group_layout.addSpacing(2)
            group_layout.addWidget(_channel_label('from', name,
                                                  CHANNEL_TINTS[row]),
                                   0, Qt.AlignHCenter)
            group.setFixedWidth(112)
            if row > 0:
                row_layout.addSpacerItem(QtWidgets.QSpacerItem(
                    28, 0, QSizePolicy.Maximum, QSizePolicy.Minimum))
            row_layout.addWidget(group)
        row_layout.addStretch(1)
        row_layout.addWidget(self._reset, 0, Qt.AlignBottom)

        samelayer.valueChanged.connect(lambda _value: self._pull())
        for tuple_editor, index in self._offdiag.values():
            tuple_editor.editors[index].valueChanged.connect(
                lambda _value: self._pull())
        self._pull()

    # ------------------------------------------------------------ sync plumbing
    def _push(self, row: int, col: int, value) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if row == col:
                values = list(self._samelayer.value)
                values[row] = float(value)
                self._samelayer.value = tuple(values)    # emits -> render
            else:
                tuple_editor, index = self._offdiag[(row, col)]
                tuple_editor.editors[index].value = float(value)
        finally:
            self._syncing = False
        self._sync_modified()

    def _pull(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            diagonal = self._samelayer.value
            for index in range(3):
                self._set_cell((index, index), diagonal[index])
            for key, (tuple_editor, index) in self._offdiag.items():
                self._set_cell(key, tuple_editor.editors[index].value)
        finally:
            self._syncing = False
        self._sync_modified()

    def _set_cell(self, key: tuple[int, int], value) -> None:
        self._cells[key].set_silent(float(value))

    # ---------------------------------------------------------------- baselines
    def capture_baselines(self, section_state, field_names=None) -> None:
        for field in self._FIELDS:
            if field_names is None or field in field_names:
                self._baselines[field] = tuple(getattr(section_state, field))
        self._reset.setEnabled(bool(self._baselines))
        self._sync_modified()

    def reset_to_baseline(self) -> None:
        for field, baseline in self._baselines.items():
            self._editors_by_field[field].value = tuple(baseline)

    def _sync_modified(self) -> None:
        modified = any(
            tuple(float(v) for v in self._editors_by_field[field].value)
            != tuple(float(v) for v in baseline)
            for field, baseline in self._baselines.items())
        self._reset.set_modified(modified)


class CouplersSection(SimpleDataclassSection):
    STATE_CLS = CouplersState
    SECTION_NAME = 'couplers'
    TITLE = 'Couplers'
    COLLAPSED_BY_DEFAULT = False
    # NAT 2026-07-22: the four gamma rows are surfaced as ONE 3x3 DIR matrix
    # grid below (editors still built + state-wired; the grid writes through).
    HIDDEN_FIELDS = {'gamma_samelayer_rgb', 'gamma_interlayer_r_to_gb',
                     'gamma_interlayer_g_to_rb', 'gamma_interlayer_b_to_rg'}

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        self.dir_matrix = CouplerMatrixEditor(
            self.gamma_samelayer_rgb, self.gamma_interlayer_r_to_gb,
            self.gamma_interlayer_g_to_rb, self.gamma_interlayer_b_to_rg)
        form.addRow(self.dir_matrix)

    def update_baselines(self, section_state, field_names=None) -> None:
        super().update_baselines(section_state, field_names)
        self.dir_matrix.capture_baselines(section_state, field_names)


class GlareSection(SimpleDataclassSection):
    STATE_CLS = GlareState
    SECTION_NAME = 'glare'
    TITLE = 'Glare'


class ChemistrySection(SimpleDataclassSection):
    STATE_CLS = ChemistryState
    SECTION_NAME = 'chemistry'
    TITLE = 'Chemistry'
    COLLAPSED_BY_DEFAULT = False
    # NAT 2026-07-22: the three per-channel gammas collapse to one RGB knob
    # row like the other triplets.
    HIDDEN_FIELDS = {'gamma_factor_red', 'gamma_factor_green',
                     'gamma_factor_blue'}

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        self.gamma_rgb_row = _knob_row('chemistry', (
            (self.gamma_factor_red, 'gamma_factor_red', 'r', None),
            (self.gamma_factor_green, 'gamma_factor_green', 'g', None),
            (self.gamma_factor_blue, 'gamma_factor_blue', 'b', None),
        ))
        form.addRow(QLabel(_normalize_ui_text('Gamma rgb')),
                    self.gamma_rgb_row)


class PushPullSection(SimpleDataclassSection):
    STATE_CLS = PushPullState
    SECTION_NAME = 'push_pull'
    TITLE = 'Push / Pull'
    ENUM_FIELDS_KEY = 'push_pull'
    COLLAPSED_BY_DEFAULT = False


class AgingSection(SimpleDataclassSection):
    # Film expiration (emulsion aging) — sibling of Push/Pull on the FILM tab.
    STATE_CLS = AgingState
    SECTION_NAME = 'aging'
    TITLE = 'Aging'
    ENUM_FIELDS_KEY = 'aging'


class ScanFinishingSection(DataclassSection):
    """SCAN tab master section ('Scan exposure'): the scanner's analog side â€”
    master bypass, exposure and channel gains. The remaining editors are
    hidden here and BORROWED by the sibling wrapper sections (Levels /
    Response / Color / Toning), all writing into the one scan_finishing
    state. Auto levels asks the controller to measure the rendered frame's
    per-channel floors/ceilings â€” the analysis vocabulary the future
    real-film inversion will build on."""

    auto_levels_requested = Signal()

    def __init__(self):
        super().__init__(
            state_cls=ScanFinishingState,
            section_name='scan_finishing',
            title='Exposure',
            collapsed_by_default=False,
            hidden_fields={
                'crop', 'crop_center', 'crop_size', 'skew_deg', 'desqueeze',
                # NAT 2026-07-22: the three channel gains collapse to one RGB
                # knob row (opponent-axis texts) below.
                'gain_red_ev', 'gain_green_ev', 'gain_blue_ev',
                'black_point', 'white_point', 'black_point_rgb', 'white_point_rgb',
                'gamma', 'contrast', 'toe', 'shoulder', 'micro_contrast',
                'color_separation', 'saturation', 'vibrance',
                'hue_saturation_ryg', 'hue_saturation_cbm',
                'shadow_tint_hue', 'shadow_tint_strength',
                'highlight_tint_hue', 'highlight_tint_strength',
                'curve_luma', 'curve_r', 'curve_g', 'curve_b',
            },
        )

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        self.channel_gain_row = _knob_row('scan_finishing', (
            (self.gain_red_ev, 'gain_red_ev', 'r', 'c↔r'),
            (self.gain_green_ev, 'gain_green_ev', 'g', 'm↔g'),
            (self.gain_blue_ev, 'gain_blue_ev', 'b', 'y↔b'),
        ))
        form.addRow(QLabel(_normalize_ui_text('Gains ev')),
                    self.channel_gain_row)

    _CURVE_FIELDS = ('curve_luma', 'curve_r', 'curve_g', 'curve_b')

    def _build_editor(self, field_name: str, annotation: Any) -> QWidget:
        # The point-curve fields are tuples of (x, y) POINTS, not numbers —
        # they get the invisible holder the ScanCurveSection's plot binds to.
        if field_name in self._CURVE_FIELDS:
            from spektrafilm_gui.curve_widget import CurvePointsEditor

            return CurvePointsEditor()
        # NAT 2026-07-22: one knob PER HUE with its own accent color.
        if field_name in ('hue_saturation_ryg', 'hue_saturation_cbm'):
            from spektrafilm_gui.widget_slider import HUE_TINTS, KnobTripletEditor

            keys = 'ryg' if field_name.endswith('ryg') else 'cbm'
            return KnobTripletEditor(
                3, decimals=2, tints=[HUE_TINTS[key] for key in keys])
        return super()._build_editor(field_name, annotation)

    def _init_extra_widgets(self) -> None:
        self.auto_levels_button = _build_button(
            'Auto levels', self.auto_levels_requested.emit,
            tooltip='Measure the rendered output (0.1 / 99.9 percentiles) and set the master '
                    'black/white points + a midtone gamma from it. Luminance only: the '
                    'per-channel offsets are reset to zero, so no cast is introduced.',
            role='compactAction',
        )


class ScanLevelsSection(QWidget):
    def __init__(self, scan_finishing: ScanFinishingSection):
        super().__init__()
        self.setLayout(_build_linked_form_section(
            'Levels',
            [
                _spec_row('scan_finishing', 'black_point', scan_finishing.black_point),
                _spec_row('scan_finishing', 'white_point', scan_finishing.white_point),
                _spec_row('scan_finishing', 'black_point_rgb', scan_finishing.black_point_rgb),
                _spec_row('scan_finishing', 'white_point_rgb', scan_finishing.white_point_rgb),
                _spec_row('scan_finishing', 'gamma', scan_finishing.gamma),
            ],
            expanded=True,
            extra_top_widget=scan_finishing.auto_levels_button,
        ))


class ScanResponseSection(QWidget):
    def __init__(self, scan_finishing: ScanFinishingSection):
        super().__init__()
        self.setLayout(_build_linked_form_section(
            'Response',
            [
                _spec_row('scan_finishing', 'contrast', scan_finishing.contrast),
                _spec_row('scan_finishing', 'toe', scan_finishing.toe),
                _spec_row('scan_finishing', 'shoulder', scan_finishing.shoulder),
                _spec_row('scan_finishing', 'micro_contrast', scan_finishing.micro_contrast),
            ],
            expanded=False,
        ))


class ScanColorSection(QWidget):
    def __init__(self, scan_finishing: ScanFinishingSection):
        super().__init__()
        self.setLayout(_build_linked_form_section(
            'Color',
            [
                _spec_row('scan_finishing', 'color_separation', scan_finishing.color_separation),
                _spec_row('scan_finishing', 'saturation', scan_finishing.saturation),
                _spec_row('scan_finishing', 'vibrance', scan_finishing.vibrance),
                _spec_row('scan_finishing', 'hue_saturation_ryg', scan_finishing.hue_saturation_ryg),
                _spec_row('scan_finishing', 'hue_saturation_cbm', scan_finishing.hue_saturation_cbm),
            ],
            expanded=False,
        ))


class ScanToningSection(QWidget):
    def __init__(self, scan_finishing: ScanFinishingSection):
        super().__init__()
        self.setLayout(_build_linked_form_section(
            'Toning',
            [
                _spec_row('scan_finishing', 'shadow_tint_hue', scan_finishing.shadow_tint_hue),
                _spec_row('scan_finishing', 'shadow_tint_strength', scan_finishing.shadow_tint_strength),
                _spec_row('scan_finishing', 'highlight_tint_hue', scan_finishing.highlight_tint_hue),
                _spec_row('scan_finishing', 'highlight_tint_strength', scan_finishing.highlight_tint_strength),
            ],
            expanded=False,
        ))


class ScanCurveSection(QWidget):
    """Phase 10A: the finishing point curve (below Toning; the engine applies
    it after split toning and before micro_contrast)."""

    def __init__(self, scan_finishing: ScanFinishingSection):
        super().__init__()
        from spektrafilm_gui.curve_widget import CurveEditorView

        self.curve_view = CurveEditorView({
            'luma': scan_finishing.curve_luma,
            'r': scan_finishing.curve_r,
            'g': scan_finishing.curve_g,
            'b': scan_finishing.curve_b,
        })
        self.curve_view.setToolTip(
            'Lightroom-style point curve on the encoded scan: Luma applies to '
            'all channels, then the per-channel curves. Click to add a point, '
            'drag to move, double-click to remove.'
        )
        form = _new_form_layout()
        form.addRow(self.curve_view)
        self.setLayout(_build_collapsible_form_section('Curve', form, expanded=False))


class InputGamutCompressSection(SimpleDataclassSection):
    STATE_CLS = InputGamutCompressState
    SECTION_NAME = 'input_gamut_compress'
    TITLE = 'Input gamut compress'
    ENUM_FIELDS_KEY = 'input_gamut_compress'


class OutputGamutCompressSection(SimpleDataclassSection):
    STATE_CLS = OutputGamutCompressState
    SECTION_NAME = 'output_gamut_compress'
    TITLE = 'Output gamut compress'
    ENUM_FIELDS_KEY = 'output_gamut_compress'
    COLLAPSED_BY_DEFAULT = False


class SpecialSection(DataclassSection):
    def __init__(self, simulation_section: 'SimulationSection'):
        self._simulation_section = simulation_section
        super().__init__(
            state_cls=SpecialState,
            section_name='special',
            title='Experimental',
            collapsed_by_default=True,
            hidden_fields={
                'film_gamma_factor',
                'print_gamma_factor',
            },
        )

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        form.addRow(_build_widget_label('simulation', 'print_illuminant'), self._simulation_section.print_illuminant)


class SpectralUpsamplingSection(QWidget):
    # NAT 2026-07-22: the method + taking filter moved to MAIN under
    # Profiles (SimulationSection.adopt_spectral_rows); the tuning knobs
    # stay here on ADVANCED.
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Spectral upsampling',
                [
                    _spec_row('input_image', 'apply_hanatos2025_adaptation_window', input_image_section.apply_hanatos2025_adaptation_window),
                    _spec_row('input_image', 'apply_hanatos2025_adaptation_surface', input_image_section.apply_hanatos2025_adaptation_surface),
                    _spec_row('input_image', 'spectral_gaussian_blur', input_image_section.spectral_gaussian_blur),
                    _spec_row('input_image', 'filter_uv', input_image_section.filter_uv),
                    _spec_row('input_image', 'filter_ir', input_image_section.filter_ir),
                ],
                expanded=True,
            ),
        )


class TuneSection(QWidget):
    def __init__(self, special_section: SpecialSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Tune',
                [
                    _spec_row('special', 'film_gamma_factor', special_section.film_gamma_factor),
                    _spec_row('special', 'print_gamma_factor', special_section.print_gamma_factor),
                ],
                expanded=True,
            ),
        )


class FilePickerSection(DataclassSection):
    """Import RGB panel: file picker plus raw-style white balance controls."""

    load_requested = Signal(str)

    def __init__(self):
        super().__init__(
            state_cls=LoadRGBState,
            section_name='load_rgb',
            title='Import RGB',
            enum_fields=GUI_SECTION_ENUMS['load_rgb'],
            collapsed_by_default=True,    # Import Raw is default-open instead
            # 'Color adaptation' (recover_16bit) removed per NAT 2026-07-12;
            # the state field survives so old .sfroll files still load.
            hidden_fields={'recover_16bit'},
        )
        self._connect_auto_reprocess_signals()

    def _init_extra_widgets(self) -> None:
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text('No image selected'))
        self.reprocess_button = _build_button('reprocess rgb', self._reprocess_rgb, role='compactAction')
        self.reprocess_button.setEnabled(False)
        # White balance / tint / recovery apply at import time, so a settings
        # change must re-import the file, not merely re-render the simulation
        # from the unchanged stored image. Debounced so spinner scrubbing does
        # not queue one reload per step.
        self._reprocess_timer = QtCore.QTimer(self)
        self._reprocess_timer.setSingleShot(True)
        self._reprocess_timer.setInterval(400)
        self._reprocess_timer.timeout.connect(self._reprocess_rgb)

    def _connect_auto_reprocess_signals(self) -> None:
        for field_info in fields(LoadRGBState):
            editor = getattr(self, field_info.name)
            for signal_name in ('toggled', 'currentTextChanged', 'valueChanged'):
                signal = getattr(editor, signal_name, None)
                if signal is not None and hasattr(signal, 'connect'):
                    signal.connect(self.schedule_reprocess)
                    break

    def schedule_reprocess(self) -> None:
        if self.file_path.text().strip():
            self._reprocess_timer.start()

    def cancel_scheduled_reprocess(self) -> None:
        self._reprocess_timer.stop()

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        browse_button = _build_button(
            'Select file',
            self._choose_file,
            tooltip='Load an RGB image (e.g. TIFF/PNG/JPEG), applying the white balance settings below',
            role='compactAction',
        )
        form.addRow(_build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=0))

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        form.addRow(self.reprocess_button)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text('Select input image'), load_dialog_dir('rgb_input'))
        if not path:
            return
        save_dialog_dir('rgb_input', str(Path(path).parent))
        self.set_path(path)
        self.load_requested.emit(path)

    def _reprocess_rgb(self) -> None:
        path = self.file_path.text().strip()
        if not path:
            return
        self.load_requested.emit(path)

    def set_path(self, path: str) -> None:
        # A new path supersedes any pending debounced reprocess of the old one.
        self._reprocess_timer.stop()
        self.file_path.setText(path)
        self.reprocess_button.setEnabled(bool(path.strip()))


class RollSection(QWidget):
    """Multi-file roll panel (Phase 5 / B1): file list, position, navigation,
    and reference-look propagation. Pure view -- the controller owns the
    RollSession and reacts to these signals."""

    files_added = Signal(list)          # list[str] absolute paths
    file_activated = Signal(int)        # index the user wants to switch to
    remove_requested = Signal(int)      # index to remove
    clear_requested = Signal()          # empty the whole roll
    apply_look_requested = Signal()     # stamp current look onto every file
    analyze_requested = Signal()        # measure every frame's auto EV (roll average)
    apply_roll_exposure_requested = Signal()  # stamp the roll-average EV onto all frames
    save_roll_requested = Signal()      # save the roll session to a .sfroll file
    open_roll_requested = Signal()      # open a .sfroll roll session file

    _DIALOG_FILTER = (
        'Images (*.raf *.nef *.nrw *.cr2 *.cr3 *.crw *.dng *.arw *.srf *.sr2 '
        '*.orf *.rw2 *.pef *.srw *.kdc *.dcr *.3fr *.iiq *.erf *.mef *.mos '
        '*.mrw *.x3f *.fff *.rwl *.tif *.tiff *.png *.jpg *.jpeg *.exr *.bmp '
        '*.webp);;All files (*)'
    )

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setObjectName('rollFileList')
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list.itemActivated.connect(self._emit_activated)
        self.file_list.itemClicked.connect(self._emit_activated)
        self._update_list_height()

        self.position_label = QLabel(_normalize_ui_text('0 / 0'))
        self.position_label.setObjectName('rollPositionLabel')

        add_button = _build_button(
            'Add files', self._choose_files,
            tooltip='Add RAW or RGB images to the roll (multi-select). RAWs go through Import Raw, the rest through Import RGB.',
            role='compactAction',
        )
        remove_button = _build_button(
            'Remove', self._emit_remove,
            tooltip='Remove the selected file from the roll (the file on disk is untouched)',
            role='compactAction',
        )
        clear_button = _build_button(
            'Clear', self.clear_requested.emit,
            tooltip='Empty the whole roll (files on disk are untouched; per-frame looks are dropped)',
            role='compactAction',
        )
        # NAT 2026-07-22: arrows match the top-bar stepper (native, 20x24)
        def _arrow_button(arrow, callback, tip):
            button = QtWidgets.QToolButton()
            button.setArrowType(arrow)
            button.setToolTip(tip)
            button.setFixedSize(20, 24)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(callback)
            return button

        prev_button = _arrow_button(Qt.LeftArrow, lambda: self._step(-1), 'Previous frame')
        next_button = _arrow_button(Qt.RightArrow, lambda: self._step(+1), 'Next frame')
        self.apply_look_button = _build_button(
            'Apply look to all', self.apply_look_requested.emit,
            tooltip='Stamp the current settings (the reference look) onto every file in the roll. '
                    'Frames without a stamped look inherit whatever the controls currently show.',
            role='compactAction',
        )
        self.apply_look_button.setEnabled(False)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(4)
        nav_row.addWidget(prev_button)
        nav_row.addWidget(self.position_label)
        nav_row.addWidget(next_button)
        nav_row.addStretch(1)
        nav_row.addWidget(add_button)
        nav_row.addWidget(remove_button)
        nav_row.addWidget(clear_button)
        nav_container = QWidget()
        nav_container.setLayout(nav_row)
        # Fixed vertical policies + a top-aligned stack: otherwise any extra
        # vertical space handed to the section is distributed INTO the rows,
        # which reads as empty gaps around the frame counter.
        nav_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_look_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # ---- roll normalization (NegPy Batch Analysis) -----------------
        self.analyze_button = _build_button(
            'Analyze roll', self.analyze_requested.emit,
            tooltip='Measure every frame\'s auto-exposure EV (off the UI thread) and compute the '
                    'roll average with outliers discarded. Nothing is changed until you apply it.',
            role='compactAction',
        )
        self.apply_roll_exposure_button = _build_button(
            'Use roll exposure', self.apply_roll_exposure_requested.emit,
            tooltip='Stamp the roll-average EV onto every frame: per-frame auto exposure OFF, one '
                    'shared exposure for the whole roll (consistent brightness relationship, like '
                    'printing the roll with one enlarger setting). Re-applying after a new analysis '
                    'replaces the previous roll EV, it never compounds. Your own exposure '
                    'compensation on top of it is preserved.',
            role='compactAction',
        )
        self.apply_roll_exposure_button.setEnabled(False)
        analysis_row = QHBoxLayout()
        analysis_row.setContentsMargins(0, 0, 0, 0)
        analysis_row.setSpacing(4)
        analysis_row.addWidget(self.analyze_button)
        analysis_row.addWidget(self.apply_roll_exposure_button)
        analysis_container = QWidget()
        analysis_container.setLayout(analysis_row)
        analysis_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # ---- roll session file (Phase 8: .sfroll save/open) -------------
        self.save_roll_button = _build_button(
            'Save roll', self.save_roll_requested.emit,
            tooltip='Save the roll as a .sfroll session file: the photo locations, every '
                    'frame\'s edits and the roll exposure — never the photos themselves. '
                    'Double-click the file later to reopen the roll exactly as left.',
            role='compactAction',
        )
        self.open_roll_button = _build_button(
            'Open roll', self.open_roll_requested.emit,
            tooltip='Open a .sfroll roll session file (replaces the current roll; source '
                    'images are found relative to the roll file first, so a synced or '
                    'moved folder keeps working).',
            role='compactAction',
        )
        session_row = QHBoxLayout()
        session_row.setContentsMargins(0, 0, 0, 0)
        session_row.setSpacing(4)
        session_row.addWidget(self.save_roll_button)
        session_row.addWidget(self.open_roll_button)
        session_container = QWidget()
        session_container.setLayout(session_row)
        session_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.analysis_label = QLabel('')
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet('color: #888888;')
        self.analysis_label.setVisible(False)
        self.analysis_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        content_layout.setAlignment(Qt.AlignTop)
        content_layout.addWidget(nav_container)
        content_layout.addWidget(self.file_list)
        content_layout.addWidget(self.apply_look_button)
        content_layout.addWidget(analysis_container)
        content_layout.addWidget(session_container)
        content_layout.addWidget(self.analysis_label)
        content = QWidget()
        content.setLayout(content_layout)
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(CollapsibleSection(_normalize_ui_text('Roll'), content, expanded=True))
        self.setLayout(root)

    # ------------------------------------------------------------- slots
    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, _normalize_ui_text('Add files to roll'),
            load_dialog_dir('roll_input'), self._DIALOG_FILTER,
        )
        if not paths:
            return
        save_dialog_dir('roll_input', str(Path(paths[0]).parent))
        self.files_added.emit(list(paths))

    def _emit_activated(self, item) -> None:
        row = self.file_list.row(item)
        if row >= 0:
            self.file_activated.emit(row)

    def _emit_remove(self) -> None:
        row = self.file_list.currentRow()
        if row >= 0:
            self.remove_requested.emit(row)

    def _step(self, delta: int) -> None:
        count = self.file_list.count()
        if count == 0:
            return
        row = self.file_list.currentRow()
        new_row = max(0, min(count - 1, (row if row >= 0 else 0) + delta))
        if new_row != row:
            self.file_activated.emit(new_row)

    # ------------------------------------------------------------- view API
    def _update_list_height(self) -> None:
        # Size the list to its contents (1..8 rows) so a short roll does not
        # leave a block of empty space inside the section.
        count = self.file_list.count()
        row_h = self.file_list.sizeHintForRow(0) if count > 0 else self.file_list.fontMetrics().height() + 6
        rows = max(1, min(count, 8))
        frame = 2 * self.file_list.frameWidth()
        self.file_list.setFixedHeight(rows * max(row_h, 12) + frame + 2)

    def set_files(self, names: list[str], current_index: int, position: str) -> None:
        """Rebuild the list without re-emitting selection signals."""
        was_blocked = self.file_list.blockSignals(True)
        try:
            self.file_list.clear()
            for name in names:
                self.file_list.addItem(_normalize_ui_text(name))
            if 0 <= current_index < len(names):
                self.file_list.setCurrentRow(current_index)
        finally:
            self.file_list.blockSignals(was_blocked)
        self._update_list_height()
        self.position_label.setText(_normalize_ui_text(position))
        self.apply_look_button.setEnabled(len(names) > 0)

    def set_current(self, index: int, position: str) -> None:
        was_blocked = self.file_list.blockSignals(True)
        try:
            self.file_list.setCurrentRow(index)
        finally:
            self.file_list.blockSignals(was_blocked)
        self.position_label.setText(_normalize_ui_text(position))

    def set_analyzing(self, analyzing: bool) -> None:
        self.analyze_button.setText(_normalize_ui_text('Cancel analysis' if analyzing else 'Analyze roll'))

    def set_analysis_result(self, text: str, apply_enabled: bool) -> None:
        self.analysis_label.setText(_normalize_ui_text(text))
        self.analysis_label.setVisible(bool(text))
        self.apply_roll_exposure_button.setEnabled(apply_enabled)


class FilmStripWidget(QWidget):
    """Horizontal strip of small UNEDITED source thumbnails under the image
    (like frames on a contact sheet). Pure view: the controller feeds entries
    and thumbnails, clicking a frame asks the controller to switch. Hidden
    while the roll is empty so single-image workflows are unchanged."""

    file_activated = Signal(int)

    THUMB_W = 125  # 96 * 1.3
    THUMB_H = 83   # 64 * 1.3

    def __init__(self):
        super().__init__()
        self.strip = QtWidgets.QListWidget()
        self.strip.setObjectName('filmStripList')
        self.strip.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.strip.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QtWidgets.QListView.Movement.Static)
        self.strip.setIconSize(QtCore.QSize(self.THUMB_W, self.THUMB_H))
        self.strip.setUniformItemSizes(True)
        self.strip.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # NAT 2026-07-22: the card's scrollbar is 6px now — tighter allowance
        self.strip.setFixedHeight(self.THUMB_H + 12)
        self.strip.itemClicked.connect(self._emit_activated)
        self.strip.itemActivated.connect(self._emit_activated)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.strip)
        self.setLayout(layout)
        self.setVisible(False)

        pm = QtGui.QPixmap(self.THUMB_W, self.THUMB_H)
        pm.fill(QtGui.QColor(60, 60, 60))
        self._placeholder = QtGui.QIcon(pm)

    def _emit_activated(self, item) -> None:
        row = self.strip.row(item)
        if row >= 0:
            self.file_activated.emit(row)

    # ------------------------------------------------------------- view API
    def set_entries(self, names: list[str], pixmaps: list, current_index: int,
                    dimmed: list | None = None) -> None:
        """Rebuild the strip (signal-blocked). ``pixmaps`` holds a QPixmap or
        None (placeholder) per entry; ``dimmed`` flags frames filtered out by
        the star filter (disabled = grayed, not clickable, nav skips them)."""
        was_blocked = self.strip.blockSignals(True)
        try:
            self.strip.clear()
            for index, (name, pm) in enumerate(zip(names, pixmaps)):
                item = QtWidgets.QListWidgetItem()
                item.setIcon(QtGui.QIcon(pm) if pm is not None else self._placeholder)
                item.setToolTip(_normalize_ui_text(name))
                item.setSizeHint(QtCore.QSize(self.THUMB_W + 8, self.THUMB_H + 8))
                if dimmed is not None and index < len(dimmed) and dimmed[index]:
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                self.strip.addItem(item)
            if 0 <= current_index < len(names):
                self.strip.setCurrentRow(current_index)
        finally:
            self.strip.blockSignals(was_blocked)
        self.setVisible(len(names) > 0)

    def set_thumbnail(self, index: int, pixmap) -> None:
        item = self.strip.item(index)
        if item is not None:
            item.setIcon(QtGui.QIcon(pixmap))

    def set_current(self, index: int) -> None:
        was_blocked = self.strip.blockSignals(True)
        try:
            self.strip.setCurrentRow(index)
        finally:
            self.strip.blockSignals(was_blocked)


class GuiConfigSection(QWidget):
    save_current_as_default_requested = Signal()
    save_current_to_file_requested = Signal()
    load_from_file_requested = Signal()
    restore_factory_default_requested = Signal()
    export_lut_requested = Signal()
    open_logs_requested = Signal()
    # Phase 14 preset browser
    preset_selected = Signal(str)
    save_preset_requested = Signal()
    update_preset_requested = Signal(str)   # NAT 2026-07-22
    delete_preset_requested = Signal(str)

    _PRESET_PLACEHOLDER = '(preset)'

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ------------------------------------------------------- preset browser
    def refresh_presets(self, select: str | None = None) -> None:
        """Rebuild the dropdown from the user presets dir. ``select`` keeps /
        sets the shown entry (e.g. right after saving)."""
        from spektrafilm_gui.preset_store import list_presets

        current = select if select is not None else self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            self.preset_combo.addItem(self._PRESET_PLACEHOLDER)
            for name in list_presets():
                self.preset_combo.addItem(name)
            index = self.preset_combo.findText(current)
            self.preset_combo.setCurrentIndex(index if index > 0 else 0)
        finally:
            self.preset_combo.blockSignals(False)
        self._sync_delete_enabled()

    def current_preset(self) -> str:
        text = self.preset_combo.currentText()
        return '' if text == self._PRESET_PLACEHOLDER else text

    def _on_preset_activated(self, index: int) -> None:
        del index
        name = self.current_preset()
        self._sync_delete_enabled()
        if name:
            self.preset_selected.emit(name)

    def _on_delete_preset(self) -> None:
        name = self.current_preset()
        if name:
            self.delete_preset_requested.emit(name)

    def _on_update_preset(self) -> None:
        name = self.current_preset()
        if name:
            self.update_preset_requested.emit(name)

    def _sync_delete_enabled(self) -> None:
        self.delete_preset_button.setEnabled(bool(self.current_preset()))
        self.update_preset_button.setEnabled(bool(self.current_preset()))

    def _build_ui(self) -> None:
        # Phase 14: OFX-style preset browser over %APPDATA%/spektrafilm/presets
        # NAT 2026-07-25: the popup is a CASCADING menu with one submenu
        # per preset folder, flying out on hover, instead of one flat list of
        # every folder's presets. The item model stays flat and qualified
        # ('Folder / Name'), so selection, refresh and the signals below are
        # unchanged.
        self.preset_combo = FolderMenuComboBox(
            separator=_PRESET_FOLDER_SEPARATOR)
        self.preset_combo.setToolTip(
            'Saved looks (.sfpreset in your user presets folder). Folders open '
            'as submenus. Selecting one applies it to every control, like Load '
            'from file. The OFX plugin\'s .spkpreset files cannot be read '
            '(their payload is obfuscated).')
        self.preset_combo.activated.connect(self._on_preset_activated)
        self.save_preset_button = _build_button(
            'Save preset...',
            self.save_preset_requested.emit,
            tooltip='Save the current look into the presets dropdown (user folder, '
                    'kept across sessions and app updates).',
            role='compactAction',
        )
        self.update_preset_button = _build_button(
            'Update',
            self._on_update_preset,
            tooltip='Overwrite the selected preset with the current look.',
            role='compactAction',
        )
        self.update_preset_button.setEnabled(False)
        self.delete_preset_button = _build_button(
            'Delete',
            self._on_delete_preset,
            tooltip='Delete the selected preset file.',
            role='compactAction',
        )
        self.delete_preset_button.setEnabled(False)

        self.save_current_as_default_button = _build_button(
            'Save current as default',
            self.save_current_as_default_requested.emit,
        )
        self.save_current_to_file_button = _build_button(
            'Save current to file',
            self.save_current_to_file_requested.emit,
        )
        self.load_from_file_button = _build_button('Load from file', self.load_from_file_requested.emit)
        self.restore_factory_default_button = _build_button(
            'Restore factory default',
            self.restore_factory_default_requested.emit,
        )
        self.export_lut_button = _build_button(
            'Export LUT...',
            self.export_lut_requested.emit,
            tooltip='Bake the current film/print/scan look into a 3D LUT (.cube / .3dl / HaldCLUT) '
                    'and an optional OCIO config. Spatial and stochastic effects (grain, halation) '
                    'are excluded -- a LUT is a per-pixel transform.',
        )
        self.open_logs_button = _build_button(
            'Open logs',
            self.open_logs_requested.emit,
            tooltip='Open the application log folder (rotating spektrafilm.log). Attach the '
                    'latest log when reporting a problem.',
        )

        preset_row = QWidget()
        preset_layout = QHBoxLayout()
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)
        preset_layout.addWidget(self.preset_combo, 1)
        preset_layout.addWidget(self.save_preset_button)
        preset_layout.addWidget(self.update_preset_button)
        preset_layout.addWidget(self.delete_preset_button)
        preset_row.setLayout(preset_layout)

        # NAT 2026-07-21 (v1.0.0 polish): the preset dropdown replaced the
        # save/load-file dialogs — those buttons are GONE (their signals stay
        # for the wiring). Save-as-default / factory-reset moved to the CONFIG
        # tab (ConfigDefaultsSection below reuses this section's signals).
        content = _build_vertical_container(
            preset_row,
            _build_button_row(self.export_lut_button, self.open_logs_button),
        )
        _set_single_collapsible_layout(self, 'Presets', content, expanded=True)
        self.refresh_presets()


class ConfigDefaultsSection(QWidget):
    """NAT 2026-07-21 (v1.0.0 polish): 'Defaults' group on the CONFIG tab —
    borrows the Presets section's save-as-default / factory-reset buttons
    (and therefore its signals; app wiring unchanged)."""

    def __init__(self, gui_config_section: 'GuiConfigSection'):
        super().__init__()
        content = _build_vertical_container(
            _build_button_row(gui_config_section.save_current_as_default_button,
                              gui_config_section.restore_factory_default_button),
        )
        _set_single_collapsible_layout(self, 'Defaults', content, expanded=True)


class LocationsSection(QWidget):
    """NAT 2026-07-21: CONFIG tab shortcuts to the app's user folders."""

    def __init__(self):
        super().__init__()
        import os

        def _open(path_fn):
            def run():
                path = str(path_fn())
                os.makedirs(path, exist_ok=True)
                os.startfile(path)
            return run

        def _presets_dir():
            from spektrafilm_gui.preset_store import user_presets_dir
            return user_presets_dir()

        def _stocks_dir():
            from spektrafilm.profiles.custom_stocks import user_stocks_dir
            return user_stocks_dir()

        def _logs_dir():
            from spektrafilm_gui.app_logging import log_directory
            return log_directory()

        content = _build_vertical_container(
            _build_button_row(
                _build_button('Presets folder', _open(_presets_dir),
                              tooltip='Open the .sfpreset folder in Explorer'),
                _build_button('Film stocks folder', _open(_stocks_dir),
                              tooltip='Open the custom .sfstock folder in Explorer'),
            ),
            _build_button_row(
                _build_button('Logs folder', _open(_logs_dir),
                              tooltip='Open the log folder in Explorer'),
            ),
        )
        _set_single_collapsible_layout(self, 'Locations', content, expanded=True)


class PreviewSizeStepper(QWidget):
    """Top-bar preview max size (NAT 2026-07-22): a box plus ONE up / ONE
    down arrow stepping a coherent size ladder — no slider. House editor
    contract (.value + valueChanged); specs forward via _editors."""

    LADDER = (128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048,
              2560, 3072, 4096, 5120, 6144, 8192, 10240)

    valueChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self._box = IntEditor(minimum=128, maximum=10240)
        # NAT 2026-07-25: start at the project default (1024), NOT the spinbox
        # minimum (128). preview_max_size is a machine preference that
        # apply_gui_state PRESERVES on every load, so the constructed value is
        # what sticks — an IntEditor defaulting to its minimum meant every
        # session opened at 128 until the user stepped it up by hand.
        self._box.value = 1024
        self._box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self._box.setFixedWidth(56)
        self._box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._box.valueChanged.connect(self.valueChanged.emit)
        self._editors = (self._box,)   # _apply_numeric_attr forwards specs here
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._box)
        # NAT 2026-07-22 'match arrows': native Qt arrow glyphs — one UP, one
        # DOWN (the text triangles rendered as twins in some fonts).
        for arrow, direction, tip in (
                (Qt.DownArrow, -1, 'Smaller preview (next size down the ladder)'),
                (Qt.UpArrow, +1, 'Larger preview (next size up the ladder)')):
            button = QtWidgets.QToolButton()
            button.setArrowType(arrow)
            button.setToolTip(tip)
            button.setFixedSize(20, 24)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, d=direction: self._step(d))
            layout.addWidget(button)

    def _step(self, direction: int) -> None:
        current = self._box.value
        if direction > 0:
            larger = [size for size in self.LADDER if size > current]
            target = larger[0] if larger else self._box.maximum()
        else:
            smaller = [size for size in self.LADDER if size < current]
            target = smaller[-1] if smaller else self._box.minimum()
        self._box.value = min(max(target, self._box.minimum()), self._box.maximum())

    @property
    def value(self) -> int:
        return self._box.value

    @value.setter
    def value(self, value: int) -> None:
        self._box.value = value


class DisplaySection(SimpleDataclassSection):
    STATE_CLS = DisplayState
    SECTION_NAME = 'display'
    TITLE = 'Display'
    COLLAPSED_BY_DEFAULT = False
    ENUM_FIELDS_KEY = 'display'
    # NAT 2026-07-21: preview_max_size + white_padding live on the viewer's
    # TOP BAR now (app.py builds the strip from these editors).
    HIDDEN_FIELDS = {'preview_max_size', 'white_padding'}

    update_preview_requested = Signal()

    def _init_extra_widgets(self) -> None:
        self.update_preview_button = _build_button(
            'update',
            self.update_preview_requested.emit,
            preserve_case=True,
            role='compactAction',
        )
        self.update_preview_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def _build_editor(self, field_name: str, annotation: Any) -> QWidget:
        # NAT 2026-07-22: on the top bar the size is stepped with arrows
        # through a ladder, not slid.
        if field_name == 'preview_max_size':
            return PreviewSizeStepper()
        return super()._build_editor(field_name, annotation)


class GpuBackendWidget(QWidget):
    """GPU status + live CPU/GPU toggle, shown at the top of the left panel.
    The tick sets the SPEKTRAFILM_GPU env var via the controller; every kernel
    re-reads it per call, so it applies on the next preview/scan (a live A/B:
    toggle and re-Scan to compare)."""

    gpu_backend_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.gpu_backend_toggle = BoolEditor()
        self.gpu_backend_toggle.setChecked(True)
        self.gpu_backend_toggle.setToolTip(
            '勾选后使用 Vulkan GPU 加速；取消勾选后使用 CPU。下次预览或扫描时生效。'
        )
        self.gpu_backend_toggle.toggled.connect(self.gpu_backend_toggled.emit)
        self.gpu_status_label = QLabel(_normalize_ui_text('GPU acceleration (Vulkan)'))
        self.gpu_status_label.setToolTip('当前光谱计算后端，由右侧开关控制。')

        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        row.addWidget(self.gpu_status_label)
        row.addStretch(1)
        row.addWidget(self.gpu_backend_toggle)
        self.setLayout(row)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def set_gpu_status_text(self, text: str) -> None:
        self.gpu_status_label.setText(_normalize_ui_text(text))


class ImageBannerWidget(QWidget):
    """Very small banner above the viewer: image name (left) and pixel
    dimensions (right), NegPy-style."""

    def __init__(self):
        super().__init__()
        self.name_label = QLabel('')
        self.name_label.setObjectName('bannerNameLabel')
        self.dims_label = QLabel('')
        self.dims_label.setObjectName('bannerDimsLabel')
        self.dims_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)
        row.addWidget(self.name_label, 1)
        row.addWidget(self.dims_label, 0)
        self.setLayout(row)
        self.setFixedHeight(18)

    def set_image_info(self, name: str, width: int = 0, height: int = 0) -> None:
        self.name_label.setText(_normalize_ui_text(name or ''))
        self.dims_label.setText(_normalize_ui_text(f'{width} x {height} px' if width and height else ''))


class MetadataPanel(QWidget):
    """Bottom-left Metadata panel: a few editable fields written into every
    export (single and roll). Camera / Lens Model default to the loaded file's
    own values (editable); the EXIF is not displayed."""

    _FIELDS = (
        ('artist', 'Artist', 'Exif.Image.Artist'),
        ('copyright', 'Copyright', 'Exif.Image.Copyright'),
        ('description', 'Description', 'Exif.Image.ImageDescription'),
        ('camera_model', 'Camera Model', 'Exif.Image.Model'),
        ('lens_model', 'Lens Model', 'Exif.Photo.LensModel'),
    )

    rating_changed = Signal(int)   # 0-5 stars for the CURRENT frame

    def __init__(self):
        super().__init__()
        from spektrafilm_gui.widget_primitives import StarRating

        form = _new_form_layout()
        self._editors: dict[str, QLineEdit] = {}
        for attr, label, key in self._FIELDS:
            editor = QLineEdit()
            editor.setToolTip(f'Optional; applied to every exported file ({key}). Leave empty to skip.')
            setattr(self, attr, editor)
            self._editors[key] = editor
            form.addRow(QLabel(_normalize_ui_text(label)), editor)
        # Per-frame star rating (Exif.Image.Rating on export; drives the
        # nav-bar filter and the roll-export selection).
        self.rating = StarRating(tooltip=_normalize_ui_text(
            'Star rating of the CURRENT frame (0-5). Written as Exif.Image.Rating on '
            'export; the nav-bar star filter and roll export select by it.'))
        self.rating.valueChanged.connect(self.rating_changed.emit)
        form.addRow(QLabel(_normalize_ui_text('Rating')), self.rating)

        content = QWidget()
        content.setLayout(form)
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(CollapsibleSection(_normalize_ui_text('Metadata'), content, expanded=False))
        self.setLayout(root)

    def set_rating(self, rating: int) -> None:
        """Reflect the current frame's rating (no signal emission)."""
        self.rating.set_value(rating)

    def custom_tags(self) -> dict:
        """Non-empty editable fields as exiv2 EXIF keys (the per-frame Rating
        is merged controller-side so batch export stays per-frame)."""
        return {key: editor.text().strip() for key, editor in self._editors.items() if editor.text().strip()}

    def set_camera_lens(self, camera_model: str = '', lens_model: str = '') -> None:
        """Prefill Camera/Lens Model from the loaded file (does not overwrite a
        value the user has typed). Values are raw metadata, not UI labels, so
        they are inserted verbatim."""
        if camera_model and not self.camera_model.text().strip():
            self.camera_model.setText(str(camera_model))
        if lens_model and not self.lens_model.text().strip():
            self.lens_model.setText(str(lens_model))


class SimulationSection(DataclassSection):
    preview_requested = Signal()
    scan_requested = Signal()
    save_requested = Signal()
    edit_stock_requested = Signal()   # Phase 10C: reopen the stock editor
    _glare_section: 'GlareSection | None'
    _scan_for_print_restore_state: dict[str, object] | None

    def adopt_spectral_rows(self, input_image_section: 'InputImageSection') -> None:
        """NAT 2026-07-22: spectral upsampling + taking filter live on the
        MAIN tab under Profiles. Editors stay input_image-owned (state and
        profile sync untouched); the rows just render here."""
        for field_name in ('spectral_upsampling_method', 'color_filter'):
            self._form.addRow(_build_widget_label('input_image', field_name),
                              getattr(input_image_section, field_name))

    def _row_widget(self, field_name: str, widget: QWidget) -> QWidget:
        if field_name != 'film_stock':
            return super()._row_widget(field_name, widget)
        # Phase 10C: the film selector row carries the 'Edit' affordance.
        # ALWAYS enabled (NAT 2026-07-20 — the disabled state read as a dead
        # button): a custom stock reopens its recipe, a bundled stock starts
        # a fresh recipe on that base, exactly like the 'Custom...' item.
        row = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(widget, 1)
        self.edit_stock_button = _build_button(
            'Edit', self.edit_stock_requested.emit,
            tooltip='Edit the selected custom stock, or design a new custom '
                    'stock based on the selected bundled stock.',
            role='compactAction',
        )
        layout.addWidget(self.edit_stock_button)
        row.setLayout(layout)
        return row

    def __init__(self):
        super().__init__(
            state_cls=SimulationState,
            section_name='simulation',
            title='Profiles',
            enum_fields=GUI_SECTION_ENUMS['simulation'],
            hidden_fields={
                'film_format_mm',
                'camera_lens_blur_um',
                'camera_diffusion_filter_active',
                'camera_diffusion_filter_family',
                'camera_diffusion_filter_strength',
                'camera_diffusion_filter_spatial_scale',
                'camera_diffusion_filter_halo_warmth',
                'camera_diffusion_filter_core_intensity',
                'camera_diffusion_filter_core_size',
                'camera_diffusion_filter_halo_intensity',
                'camera_diffusion_filter_halo_size',
                'camera_diffusion_filter_bloom_intensity',
                'camera_diffusion_filter_bloom_size',
                'exposure_compensation_ev',
                'auto_exposure',
                'auto_exposure_method',
                'print_exposure',
                'print_exposure_compensation',
                'print_y_filter_shift',
                'print_m_filter_shift',
                'print_c_filter_shift',
                'film_development_time',
                'print_shadow_shape',
                'print_highlight_shape',
                'negative_bleach_bypass',
                'negative_leuco_cyan_coupling',
                'print_bleach_bypass',
                'diffusion_filter_active',
                'diffusion_filter_family',
                'diffusion_filter_strength',
                'diffusion_filter_spatial_scale',
                'diffusion_filter_halo_warmth',
                'diffusion_filter_core_intensity',
                'diffusion_filter_core_size',
                'diffusion_filter_halo_intensity',
                'diffusion_filter_halo_size',
                'diffusion_filter_bloom_intensity',
                'diffusion_filter_bloom_size',
                'print_illuminant',
                'scan_lens_blur',
                'scan_white_correction',
                'scan_white_level',
                'scan_black_correction',
                'scan_black_level',
                'scan_unsharp_mask',
                'auto_preview',
                'scan_film',
                'output_color_space',
                'saving_color_space',
                'saving_cctf_encoding',
                'export_format',
                'export_full_precision',
            },
        )

    def _init_extra_widgets(self) -> None:
        self._glare_section = None
        self._scan_for_print_restore_state = None
        self.bottom_auto_preview = BoolEditor()
        # On by default; apply_gui_state never overwrites it (session-scoped
        # toggle, dormant until the first render — NAT 2026-07-18).
        self.bottom_auto_preview.setChecked(True)
        self.bottom_scan_film = BoolEditor()
        # Grain-in-preview (NAT 2026-07-26): session toggle; off = fast scrub
        # without grain in the live preview (Scan/export still grain). Never
        # written by apply_gui_state (session-scoped, like auto_preview).
        self.bottom_grain_preview = BoolEditor()
        self.bottom_grain_preview.setChecked(True)
        self.bottom_grain_preview.setToolTip(
            'Show grain in the live preview. Turn off for fast scrubbing — grain '
            'still renders on Scan and export, just not while you adjust.')
        self.bottom_scan_for_print = BoolEditor()
        scan_for_print_spec = get_auxiliary_spec('scan_for_print')
        if scan_for_print_spec.tooltip:
            self.bottom_scan_for_print.setToolTip(scan_for_print_spec.tooltip)
        self.bottom_scan_for_print.toggled.connect(self._apply_scan_for_print_mode)
        preview_button_spec = get_button_spec('preview')
        self.preview_button = _build_button(
            preview_button_spec.text,
            self.preview_requested.emit,
            tooltip=preview_button_spec.tooltip,
            preserve_case=preview_button_spec.preserve_case,
            role='accentAction',
        )
        scan_button_spec = get_button_spec('scan')
        self.scan_button = _build_button(
            scan_button_spec.text,
            self.scan_requested.emit,
            tooltip=scan_button_spec.tooltip,
            preserve_case=scan_button_spec.preserve_case,
            role='accentAction',
        )
        save_button_spec = get_button_spec('save')
        self.save_button = _build_button(
            save_button_spec.text,
            self.save_requested.emit,
            tooltip=save_button_spec.tooltip,
            preserve_case=save_button_spec.preserve_case,
            role='accentAction',
        )

        # (The GPU status/toggle moved to GpuBackendWidget at the top of the
        # left panel.)
        scan_film_row = QHBoxLayout()
        scan_film_row.setContentsMargins(0, 0, 0, 0)
        scan_film_row.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_widget_label('simulation', 'auto_preview'))
        scan_film_row.addWidget(self.bottom_auto_preview)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_widget_label('simulation', 'scan_film'))
        scan_film_row.addWidget(self.bottom_scan_film)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        grain_prev_label = QLabel('grain preview')
        grain_prev_label.setToolTip(self.bottom_grain_preview.toolTip())
        scan_film_row.addWidget(grain_prev_label)
        scan_film_row.addWidget(self.bottom_grain_preview)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_auxiliary_label('scan_for_print'))
        scan_film_row.addWidget(self.bottom_scan_for_print)
        scan_film_row.addStretch(1)

        action_buttons = QWidget()
        action_buttons.setLayout(
            _build_button_row(
                self.preview_button,
                self.scan_button,
                self.save_button,
                stretch=1,
                spacing=SIZE_FOOTER_ITEM_SPACING,
            ),
        )

        self.bottom_bar = QWidget()
        bottom_bar_layout = QVBoxLayout(self.bottom_bar)
        bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        bottom_bar_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        bottom_bar_layout.addLayout(scan_film_row)
        bottom_bar_layout.addWidget(action_buttons)

    def set_gpu_backend_checked(self, enabled: bool) -> None:
        toggle = getattr(self, 'gpu_backend_toggle', None)
        if toggle is None:
            return
        was_blocked = toggle.blockSignals(True)
        toggle.setChecked(enabled)
        toggle.blockSignals(was_blocked)

    def action_bar(self) -> QWidget:
        return self.bottom_bar

    def set_auto_preview_value(self, value: bool) -> None:
        self.bottom_auto_preview.setChecked(value)

    def auto_preview_value(self) -> bool:
        return self.bottom_auto_preview.isChecked()

    def set_scan_film_value(self, value: bool) -> None:
        self.bottom_scan_film.setChecked(value)

    def scan_film_value(self) -> bool:
        return self.bottom_scan_film.isChecked()

    def bind_scan_for_print_glare_section(self, glare_section: 'GlareSection') -> None:
        self._glare_section = glare_section

    def reset_scan_for_print_value(self) -> None:
        was_blocked = self.bottom_scan_for_print.blockSignals(True)
        self.bottom_scan_for_print.setChecked(False)
        self.bottom_scan_for_print.blockSignals(was_blocked)
        self._scan_for_print_restore_state = None

    def _apply_scan_for_print_mode(self, active: bool) -> None:
        if active:
            if self._scan_for_print_restore_state is None:
                self._scan_for_print_restore_state = {
                    'scan_white_correction': self.scan_white_correction.value,
                    'scan_black_correction': self.scan_black_correction.value,
                    'glare_active': None if self._glare_section is None else self._glare_section.active.value,
                }
            self.scan_white_correction.value = True
            self.scan_black_correction.value = True
            if self._glare_section is not None:
                self._glare_section.active.value = False
            return

        restore_state = self._scan_for_print_restore_state
        if restore_state is None:
            return
        self.scan_white_correction.value = restore_state['scan_white_correction']
        self.scan_black_correction.value = restore_state['scan_black_correction']
        glare_active = restore_state['glare_active']
        if self._glare_section is not None and glare_active is not None:
            self._glare_section.active.value = glare_active
        self._scan_for_print_restore_state = None


class ExportSection(QWidget):
    """Export panel (the 'film scanner' output). Groups the independent,
    non-baked export choices -- format/bit-depth, encoding (flat-linear vs
    tonally-normal), and output colour profile -- plus a full-precision toggle
    and the Export/Cancel action. The render is full-resolution and written off
    the UI thread by the controller; 16-bit comes straight from the float
    render. Warns on 8-bit + wide-gamut (posterisation) and notes when a
    chosen (profile, encoding) pair has no embeddable ICC.
    """

    export_requested = Signal()
    export_roll_requested = Signal()

    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self._simulation = simulation_section
        self._format = simulation_section.export_format
        self._profile = simulation_section.saving_color_space
        self._encoding = simulation_section.saving_cctf_encoding
        self._full_precision = simulation_section.export_full_precision

        form = _new_form_layout()
        _add_form_rows(
            form,
            [
                _spec_row('simulation', 'export_format', self._format),
                _spec_row('simulation', 'saving_color_space', self._profile),
            ],
        )

        # Encoding is shown as a plain "Linear tones" tick: ticked = flat/linear
        # (cctf OFF, max headroom), unticked = tonally-normal. It drives the
        # hidden saving_cctf_encoding editor INVERSELY (linear == encoding off),
        # which is the field the rest of the app + the saved state read.
        self._linear = BoolEditor()
        linear_label = QLabel(_normalize_ui_text('Linear tones'))
        _linear_tip = (
            'Ticked = flat/linear: no tone curve, maximum grading headroom, looks flat on open (best for Resolve). '
            'Unticked = tonally-normal: the colour space curve is applied so the file looks correct on open '
            '(Lightroom/Capture One). The matching ICC variant is embedded either way.'
        )
        linear_label.setToolTip(_linear_tip)
        self._linear.setToolTip(_linear_tip)
        form.addRow(linear_label, self._linear)

        _add_form_rows(form, [_spec_row('simulation', 'export_full_precision', self._full_precision)])

        self._warning_label = QLabel('')
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet('color: #d08a2a;')  # amber, advisory
        self._warning_label.setVisible(False)
        form.addRow(self._warning_label)

        self._icc_note_label = QLabel('')
        self._icc_note_label.setWordWrap(True)
        self._icc_note_label.setStyleSheet('color: #888888;')  # muted, informational
        self._icc_note_label.setVisible(False)
        form.addRow(self._icc_note_label)

        self.export_button = _build_button(
            'Export',
            self.export_requested.emit,
            tooltip='Write the image with these settings, off the UI thread. Full-precision re-renders the input; otherwise the current scan is exported as-is.',
            role='accentAction',
        )
        form.addRow(self.export_button)

        # ---- batch (roll) export --------------------------------------
        self.sync_export_settings = BoolEditor()
        self.sync_export_settings.setChecked(True)
        sync_label = QLabel(_normalize_ui_text('Sync export settings'))
        _sync_tip = (
            'On (default): every frame in the roll is written with the format / profile / '
            'encoding chosen ABOVE. Off: each frame uses the export settings stored with '
            'its own look. The per-frame LOOK settings are always the frame\'s own.'
        )
        sync_label.setToolTip(_sync_tip)
        self.sync_export_settings.setToolTip(_sync_tip)
        form.addRow(sync_label, self.sync_export_settings)

        self.sync_metadata = BoolEditor()
        self.sync_metadata.setChecked(True)
        sync_metadata_label = QLabel(_normalize_ui_text('Sync metadata'))
        _sync_metadata_tip = (
            'On (default): the Metadata panel values (Artist / Copyright / Description / '
            'Camera / Lens) as currently shown are embedded in EVERY exported frame. '
            'Off: each frame keeps only the metadata read from its own source file.'
        )
        sync_metadata_label.setToolTip(_sync_metadata_tip)
        self.sync_metadata.setToolTip(_sync_metadata_tip)
        form.addRow(sync_metadata_label, self.sync_metadata)

        # Live note on what 'Export roll' will write under the nav-bar star
        # filter (the filter is CONTROLLED there; this line only reports it).
        self.rating_filter_info = QLabel(_normalize_ui_text('Exports every frame in the roll.'))
        self.rating_filter_info.setToolTip(_normalize_ui_text(
            'Export roll only writes frames passing the star filter on the '
            'navigation bar (bottom left).'))
        self.rating_filter_info.setWordWrap(True)
        self.rating_filter_info.setStyleSheet('color: #888888;')  # muted, informational
        form.addRow(self.rating_filter_info)

        self.export_roll_button = _build_button(
            'Export roll',
            self.export_roll_requested.emit,
            tooltip='Render and write every frame PASSING the star filter above (full precision, each '
                    'with its own look) into a folder you choose. Runs off the UI thread with progress; '
                    'cancel between frames with the Cancel export button.',
            role='accentAction',
        )
        form.addRow(self.export_roll_button)

        self.setLayout(_build_collapsible_form_section('Export', form, expanded=False))

        # The (removed) Output section's colour space now MIRRORS the export
        # saving colour space: the render lands directly in the space the file
        # is written in. output_color_space is the hidden editor the pipeline
        # reads; keep it locked to the saving choice both ways.
        self._output = simulation_section.output_color_space
        self._mirror_output_color_space(self._profile.currentText())

        # Sync the linear tick from the canonical encoding field, then wire both ways.
        self._linear.value = not self._encoding.value
        self._format.currentTextChanged.connect(self._refresh_state)
        self._profile.currentTextChanged.connect(self._refresh_state)
        self._profile.currentTextChanged.connect(self._mirror_output_color_space)
        self._linear.toggled.connect(self._on_linear_toggled)
        self._encoding.toggled.connect(self._sync_linear_from_encoding)
        self._refresh_state()

    def _mirror_output_color_space(self, text: str) -> None:
        was_blocked = self._output.blockSignals(True)
        try:
            self._output.value = text
        finally:
            self._output.blockSignals(was_blocked)

    def set_exporting(self, exporting: bool) -> None:
        """Relabel the action button Export <-> Cancel while a job runs."""
        self.export_button.setText(_normalize_ui_text('Cancel export' if exporting else 'Export'))
        self.export_roll_button.setEnabled(not exporting)

    def sync_export_settings_value(self) -> bool:
        return self.sync_export_settings.isChecked()

    def sync_metadata_value(self) -> bool:
        return self.sync_metadata.isChecked()

    def _on_linear_toggled(self, checked: bool) -> None:
        # Linear ticked == cctf encoding OFF.
        was_blocked = self._encoding.blockSignals(True)
        self._encoding.value = not checked
        self._encoding.blockSignals(was_blocked)
        self._update_posterization_warning()

    def _sync_linear_from_encoding(self, encoded: bool) -> None:
        # External / state-load changes to the canonical field reflect into the tick.
        was_blocked = self._linear.blockSignals(True)
        self._linear.value = not encoded
        self._linear.blockSignals(was_blocked)

    def _refresh_state(self, *_args) -> None:
        self._update_encoding_availability()
        self._update_posterization_warning()

    def _update_encoding_availability(self) -> None:
        profile = self._profile.currentText()
        has_encoded, has_linear = self._icc_availability(profile)

        if has_encoded and has_linear:
            self._linear.setEnabled(True)
            self._set_icc_note('')
            return
        if has_encoded and not has_linear:
            self._force_linear(False)
            self._set_icc_note(f'No bundled ICC for linear {profile}; using normal tones and tagging via metadata.')
            return
        if has_linear and not has_encoded:
            self._force_linear(True)
            self._set_icc_note(f'No bundled ICC for normal {profile}; using linear tones and tagging via metadata.')
            return
        # Neither variant has a bundled ICC: leave the choice, note metadata-only.
        self._linear.setEnabled(True)
        self._set_icc_note(f'No bundled ICC for {profile}; the file is tagged via metadata only (no embedded profile).')

    def _force_linear(self, linear: bool) -> None:
        lb = self._linear.blockSignals(True)
        eb = self._encoding.blockSignals(True)
        self._linear.value = linear
        self._encoding.value = not linear
        self._encoding.blockSignals(eb)
        self._linear.blockSignals(lb)
        self._linear.setEnabled(False)

    def _update_posterization_warning(self) -> None:
        _ext, bit_depth = _export_format_spec(self._format.currentText())
        profile = self._profile.currentText()
        if bit_depth <= 8 and profile in _WIDE_GAMUT_COLOR_SPACES:
            self._set_warning(
                f'{bit_depth}-bit + {profile}: a wide gamut in 8-bit can posterise smooth gradients. '
                'Prefer 16-bit TIFF/PNG, or sRGB for 8-bit JPEG.'
            )
        else:
            self._set_warning('')

    def _set_warning(self, text: str) -> None:
        self._warning_label.setText(text)
        self._warning_label.setVisible(bool(text))

    def _set_icc_note(self, text: str) -> None:
        self._icc_note_label.setText(text)
        self._icc_note_label.setVisible(bool(text))

    @staticmethod
    def _icc_availability(profile: str) -> tuple[bool, bool]:
        try:
            from spektrafilm.utils.io import icc_pair_available
        except Exception:
            return True, True  # never block the UI if io import fails
        return icc_pair_available(profile, True), icc_pair_available(profile, False)


class ExposureControlSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = _new_form_layout()
        self._add_spec_row(form, 'simulation', 'auto_exposure', simulation_section.auto_exposure)
        self._add_spec_row(form, 'simulation', 'exposure_compensation_ev', simulation_section.exposure_compensation_ev)
        self._add_spec_row(form, 'simulation', 'print_exposure_compensation', simulation_section.print_exposure_compensation)
        self._add_spec_row(form, 'simulation', 'print_exposure', simulation_section.print_exposure)

        self.setLayout(_build_collapsible_form_section('Exposure control', form, expanded=True))

    def _add_spec_row(self, form: QFormLayout, section_name: str, field_name: str, widget: QWidget) -> None:
        spec = get_widget_spec(section_name, field_name)
        if spec.tooltip:
            widget.setToolTip(spec.tooltip)
        form.addRow(_build_widget_label(section_name, field_name), widget)


class EnlargerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        # NAT 2026-07-22: the three filter shifts collapse to ONE row of
        # C/M/Y-tinted knobs (write-through to the simulation editors).
        self.filter_shift_row = _knob_row('simulation', (
            (simulation_section.print_c_filter_shift, 'print_c_filter_shift', 'c', 'c'),
            (simulation_section.print_m_filter_shift, 'print_m_filter_shift', 'm', 'm'),
            (simulation_section.print_y_filter_shift, 'print_y_filter_shift', 'y', 'y'),
        ))
        self.setLayout(
            _build_linked_form_section(
                'Enlarger',
                [
                    (QLabel(_normalize_ui_text('Filter shifts')), self.filter_shift_row),
                ],
                expanded=True,
            ),
        )


class DevelopmentSection(QWidget):
    """Phase 12B: the development-time picker (FILM tab), borrowing the
    simulation section's hidden editor — B&W stocks only, 0 = default."""

    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Development',
                [
                    _spec_row('simulation', 'film_development_time',
                              simulation_section.film_development_time),
                ],
                expanded=False,
            ),
        )


class PaperShapingSection(QWidget):
    """Phase 13A + print-side 13B (PRINT tab): shadow/highlight shape through
    the paper response + print bleach bypass, borrowing simulation editors."""

    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Paper shaping',
                [
                    _spec_row('simulation', 'print_shadow_shape',
                              simulation_section.print_shadow_shape),
                    _spec_row('simulation', 'print_highlight_shape',
                              simulation_section.print_highlight_shape),
                    _spec_row('simulation', 'print_bleach_bypass',
                              simulation_section.print_bleach_bypass),
                ],
                expanded=False,
            ),
        )


class BleachBypassSection(QWidget):
    """Phase 13B (FILM tab): negative bleach bypass (modeled extension)."""

    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Bleach bypass',
                [
                    _spec_row('simulation', 'negative_bleach_bypass',
                              simulation_section.negative_bleach_bypass),
                    _spec_row('simulation', 'negative_leuco_cyan_coupling',
                              simulation_section.negative_leuco_cyan_coupling),
                ],
                expanded=False,
            ),
        )


class DiffusionSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Diffusion',
                [
                    _spec_row('simulation', 'diffusion_filter_active', simulation_section.diffusion_filter_active),
                    _spec_row('simulation', 'diffusion_filter_family', simulation_section.diffusion_filter_family),
                    _spec_row('simulation', 'diffusion_filter_strength', simulation_section.diffusion_filter_strength),
                    _spec_row('simulation', 'diffusion_filter_spatial_scale', simulation_section.diffusion_filter_spatial_scale),
                    _spec_row('simulation', 'diffusion_filter_halo_warmth', simulation_section.diffusion_filter_halo_warmth),
                    _spec_row('simulation', 'diffusion_filter_core_intensity', simulation_section.diffusion_filter_core_intensity),
                    _spec_row('simulation', 'diffusion_filter_core_size', simulation_section.diffusion_filter_core_size),
                    _spec_row('simulation', 'diffusion_filter_halo_intensity', simulation_section.diffusion_filter_halo_intensity),
                    _spec_row('simulation', 'diffusion_filter_halo_size', simulation_section.diffusion_filter_halo_size),
                    _spec_row('simulation', 'diffusion_filter_bloom_intensity', simulation_section.diffusion_filter_bloom_intensity),
                    _spec_row('simulation', 'diffusion_filter_bloom_size', simulation_section.diffusion_filter_bloom_size),
                ],
                expanded=False,
            ),
        )


class CameraDiffusionSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Diffusion',
                [
                    _spec_row('simulation', 'camera_diffusion_filter_active', simulation_section.camera_diffusion_filter_active),
                    _spec_row('simulation', 'camera_diffusion_filter_family', simulation_section.camera_diffusion_filter_family),
                    _spec_row('simulation', 'camera_diffusion_filter_strength', simulation_section.camera_diffusion_filter_strength),
                    _spec_row('simulation', 'camera_diffusion_filter_spatial_scale', simulation_section.camera_diffusion_filter_spatial_scale),
                    _spec_row('simulation', 'camera_diffusion_filter_halo_warmth', simulation_section.camera_diffusion_filter_halo_warmth),
                    _spec_row('simulation', 'camera_diffusion_filter_core_intensity', simulation_section.camera_diffusion_filter_core_intensity),
                    _spec_row('simulation', 'camera_diffusion_filter_core_size', simulation_section.camera_diffusion_filter_core_size),
                    _spec_row('simulation', 'camera_diffusion_filter_halo_intensity', simulation_section.camera_diffusion_filter_halo_intensity),
                    _spec_row('simulation', 'camera_diffusion_filter_halo_size', simulation_section.camera_diffusion_filter_halo_size),
                    _spec_row('simulation', 'camera_diffusion_filter_bloom_intensity', simulation_section.camera_diffusion_filter_bloom_intensity),
                    _spec_row('simulation', 'camera_diffusion_filter_bloom_size', simulation_section.camera_diffusion_filter_bloom_size),
                ],
                expanded=False,
            ),
        )


class ScannerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Scanner',
                [
                    _spec_row('simulation', 'scan_lens_blur', simulation_section.scan_lens_blur),
                    _compound_spec_row(
                        'simulation',
                        'scan_white_correction',
                        simulation_section.scan_white_correction,
                        simulation_section.scan_white_level,
                    ),
                    _compound_spec_row(
                        'simulation',
                        'scan_black_correction',
                        simulation_section.scan_black_correction,
                        simulation_section.scan_black_level,
                    ),
                    _spec_row('simulation', 'scan_unsharp_mask', simulation_section.scan_unsharp_mask),
                ],
                expanded=True,
            ),
        )


class FilmFormatPresetCombo(QComboBox):
    """Film-size preset picker (NAT 2026-07-25). Selecting a preset writes the
    long-edge mm into the existing film_format_mm box; a hand-typed value the
    presets don't cover shows as 'Other…'. The mm box stays the source of truth
    so state / profile-sync / parity are untouched."""

    def __init__(self, mm_editor: QWidget):
        super().__init__()
        from spektrafilm_gui.film_formats import FILM_FORMAT_PRESETS, is_separator
        self._mm_editor = mm_editor
        self.setToolTip('Pick a film format; it sets the long-edge mm below. '
                        'Choose Other… to type a custom size.')
        model = self.model()
        for index, (label, _mm) in enumerate(FILM_FORMAT_PRESETS):
            self.addItem(label)
            if is_separator(label):
                item = model.item(index)
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
        self.activated.connect(self._on_activated)
        getattr(mm_editor, 'valueChanged').connect(self._sync_from_mm)
        self._sync_from_mm()

    def _on_activated(self, index: int) -> None:
        from spektrafilm_gui.film_formats import mm_for_label
        mm = mm_for_label(self.itemText(index))
        if mm is not None:
            self._mm_editor.value = mm     # fires valueChanged -> preview + re-sync
        else:
            self._sync_from_mm()           # 'Other…' / separator: keep box, type below

    def _sync_from_mm(self, *_args) -> None:
        from spektrafilm_gui.film_formats import label_for_mm
        try:
            label = label_for_mm(float(self._mm_editor.value))
        except (TypeError, ValueError):
            return
        idx = self.findText(label)
        if idx >= 0:
            was = self.blockSignals(True)
            self.setCurrentIndex(idx)
            self.blockSignals(was)


class CameraSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.film_format_preset = FilmFormatPresetCombo(simulation_section.film_format_mm)
        self.setLayout(
            _build_linked_form_section(
                'Camera',
                [
                    ('Film format', self.film_format_preset),
                    _spec_row('simulation', 'film_format_mm', simulation_section.film_format_mm),
                    _spec_row('simulation', 'auto_exposure_method', simulation_section.auto_exposure_method),
                    _spec_row('simulation', 'camera_lens_blur_um', simulation_section.camera_lens_blur_um),
                ],
                expanded=True,
            ),
        )

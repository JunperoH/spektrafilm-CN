"""Film-format presets (NAT 2026-07-25).

The engine takes a single scalar: the long edge of the film frame in mm
(``film_format_mm``), which scales grain size relative to the frame. Users
don't think in millimetres, they think '135' or '4x5', so the FILM tab offers
a preset picker that just writes the mm value. 'Other' keeps the manual box.

Values are the IMAGE-area long edge (what governs grain scale), rounded to a
sensible round number; '120' uses the common 6x7 back. Calibrate by eye and
retune the box if a stock looks off — these are starting points, not gospel.
"""
from __future__ import annotations

# (display label, long-edge mm). Order = dropdown order. Grouped by a divider
# label (mm = None) the widget renders as a non-selectable separator.
# mm are integers (the field is whole-mm) so a preset round-trips to its label.
FILM_FORMAT_PRESETS: tuple[tuple[str, float | None], ...] = (
    ("— Motion —", None),
    ("Super 8", 6.0),
    ("16mm", 10.0),
    ("Super 16", 13.0),
    ("Motion 35", 22.0),
    ("Motion 65/70", 52.0),
    ("— Stills —", None),
    ("110", 17.0),
    # 35.0 = the app's existing default film_format_mm, so a fresh session
    # reads as '135' in the picker with no render change (NAT 2026-07-26).
    ("135 (35mm)", 35.0),
    ("120 (6x7)", 70.0),
    ("— Large format —", None),
    ("4x5", 127.0),
    ("8x10", 254.0),
    ("— Custom —", None),
    ("Other…", None),
)

CUSTOM_LABEL = "Other…"

# label -> mm, only the selectable (real) presets
_LABEL_TO_MM = {label: mm for label, mm in FILM_FORMAT_PRESETS if mm is not None}


def preset_labels() -> tuple[str, ...]:
    return tuple(label for label, _mm in FILM_FORMAT_PRESETS)

def is_separator(label: str) -> bool:
    return _LABEL_TO_MM.get(label) is None and label != CUSTOM_LABEL

def mm_for_label(label: str) -> float | None:
    return _LABEL_TO_MM.get(label)

def label_for_mm(mm: float, *, tol: float = 0.05) -> str:
    """The preset whose mm matches, else the custom label (so a hand-typed or
    profile-supplied value shows as 'Other…')."""
    for label, value in _LABEL_TO_MM.items():
        if abs(value - float(mm)) <= tol:
            return label
    return CUSTOM_LABEL

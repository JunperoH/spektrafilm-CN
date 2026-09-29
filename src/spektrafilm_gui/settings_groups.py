"""User-facing groupings of the gui_state sections (NAT 2026-07-25).

The copy/paste, apply-to-all and preset-save flows all used to move the WHOLE
gui_state. NAT asked to pick WHICH settings travel ("not always populate the
whole edit"). Rather than expose the 17 raw dataclass sections, this module
folds them into a handful of friendly groups that match how the app is used.

Every gui_state section belongs to exactly one group, so selecting all groups
is byte-identical to moving the full state (the old behavior). Adding a new
section to GuiState without listing it here raises in the covering test.
"""
from __future__ import annotations

from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES

# label -> the gui_state section names it covers. Order is display order.
SETTINGS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Film & chemistry", ("simulation", "chemistry", "push_pull", "aging")),
    ("Grain", ("grain",)),
    ("Halation, glare & preflash", ("halation", "glare", "preflashing", "couplers")),
    ("Color / gamut", ("input_gamut_compress", "output_gamut_compress")),
    ("Scan & finishing", ("scan_finishing", "paper")),
    ("Special effects", ("special",)),
    ("Crop & geometry", ("input_image",)),
    ("Loading (RAW / RGB)", ("load_raw", "load_rgb")),
    ("Display preferences", ("display",)),
)

# Groups that are NOT part of a look by default: loading options and machine
# display prefs. Pre-unchecked in the picker so the common case ("copy the
# look") does not drag them along, but still available if the user ticks them.
NON_LOOK_GROUPS: frozenset[str] = frozenset(
    ("Loading (RAW / RGB)", "Display preferences"))


def group_labels() -> tuple[str, ...]:
    return tuple(label for label, _ in SETTINGS_GROUPS)


def sections_for_groups(selected_labels: object) -> tuple[str, ...]:
    """Flatten the chosen group labels to the gui_state section names they
    cover, preserving GuiState field order (so downstream apply is stable)."""
    chosen = set(selected_labels)
    covered: set[str] = set()
    for label, sections in SETTINGS_GROUPS:
        if label in chosen:
            covered.update(sections)
    return tuple(name for name in GUI_STATE_SECTION_NAMES if name in covered)


def default_look_labels() -> tuple[str, ...]:
    """Every group except the non-look ones (the sensible copy-a-look set)."""
    return tuple(label for label, _ in SETTINGS_GROUPS
                 if label not in NON_LOOK_GROUPS)

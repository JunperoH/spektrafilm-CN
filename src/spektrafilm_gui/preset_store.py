"""Phase 14: the .sfpreset store — an OFX-style preset browser's backend.

A preset is a versioned JSON file mirroring the OFX .spkpreset LAYOUT (a
clear-text metadata block for browsing plus the full parameter payload) but
with a READABLE payload: the complete gui_state dict, loaded back through
``gui_state_from_dict`` so presets from older app versions merge-inject
missing fields exactly like rolls and saved states do.

NOTE the OFX plugin's own .spkpreset files can NOT be imported: their payload
is an XOR-scrambled hex snapshot whose key is private to the plugin
(established 2026-06-24, presets/output/CONVERSION_FINDINGS.md). Only their
clear metadata is readable — never guess parameter values.

Storage is a USER-WRITABLE directory (never package resources — the frozen
build's install folder is not for user data):

    %APPDATA%/spektrafilm/presets/<slug>.sfpreset    (Windows)
    ~/.spektrafilm/presets/<slug>.sfpreset           (fallback)
    $SPEKTRAFILM_PRESETS_DIR                         (override, also the test hook)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PRESET_FILE_SUFFIX = ".sfpreset"
SCHEMA_VERSION = 1

__all__ = [
    "PRESET_FILE_SUFFIX",
    "SCHEMA_VERSION",
    "delete_preset",
    "join_qualified_name",
    "list_preset_folders",
    "list_presets",
    "load_preset_payload",
    "load_preset_sections",
    "make_preset_slug",
    "preset_path",
    "save_preset",
    "split_qualified_name",
    "user_presets_dir",
]


def user_presets_dir() -> Path:
    override = os.environ.get("SPEKTRAFILM_PRESETS_DIR")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "spektrafilm" / "presets"
    return Path.home() / ".spektrafilm" / "presets"


def make_preset_slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("Preset name must contain letters or digits.")
    return cleaned


# Preset folders (NAT 2026-07-25): a preset is identified by a qualified name
# 'Folder / Name'. The folder maps to a one-level subdirectory of the presets
# dir; a bare name (no separator) stays a flat file in the root, exactly as
# before — so old presets and old callers keep working unchanged.
FOLDER_SEPARATOR = " / "


def split_qualified_name(qualified: str) -> tuple[str, str]:
    """('Folder', 'Name') for a qualified preset, ('', 'Name') for a flat one.
    Accepts either ' / ' (the display form) or a plain '/'."""
    text = str(qualified).strip()
    for sep in (FOLDER_SEPARATOR, "/"):
        if sep in text:
            folder, name = text.split(sep, 1)
            return folder.strip(), name.strip()
    return "", text


def join_qualified_name(folder: str, name: str) -> str:
    folder = str(folder).strip()
    name = str(name).strip()
    return f"{folder}{FOLDER_SEPARATOR}{name}" if folder else name


def preset_path(name: str) -> Path:
    folder, leaf = split_qualified_name(name)
    directory = user_presets_dir()
    if folder:
        directory = directory / make_preset_slug(folder)
    return directory / f"{make_preset_slug(leaf)}{PRESET_FILE_SUFFIX}"


def list_preset_folders() -> list[str]:
    """The display names of existing preset folders (subdirectories that hold
    at least one preset), sorted."""
    root = user_presets_dir()
    if not root.is_dir():
        return []
    folders: set[str] = set()
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        for file in entry.iterdir():
            if file.suffix == PRESET_FILE_SUFFIX and file.is_file():
                folders.add(_folder_display_for(entry, file))
                break
    return sorted(folders, key=str.lower)


def _folder_display_for(subdir: Path, sample_file: Path) -> str:
    """The folder's display name: the metadata 'folder' if the file records
    one, else the subdirectory name (title-cased from the slug)."""
    try:
        with sample_file.open("r", encoding="utf-8") as handle:
            folder = json.load(handle).get("folder")
        if folder:
            return str(folder)
    except (OSError, ValueError):
        pass
    return subdir.name.replace("_", " ").strip()


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("spektrafilm")
    except Exception:
        return "0+unknown"


def save_preset(name: str, state_dict: dict[str, Any],
                *, sections: Any = None) -> Path:
    """Write a preset. The clear metadata block mirrors what the OFX preset
    browser shows (name / stock / paper / created); the payload is the full
    state dict.

    ``sections`` (NAT 2026-07-25): an optional allow-list of gui_state section
    names. When given, applying the preset touches ONLY those sections (a
    partial preset — 'just the grain', etc.). Omitted / None = a full preset,
    exactly the old behavior."""
    if not str(name).strip():
        raise ValueError("Preset name must not be empty.")
    if not isinstance(state_dict, dict):
        raise ValueError("Preset payload must be a state dict.")
    simulation = state_dict.get("simulation", {}) or {}
    folder, leaf = split_qualified_name(name)
    payload = {
        "format": "sfpreset",
        "schema_version": SCHEMA_VERSION,
        "name": leaf,
        "created": datetime.now().isoformat(timespec="seconds"),
        "app_version": _package_version(),
        "film_stock": str(simulation.get("film_stock", "")),
        "print_paper": str(simulation.get("print_paper", "")),
        "state": state_dict,
    }
    if folder:
        payload["folder"] = folder
    if sections is not None:
        payload["sections"] = list(sections)
    path = preset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def _display_name_for(file: Path, folder: str) -> str:
    """The qualified display name for one preset file (falling back to the
    stem when metadata is unreadable — never hide a file)."""
    try:
        with file.open("r", encoding="utf-8") as handle:
            name = str(json.load(handle).get("name") or file.stem)
    except (OSError, ValueError):
        name = file.stem
    return join_qualified_name(folder, name)


def list_presets() -> list[str]:
    """Sorted qualified names of the saved presets. Root-level files stay flat
    ('Name'); files inside a subdirectory read as 'Folder / Name'. Sorted so
    same-folder presets cluster (folder first, then name)."""
    root = user_presets_dir()
    if not root.is_dir():
        return []
    names: list[str] = []
    for entry in root.iterdir():
        if entry.suffix == PRESET_FILE_SUFFIX and entry.is_file():
            names.append(_display_name_for(entry, ""))
        elif entry.is_dir():
            folder = None
            for file in sorted(entry.iterdir()):
                if file.suffix == PRESET_FILE_SUFFIX and file.is_file():
                    if folder is None:
                        folder = _folder_display_for(entry, file)
                    names.append(_display_name_for(file, folder))
    def _key(qualified: str) -> tuple[str, str]:
        folder, leaf = split_qualified_name(qualified)
        return (folder.lower(), leaf.lower())

    return sorted(names, key=_key)


def load_preset_payload(name: str) -> dict[str, Any]:
    """The preset's state dict (feed it to gui_state_from_dict)."""
    path = preset_path(name)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or data.get("format", "sfpreset") != "sfpreset":
        raise ValueError(f"'{path.name}' is not a .sfpreset file.")
    # The spektrafilm macOS app ALSO uses the .sfpreset extension, with a
    # different (clear, camelCase) schema — discovered 2026-07-21 via
    # presets/NBbitmap.sfpreset. Its continuous parameters are mappable, but
    # film/paper/format/colorspace are NUMERIC INDICES into the Mac app's
    # internal lists, so a faithful import needs those tables (recorded as a
    # Phase 14 follow-up). Detect it and say so instead of a puzzling error.
    if "settings" in data and "state" not in data:
        raise ValueError(
            f"'{path.name}' is a preset from the spektrafilm macOS app — a "
            "different (readable) schema whose film/paper entries are "
            "numeric indices. Importing it is not supported yet; recreate "
            "the look and save it here.")
    version = int(data.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Preset schema_version {version} is newer than this app "
            f"supports ({SCHEMA_VERSION}).")
    state = data.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"'{path.name}' carries no state payload.")
    return state


def load_preset_sections(name: str) -> tuple[str, ...] | None:
    """The preset's section allow-list (None = a full preset). Kept separate
    from ``load_preset_payload`` so that function's return type stays a plain
    state dict for its existing callers."""
    path = preset_path(name)
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    sections = data.get("sections") if isinstance(data, dict) else None
    if isinstance(sections, list):
        return tuple(str(s) for s in sections)
    return None


def delete_preset(name: str) -> bool:
    path = preset_path(name)
    if path.is_file():
        path.unlink()
        return True
    return False

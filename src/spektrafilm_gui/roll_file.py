"""Phase 8: the .sfroll roll session file (save/load + double-click open).

A roll file is a small VERSIONED JSON document: the photo LOCATIONS (relative
to the roll file first, absolute as fallback), the per-frame edits
(gui_state_to_dict of each stored look), the applied-roll-EV bookkeeping
(restored as saved, so re-applying a roll exposure never recompounds), and an
optional small embedded JPEG thumbnail per frame for offline browsing /
relinking. It NEVER contains pixels of the sources and never touches the
originals — a 30-frame roll is well under a megabyte even with thumbnails.

Pure logic + file IO, no Qt: the controller owns the dialogs and prompts;
everything here is unit-testable headless. Reuses the existing persistence
vocabulary (gui_state_to_dict / gui_state_from_dict, whose merge-onto-defaults
also absorbs version drift in the per-frame looks) and RollSession's
snapshot/restore API — no parallel state system.

Reopening a roll reproduces the LOOK exactly (the GUI state round-trips);
the demosaiced pixels of a RAW may differ by ~1e-2 run-to-run (known RAW
decode nondeterminism) — session restore, not bit-identity.
"""

from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from spektrafilm_gui.persistence import gui_state_from_dict, gui_state_to_dict
from spektrafilm_gui.roll_session import RollSession, is_supported_path

SFROLL_FORMAT = 'spektrafilm-roll'
SFROLL_VERSION = 1
SFROLL_EXTENSION = '.sfroll'
SFROLL_DIALOG_FILTER = 'Spektrafilm roll (*.sfroll);;All files (*)'
THUMBNAIL_MAX_DIM = 192
THUMBNAIL_JPEG_QUALITY = 80


def _app_version() -> str:
    try:
        from importlib.metadata import version
        return version('spektrafilm')
    except Exception:
        return 'unknown'


def encode_thumbnail(array) -> Optional[str]:
    """uint8 (h, w, 3) -> base64 JPEG string (fitted to THUMBNAIL_MAX_DIM),
    None on any failure — thumbnails are best-effort, never fatal."""
    try:
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.ascontiguousarray(array[..., :3]))
        img.thumbnail((THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=THUMBNAIL_JPEG_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode('ascii')
    except Exception:
        return None


def decode_thumbnail(text: str):
    """base64 JPEG string -> uint8 (h, w, 3) array, None on failure."""
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(text)))
        img.load()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def roll_to_dict(
    session: RollSession,
    *,
    roll_path: str,
    name: str = '',
    roll_settings: Optional[dict] = None,
    thumbnail_fn: Optional[Callable[[str, str], Any]] = None,
) -> dict:
    """Serialize a RollSession to the .sfroll document.

    ``thumbnail_fn(path, kind) -> uint8 array | None`` is optional (the
    controller passes roll_thumbnails.load_thumbnail_array); failures embed no
    thumbnail. ``roll_settings`` carries roll-level extras (e.g. the canvas
    background) verbatim.
    """
    roll_dir = os.path.dirname(os.path.abspath(roll_path))
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    frames = []
    for roll_file, state, applied_ev in session.snapshot_entries():
        try:
            relative = os.path.relpath(roll_file.path, roll_dir)
        except ValueError:            # different drive: no relative form
            relative = None
        thumbnail = None
        if thumbnail_fn is not None:
            array = thumbnail_fn(roll_file.path, roll_file.kind)
            if array is not None:
                thumbnail = encode_thumbnail(array)
        frames.append({
            'absolute_path': roll_file.path,
            'relative_path': relative,
            'name': roll_file.name,
            'kind': roll_file.kind,
            'look': gui_state_to_dict(state) if state is not None else None,
            'applied_roll_ev': applied_ev,
            'rating': int(session.rating_for(roll_file)),
            'thumbnail': thumbnail,
        })
    return {
        'format': SFROLL_FORMAT,
        'schema_version': SFROLL_VERSION,
        'app_version': _app_version(),
        'created': timestamp,
        'modified': timestamp,
        'name': name or os.path.splitext(os.path.basename(roll_path))[0],
        'current_index': int(session.current_index),
        'roll_settings': dict(roll_settings or {}),
        'frames': frames,
    }


def save_roll(
    session: RollSession,
    path: str,
    *,
    name: str = '',
    roll_settings: Optional[dict] = None,
    thumbnail_fn: Optional[Callable[[str, str], Any]] = None,
) -> dict:
    """Write the roll file (UTF-8 JSON). Returns the written document."""
    document = roll_to_dict(
        session, roll_path=path, name=name,
        roll_settings=roll_settings, thumbnail_fn=thumbnail_fn)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(document, fh, indent=1)
    return document


def load_roll_dict(path: str) -> dict:
    """Parse + validate a .sfroll file. Raises ValueError on a wrong format;
    NEWER schema versions than this build knows still load best-effort (the
    per-frame looks merge onto defaults, unknown top-level keys are ignored)."""
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get('format') != SFROLL_FORMAT:
        raise ValueError('Not a Spektrafilm roll file.')
    if not isinstance(data.get('frames'), list):
        raise ValueError('Roll file has no frame list.')
    return data


def resolve_frame_path(frame: dict, roll_dir: str) -> Optional[str]:
    """The frame's source on THIS machine: relative to the roll file FIRST
    (portability across machines/sync roots), absolute as recorded second.
    None when neither exists."""
    relative = frame.get('relative_path')
    if relative:
        candidate = os.path.abspath(os.path.join(roll_dir, relative))
        if os.path.isfile(candidate):
            return candidate
    absolute = frame.get('absolute_path')
    if absolute and os.path.isfile(absolute):
        return absolute
    return None


def find_in_relink_dirs(frame: dict, relink_dirs: list[str]) -> Optional[str]:
    """Locate a missing frame by FILENAME in user-supplied relink folders."""
    name = frame.get('name') or os.path.basename(frame.get('absolute_path') or '')
    if not name:
        return None
    for directory in relink_dirs:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and is_supported_path(candidate):
            return candidate
    return None


def restore_session(
    data: dict,
    roll_dir: str,
    *,
    relink_dirs: Optional[list[str]] = None,
) -> tuple[RollSession, list[dict]]:
    """Rebuild a RollSession from a parsed roll document.

    Per-frame looks come back through gui_state_from_dict (merge-onto-defaults
    absorbs schema drift); the applied-EV record is restored AS SAVED so a
    later 'Use roll exposure' replaces instead of compounding. Frames whose
    source cannot be found (relative, then absolute, then the relink folders)
    are returned as the second element — each still carrying its embedded
    thumbnail — and are NOT added to the session.
    """
    session = RollSession()
    entries: list[tuple[str, Any, Optional[float]]] = []
    missing: list[dict] = []
    for frame in data.get('frames', []):
        if not isinstance(frame, dict):
            continue
        path = resolve_frame_path(frame, roll_dir)
        if path is None and relink_dirs:
            path = find_in_relink_dirs(frame, relink_dirs)
        if path is None:
            missing.append(frame)
            continue
        look = frame.get('look')
        state = gui_state_from_dict(look) if isinstance(look, dict) else None
        applied_ev = frame.get('applied_roll_ev')
        entries.append((path, state,
                        float(applied_ev) if applied_ev is not None else None,
                        int(frame.get('rating') or 0)))
    session.restore_entries(entries, int(data.get('current_index', 0)))
    return session, missing

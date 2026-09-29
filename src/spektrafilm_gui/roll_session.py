"""Multi-file roll session (Phase 5 / B1), adapted from NegPy's WorkspaceSession.

Pure-Python state holder: an ordered list of files, the selected index, and a
per-file store of GUI states. The controller owns one instance and decides when
to snapshot/apply states; this module has no Qt dependencies so the logic is
unit-testable headless.

Semantics (NegPy-informed, adjusted for spektrafilm's roll workflow):
- Files are deduplicated by normalized absolute path (NegPy hashes because it
  persists settings across sessions in a DB; this session is in-memory).
- A file has NO stored state until the user switches away from it (the snapshot
  happens at switch time) or a look is propagated onto it. A file without a
  stored state INHERITS whatever the GUI currently shows -- so a fresh roll
  naturally shares the look you are developing, like frames on one film strip.
- "Apply look to all" stamps a deep copy of the given state onto every file
  (NegPy's copy/paste-to-roll collapsed into one action).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Extensions rawpy can decode -> the Import Raw path; everything else goes
# through the OpenImageIO Import RGB path.
RAW_EXTENSIONS = {
    '.raf', '.nef', '.nrw', '.cr2', '.cr3', '.crw', '.dng', '.arw', '.srf',
    '.sr2', '.orf', '.rw2', '.pef', '.srw', '.kdc', '.dcr', '.3fr', '.iiq',
    '.erf', '.mef', '.mos', '.mrw', '.x3f', '.fff', '.rwl',
}

RGB_EXTENSIONS = {
    '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.exr', '.bmp', '.webp',
}


def classify_path(path: str) -> str:
    """'raw' | 'rgb' by extension ('rgb' for anything OIIO might read)."""
    ext = os.path.splitext(path)[1].lower()
    return 'raw' if ext in RAW_EXTENSIONS else 'rgb'


def is_supported_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in RAW_EXTENSIONS or ext in RGB_EXTENSIONS


@dataclass
class RollFile:
    path: str
    name: str
    kind: str  # 'raw' | 'rgb'


def robust_roll_ev(evs: list, outlier_ev: float = 1.0) -> tuple:
    """Roll-average EV with outlier rejection (NegPy Batch Analysis semantics):
    frames further than ``outlier_ev`` stops from the median are discarded and
    the rest averaged. Returns (roll_ev, used_count, rejected_count); the
    median frame itself is always kept, so used_count >= 1 for non-empty input."""
    values = [float(v) for v in evs if v is not None]
    if not values:
        return 0.0, 0, 0
    median = float(sorted(values)[len(values) // 2])
    kept = [v for v in values if abs(v - median) <= float(outlier_ev)]
    return float(sum(kept) / len(kept)), len(kept), len(values) - len(kept)


@dataclass
class RollSession:
    files: list[RollFile] = field(default_factory=list)
    current_index: int = -1
    _states: dict[str, Any] = field(default_factory=dict)
    _applied_roll_ev: dict[str, float] = field(default_factory=dict)
    _ratings: dict[str, int] = field(default_factory=dict)   # 0-5 stars per path

    # ------------------------------------------------------------- files
    def add_files(self, paths: list[str]) -> list[RollFile]:
        """Adds files (deduplicated, unsupported extensions skipped); returns
        the RollFiles actually added. Selects the first file when the roll was
        previously empty."""
        added: list[RollFile] = []
        existing = {f.path for f in self.files}
        for p in paths:
            norm = os.path.normcase(os.path.abspath(p))
            if norm in existing or not is_supported_path(norm):
                continue
            rf = RollFile(path=norm, name=os.path.basename(p), kind=classify_path(norm))
            self.files.append(rf)
            existing.add(norm)
            added.append(rf)
        if self.current_index < 0 and self.files:
            self.current_index = 0
        return added

    def remove(self, index: int) -> Optional[RollFile]:
        """Removes a file; keeps the selection on the nearest remaining file.
        Returns the removed RollFile (its stored state is dropped)."""
        if not (0 <= index < len(self.files)):
            return None
        removed = self.files.pop(index)
        self._states.pop(removed.path, None)
        self._applied_roll_ev.pop(removed.path, None)
        self._ratings.pop(removed.path, None)
        if not self.files:
            self.current_index = -1
        elif index < self.current_index:
            self.current_index -= 1
        elif index == self.current_index:
            self.current_index = min(index, len(self.files) - 1)
        return removed

    def clear(self) -> int:
        """Empties the whole roll (files on disk untouched). Returns the number
        of files removed; stored looks and applied-EV records are dropped."""
        count = len(self.files)
        self.files.clear()
        self._states.clear()
        self._applied_roll_ev.clear()
        self._ratings.clear()
        self.current_index = -1
        return count

    def select(self, index: int) -> Optional[RollFile]:
        if not (0 <= index < len(self.files)):
            return None
        self.current_index = index
        return self.files[index]

    @property
    def current(self) -> Optional[RollFile]:
        if 0 <= self.current_index < len(self.files):
            return self.files[self.current_index]
        return None

    def position_label(self) -> str:
        if not self.files:
            return '0 / 0'
        return f'{self.current_index + 1} / {len(self.files)}'

    # ------------------------------------------------------------- states
    def store_state_for_current(self, state: Any) -> None:
        cur = self.current
        if cur is not None:
            self._states[cur.path] = copy.deepcopy(state)

    def state_for(self, roll_file: RollFile) -> Optional[Any]:
        stored = self._states.get(roll_file.path)
        return copy.deepcopy(stored) if stored is not None else None

    def apply_state_to_all(self, state: Any) -> int:
        """Stamps a deep copy of ``state`` onto every file; returns the count."""
        for f in self.files:
            self._states[f.path] = copy.deepcopy(state)
        return len(self.files)

    def apply_sections_to_all(self, state: Any, section_names: Any,
                              *, default_state: Any) -> int:
        """Selective apply-to-all (NAT 2026-07-25): copy ONLY ``section_names``
        from ``state`` onto every frame, keeping the rest of each frame's own
        edit. A frame with no stored look starts from ``default_state`` so its
        untouched sections stay at the app default, not at the source look.
        Returns the count of frames updated."""
        names = list(section_names)
        if not names:
            return 0
        for f in self.files:
            base = self._states.get(f.path)
            base = copy.deepcopy(base if base is not None else default_state)
            for name in names:
                setattr(base, name, copy.deepcopy(getattr(state, name)))
            self._states[f.path] = base
        return len(self.files)

    def has_state(self, roll_file: RollFile) -> bool:
        return roll_file.path in self._states

    def rating_for(self, roll_file: RollFile) -> int:
        """The frame's 0-5 star rating (0 = unrated)."""
        return int(self._ratings.get(roll_file.path, 0))

    def set_rating(self, roll_file: RollFile, rating: int) -> None:
        rating = min(max(int(rating), 0), 5)
        if rating:
            self._ratings[roll_file.path] = rating
        else:
            self._ratings.pop(roll_file.path, None)

    def applied_ev_for(self, roll_file: RollFile) -> Optional[float]:
        """The roll-EV contribution currently stamped on a frame (None if the
        frame never received one) — the idempotency bookkeeping of
        :meth:`apply_roll_exposure`, exposed for session persistence."""
        value = self._applied_roll_ev.get(roll_file.path)
        return float(value) if value is not None else None

    def snapshot_entries(self) -> list[tuple[RollFile, Any, Optional[float]]]:
        """(file, stored look | None, applied roll EV | None) per frame, in
        order — everything a roll session file needs (Phase 8)."""
        return [
            (f, self.state_for(f), self.applied_ev_for(f))
            for f in self.files
        ]

    def restore_entries(
        self,
        entries: list[tuple[str, Any, Optional[float]]],
        current_index: int = 0,
    ) -> int:
        """Rebuild the session from (path, look | None, applied EV | None
        [, rating]) tuples (Phase 8 roll-file load). Replaces the whole
        session; paths are normalized and deduplicated exactly like
        :meth:`add_files`, and the applied-EV bookkeeping is restored AS
        SAVED so a later apply_roll_exposure never recompounds. Returns the
        frame count."""
        self.clear()
        for entry in entries:
            path, state, applied_ev = entry[0], entry[1], entry[2]
            rating = int(entry[3]) if len(entry) > 3 and entry[3] else 0
            added = self.add_files([path])
            if not added:
                continue
            key = added[0].path
            if state is not None:
                self._states[key] = copy.deepcopy(state)
            if applied_ev is not None:
                self._applied_roll_ev[key] = float(applied_ev)
            if rating:
                self._ratings[key] = min(max(rating, 0), 5)
        if self.files:
            self.current_index = min(max(int(current_index), 0), len(self.files) - 1)
        return len(self.files)

    def apply_roll_exposure(self, roll_ev: float, fallback_state: Any) -> int:
        """Stamps the roll-average exposure onto every frame: per-frame auto
        exposure OFF, exposure_compensation_ev adjusted by ``roll_ev``.

        Idempotent: the roll contribution previously applied to a frame is
        subtracted before the new one is added, so re-running an analysis and
        re-applying never compounds. The user's own compensation (anything they
        dialed on top) is preserved. Frames without a stored look are stamped
        from a deep copy of ``fallback_state`` first."""
        roll_ev = float(roll_ev)
        for f in self.files:
            state = self._states.get(f.path)
            if state is None:
                state = copy.deepcopy(fallback_state)
                self._states[f.path] = state
            previous = self._applied_roll_ev.get(f.path, 0.0)
            state.simulation.auto_exposure = False
            state.simulation.exposure_compensation_ev = float(
                state.simulation.exposure_compensation_ev
            ) - previous + roll_ev
            self._applied_roll_ev[f.path] = roll_ev
        return len(self.files)

"""Phase 8B: Windows .sfroll file association (double-click to open a roll).

Self-registration into HKCU\\Software\\Classes — per-user, NO admin rights.
Called on every FROZEN launch: re-registering is idempotent and auto-heals
the association when the portable folder moves (the recorded exe path is
refreshed). Source runs (no sys.frozen) never register: associating a dev
interpreter would be wrong on a user machine.

Registration gives double-click, drag-onto-exe and "Open with" for free; the
argv side lives in app.py main(). A proper installer (Inno/NSIS) that also
registers under HKLM is a later, optional nicety — out of scope here.

The functions accept an explicit exe path and registry base so tests can
exercise the real winreg writes under a scratch key instead of touching
HKCU\\Software\\Classes.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

SFROLL_PROGID = 'Spektrafilm.Roll'
SFROLL_DESCRIPTION = 'Spektrafilm roll session'
_CLASSES_BASE = r'Software\Classes'


def frozen_executable() -> Optional[str]:
    """The frozen app's exe path, None when running from source."""
    if getattr(sys, 'frozen', False):
        return os.path.abspath(sys.executable)
    return None


def register_sfroll_association(
    exe_path: Optional[str] = None,
    *,
    classes_base: str = _CLASSES_BASE,
) -> bool:
    """Write the per-user .sfroll association. Returns True when registered.

    No-ops (False) on non-Windows or when no exe path is available (source
    run without an explicit path). Never raises — a registry failure must not
    break app startup."""
    if sys.platform != 'win32':
        return False
    exe = exe_path or frozen_executable()
    if not exe:
        return False
    try:
        import winreg

        def set_value(subkey: str, value: str) -> None:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  f'{classes_base}\\{subkey}') as key:
                winreg.SetValueEx(key, None, 0, winreg.REG_SZ, value)

        set_value('.sfroll', SFROLL_PROGID)
        set_value(SFROLL_PROGID, SFROLL_DESCRIPTION)
        set_value(f'{SFROLL_PROGID}\\DefaultIcon', f'"{exe}",0')
        set_value(f'{SFROLL_PROGID}\\shell\\open\\command', f'"{exe}" "%1"')
        return True
    except Exception:
        return False


def sfroll_path_from_argv(argv: list[str]) -> Optional[str]:
    """The first existing .sfroll path in argv (the double-click payload),
    None when the launch carries none — a normal start."""
    for argument in argv:
        if str(argument).lower().endswith('.sfroll') and os.path.isfile(argument):
            return os.path.abspath(argument)
    return None

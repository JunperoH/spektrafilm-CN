"""File logging for the (windowed, console-less) app.

Motivated by the 2026-07-12 debugging sessions: the frozen exe has no console,
so nothing the app logs ever reaches anyone. This module writes a rotating log
to %APPDATA%/spektrafilm/logs, captures unhandled exceptions, and exposes the
folder for the GUI's 'Open logs' button. Setup is defensive: a logging failure
must never break app startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys

LOG_DIR_NAME = 'logs'
LOG_FILE_NAME = 'spektrafilm.log'
_MAX_BYTES = 1_000_000
_BACKUPS = 5
_installed = False


def log_directory() -> str:
    """%APPDATA%/spektrafilm/logs (created on demand)."""
    base = os.environ.get('APPDATA') or os.path.join(os.path.expanduser('~'), '.spektrafilm')
    directory = os.path.join(base, 'spektrafilm', LOG_DIR_NAME)
    os.makedirs(directory, exist_ok=True)
    return directory


def setup_file_logging() -> str | None:
    """Attach a rotating file handler to the root logger (INFO+) and route
    unhandled exceptions into it. Returns the log directory, None on failure.
    Idempotent: repeated calls install nothing twice."""
    global _installed
    try:
        directory = log_directory()
        if _installed:
            return directory
        handler = logging.handlers.RotatingFileHandler(
            os.path.join(directory, LOG_FILE_NAME),
            maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-7s %(name)s: %(message)s'))
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

        previous_hook = sys.excepthook

        def _log_unhandled(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                logging.getLogger('spektrafilm.crash').critical(
                    'UNHANDLED EXCEPTION', exc_info=(exc_type, exc_value, exc_traceback))
            previous_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = _log_unhandled
        _installed = True
        logging.getLogger('spektrafilm.app').info(
            'file logging started (frozen=%s, argv=%s)',
            bool(getattr(sys, 'frozen', False)), sys.argv[1:])
        return directory
    except Exception:
        return None


# Known-benign Qt console noise, demoted to DEBUG (kept out of the log file
# and the console, still visible if the root logger is put in DEBUG).
_QT_BENIGN_FRAGMENTS = (
    # Qt internals rescaling a font that a stylesheet sized in pixels
    # (pointSize() is -1 for pixel-sized fonts); cosmetic, no effect.
    'QFont::setPointSize: Point size <= 0',
    'QFont::setPointSizeF: Point size <= 0',
)
_qt_handler_installed = False


def install_qt_message_logging() -> bool:
    """Route Qt's own messages (qWarning etc.) into the ``qt`` logger instead
    of stderr, so they land in the log file with timestamps and stop littering
    the console. Known-benign noise is demoted to DEBUG. Defensive + idempotent
    like setup_file_logging."""
    global _qt_handler_installed
    if _qt_handler_installed:
        return True
    try:
        from qtpy import QtCore

        logger = logging.getLogger('qt')
        levels = {
            QtCore.QtMsgType.QtDebugMsg: logging.DEBUG,
            QtCore.QtMsgType.QtInfoMsg: logging.INFO,
            QtCore.QtMsgType.QtWarningMsg: logging.WARNING,
            QtCore.QtMsgType.QtCriticalMsg: logging.ERROR,
            QtCore.QtMsgType.QtFatalMsg: logging.CRITICAL,
        }

        def _handler(mode, _context, message):
            level = levels.get(mode, logging.WARNING)
            if any(fragment in message for fragment in _QT_BENIGN_FRAGMENTS):
                level = logging.DEBUG
            logger.log(level, message)

        QtCore.qInstallMessageHandler(_handler)
        _qt_handler_installed = True
        return True
    except Exception:
        return False


def open_log_directory() -> bool:
    """Open the log folder in Explorer (the GUI 'Open logs' action)."""
    try:
        os.startfile(log_directory())  # noqa: S606 - intentional shell open
        return True
    except Exception:
        return False

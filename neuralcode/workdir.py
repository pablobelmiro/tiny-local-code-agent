"""Picks or creates the sessions/<id>/ subdirectory a run works inside.

Called once, before anything else touches the filesystem or chdirs, so the
choice is made from the repo root the user launched neuralcode from.
"""

from datetime import datetime
from pathlib import Path

SESSIONS_DIR_NAME = "sessions"


def list_existing(root):
    """Session subdirectories under root/sessions, newest first."""
    sessions_dir = root / SESSIONS_DIR_NAME
    if not sessions_dir.exists():
        return []
    return sorted(
        (p for p in sessions_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _new_session_dir(root):
    sessions_dir = root / SESSIONS_DIR_NAME
    sessions_dir.mkdir(parents=True, exist_ok=True)
    new_dir = sessions_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    new_dir.mkdir(exist_ok=True)
    return new_dir


def choose_or_create(root, ui):
    """Ask the user to pick an existing session directory or make a new one."""
    existing = list_existing(root)
    rows = ["new session directory"] + [p.name for p in existing]
    choice = ui.pick("session working directory", rows)

    if choice is None or choice == 0:
        return _new_session_dir(root)

    return existing[choice - 1]

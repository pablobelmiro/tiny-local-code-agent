"""Kernel-enforced limits on what bash can touch.

One policy - read anything, write only inside the project, no network - and a
different enforcement mechanism per OS. The idea ports; the mechanism never does.

The macOS profile is adapted from openai/codex (Apache-2.0), simplified.
https://github.com/openai/codex
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _project_root():
    """Computed at call time, not import time, so it follows os.chdir()."""
    return Path.cwd().resolve()


def _profile(project_root):
    return f"""(version 1)
(deny default)
(allow process-exec process-fork signal)
(allow file-read*)
(allow sysctl-read)
(deny network*)
(allow file-write* (subpath "{project_root}") (literal "/dev/null"))
(deny file-write* (subpath "{project_root}/.git"))
"""


def wrap(command):
    """Wrap a shell command in an OS sandbox. None means we have no sandbox."""
    project_root = _project_root()

    if sys.platform == "darwin":
        profile = Path(tempfile.gettempdir()) / "neuralcode.sb"
        profile.write_text(_profile(project_root))
        return ["sandbox-exec", "-f", str(profile), "/bin/sh", "-c", command]

    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return [
            "bwrap",
            "--ro-bind", "/", "/",
            "--bind", str(project_root), str(project_root),
            "--dev", "/dev", "--proc", "/proc",
            "--unshare-net", "--die-with-parent",
            "/bin/sh", "-c", command,
        ]

    return None  # Windows, or Linux without bubblewrap


def name():
    if sys.platform == "darwin":
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    return "none"


def run(command, timeout=60):
    """Run a command, sandboxed when the OS lets us."""
    sandboxed = wrap(command)
    return subprocess.run(
        sandboxed or command,
        shell=sandboxed is None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

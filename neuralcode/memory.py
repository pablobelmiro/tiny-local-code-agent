"""Per-session project memory: a small on-disk file scaffold.

No embeddings, no retrieval, no new dependencies - the model reads and
writes these files with the same read_file/write_file/str_replace tools it
already has. Only INDEX.md, kept small on purpose, is loaded automatically
(see llm.system_prompt); the rest is paged in by the model on demand.
"""

from pathlib import Path

MEMORY_DIR_NAME = "memory"

TEMPLATES = {
    "INDEX.md": """# Memory index

Read this first, every session. It stays small on purpose - the rest is
paged in only when needed.

- profile.md - stable facts about this project: stack, build/test commands
- decisions.md - append-only log of decisions made and why
- gotchas.md - one line per environment quirk or approach that failed

Keep this file short. Update the other files, not this one, as you learn
things - only add a line here if a whole new file gets added to memory/.
""",
    "profile.md": """# Project profile

Stable facts about this project: stack, build/test commands, conventions.
""",
    "decisions.md": """# Decisions

Append-only. One entry per decision: what, why, when.
""",
    "gotchas.md": """# Gotchas

One line per environment quirk or approach that did not work.
""",
}


def _memory_dir():
    return Path.cwd() / MEMORY_DIR_NAME


def ensure_scaffold():
    """Create memory/ and its template files if they don't exist yet.

    Idempotent and non-destructive: never overwrites a file that is already
    there, however it got there.
    """
    memory_dir = _memory_dir()
    memory_dir.mkdir(exist_ok=True)
    for name, template in TEMPLATES.items():
        path = memory_dir / name
        if not path.exists():
            path.write_text(template)


def read_index():
    """The always-loaded index, or "" if memory/ doesn't exist yet."""
    index_path = _memory_dir() / "INDEX.md"
    if not index_path.exists():
        return ""
    return index_path.read_text()

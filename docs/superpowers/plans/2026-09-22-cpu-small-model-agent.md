# CPU Small-Model Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU-friendly small-model profiles, a hard per-session token budget with context reset, and a per-session working subdirectory to the `neuralcode` fork.

**Architecture:** Three additive components layered onto the existing loop in `neuralcode/agent.py`: (1) `profiles.py` resolves named model profiles from `models.yaml` into `config.py`'s existing env-var-driven settings; (2) `budget.py` tracks accumulated token usage and, on confirmation past the limit, resets the conversation to just the system prompt; (3) `workdir.py` picks or creates a `sessions/<id>/` subdirectory and `os.chdir`s into it before any tool runs, which requires making `sandbox.py`'s `PROJECT` and `session.py`'s `SESSION_DIR` lazy (computed at call time, not at import time).

**Tech Stack:** Python 3.10+, `pyyaml` (already a dependency), `rich` (already used for all terminal UI, no new dependency), `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-09-22-cpu-small-model-agent-design.md`

## Global Constraints

- No new dependencies beyond what's already in `pyproject.toml` (`openai`, `prompt-toolkit`, `pyyaml`, `rich`).
- `BASE_URL`/`API_KEY` stay exclusively env-var driven — never part of a profile (spec Component 1).
- Module-level state (no classes) for `budget.py`, matching the existing style in `session.py`/`history.py`.
- Terminal UI stays on `rich`, following the existing patterns in `ui.py` (`Console`, `Padding`, `Text`, `Panel`) — no new UI library.
- `sessions/` subdirectories live under the repository root (where `neuralcode` is invoked from), not under `~/.agents`.
- Existing behavior (tools, sandbox policy, permissions, `/compact`, `/rewind`, `/sessions`, `--resume`) must keep working unchanged from a user's perspective, except that they now operate inside the chosen session subdirectory.

## Review Focus

- **`models.yaml` missing or absent `profiles` key** — `profiles.py` must not crash the whole agent; it should behave as if no profile exists (spec says perfis are optional).
- **`--profile` given a name not in `models.yaml`** — should fail clearly at startup, not silently fall back to defaults (the user asked for a specific profile; silently ignoring it would be confusing, and the spec doesn't say to fall back).
- **Token budget crossed mid-turn while `should_warn()` fires between tool calls, not just before the first LLM call of a turn** — the loop must not re-prompt for every remaining tool call in the same turn once the user already said yes; only the module-level "already warned this multiple" state prevents that, so it must be exercised in a test with more than one tool call after the warning.
- **User picks "new session directory" when `sessions/` doesn't exist yet at all** — `workdir.choose_or_create` must create both `sessions/` and the new subdirectory, not assume `sessions/` pre-exists.
- **Resetting context after budget confirmation while a tool call is in-flight (i.e. mid-turn, not only between turns)** — the reset must happen at a point where there's no dangling `tool_calls` message waiting for a `role: tool` reply, otherwise the next LLM call sends an invalid transcript. The chosen point (top of the loop, before `call_llm`, same place `should_warn` is checked) is inherently between a complete assistant turn and the next LLM call, so this is a design property to verify with a test, not new logic.

---

## File Structure

- `models.yaml` (new, repo root) — model profile definitions.
- `neuralcode/profiles.py` (new) — loads and resolves profiles.
- `neuralcode/config.py` (modify) — resolve `MODEL`/`CONTEXT_WINDOW`/`TOKEN_BUDGET` through the active profile.
- `neuralcode/budget.py` (new) — session token budget tracking and reset.
- `neuralcode/sandbox.py` (modify) — make `PROJECT`/`PROFILE` lazy.
- `neuralcode/session.py` (modify) — make `SESSION_DIR` lazy.
- `neuralcode/workdir.py` (new) — pick/create the session's working subdirectory.
- `neuralcode/ui.py` (modify) — budget line in `usage()`, new `budget_warning()`, `context_reset()`.
- `neuralcode/agent.py` (modify) — wire `--profile`, workdir selection + chdir at startup, budget check/reset in the main loop.
- `tests/test_profiles.py` (new)
- `tests/test_budget.py` (new)
- `tests/test_workdir.py` (new)
- `tests/test_sandbox.py` (new) — covers the lazy `PROJECT` behavior.
- `tests/test_session.py` (new) — covers the lazy `SESSION_DIR` behavior.

No `tests/` directory exists yet in the repo; Task 1 creates it.

---

## Task 1: Model profiles (`profiles.py` + `models.yaml`)

**Files:**
- Create: `models.yaml`
- Create: `neuralcode/profiles.py`
- Test: `tests/test_profiles.py`
- Test: `tests/__init__.py` (empty, makes `tests` a package so relative test imports are consistent)

**Interfaces:**
- Produces: `profiles.load_profiles(models_file: Path) -> dict[str, dict]` and `profiles.resolve_profile(profiles: dict, name: str | None) -> dict | None`. Later tasks (`config.py`) call these two functions.

- [ ] **Step 1: Create the tests directory and `models.yaml`**

Run:
```bash
mkdir -p tests
touch tests/__init__.py
```

Create `models.yaml`:

```yaml
profiles:
  small-2b:
    model: qwen2.5-coder:1.5b
    context_window: 32000
    token_budget: 60000
  medium-8b:
    model: qwen2.5-coder:7b
    context_window: 32000
    token_budget: 120000
  large-14b:
    model: qwen2.5-coder:14b
    context_window: 32000
    token_budget: 200000
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_profiles.py`:

```python
import textwrap

import pytest

from neuralcode import profiles


def write_yaml(tmp_path, content):
    path = tmp_path / "models.yaml"
    path.write_text(textwrap.dedent(content))
    return path


def test_load_profiles_parses_named_profiles(tmp_path):
    path = write_yaml(tmp_path, """
        profiles:
          small-2b:
            model: qwen2.5-coder:1.5b
            context_window: 32000
            token_budget: 60000
    """)

    loaded = profiles.load_profiles(path)

    assert loaded == {
        "small-2b": {
            "model": "qwen2.5-coder:1.5b",
            "context_window": 32000,
            "token_budget": 60000,
        }
    }


def test_load_profiles_missing_file_returns_empty_dict(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"

    assert profiles.load_profiles(missing) == {}


def test_load_profiles_file_without_profiles_key_returns_empty_dict(tmp_path):
    path = write_yaml(tmp_path, "some_other_key: 1\n")

    assert profiles.load_profiles(path) == {}


def test_resolve_profile_returns_none_for_none_name():
    assert profiles.resolve_profile({"a": {}}, None) is None


def test_resolve_profile_returns_matching_profile():
    loaded = {"small-2b": {"model": "qwen2.5-coder:1.5b"}}

    assert profiles.resolve_profile(loaded, "small-2b") == {"model": "qwen2.5-coder:1.5b"}


def test_resolve_profile_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        profiles.resolve_profile({"small-2b": {}}, "does-not-exist")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuralcode.profiles'`

- [ ] **Step 4: Implement `neuralcode/profiles.py`**

```python
"""Named model profiles for running the agent against small CPU models.

Profiles are optional - a missing or empty models.yaml just means no
profile is active and config.py falls back to its existing defaults.
"""

import yaml


def load_profiles(models_file):
    """Read the `profiles` map out of a models.yaml. Missing file or key -> {}."""
    if not models_file.exists():
        return {}
    data = yaml.safe_load(models_file.read_text()) or {}
    return data.get("profiles") or {}


def resolve_profile(profiles, name):
    """Look up a profile by name. None name -> None. Unknown name -> KeyError.

    Unknown names raise rather than silently falling back, because the user
    asked for a specific profile by name - silently ignoring a typo would be
    confusing.
    """
    if name is None:
        return None
    return profiles[name]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add models.yaml neuralcode/profiles.py tests/__init__.py tests/test_profiles.py
git commit -m "Add model profiles (profiles.py + models.yaml)"
```

---

## Task 2: Wire profiles into `config.py`

**Files:**
- Modify: `neuralcode/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `profiles.load_profiles`, `profiles.resolve_profile` from Task 1.
- Produces: `config.TOKEN_BUDGET` (new module attribute, `int | None`), and `config.MODEL`/`config.CONTEXT_WINDOW` now profile-aware. `config.active_profile_name()` returns the resolved profile name (`str | None`) — used by `agent.py` for the `--profile` CLI arg and by `workdir.py`/`ui.py` if needed later.

Because `config.py` reads `os.environ["BASE_URL"]`/`os.environ["API_KEY"]` at import time (unchanged, per spec), and because profile resolution needs to happen before those existing lines run only for `MODEL`/`CONTEXT_WINDOW`/`TOKEN_BUDGET`, this task restructures `config.py` to compute the profile first, then layer env vars on top.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import importlib
import os

import pytest


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("API_KEY", "ollama")
    for var in ("MODEL", "CONTEXT_WINDOW", "TOKEN_BUDGET", "PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def reload_config():
    import neuralcode.config as config
    return importlib.reload(config)


def test_config_defaults_when_no_profile_and_no_models_yaml(clean_env):
    config = reload_config()

    assert config.MODEL == "deepseek/deepseek-v4-flash"
    assert config.CONTEXT_WINDOW == 128_000
    assert config.TOKEN_BUDGET is None


def test_config_uses_profile_values_when_profile_env_set(clean_env, monkeypatch):
    (clean_env / "models.yaml").write_text(
        "profiles:\n"
        "  small-2b:\n"
        "    model: qwen2.5-coder:1.5b\n"
        "    context_window: 32000\n"
        "    token_budget: 60000\n"
    )
    monkeypatch.setenv("PROFILE", "small-2b")

    config = reload_config()

    assert config.MODEL == "qwen2.5-coder:1.5b"
    assert config.CONTEXT_WINDOW == 32000
    assert config.TOKEN_BUDGET == 60000


def test_config_explicit_env_var_overrides_profile(clean_env, monkeypatch):
    (clean_env / "models.yaml").write_text(
        "profiles:\n"
        "  small-2b:\n"
        "    model: qwen2.5-coder:1.5b\n"
        "    context_window: 32000\n"
        "    token_budget: 60000\n"
    )
    monkeypatch.setenv("PROFILE", "small-2b")
    monkeypatch.setenv("MODEL", "explicit-override")

    config = reload_config()

    assert config.MODEL == "explicit-override"
    assert config.CONTEXT_WINDOW == 32000  # not overridden, still from profile
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `TOKEN_BUDGET` attribute doesn't exist, and `PROFILE` handling doesn't exist yet, so the profile-related assertions fail.

- [ ] **Step 3: Implement the change in `neuralcode/config.py`**

Replace the full contents of `neuralcode/config.py`:

```python
"""Settings: real environment variables first, then ~/.agents/env."""

import os
from pathlib import Path

from . import profiles

ENV_FILE = Path.home() / ".agents" / "env"

if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

BASE_URL = os.environ["BASE_URL"]
API_KEY = os.environ["API_KEY"]

_PROFILE_NAME = os.environ.get("PROFILE")
_PROFILES = profiles.load_profiles(Path.cwd() / "models.yaml")
_ACTIVE_PROFILE = profiles.resolve_profile(_PROFILES, _PROFILE_NAME) or {}


def active_profile_name():
    return _PROFILE_NAME


MODEL = os.environ.get("MODEL") or _ACTIVE_PROFILE.get("model", "deepseek/deepseek-v4-flash")

# How much room the model has, and how we spend it.
CONTEXT_WINDOW = int(
    os.environ.get("CONTEXT_WINDOW") or _ACTIVE_PROFILE.get("context_window", 128_000)
)
COMPACT_AT = 0.85  # compact once the prompt crosses this much of the window
COMPACT_TO = 0.35  # and cut back to this much, so it does not retrigger soon

_env_budget = os.environ.get("TOKEN_BUDGET")
TOKEN_BUDGET = (
    int(_env_budget) if _env_budget else _ACTIVE_PROFILE.get("token_budget")
)
```

Note: `Path.cwd() / "models.yaml"` is resolved at import time, which is before Task 3's `workdir.py` changes the working directory (import happens once, at process start, before `main()` runs). This is intentional — profiles describe which model/budget to use, decided once per process from the directory `neuralcode` was launched from, not from inside a `sessions/<id>/` subdirectory.

**Design note — no `--profile` CLI flag:** `agent.py` imports `from .llm import SYSTEM_PROMPT, call_llm` at module top, and `llm.py` imports `config` immediately, which resolves `PROFILE` from `os.environ` right away. By the time `argparse` would parse a `--profile` flag inside `main()`, `config.py` has already run — setting `os.environ["PROFILE"]` at that point would be a no-op. Rather than reordering imports to work around this, profile selection is via the `PROFILE` environment variable only (e.g. `PROFILE=small-2b neuralcode`), consistent with how `MODEL`/`BASE_URL`/`API_KEY` already work today. `agent.py` needs no changes in this task — profile selection is fully handled by `config.py` reading `PROFILE` from the environment, which Step 3 above already implements.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_profiles.py tests/test_config.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add neuralcode/config.py tests/test_config.py
git commit -m "Resolve MODEL/CONTEXT_WINDOW/TOKEN_BUDGET through active profile"
```

---

## Task 3: Token budget tracking (`budget.py`)

**Files:**
- Create: `neuralcode/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `config.TOKEN_BUDGET` from Task 2.
- Produces: `budget.track(usage: dict) -> None`, `budget.remaining() -> int | None`, `budget.should_warn() -> bool`, `budget.mark_warned() -> None`, `budget.reset() -> None`, `budget.total() -> int`. `agent.py` (Task 6) and `ui.py` (Task 5) consume all of these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_budget.py`:

```python
import pytest

from neuralcode import budget


@pytest.fixture(autouse=True)
def reset_budget_state():
    budget.reset()
    yield
    budget.reset()


@pytest.fixture
def with_budget(monkeypatch):
    monkeypatch.setattr(budget.config, "TOKEN_BUDGET", 1000)


@pytest.fixture
def without_budget(monkeypatch):
    monkeypatch.setattr(budget.config, "TOKEN_BUDGET", None)


def usage(prompt_tokens, completion_tokens):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def test_track_accumulates_prompt_and_completion_tokens(with_budget):
    budget.track(usage(100, 50))
    budget.track(usage(200, 25))

    assert budget.total() == 375


def test_remaining_is_none_when_no_budget_configured(without_budget):
    budget.track(usage(100, 50))

    assert budget.remaining() is None


def test_remaining_counts_down_from_configured_budget(with_budget):
    budget.track(usage(100, 50))

    assert budget.remaining() == 850


def test_should_warn_false_below_budget(with_budget):
    budget.track(usage(500, 0))

    assert budget.should_warn() is False


def test_should_warn_true_once_budget_crossed(with_budget):
    budget.track(usage(900, 200))  # total 1100 > 1000

    assert budget.should_warn() is True


def test_should_warn_false_again_after_mark_warned_until_next_multiple(with_budget):
    budget.track(usage(900, 200))  # total 1100, over budget
    assert budget.should_warn() is True
    budget.mark_warned()

    assert budget.should_warn() is False

    budget.track(usage(0, 500))  # total 1600, still same "over budget" multiple (< 2x)
    assert budget.should_warn() is False


def test_should_warn_true_again_after_next_multiple_crossed(with_budget):
    budget.track(usage(900, 200))  # total 1100
    budget.mark_warned()

    budget.track(usage(1000, 0))  # total 2100, crossed the 2x=2000 multiple

    assert budget.should_warn() is True


def test_should_warn_false_when_no_budget_configured(without_budget):
    budget.track(usage(10_000, 10_000))

    assert budget.should_warn() is False


def test_reset_clears_total_and_warn_state(with_budget):
    budget.track(usage(900, 200))
    budget.mark_warned()

    budget.reset()

    assert budget.total() == 0
    assert budget.should_warn() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuralcode.budget'`

- [ ] **Step 3: Implement `neuralcode/budget.py`**

```python
"""Hard per-session token budget.

One accumulated counter for the whole session (prompt + completion tokens
across every LLM call). Crossing config.TOKEN_BUDGET, or any further
multiple of it, asks the user to confirm before the next call - see
should_warn()/mark_warned(). Module-level state, same pattern history.py and
session.py use for session-scoped counters.
"""

from . import config

_total = 0
_warned_at_multiple = 0  # last multiple of TOKEN_BUDGET we already warned for


def track(usage):
    global _total
    _total += (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)


def total():
    return _total


def remaining():
    if config.TOKEN_BUDGET is None:
        return None
    return config.TOKEN_BUDGET - _total


def should_warn():
    if config.TOKEN_BUDGET is None or config.TOKEN_BUDGET <= 0:
        return False
    current_multiple = _total // config.TOKEN_BUDGET
    return current_multiple >= 1 and current_multiple > _warned_at_multiple


def mark_warned():
    global _warned_at_multiple
    if config.TOKEN_BUDGET:
        _warned_at_multiple = _total // config.TOKEN_BUDGET


def reset():
    global _total, _warned_at_multiple
    _total = 0
    _warned_at_multiple = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_budget.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add neuralcode/budget.py tests/test_budget.py
git commit -m "Add session token budget tracking (budget.py)"
```

---

## Task 4: Lazy project path in `sandbox.py`

**Files:**
- Modify: `neuralcode/sandbox.py`
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Produces: `sandbox.wrap(command)`, `sandbox.name()`, `sandbox.run(command, timeout=60)` — same public signatures as before, but now reflecting `Path.cwd()` at call time rather than at import time.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox.py`:

```python
import sys

import pytest

from neuralcode import sandbox


@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt profile is macOS-only")
def test_wrap_profile_uses_current_cwd_not_import_time_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    wrapped = sandbox.wrap("echo hi")

    profile_path = wrapped[2]
    from pathlib import Path

    profile_text = Path(profile_path).read_text()
    assert str(tmp_path) in profile_text


def test_run_executes_in_current_cwd(tmp_path, monkeypatch):
    (tmp_path / "marker.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)

    result = sandbox.run("cat marker.txt")

    assert "hello" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail (or trivially pass for the wrong reason)**

Run: `python -m pytest tests/test_sandbox.py -v`

Expected on Linux with `bwrap` installed or no sandbox: `test_run_executes_in_current_cwd` likely already passes today (bash `cat` runs in `subprocess`'s cwd regardless), but on macOS `test_wrap_profile_uses_current_cwd_not_import_time_cwd` FAILS today, because `PROJECT`/`PROFILE` are computed once at import time using whatever `cwd` was active during the *first* import of `sandbox` in the test run (likely the repo root, not `tmp_path`). This confirms the bug this task fixes. If running on Linux, this step is a no-op check (the macOS test is skipped) — the fix is still required in Task 6, since the same lazy pattern is needed for `session.py` and for the `sessions/<id>/` chdir to work correctly regardless of OS.

- [ ] **Step 3: Implement the fix in `neuralcode/sandbox.py`**

Replace the full contents of `neuralcode/sandbox.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sandbox.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest -v`
Expected: PASS (all tests from Tasks 1-4)

- [ ] **Step 6: Commit**

```bash
git add neuralcode/sandbox.py tests/test_sandbox.py
git commit -m "Make sandbox.py compute project root lazily, not at import time"
```

---

## Task 5: Lazy session directory in `session.py`

**Files:**
- Modify: `neuralcode/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: same public functions as before (`path_for`, `save`, `rewind_to`, `compacted`, `load`, `open_session`, `title`, `all_sessions`), now resolving the storage directory from `Path.cwd()` at call time. Adds `session.reset_context() -> list[dict]` — used by Task 6/7 for the budget-triggered context reset; returns a fresh `messages` list containing only the system prompt entry recorded to disk as a new resettable point (reuses the existing `compacted()` persistence shape, since both are "replace messages going forward" operations).

Note: `session.py`'s existing module-level globals `CURRENT` (session id / timestamp) and `WRITTEN` (message count already flushed) stay as module state — they are per-run session identity, not a project path, so they don't need to become lazy. Only `SESSION_DIR`/`PROJECT` (the two path-only globals) move to call-time.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session.py`:

```python
from neuralcode import session


def test_session_dir_reflects_current_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(session.Path, "home", lambda: tmp_path)
    project_a = tmp_path / "project-a"
    project_a.mkdir()
    monkeypatch.chdir(project_a)

    session.save([{"role": "user", "content": "hi"}])

    expected_dir = tmp_path / ".agents" / "sessions" / str(project_a.resolve()).replace("/", "-")
    assert expected_dir.exists()
    assert list(expected_dir.glob("*.jsonl"))


def test_session_dir_changes_when_cwd_changes_between_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(session.Path, "home", lambda: tmp_path)
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    monkeypatch.chdir(project_a)
    session.CURRENT = "20260101-000000"
    session.WRITTEN = 0
    session.save([{"role": "user", "content": "in a"}])

    monkeypatch.chdir(project_b)
    session.CURRENT = "20260101-000001"
    session.WRITTEN = 0
    session.save([{"role": "user", "content": "in b"}])

    dir_a = tmp_path / ".agents" / "sessions" / str(project_a.resolve()).replace("/", "-")
    dir_b = tmp_path / ".agents" / "sessions" / str(project_b.resolve()).replace("/", "-")
    assert list(dir_a.glob("*.jsonl"))
    assert list(dir_b.glob("*.jsonl"))


def test_reset_context_returns_only_system_prompt_and_persists_it(tmp_path, monkeypatch):
    monkeypatch.setattr(session.Path, "home", lambda: tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    session.CURRENT = "20260101-000002"
    session.WRITTEN = 0
    session.save([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])

    reset_messages = session.reset_context("sys")

    assert reset_messages == [{"role": "system", "content": "sys"}]
    reloaded = session.load(session.CURRENT)
    assert reloaded == [{"role": "system", "content": "sys"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -v`
Expected: FAIL — `test_session_dir_changes_when_cwd_changes_between_calls` fails because today's `SESSION_DIR`/`PROJECT` are fixed at import time and never move with `chdir`; `reset_context` doesn't exist yet.

- [ ] **Step 3: Implement the fix in `neuralcode/session.py`**

Replace the full contents of `neuralcode/session.py`:

```python
"""Transcripts on disk. One JSONL file per chat."""

import json
from datetime import datetime
from pathlib import Path

CURRENT = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITTEN = 0  # how many messages are already on disk


def _session_dir():
    """Computed at call time, not import time, so it follows os.chdir()."""
    project = str(Path.cwd().resolve()).replace("/", "-")
    return Path.home() / ".agents" / "sessions" / project


def path_for(session_id):
    return _session_dir() / f"{session_id}.jsonl"


def save(messages):
    """Append what is new. Never rewrite what is already on disk."""
    global WRITTEN
    _session_dir().mkdir(parents=True, exist_ok=True)
    with path_for(CURRENT).open("a") as f:
        for message in messages[WRITTEN:]:
            f.write(json.dumps(message) + "\n")
    WRITTEN = len(messages)


def rewind_to(count):
    """Record a rewind as an entry, so the old messages stay in the file."""
    global WRITTEN
    with path_for(CURRENT).open("a") as f:
        f.write(json.dumps({"rewind_to": count}) + "\n")
    WRITTEN = count


def compacted(messages):
    """Compaction rewrites history, so record the result and start from it."""
    global WRITTEN
    with path_for(CURRENT).open("a") as f:
        f.write(json.dumps({"compacted": messages}) + "\n")
    WRITTEN = len(messages)


def reset_context(system_prompt):
    """Token budget forced a clean slate: keep only the system prompt.

    Reuses the same on-disk shape as compaction (a `compacted` entry) since
    both mean "replace the messages going forward with this list".
    """
    fresh = [{"role": "system", "content": system_prompt}]
    compacted(fresh)
    return fresh


def load(session_id):
    """Replay the log: messages accumulate, rewinds cut them back."""
    messages = []
    for line in path_for(session_id).read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # A half-written last line, usually from a kill mid-save. Skipping
            # it costs one message; raising would break /sessions for every
            # chat in the project, because listing them all calls load().
            continue
        if "rewind_to" in entry:
            del messages[entry["rewind_to"]:]
        elif "compacted" in entry:
            messages = list(entry["compacted"])
        else:
            messages.append(entry)
    return messages


def open_session(session_id):
    """Switch to a past chat and become it."""
    global CURRENT, WRITTEN
    CURRENT = session_id
    messages = load(session_id)
    WRITTEN = len(messages)
    return messages


def title(messages):
    for message in messages:
        if message["role"] == "user":
            return " ".join(str(message.get("content") or "").split())[:60]
    return "(empty)"


def all_sessions():
    """Newest first."""
    session_dir = _session_dir()
    if not session_dir.exists():
        return []
    files = sorted(
        session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [{"id": p.stem, "title": title(load(p.stem))} for p in files]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `python -m pytest -v`
Expected: PASS (all tests from Tasks 1-5)

- [ ] **Step 6: Commit**

```bash
git add neuralcode/session.py tests/test_session.py
git commit -m "Make session.py compute session dir lazily, add reset_context()"
```

---

## Task 6: Per-session working directory (`workdir.py`)

**Files:**
- Create: `neuralcode/workdir.py`
- Test: `tests/test_workdir.py`

**Interfaces:**
- Consumes: `session.all_sessions()` from Task 5 (called only after `chdir`, i.e. from inside `choose_or_create`'s caller — see Task 7) — actually not consumed directly by `workdir.py` itself, see Step 3 design note below.
- Produces: `workdir.list_existing(root: Path) -> list[Path]`, `workdir.choose_or_create(root: Path, ui) -> Path`. Task 7 (`agent.py`) consumes both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workdir.py`:

```python
from neuralcode import workdir


class FakeUI:
    def __init__(self, choice):
        self.choice = choice
        self.picked_rows = None

    def pick(self, title, rows):
        self.picked_rows = rows
        return self.choice


def test_list_existing_returns_empty_when_sessions_dir_missing(tmp_path):
    assert workdir.list_existing(tmp_path) == []


def test_list_existing_lists_subdirectories_newest_first(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    older = sessions / "20260101-000000"
    newer = sessions / "20260101-000001"
    older.mkdir()
    newer.mkdir()
    import os
    import time

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    result = workdir.list_existing(tmp_path)

    assert result == [newer, older]


def test_choose_or_create_new_creates_sessions_dir_and_timestamped_subdir(tmp_path):
    ui = FakeUI(choice=0)  # index 0 is always "new session directory"

    chosen = workdir.choose_or_create(tmp_path, ui)

    assert chosen.parent == tmp_path / "sessions"
    assert chosen.exists()
    assert ui.picked_rows[0].startswith("new session directory")


def test_choose_or_create_existing_returns_the_picked_directory(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    existing = sessions / "20260101-000000"
    existing.mkdir()

    ui = FakeUI(choice=1)  # index 0 = "new", index 1 = the one existing dir

    chosen = workdir.choose_or_create(tmp_path, ui)

    assert chosen == existing


def test_choose_or_create_none_picked_falls_back_to_new(tmp_path):
    ui = FakeUI(choice=None)

    chosen = workdir.choose_or_create(tmp_path, ui)

    assert chosen.parent == tmp_path / "sessions"
    assert chosen.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workdir.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuralcode.workdir'`

- [ ] **Step 3: Implement `neuralcode/workdir.py`**

```python
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
```

Design note on the "Interfaces / Consumes" line above: `workdir.py` deliberately does **not** import `session.py` to show titles next to each directory (e.g. "20260101-000000 — fix login bug"), even though the spec mentions reusing `session.title`. Doing that from `workdir.py` would require temporarily `chdir`-ing into each candidate directory just to list its sessions, before the real chdir happens — extra complexity for a cosmetic label. Task 7 (`agent.py`) is the place `workdir.py`'s result is used to `chdir`, so richer per-directory session titles are deferred; showing just the directory name (a timestamp) is sufficient for choosing which one to reuse, and is what the tests above pin.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workdir.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add neuralcode/workdir.py tests/test_workdir.py
git commit -m "Add workdir.py to pick or create a per-session working directory"
```

---

## Task 7: UI additions for budget and context reset

**Files:**
- Modify: `neuralcode/ui.py`

**Interfaces:**
- Consumes: `budget.remaining()`, `budget.total()` from Task 3 (called by `agent.py`, passed as plain values into these new `ui` methods — `ui.py` stays free of imports from `budget.py`/`config.py`, matching its existing "knows nothing about LLMs, providers or tools" docstring).
- Produces: `ui.usage(stats, budget_remaining=None, budget_total=None)` (extends the existing method — new params are optional so Tasks that don't pass them keep working), `ui.budget_warning(total, limit) -> bool` (prints the warning and returns the user's y/n as a bool, mirroring `ui.approve`), `ui.context_reset() -> None`.

No test file for this task: `ui.py` is a thin `rich`-printing layer with no existing test coverage in the repo (it wasn't tested before this plan either), and its correctness is verified by the manual end-to-end run in Task 9. This matches the "Testes" section of the spec, which scopes automated tests to pure logic (`profiles`, `budget`, `workdir`), not the printing layer.

- [ ] **Step 1: Modify `usage()` in `neuralcode/ui.py`**

Find:

```python
    def usage(self, stats):
        for key, value in stats.items():
            self._totals[key] = self._totals.get(key, 0) + (value or 0)

        parts = " · ".join(
            f"{value:,} {key.replace('_tokens', '')}"
            for key, value in stats.items()
            if value
        )
        self.console.print(Padding(Text(parts, style=MUTED), (1, 0, 0, 2)))
```

Replace with:

```python
    def usage(self, stats, budget_remaining=None, budget_total=None):
        for key, value in stats.items():
            self._totals[key] = self._totals.get(key, 0) + (value or 0)

        parts = " · ".join(
            f"{value:,} {key.replace('_tokens', '')}"
            for key, value in stats.items()
            if value
        )
        self.console.print(Padding(Text(parts, style=MUTED), (1, 0, 0, 2)))

        if budget_total is not None:
            used = budget_total - (budget_remaining or 0)
            pct = int(100 * used / budget_total) if budget_total else 0
            budget_line = f"budget: {used:,} / {budget_total:,} tokens ({pct}%)"
            self.console.print(Padding(Text(budget_line, style=MUTED), (0, 0, 0, 2)))
```

- [ ] **Step 2: Add `budget_warning()` and `context_reset()` to `neuralcode/ui.py`**

Directly below the `approve()` method (which follows the same "print then read y/n" shape), add:

```python
    def budget_warning(self, total, limit):
        self.console.print(
            Padding(
                Text(
                    f"token budget reached: {total:,} used, limit is {limit:,}",
                    style=f"bold {TOOL}",
                ),
                (1, 0, 0, 2),
            )
        )
        try:
            answer = prompt.read("  clear context and continue? (y/N)> ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.lower().startswith("y")
```

Directly below `note()`, add:

```python
    def context_reset(self):
        self.note("context cleared - starting a fresh conversation in this working directory")
```

- [ ] **Step 3: Verify the module still imports cleanly**

Run: `python -c "from neuralcode.ui import ui; print('ok')"`
Expected: prints `ok` with no traceback.

- [ ] **Step 4: Commit**

```bash
git add neuralcode/ui.py
git commit -m "Add budget display, budget_warning() and context_reset() to ui.py"
```

---

## Task 8: Wire everything into `agent.py`

**Files:**
- Modify: `neuralcode/agent.py`

**Interfaces:**
- Consumes: `workdir.choose_or_create` (Task 6), `budget.track/should_warn/mark_warned/remaining/total/reset/limit` (Task 3, `limit()` added in this task's Step 1), `session.reset_context` (Task 5), `ui.budget_warning/context_reset/usage` (Task 7).

This task has no isolated unit test — it's the integration point, verified by Task 9's manual end-to-end run plus a lightweight smoke test here that the module still imports and `main` is callable without executing the interactive loop.

- [ ] **Step 1: Add `budget.limit()`**

`agent.py` needs the configured budget ceiling itself (not just tokens remaining) to pass to `ui.budget_warning()` and `ui.usage()`. Add this to `neuralcode/budget.py`, directly below `remaining()`:

```python
def limit():
    """The configured TOKEN_BUDGET itself (not remaining), or None."""
    return config.TOKEN_BUDGET
```

- [ ] **Step 2: Replace `neuralcode/agent.py`**

Replace the full contents of `neuralcode/agent.py`:

```python
import argparse
import os
from pathlib import Path

from . import budget
from . import commands
from . import compact
from . import history
from . import session
from . import workdir
from .context import reminder
from .llm import SYSTEM_PROMPT, call_llm
from . import sandbox
from .todos import active_form
from .tools import execute
from .ui import ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="continue the last session")
    parser.add_argument("--debug", action="store_true", help="show the raw model response")
    cli = parser.parse_args()

    repo_root = Path.cwd()
    session_dir = workdir.choose_or_create(repo_root, ui)
    os.chdir(session_dir)

    ui.banner(sandbox.name())

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if cli.resume:
        saved = session.all_sessions()
        if saved:
            messages = session.open_session(saved[0]["id"])
            history.strip(messages)
            ui.resumed(messages)
            ui.replay(messages)

    while True:
        user_input = ui.ask()
        if not user_input:
            break

        if user_input.startswith("/"):
            messages = commands.handle(user_input, messages)
            session.save(messages)
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            if budget.should_warn():
                confirmed = ui.budget_warning(budget.total(), budget.limit())
                if not confirmed:
                    break
                messages = session.reset_context(SYSTEM_PROMPT)
                budget.reset()
                ui.context_reset()

            injection = reminder()
            ui.injection(injection["content"])

            if history.fit(messages):
                ui.note("dropped old tool output to make this request fit")

            with ui.working(active_form()):
                message, usage = call_llm(messages + [injection])

            messages.append(message.model_dump(exclude_none=True))
            session.save(messages)
            budget.track(usage)
            ui.usage(usage, budget_remaining=budget.remaining(), budget_total=budget.limit())

            if cli.debug:
                ui.debug(message.model_dump(exclude_none=True))

            if message.content:
                ui.agent(message.content)

            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                args, result = execute(tool_call)
                ui.tool(tool_call.function.name, args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                session.save(messages)

        history.sweep()   # the turn is over: bin its temp files
        history.strip(messages)  # ...and shrink the tool output it produced

        if compact.needed(usage):
            messages = commands.compact(messages)

    ui.summary()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write a smoke test for `budget.limit()`**

Create/extend `tests/test_budget.py` by adding (append to the file from Task 3):

```python
def test_limit_returns_configured_budget(with_budget):
    assert budget.limit() == 1000


def test_limit_returns_none_when_no_budget_configured(without_budget):
    assert budget.limit() is None
```

- [ ] **Step 4: Run the budget tests to verify `limit()` works**

Run: `python -m pytest tests/test_budget.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Verify `agent.py` imports cleanly**

Run: `BASE_URL=http://localhost:11434/v1 API_KEY=ollama python -c "from neuralcode.agent import main; print('ok')"`
Expected: prints `ok` with no traceback (this only imports the module; it does not run the interactive loop since `main()` isn't called here).

- [ ] **Step 6: Run the full automated test suite**

Run: `python -m pytest -v`
Expected: PASS (all tests from Tasks 1-8)

- [ ] **Step 7: Commit**

```bash
git add neuralcode/agent.py neuralcode/budget.py tests/test_budget.py
git commit -m "Wire session workdir selection and token budget reset into agent.py"
```

---

## Task 9: Manual end-to-end verification against Ollama

**Files:** none (verification only, no code changes expected unless a bug is found — if one is, fix it in the relevant file from Tasks 1-8 and commit separately with a description of the bug).

- [ ] **Step 1: Start the local Ollama container (coordinate with the user first — see chat)**

This step is intentionally left for interactive execution, not scripted here: the user asked to check the host's available memory before unpausing their existing Ollama container, and to do that check together. Do not run `docker start`/`docker unpause` as part of this plan without that check happening first in the conversation.

- [ ] **Step 2: Pull a small profile's model into Ollama**

Run (adjust container exec prefix to match the user's actual container name/setup, confirmed in chat):
```bash
docker exec <ollama-container> ollama pull qwen2.5-coder:1.5b
```

- [ ] **Step 3: Run the agent with the small-2b profile**

Run:
```bash
BASE_URL=http://localhost:11434/v1 API_KEY=ollama PROFILE=small-2b python -m neuralcode.agent
```

Expected: the workdir prompt appears first ("session working directory", offering "new session directory"), picking it creates `sessions/<timestamp>/` and the banner then shows. Ask the agent to do a small task (e.g. "create a file hello.py that prints hello world") and confirm: the file appears inside `sessions/<timestamp>/`, not the repo root; the usage line after each response shows a `budget:` line; tool calls work end-to-end.

- [ ] **Step 4: Force a budget warning**

Set an artificially low budget to trigger the flow without a long session:
```bash
BASE_URL=http://localhost:11434/v1 API_KEY=ollama PROFILE=small-2b TOKEN_BUDGET=200 python -m neuralcode.agent
```

Expected: after the first response (which alone likely exceeds 200 tokens), the next turn shows the `budget_warning` prompt. Answering `y` should print the "context cleared" note and continue in the same directory; answering `n` should abort that turn without calling the LLM, and the next `> ` prompt should let the same or a new message be typed.

- [ ] **Step 5: Verify reusing an existing session directory**

Run the agent again, pick the existing `sessions/<timestamp>/` directory from the list this time instead of "new session directory", and confirm it starts a fresh conversation but the `hello.py` file from Step 3 is still present in that directory (`ls sessions/<timestamp>/`).

- [ ] **Step 6: Report findings back in chat**

Summarize what worked, what didn't (if anything), and whether tool-calling quality with `qwen2.5-coder:1.5b` seemed usable for real tasks — this is a judgment call for the user, not something to auto-fix.

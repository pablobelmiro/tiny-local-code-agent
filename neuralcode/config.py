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

# models.yaml lives next to wherever neuralcode was launched from. Captured
# here, at import time - before agent.py's chdir into the session directory -
# so reload() still finds it later even though the process has since moved.
_MODELS_ROOT = Path.cwd()

COMPACT_AT = 0.85  # compact once the prompt crosses this much of the window
COMPACT_TO = 0.35  # and cut back to this much, so it does not retrigger soon


def active_profile_name():
    return _PROFILE_NAME


def _resolve():
    """Compute settings from the current environment and models.yaml.

    Raises ValueError on an unknown PROFILE, without touching any module
    state - the caller decides what an invalid reload should do.
    """
    base_url = os.environ["BASE_URL"]
    api_key = os.environ["API_KEY"]

    profile_name = os.environ.get("PROFILE")
    all_profiles = profiles.load_profiles(_MODELS_ROOT / "models.yaml")
    try:
        active_profile = profiles.resolve_profile(all_profiles, profile_name) or {}
    except KeyError:
        raise ValueError(
            f"unknown PROFILE {profile_name!r}; available: {sorted(all_profiles)}"
        )

    model = os.environ.get("MODEL") or active_profile.get(
        "model", "deepseek/deepseek-v4-flash"
    )
    context_window = int(
        os.environ.get("CONTEXT_WINDOW") or active_profile.get("context_window", 128_000)
    )
    env_budget = os.environ.get("TOKEN_BUDGET")
    token_budget = int(env_budget) if env_budget else active_profile.get("token_budget")

    return {
        "BASE_URL": base_url,
        "API_KEY": api_key,
        "_PROFILE_NAME": profile_name,
        "MODEL": model,
        "CONTEXT_WINDOW": context_window,
        "TOKEN_BUDGET": token_budget,
    }


def _apply(settings):
    globals().update(settings)


def reload():
    """Re-read env vars and models.yaml, picking up changes without a restart.

    Validates everything before changing any setting, so a bad PROFILE typo
    raises ValueError and leaves the running session on its old settings
    rather than crashing it. Returns the new settings dict.
    """
    settings = _resolve()
    _apply(settings)
    return settings


try:
    _apply(_resolve())
except ValueError as failure:
    # Startup is the one place an invalid PROFILE should stop the process
    # outright, with a message instead of a raw traceback.
    raise SystemExit(str(failure))

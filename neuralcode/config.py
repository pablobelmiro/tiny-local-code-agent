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
try:
    _ACTIVE_PROFILE = profiles.resolve_profile(_PROFILES, _PROFILE_NAME) or {}
except KeyError:
    raise SystemExit(
        f"unknown PROFILE {_PROFILE_NAME!r}; available: {sorted(_PROFILES)}"
    )


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

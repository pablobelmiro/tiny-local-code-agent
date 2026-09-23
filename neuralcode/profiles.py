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

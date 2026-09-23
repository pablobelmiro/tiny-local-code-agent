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

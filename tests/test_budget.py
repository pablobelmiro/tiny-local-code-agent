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


def test_limit_returns_configured_budget(with_budget):
    assert budget.limit() == 1000


def test_limit_returns_none_when_no_budget_configured(without_budget):
    assert budget.limit() is None

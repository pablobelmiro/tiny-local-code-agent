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


def test_config_unknown_profile_raises_clear_system_exit(clean_env, monkeypatch):
    (clean_env / "models.yaml").write_text(
        "profiles:\n"
        "  small-2b:\n"
        "    model: qwen2.5-coder:1.5b\n"
    )
    monkeypatch.setenv("PROFILE", "does-not-exist")

    with pytest.raises(SystemExit, match="does-not-exist"):
        reload_config()


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


def test_reload_picks_up_env_changes_without_reimporting(clean_env, monkeypatch):
    config = reload_config()
    assert config.MODEL == "deepseek/deepseek-v4-flash"

    monkeypatch.setenv("MODEL", "new-model")
    settings = config.reload()

    assert config.MODEL == "new-model"
    assert settings["MODEL"] == "new-model"


def test_reload_unknown_profile_raises_value_error_and_keeps_old_settings(clean_env, monkeypatch):
    config = reload_config()
    assert config.MODEL == "deepseek/deepseek-v4-flash"

    monkeypatch.setenv("PROFILE", "does-not-exist")

    with pytest.raises(ValueError, match="does-not-exist"):
        config.reload()

    assert config.MODEL == "deepseek/deepseek-v4-flash"

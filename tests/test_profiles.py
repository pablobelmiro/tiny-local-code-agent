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

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

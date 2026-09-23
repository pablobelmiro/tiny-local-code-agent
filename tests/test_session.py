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

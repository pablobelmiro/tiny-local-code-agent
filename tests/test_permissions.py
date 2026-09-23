from neuralcode import permissions


def test_inside_project_reflects_current_cwd_not_import_time_cwd(tmp_path, monkeypatch):
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    monkeypatch.chdir(project_a)
    assert permissions.inside_project(project_a / "file.txt") is True
    assert permissions.inside_project(project_b / "file.txt") is False

    monkeypatch.chdir(project_b)
    assert permissions.inside_project(project_b / "file.txt") is True
    assert permissions.inside_project(project_a / "file.txt") is False


def test_check_write_file_outside_current_cwd_asks(tmp_path, monkeypatch):
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    monkeypatch.chdir(project_a)

    action, reason = permissions.check("write_file", {"path": str(project_b / "f.txt")})

    assert action == "ask"


def test_check_write_file_inside_current_cwd_allows(tmp_path, monkeypatch):
    project_a = tmp_path / "project-a"
    project_a.mkdir()
    monkeypatch.chdir(project_a)

    action, reason = permissions.check("write_file", {"path": str(project_a / "f.txt")})

    assert action == "allow"

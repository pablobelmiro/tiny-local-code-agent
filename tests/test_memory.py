from neuralcode import memory


def test_ensure_scaffold_creates_memory_dir_and_template_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    memory.ensure_scaffold()

    memory_dir = tmp_path / "memory"
    assert memory_dir.is_dir()
    assert (memory_dir / "INDEX.md").exists()
    assert (memory_dir / "profile.md").exists()
    assert (memory_dir / "decisions.md").exists()
    assert (memory_dir / "gotchas.md").exists()


def test_ensure_scaffold_does_not_overwrite_existing_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "INDEX.md").write_text("my own notes")

    memory.ensure_scaffold()

    assert (memory_dir / "INDEX.md").read_text() == "my own notes"


def test_read_index_returns_index_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory.ensure_scaffold()

    index = memory.read_index()

    assert "profile.md" in index
    assert "decisions.md" in index
    assert "gotchas.md" in index


def test_read_index_returns_empty_string_when_no_memory_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert memory.read_index() == ""

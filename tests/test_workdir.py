from neuralcode import workdir


class FakeUI:
    def __init__(self, choice):
        self.choice = choice
        self.picked_rows = None

    def pick(self, title, rows):
        self.picked_rows = rows
        return self.choice


def test_list_existing_returns_empty_when_sessions_dir_missing(tmp_path):
    assert workdir.list_existing(tmp_path) == []


def test_list_existing_lists_subdirectories_newest_first(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    older = sessions / "20260101-000000"
    newer = sessions / "20260101-000001"
    older.mkdir()
    newer.mkdir()
    import os
    import time

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    result = workdir.list_existing(tmp_path)

    assert result == [newer, older]


def test_choose_or_create_new_creates_sessions_dir_and_timestamped_subdir(tmp_path):
    ui = FakeUI(choice=0)  # index 0 is always "new session directory"

    chosen = workdir.choose_or_create(tmp_path, ui)

    assert chosen.parent == tmp_path / "sessions"
    assert chosen.exists()
    assert ui.picked_rows[0].startswith("new session directory")


def test_choose_or_create_existing_returns_the_picked_directory(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    existing = sessions / "20260101-000000"
    existing.mkdir()

    ui = FakeUI(choice=1)  # index 0 = "new", index 1 = the one existing dir

    chosen = workdir.choose_or_create(tmp_path, ui)

    assert chosen == existing


def test_choose_or_create_none_picked_falls_back_to_new(tmp_path):
    ui = FakeUI(choice=None)

    chosen = workdir.choose_or_create(tmp_path, ui)

    assert chosen.parent == tmp_path / "sessions"
    assert chosen.exists()

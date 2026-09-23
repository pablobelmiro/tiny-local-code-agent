from neuralcode import commands


def test_reload_applies_new_settings_and_reloads_client(monkeypatch):
    notes = []
    monkeypatch.setattr(commands.ui, "note", lambda text: notes.append(text))
    monkeypatch.setattr(
        commands.config,
        "reload",
        lambda: {"MODEL": "new-model", "CONTEXT_WINDOW": 32000, "BASE_URL": "http://x/v1"},
    )
    reload_client_calls = []
    monkeypatch.setattr(commands.llm, "reload_client", lambda: reload_client_calls.append(True))

    messages = [{"role": "system", "content": "sys"}]
    result = commands.reload(messages)

    assert result == messages  # conversation untouched
    assert reload_client_calls == [True]
    assert any("new-model" in note for note in notes)


def test_reload_failure_keeps_messages_and_reports_note(monkeypatch):
    notes = []
    monkeypatch.setattr(commands.ui, "note", lambda text: notes.append(text))

    def fail():
        raise ValueError("unknown PROFILE 'nope'; available: ['small-2b']")

    monkeypatch.setattr(commands.config, "reload", fail)
    reload_client_calls = []
    monkeypatch.setattr(commands.llm, "reload_client", lambda: reload_client_calls.append(True))

    messages = [{"role": "system", "content": "sys"}]
    result = commands.reload(messages)

    assert result == messages
    assert reload_client_calls == []  # never rebuilt the client on a failed reload
    assert any("nope" in note for note in notes)


def test_handle_dispatches_slash_reload(monkeypatch):
    called = []
    monkeypatch.setattr(commands, "reload", lambda messages: called.append(messages) or messages)

    messages = [{"role": "system", "content": "sys"}]
    result = commands.handle("/reload", messages)

    assert result == messages
    assert called == [messages]


def test_reload_listed_in_commands_help():
    assert "/reload" in commands.COMMANDS

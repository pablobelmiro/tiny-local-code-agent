from neuralcode import llm


def test_system_prompt_reflects_current_cwd_not_import_time_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    prompt = llm.system_prompt()

    assert str(tmp_path) in prompt


def test_reload_client_rebuilds_client_from_current_config(monkeypatch):
    monkeypatch.setattr(llm.config, "BASE_URL", "http://example-changed:11434/v1")
    monkeypatch.setattr(llm.config, "API_KEY", "changed-key")

    old_client = llm.client
    llm.reload_client()

    assert llm.client is not old_client
    assert str(llm.client.base_url) == "http://example-changed:11434/v1/"

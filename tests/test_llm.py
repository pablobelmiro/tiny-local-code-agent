from neuralcode import llm


def test_system_prompt_reflects_current_cwd_not_import_time_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    prompt = llm.system_prompt()

    assert str(tmp_path) in prompt


def test_system_prompt_omits_skills_section_when_no_skills(monkeypatch):
    monkeypatch.setattr(llm, "skills_prompt", lambda: "")

    prompt = llm.system_prompt()

    assert "skills available" not in prompt
    assert "read_skill" not in prompt


def test_system_prompt_includes_skills_section_when_skills_exist(monkeypatch):
    monkeypatch.setattr(llm, "skills_prompt", lambda: "- foo: does foo things")

    prompt = llm.system_prompt()

    assert "skills available" in prompt
    assert "- foo: does foo things" in prompt


def test_system_prompt_omits_memory_section_when_no_index(monkeypatch):
    monkeypatch.setattr(llm, "read_memory_index", lambda: "")

    prompt = llm.system_prompt()

    assert "memory/INDEX.md" not in prompt


def test_system_prompt_includes_memory_section_when_index_exists(monkeypatch):
    monkeypatch.setattr(llm, "read_memory_index", lambda: "- profile.md - stack info")

    prompt = llm.system_prompt()

    assert "memory/INDEX.md" in prompt
    assert "- profile.md - stack info" in prompt


def test_reload_client_rebuilds_client_from_current_config(monkeypatch):
    monkeypatch.setattr(llm.config, "BASE_URL", "http://example-changed:11434/v1")
    monkeypatch.setattr(llm.config, "API_KEY", "changed-key")

    old_client = llm.client
    llm.reload_client()

    assert llm.client is not old_client
    assert str(llm.client.base_url) == "http://example-changed:11434/v1/"

from neuralcode import llm


def test_system_prompt_reflects_current_cwd_not_import_time_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    prompt = llm.system_prompt()

    assert str(tmp_path) in prompt

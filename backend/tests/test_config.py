import importlib

import dotenv
import pytest

from app import config


@pytest.fixture
def reload_with_env_file(monkeypatch, tmp_path):
    """Reload app.config as if backend/.env held `content`; restore everything afterwards."""
    real_values = dotenv.dotenv_values

    def run(content):
        env_file = tmp_path / ".env"
        env_file.write_text(content)
        monkeypatch.setattr(dotenv, "dotenv_values", lambda *_a, **k: real_values(env_file, **k))
        importlib.reload(config)

    yield run
    monkeypatch.undo()  # restores os.environ and dotenv before the final reload
    importlib.reload(config)


def test_env_file_is_loaded(monkeypatch, reload_with_env_file):
    monkeypatch.setenv("LLM_PROVIDER", "x")
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.setenv("LLM_MODEL", "x")
    monkeypatch.delenv("LLM_MODEL")
    reload_with_env_file("LLM_PROVIDER=ollama\nLLM_MODEL=llama3.1\n")
    assert config.LLM_PROVIDER == "ollama"
    assert config.LLM_MODEL == "llama3.1"


def test_env_file_wins_over_environment(monkeypatch, reload_with_env_file):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    reload_with_env_file("LLM_PROVIDER=ollama\n")
    assert config.LLM_PROVIDER == "ollama"


def test_environment_used_when_env_file_lacks_var(monkeypatch, reload_with_env_file):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    reload_with_env_file("LLM_MODEL=m\n")
    assert config.LLM_PROVIDER == "gemini"


def test_blank_env_file_value_does_not_wipe_environment(monkeypatch, reload_with_env_file):
    monkeypatch.setenv("LLM_API_KEY", "from-environment")
    reload_with_env_file("LLM_API_KEY=\n")
    assert config.LLM_API_KEY == "from-environment"

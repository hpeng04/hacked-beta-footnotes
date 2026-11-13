import importlib

import dotenv
import pytest

from app import config


@pytest.fixture
def reload_with_env_file(monkeypatch, tmp_path):
    """Reload app.config as if backend/.env held `content`; restore everything afterwards."""
    real_load = dotenv.load_dotenv

    def run(content):
        env_file = tmp_path / ".env"
        env_file.write_text(content)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_k: real_load(env_file))
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


def test_real_env_wins_over_env_file(monkeypatch, reload_with_env_file):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    reload_with_env_file("LLM_PROVIDER=ollama\n")
    assert config.LLM_PROVIDER == "gemini"

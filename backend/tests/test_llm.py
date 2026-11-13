import httpx
import pytest

from app import config
from app.schemas import Chunk
from app.search import llm


# --- complete(): primary provider ---

def test_returns_text_from_primary(fake_llm):
    fake_llm.reply("groq", "hello")
    assert llm.complete("hi") == "hello"
    assert len(fake_llm.calls) == 1


def test_json_mode_parses_and_requests_json(fake_llm):
    fake_llm.reply("groq", '{"sections": []}')
    assert llm.complete("hi", json=True) == {"sections": []}
    assert fake_llm.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_text_mode_omits_response_format(fake_llm):
    fake_llm.reply("groq", "x")
    llm.complete("hi")
    assert "response_format" not in fake_llm.calls[0]["json"]


def test_request_shape(fake_llm):
    fake_llm.reply("groq", "x")
    llm.complete("the prompt")
    call = fake_llm.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer primary-key"}
    assert call["json"]["model"] == llm.PROVIDERS["groq"][1]
    assert call["json"]["messages"] == [{"role": "user", "content": "the prompt"}]


def test_model_override(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "custom-model")
    fake_llm.reply("groq", "x")
    llm.complete("hi")
    assert fake_llm.calls[0]["json"]["model"] == "custom-model"


def test_no_auth_header_without_key(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    fake_llm.reply("localhost:11434", "x")
    llm.complete("hi")
    assert fake_llm.calls[0]["headers"] == {}


# --- complete(): fallback ---

def test_falls_back_on_http_error(fake_llm):
    fake_llm.reply("groq", httpx.ConnectError("rate limited"))
    fake_llm.reply("generativelanguage", "from gemini")
    assert llm.complete("hi") == "from gemini"
    assert fake_llm.calls[1]["headers"] == {"Authorization": "Bearer backup-key"}


def test_falls_back_on_bad_json(fake_llm):
    fake_llm.reply("groq", "not json")
    fake_llm.reply("generativelanguage", '{"ok": 1}')
    assert llm.complete("hi", json=True) == {"ok": 1}


def test_both_fail_raises_with_both_errors(fake_llm):
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("also down"))
    with pytest.raises(llm.LLMError, match="groq.*gemini"):
        llm.complete("hi")


def test_no_backup_configured(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKUP_PROVIDER", "")
    fake_llm.reply("groq", httpx.ConnectError("down"))
    with pytest.raises(llm.LLMError):
        llm.complete("hi")
    assert len(fake_llm.calls) == 1


# --- answer() ---

SOURCES = [
    Chunk(lecture_id="demo", start=130, end=168, text="Deadlock needs four conditions."),
    Chunk(lecture_id="demo", start=95, end=130, text="Locks create deadlock."),
]


@pytest.mark.parametrize("secs,expected", [(0, "0:00"), (95, "1:35"), (59.9, "0:59"), (3725, "1:02:05")])
def test_fmt_time(secs, expected):
    assert llm.fmt_time(secs) == expected


def test_answer_without_sources_skips_llm(fake_llm):
    assert "Nothing" in llm.answer("q", [])
    assert fake_llm.calls == []


def test_answer_returns_llm_text(fake_llm):
    fake_llm.reply("groq", "  Deadlock is when threads wait forever [1:35].\n")
    assert llm.answer("what is deadlock", SOURCES) == "Deadlock is when threads wait forever [1:35]."


def test_answer_prompt_has_question_and_timestamped_sources_in_order(fake_llm):
    fake_llm.reply("groq", "x")
    llm.answer("what is deadlock", SOURCES)
    prompt = fake_llm.calls[0]["json"]["messages"][0]["content"]
    assert "Question: what is deadlock" in prompt
    assert prompt.index("[1:35] Locks create deadlock.") < prompt.index("[2:10] Deadlock needs four conditions.")


def test_answer_falls_back_to_top_source_when_llm_down(fake_llm):
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    assert llm.answer("q", SOURCES) == "[2:10] Deadlock needs four conditions."

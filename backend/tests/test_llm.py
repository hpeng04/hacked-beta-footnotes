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
    assert prompt.index("(Excerpt at [1:35])\nLocks create deadlock.") < prompt.index("(Excerpt at [2:10])\nDeadlock needs four conditions.")


def test_answer_falls_back_to_top_source_when_llm_down(fake_llm):
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    assert llm.answer("q", SOURCES) == "[2:10] Deadlock needs four conditions."


# --- live-run findings: citation format, retries ---

@pytest.mark.parametrize("raw,clean", [
    ("Water is split 【1:33】.", "Water is split [1:33]."),
    ("See (4:03) and （1:02:05）.", "See [4:03] and [1:02:05]."),
    ("Already fine [1:33].", "Already fine [1:33]."),
    ("Not a time (see above).", "Not a time (see above)."),
])
def test_answer_normalizes_citations(fake_llm, raw, clean):
    fake_llm.reply("groq", raw)
    assert llm.answer("q", SOURCES) == clean


def test_answer_prompt_says_cite_nothing_when_unanswerable(fake_llm):
    fake_llm.reply("groq", "x")
    llm.answer("q", SOURCES)
    assert "cite nothing" in fake_llm.calls[0]["json"]["messages"][0]["content"]


def status_error(code, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.HTTPStatusError(str(code), request=req, response=httpx.Response(code, headers=headers, request=req))


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    return waits


@pytest.mark.parametrize("code", [429, 503])
def test_retries_same_provider_once_on_rate_limit(fake_llm, sleeps, code):
    fake_llm.reply("groq", status_error(code, {"retry-after": "3"}), "second try")
    assert llm.complete("hi") == "second try"
    assert sleeps == [3.0]
    assert all("groq" in c["url"] for c in fake_llm.calls)


def test_retry_wait_is_capped(fake_llm, sleeps):
    fake_llm.reply("groq", status_error(429, {"retry-after": "600"}), "ok")
    llm.complete("hi")
    assert sleeps == [llm.MAX_RETRY_WAIT]


def test_falls_back_after_second_rate_limit(fake_llm, sleeps):
    fake_llm.reply("groq", status_error(429), status_error(429))
    fake_llm.reply("generativelanguage", "from gemini")
    assert llm.complete("hi") == "from gemini"
    assert sleeps == [2.0]


def test_no_retry_on_other_errors(fake_llm, sleeps):
    fake_llm.reply("groq", status_error(401))
    fake_llm.reply("generativelanguage", "from gemini")
    assert llm.complete("hi") == "from gemini"
    assert sleeps == []


def test_ollama_gets_longer_timeout(fake_llm, monkeypatch):
    seen = []
    real = fake_llm.__call__
    monkeypatch.setattr(httpx, "post", lambda url, json, headers, timeout: seen.append(timeout) or real(url, json, headers, timeout))
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    fake_llm.reply("localhost:11434", "x")
    fake_llm.reply("groq", "y")
    llm.complete("hi")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    llm.complete("hi")
    assert seen == [llm.TIMEOUTS["ollama"], llm.DEFAULT_TIMEOUT]


# --- citation grounding (small models, e.g. gemma on Ollama) ---

ROUND_ROBIN = [
    Chunk(lecture_id="os", start=213.7, end=243.7, text="Okay, round robin. Each process gets a small time slice, called a quantum."),
    Chunk(lecture_id="os", start=243.7, end=273.7, text="Choosing the quantum is the key design decision. Too large and round robin degenerates into FCFS."),
    Chunk(lecture_id="os", start=367.8, end=397.8, text="Homework three: implement gradient descent from scratch. Due next Tuesday."),
]


@pytest.mark.parametrize("raw,grounded", [
    # live failure 1: no citation at all -> best matching source added
    ("Due next Tuesday.", "Due next Tuesday [6:07]."),
    # live failure 2: copied excerpt-label style, cited the neighbouring chunk -> moved
    ("[3:33] Choosing the quantum is the key design decision.", "[4:03] Choosing the quantum is the key design decision."),
    # correct citations are left alone
    ("Too large and it degenerates into FCFS [4:03].", "Too large and it degenerates into FCFS [4:03]."),
    # a reasonable citation that is not the single best match is kept
    ("Round robin gives each process a quantum [3:33].", "Round robin gives each process a quantum [3:33]."),
    # off-topic refusal stays uncited
    ("The excerpts do not cover virtual memory paging.", "The excerpts do not cover virtual memory paging."),
    # once the reply cites something, uncited sentences are not given extra citations
    ("Due next Tuesday [6:07]. Implement gradient descent from scratch.", "Due next Tuesday [6:07]. Implement gradient descent from scratch."),
])
def test_answer_grounds_citations(fake_llm, raw, grounded):
    fake_llm.reply("groq", raw)
    assert llm.answer("q", ROUND_ROBIN) == grounded

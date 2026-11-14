import json

import httpx
import pytest

from app import config
from app.ingestion.timing import source_id
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


def reply_sentence(text, *sources):
    return json.dumps({"sentences": [{"text": text, "source_ids": [source_id(c) for c in sources]}]})


def test_answer_resolves_ids_to_stored_timestamps(fake_llm):
    fake_llm.reply("groq", reply_sentence("Threads wait forever.", SOURCES[1]))
    result = llm.answer_result("what is deadlock", SOURCES)
    assert result.answer == "Threads wait forever. [1:35]"
    assert result.sources == [SOURCES[1]]
    assert result.sentences[0].sources == [SOURCES[1]]


def test_answer_prompt_uses_stable_ids_without_generated_times(fake_llm):
    fake_llm.reply("groq", '{"sentences": []}')
    llm.answer("what is deadlock", SOURCES)
    call = fake_llm.calls[0]["json"]
    prompt = call["messages"][0]["content"]
    assert "Question: what is deadlock" in prompt
    assert prompt.index(source_id(SOURCES[1])) < prompt.index(source_id(SOURCES[0]))
    assert "[1:35]" not in prompt
    assert call["response_format"] == {"type": "json_object"}


def test_answer_failure_does_not_dump_transcript(fake_llm):
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    result = llm.answer_result("q", SOURCES)
    assert result.answer == llm.UNAVAILABLE
    assert result.sources == result.sentences == []


def test_unanswerable_cites_nothing(fake_llm):
    fake_llm.reply("groq", '{"sentences": []}')
    result = llm.answer_result("q", SOURCES)
    assert result.answer == llm.NOT_COVERED and not result.sources


@pytest.mark.parametrize("bad", [
    {"sentences": [{"text": "Claim.", "source_ids": ["made-up"]}]},
    {"sentences": [{"text": "Claim.", "source_ids": []}]},
    {"sentences": [{"text": "Claim.", "source_ids": "s_123"}]},
    {"sentences": [{"text": "Claim.", "source_ids": [{}]}]},
    {"sentences": [{"text": "Claim [1:35].", "source_ids": []}]},
    {"sentences": [None]}, {"sentences": "wrong"}, [], {},
])
def test_invalid_answers_never_invent_or_relocate_citations(fake_llm, bad):
    fake_llm.reply("groq", json.dumps(bad))
    result = llm.answer_result("q", SOURCES)
    assert result.answer == llm.UNAVAILABLE
    assert result.sources == result.sentences == []


def test_same_display_second_retains_independent_seek_targets(fake_llm):
    a = Chunk(lecture_id="lec", start=111.1, end=111.4, text="Yes.")
    b = Chunk(lecture_id="lec", start=111.5, end=111.9, text="No.")
    fake_llm.reply("groq", json.dumps({"sentences": [
        {"text": "First answer.", "source_ids": [source_id(a)]},
        {"text": "Second answer.", "source_ids": [source_id(b)]},
    ]}))
    result = llm.answer_result("q", [b, a])
    assert result.answer.count("[1:51]") == 2
    assert [s.sources[0].start for s in result.sentences] == [111.1, 111.5]


def test_multiple_supporting_excerpts_are_preserved_without_word_overlap_rewrite(fake_llm):
    fake_llm.reply("groq", reply_sentence("Yes, both conditions apply.", *SOURCES))
    result = llm.answer_result("q", SOURCES)
    assert result.sentences[0].sources == SOURCES


@pytest.mark.parametrize("wrapper", ["({id})", "[{id}]", "{id}", "（{id}）"])
def test_internal_ids_are_removed_from_prose_without_changing_citation(fake_llm, wrapper):
    source = SOURCES[1]
    leaked = wrapper.format(id=source_id(source))
    fake_llm.reply("groq", reply_sentence(f"Memory is assigned to variables {leaked}.", source))
    result = llm.answer_result("where is memory mentioned?", SOURCES)
    assert result.sentences[0].text == "Memory is assigned to variables."
    assert result.sentences[0].sources == [source]
    assert "s_" not in result.answer


def test_unknown_id_in_prose_is_rejected_instead_of_guessed(fake_llm):
    fake_llm.reply("groq", reply_sentence("Memory is assigned (s_madeup).", SOURCES[0]))
    assert llm.answer_result("q", SOURCES).answer == llm.UNAVAILABLE


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



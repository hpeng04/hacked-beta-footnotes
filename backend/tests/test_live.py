"""Live LLM tests against the sample lectures in data/mock/samples.

Skipped by default (see pytest.ini). Settings come from backend/.env, as in .env.example.
    python -m pytest -m live      primary provider (LLM_PROVIDER / LLM_MODEL / LLM_API_KEY),
                                  plus the configured primary -> backup chain
    python -m pytest -m ollama    local Ollama (`ollama serve`, model pulled). Uses
                                  LLM_BACKUP_MODEL when LLM_BACKUP_PROVIDER=ollama.

Each target runs with no backup, so a failure is that provider's own.
Outputs are model-dependent: a failing citation test means "check the prompt", not
necessarily a code bug. Ollama on CPU is slow (minutes per timeline).
"""
import json
import re

import httpx
import pytest

from app import config, mock
from app.search import llm, retrieval, timeline

QUESTIONS = json.loads((config.MOCK_DIR / "sample_questions.json").read_text())
LECTURES = sorted(QUESTIONS)
CITATION = re.compile(r"\[(?:\d+:)?\d+:\d\d\]")


def _ollama_model() -> str:
    if config.LLM_PROVIDER == "ollama" and config.LLM_MODEL:
        return config.LLM_MODEL
    if config.LLM_BACKUP_PROVIDER == "ollama" and config.LLM_BACKUP_MODEL:
        return config.LLM_BACKUP_MODEL
    return llm.PROVIDERS["ollama"][1]


def _ollama_ready(model: str) -> str | None:
    """None if usable, else the reason to skip."""
    try:
        tags = httpx.get("http://localhost:11434/api/tags", timeout=3).json()
    except Exception:
        return "Ollama not running on localhost:11434 (start it with `ollama serve`)"
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return f"model {model} not pulled (run `ollama pull {model}`)"
    return None


@pytest.fixture(params=[
    pytest.param("primary", marks=pytest.mark.live),
    pytest.param("ollama", marks=pytest.mark.ollama),
])
def target(request, monkeypatch):
    """Point app.config at one provider, with no backup."""
    if request.param == "primary":
        if config.LLM_PROVIDER != "ollama" and not config.LLM_API_KEY:
            pytest.skip("LLM_API_KEY not set in backend/.env")
        provider, model, key = config.LLM_PROVIDER, config.LLM_MODEL, config.LLM_API_KEY
    else:
        model = _ollama_model()
        if reason := _ollama_ready(model):
            pytest.skip(reason)
        provider, key = "ollama", ""
    monkeypatch.setattr(config, "LLM_PROVIDER", provider)
    monkeypatch.setattr(config, "LLM_MODEL", model)
    monkeypatch.setattr(config, "LLM_API_KEY", key)
    monkeypatch.setattr(config, "LLM_BACKUP_PROVIDER", "")
    return provider


def _sources(lecture_id: str, question: str, expect_start: float | None):
    """Retrieved chunks, plus the expected one, so these tests measure the LLM, not retrieval."""
    sources = retrieval.search(lecture_id, question)
    if expect_start is not None and all(c.start != expect_start for c in sources):
        sources.append(next(c for c in mock.load_chunks(lecture_id) if c.start == expect_start))
    return sources


def _check_sections(sections, chunks):
    assert 2 <= len(sections) <= timeline.MAX_SECTIONS
    assert sections[0].start == chunks[0].start and sections[-1].end == chunks[-1].end
    for a, b in zip(sections, sections[1:]):
        assert a.end == b.start and a.start < a.end
    assert all(0 < len(s.title) <= timeline.MAX_TITLE_CHARS for s in sections)


# --- raw client ---

def test_complete_text(target):
    assert llm.complete("Reply with the single word: pong").strip()


def test_complete_json(target):
    reply = llm.complete('Return JSON only: {"answer": 4} where answer is 2 + 2.', json=True)
    assert isinstance(reply, dict) and int(reply["answer"]) == 4


# --- Q&A ---

ANSWERABLE = [(lid, i["q"], i["expect_start"]) for lid in LECTURES for i in QUESTIONS[lid] if i["expect_start"] is not None]
OFF_TOPIC = [(lid, i["q"]) for lid in LECTURES for i in QUESTIONS[lid] if i["expect_start"] is None]


@pytest.mark.parametrize("lecture_id,question,expect_start", ANSWERABLE)
def test_answer_cites_expected_chunk(target, lecture_id, question, expect_start):
    sources = _sources(lecture_id, question, expect_start)
    reply = llm.answer(question, sources)
    assert reply != f"[{llm.fmt_time(sources[0].start)}] {sources[0].text}", "LLM failed, got fallback answer"
    assert f"[{llm.fmt_time(expect_start)}]" in reply, reply
    cited = set(CITATION.findall(reply))
    allowed = {f"[{llm.fmt_time(c.start)}]" for c in sources}
    assert cited <= allowed, f"cited a timestamp not in the sources: {reply}"


@pytest.mark.parametrize("lecture_id,question", OFF_TOPIC)
def test_off_topic_question_cites_nothing(target, lecture_id, question):
    reply = llm.answer(question, _sources(lecture_id, question, None) or mock.load_chunks(lecture_id)[:3])
    assert not CITATION.search(reply), reply


# --- timeline ---

@pytest.mark.parametrize("lecture_id", LECTURES)
def test_timeline_generates_valid_sections(target, lecture_id):
    chunks = retrieval.transcript(lecture_id)
    _check_sections(timeline.generate(chunks), chunks)


def test_timeline_build_caches_real_result(target):
    sections = timeline.build("os-scheduling")
    assert timeline.cache_path("os-scheduling").exists(), "fell back to fixed sections"
    assert timeline.build("os-scheduling") == sections


# --- configured chain (primary -> backup, as in .env) ---

@pytest.mark.live
def test_backup_takes_over_when_primary_fails(monkeypatch):
    if not config.LLM_BACKUP_PROVIDER:
        pytest.skip("LLM_BACKUP_PROVIDER not set in backend/.env")
    if config.LLM_BACKUP_PROVIDER == "ollama" and (reason := _ollama_ready(config.LLM_BACKUP_MODEL or llm.PROVIDERS["ollama"][1])):
        pytest.skip(reason)
    elif config.LLM_BACKUP_PROVIDER != "ollama" and not config.LLM_BACKUP_API_KEY:
        pytest.skip("LLM_BACKUP_API_KEY not set in backend/.env")
    monkeypatch.setattr(config, "LLM_API_KEY", "invalid-key")  # primary now fails with 401
    if config.LLM_PROVIDER == "ollama":
        monkeypatch.setattr(config, "LLM_MODEL", "no-such-model")
    assert llm.complete("Reply with the single word: pong").strip()

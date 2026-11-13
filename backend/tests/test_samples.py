"""Sample lectures for live LLM testing, and the live test script (LLM faked here)."""
import json

import pytest

from app import config, mock
from app.search import retrieval
from scripts import live_llm_test

QUESTIONS = json.loads((config.MOCK_DIR / "sample_questions.json").read_text())


@pytest.mark.parametrize("lecture_id", sorted(QUESTIONS))
def test_sample_is_valid_and_contiguous(lecture_id):
    chunks = mock.load_chunks(lecture_id)
    assert len(chunks) >= 10 and all(c.lecture_id == lecture_id and c.text for c in chunks)
    assert chunks[0].start == 0
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start and a.start < a.end


@pytest.mark.parametrize("lecture_id", sorted(QUESTIONS))
def test_expected_starts_exist(lecture_id):
    starts = {c.start for c in mock.load_chunks(lecture_id)}
    for item in QUESTIONS[lecture_id]:
        assert item["expect_start"] is None or item["expect_start"] in starts


def test_demo_still_loads():
    assert mock.load_chunks("demo")[-1].end == 270


def test_keyword_retrieval_finds_most_expected_chunks():
    # Keyword search is a stopgap; some questions (e.g. "light dependent reactions" vs
    # "stage one") need semantic search, so only most must hit.
    items = [(lid, i) for lid, items in QUESTIONS.items() for i in items if i["expect_start"] is not None]
    hits = sum(i["expect_start"] in {c.start for c in retrieval.search(lid, i["q"])} for lid, i in items)
    assert hits >= len(items) - 2


def test_live_script_passes_with_good_llm(fake_llm, capsys):
    lid = "os-scheduling"
    items = QUESTIONS[lid]
    fake_llm.reply("groq", json.dumps({"sections": [{"title": "Intro", "start": 0}]}))
    for item in items:
        cite = "" if item["expect_start"] is None else f" [{live_llm_test.llm.fmt_time(item['expect_start'])}]"
        fake_llm.reply("groq", "Answer." + cite if cite else "The lecture does not cover that.")
    assert live_llm_test.main([lid]) == 0
    assert f"{len(items) + 1}/{len(items) + 1} passed" in capsys.readouterr().out


def test_live_script_reports_llm_down(fake_llm, capsys):
    assert live_llm_test.main(["os-scheduling"]) == 1  # no replies queued: every call fails
    out = capsys.readouterr().out
    assert "timeline: FAILED" in out and "fallback answer" in out
    assert not (config.CACHE_DIR / "os-scheduling.timeline.json").exists()


def test_samples_dir_holds_only_transcripts():
    for path in (config.MOCK_DIR / "samples").glob("*.json"):
        chunks = mock.load_chunks(path.stem)
        assert chunks and all(c.lecture_id == path.stem for c in chunks), path.name

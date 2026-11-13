import json

import httpx
import pytest

from app import config
from app.ingestion import store
from app.schemas import Chunk, Section
from app.search import timeline

CHUNKS = [
    Chunk(lecture_id="lec", start=0, end=60, text="intro to races"),
    Chunk(lecture_id="lec", start=60, end=120, text="locks and mutexes"),
    Chunk(lecture_id="lec", start=120, end=200, text="deadlock conditions"),
]


@pytest.fixture
def chunks(monkeypatch):
    monkeypatch.setattr(store, "get_chunks", lambda lid: list(CHUNKS) if lid == "lec" else [], raising=False)


def reply(*sections):
    return json.dumps({"sections": [{"title": t, "start": s} for t, s in sections]})


def as_tuples(sections):
    return [(s.title, s.start, s.end) for s in sections]


# --- happy path and cache ---

def test_builds_contiguous_sections(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Races", 0), ("Locks", 60), ("Deadlock", 120)))
    assert as_tuples(timeline.build("lec")) == [("Races", 0, 60), ("Locks", 60, 120), ("Deadlock", 120, 200)]
    assert fake_llm.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_writes_cache_and_reuses_it(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Races", 0), ("Locks", 60)))
    first = timeline.build("lec")
    assert timeline.cache_path("lec").exists()
    assert timeline.build("lec") == first
    assert len(fake_llm.calls) == 1  # second call served from cache


def test_reads_existing_cache_without_llm(fake_llm, chunks):
    path = timeline.cache_path("lec")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"title": "Cached", "start": 0, "end": 200}]))
    assert timeline.build("lec") == [Section(title="Cached", start=0, end=200)]
    assert fake_llm.calls == []


def test_prompt_has_timestamped_lines(fake_llm, chunks):
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("lec")
    prompt = fake_llm.calls[0]["json"]["messages"][0]["content"]
    assert "0 intro to races\n60 locks and mutexes\n120 deadlock conditions" in prompt


def transcript_part(prompt):
    return prompt.split("Transcript:\n", 1)[1]


@pytest.mark.parametrize("n,seconds,words", [
    (720, 5, 30),   # one hour of 5-second chunks: many short lines
    (200, 30, 500), # few chunks, very long text
    (5000, 1, 3),   # extreme chunk count
])
def test_prompt_transcript_within_budget(fake_llm, monkeypatch, n, seconds, words):
    long = [Chunk(lecture_id="big", start=i * seconds, end=(i + 1) * seconds, text="word " * words) for i in range(n)]
    monkeypatch.setattr(store, "get_chunks", lambda lid: long, raising=False)
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("big")
    text = transcript_part(fake_llm.calls[0]["json"]["messages"][0]["content"])
    assert len(text) <= timeline.MAX_TRANSCRIPT_CHARS
    lines = text.split("\n")
    assert lines[0].startswith("0 ")
    assert all(len(line) >= timeline.MIN_LINE_CHARS // 2 for line in lines)  # merged, not cut to nothing
    starts = [int(line.split(" ", 1)[0]) for line in lines]
    assert starts == sorted(starts) and set(starts) <= {c.start for c in long}


def test_short_transcript_not_truncated(fake_llm, chunks):
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("lec")
    text = transcript_part(fake_llm.calls[0]["json"]["messages"][0]["content"])
    assert text == "0 intro to races\n60 locks and mutexes\n120 deadlock conditions"


# --- rebuild() ---

def test_rebuild_ignores_cache_and_overwrites(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Old", 0)), reply(("New", 0)))
    timeline.build("lec")
    assert timeline.rebuild("lec")[0].title == "New"
    assert timeline.build("lec")[0].title == "New"


def test_rebuild_raises_and_keeps_cache_on_failure(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Old", 0)), "not json")
    fake_llm.reply("generativelanguage", "not json")
    timeline.build("lec")
    with pytest.raises(timeline.LLM_FAILURES):
        timeline.rebuild("lec")
    assert timeline.build("lec")[0].title == "Old"


def test_rebuild_unknown_lecture_raises(fake_llm, chunks):
    with pytest.raises(LookupError):
        timeline.rebuild("nope")


def test_no_tmp_file_left_behind(fake_llm, chunks):
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("lec")
    assert [p.name for p in timeline.cache_path("lec").parent.iterdir()] == ["lec.timeline.json"]


# --- cleaning the LLM reply ---

def test_sorts_out_of_order_sections(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Deadlock", 120), ("Races", 0), ("Locks", 60)))
    assert [s.title for s in timeline.build("lec")] == ["Races", "Locks", "Deadlock"]


def test_first_section_starts_at_lecture_start(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Locks", 60), ("Deadlock", 120)))
    assert as_tuples(timeline.build("lec"))[0] == ("Locks", 0, 120)


def test_clamps_and_drops_bad_items(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Races", -5), ("Dup", 0), ("", 60), ("Past end", 999), ("Deadlock", 120)))
    assert as_tuples(timeline.build("lec")) == [("Races", 0, 120), ("Deadlock", 120, 200)]


def test_truncates_long_titles(fake_llm, chunks):
    fake_llm.reply("groq", reply(("x" * 200, 0)))
    assert len(timeline.build("lec")[0].title) == timeline.MAX_TITLE_CHARS


def test_accepts_numeric_strings(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Races", "0"), ("Locks", "60.0")))
    assert as_tuples(timeline.build("lec"))[1] == ("Locks", 60, 200)


# --- fallback (never cached) ---

@pytest.mark.parametrize("bad", [
    '{"wrong": []}',
    '{"sections": []}',
    '{"sections": [{"title": "x"}]}',
    '{"sections": [{"title": "x", "start": "soon"}]}',
    '{"sections": "nope"}',
])
def test_bad_reply_falls_back_to_fixed_sections(fake_llm, chunks, bad):
    fake_llm.reply("groq", bad)
    sections = timeline.build("lec")
    assert [s.title for s in sections] == ["Part 1"]
    assert not timeline.cache_path("lec").exists()


def test_llm_down_falls_back_in_5_minute_parts(fake_llm, monkeypatch):
    long = [Chunk(lecture_id="l", start=i * 100, end=i * 100 + 100, text="t") for i in range(7)]  # 700s
    monkeypatch.setattr(store, "get_chunks", lambda lid: long, raising=False)
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    assert as_tuples(timeline.build("l")) == [("Part 1", 0, 300), ("Part 2", 300, 600), ("Part 3", 600, 700)]
    assert not timeline.cache_path("l").exists()


# --- transcript source ---

def test_unknown_lecture_returns_empty(fake_llm, chunks):
    assert timeline.build("nope") == []
    assert fake_llm.calls == []


def test_uses_mock_transcript_until_store_ready(fake_llm):
    fake_llm.reply("groq", reply(("Concurrency", 0)))
    sections = timeline.build("demo")
    assert sections[0].start == 0 and sections[-1].end == 270  # mock demo lecture length


def test_demo_cache_is_committed():
    path = config.DATA_DIR / "cache" / "demo.timeline.json"
    sections = [Section(**s) for s in json.loads(path.read_text())]
    assert sections[0].start == 0 and sections[-1].end == 270

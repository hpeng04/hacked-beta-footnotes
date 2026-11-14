import json
import threading
import time

import httpx
import pytest

from app import config
from app.ingestion import pipeline, store
from app.ingestion.timing import TimedSegment, source_id
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


def reply(*sections, sources=CHUNKS):
    return json.dumps({"sections": [{"title": t, "start_id": source_id(next(c for c in sources if c.start == s))} for t, s in sections]})


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


def test_legacy_cache_is_regenerated_once(fake_llm, chunks):
    path = timeline.cache_path("lec")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"title": "Cached", "start": 0, "end": 200}]))
    fake_llm.reply("groq", reply(("Updated", 0)))
    assert timeline.build("lec") == [Section(title="Updated", start=0, end=200)]
    assert timeline.build("lec")[0].title == "Updated"
    assert len(fake_llm.calls) == 1


def test_prompt_has_timestamped_lines(fake_llm, chunks):
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("lec")
    prompt = fake_llm.calls[0]["json"]["messages"][0]["content"]
    assert all(f"{source_id(c)} {c.text}" in prompt for c in CHUNKS)


def transcript_part(prompt):
    return prompt.split("Transcript:\n", 1)[1]


@pytest.mark.parametrize("n,seconds,words", [
    (720, 5, 30),   # one hour of 5-second chunks: many short lines
    (200, 30, 500), # few chunks, very long text
    (5000, 1, 3),   # extreme chunk count
])
def test_prompt_transcript_within_budget(fake_llm, monkeypatch, n, seconds, words):
    long = [Chunk(lecture_id="big", start=i * seconds, end=(i + 1) * seconds, text="word " * words) for i in range(n)]
    lines = []
    for batch in timeline._batches(long):
        text = transcript_part(timeline._prompt(batch))
        assert len(text) <= timeline.MAX_TRANSCRIPT_CHARS
        lines.extend(text.splitlines())
    # Every sentence retains its own anchor; no merging to fit the prompt.
    assert [line.split(" ", 1)[0] for line in lines] == [source_id(c) for c in long]


def test_short_transcript_not_truncated(fake_llm, chunks):
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("lec")
    text = transcript_part(fake_llm.calls[0]["json"]["messages"][0]["content"])
    assert text == "\n".join(f"{source_id(c)} {c.text}" for c in CHUNKS)


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
    assert as_tuples(timeline.build("lec"))[:2] == [("Opening", 0, 60), ("Locks", 60, 120)]


def test_duplicate_and_blank_titles_are_dropped(fake_llm, chunks):
    fake_llm.reply("groq", reply(("Races", 0), ("Dup", 0), ("", 60), ("Deadlock", 120)))
    assert as_tuples(timeline.build("lec")) == [("Races", 0, 120), ("Deadlock", 120, 200)]


def test_truncates_long_titles(fake_llm, chunks):
    fake_llm.reply("groq", reply(("x" * 200, 0)))
    assert len(timeline.build("lec")[0].title) == timeline.MAX_TITLE_CHARS


def test_unknown_ids_do_not_snap_to_unrelated_times(fake_llm, chunks):
    fake_llm.reply("groq", '{"sections": [{"title": "Invented", "start_id": "s_missing"}]}')
    assert timeline.build("lec")[0].title == "intro to races"
    assert not timeline.cache_path("lec").exists()


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
    assert [s.title for s in sections] == ["intro to races"]
    assert not timeline.cache_path("lec").exists()


def test_llm_down_falls_back_in_5_minute_parts(fake_llm, monkeypatch):
    long = [Chunk(lecture_id="l", start=i * 100, end=i * 100 + 100, text="t") for i in range(7)]  # 700s
    monkeypatch.setattr(store, "get_chunks", lambda lid: long, raising=False)
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    assert as_tuples(timeline.build("l")) == [("t", 0, 300), ("t", 300, 600), ("t", 600, 700)]
    assert not timeline.cache_path("l").exists()


# --- transcript source ---

def test_unknown_lecture_returns_empty(fake_llm, chunks):
    assert timeline.build("nope") == []
    assert fake_llm.calls == []


def test_uses_mock_transcript_until_store_ready(fake_llm):
    fake_llm.reply("groq", reply(("Concurrency", 0), sources=timeline.retrieval.transcript("demo")))
    sections = timeline.build("demo")
    assert sections[0].start == 0 and sections[-1].end == 270  # mock demo lecture length


def test_demo_cache_is_committed():
    path = config.DATA_DIR / "cache" / "demo.timeline.json"
    sections = [Section(**s) for s in json.loads(path.read_text())["sections"]]
    assert sections[0].start == 0 and sections[-1].end == 270


# --- live-run findings (small models) ---

def test_ids_preserve_fractional_sentence_starts(fake_llm, monkeypatch):
    odd = [Chunk(lecture_id="o", start=st, end=en, text="t") for st, en in [(0, 31.2), (31.2, 59.6), (59.6, 90)]]
    monkeypatch.setattr(store, "get_chunks", lambda lid: odd, raising=False)
    fake_llm.reply("groq", reply(("A", 0), ("B", 31.2), ("C", 59.6), sources=odd))
    assert as_tuples(timeline.build("o")) == [("A", 0, 31.2), ("B", 31.2, 59.6), ("C", 59.6, 90)]


def test_one_section_per_chunk_is_merged_to_max(fake_llm, monkeypatch):
    many = [Chunk(lecture_id="m", start=i * 30, end=i * 30 + 30, text="t") for i in range(14)]
    monkeypatch.setattr(store, "get_chunks", lambda lid: many, raising=False)
    fake_llm.reply("groq", reply(*[(f"T{i}", i * 30) for i in range(13)], sources=many))  # live failure: 13 sections
    sections = timeline.build("m")
    assert len(sections) == timeline.MAX_SECTIONS
    assert sections[0].start == 0 and sections[-1].end == 420
    assert all(a.end == b.start for a, b in zip(sections, sections[1:]))


def test_merge_keeps_longer_sections_title(fake_llm, chunks, monkeypatch):
    monkeypatch.setattr(timeline, "MAX_SECTIONS", 2)
    fake_llm.reply("groq", reply(("Races", 0), ("Locks", 60), ("Deadlock", 120)))
    # Races and Locks are both 60s, Deadlock 80s: Races (first shortest) merges into Locks
    assert as_tuples(timeline.build("lec")) == [("Races", 0, 120), ("Deadlock", 120, 200)]


def test_prompt_states_section_limits(fake_llm, chunks):
    fake_llm.reply("groq", reply(("All", 0)))
    timeline.build("lec")
    prompt = fake_llm.calls[0]["json"]["messages"][0]["content"]
    assert f"into {timeline.MIN_SECTIONS} to {timeline.MAX_SECTIONS} topic sections" in prompt


# --- review findings: concurrency ---


def test_concurrent_builds_call_llm_once(fake_llm, chunks, monkeypatch):
    calls = []

    def slow_complete(prompt, json=False):
        calls.append(1)
        time.sleep(0.2)  # both requests arrive while the first is still generating
        return {"sections": [{"title": "All", "start_id": source_id(CHUNKS[0])}]}

    monkeypatch.setattr(timeline.llm, "complete", slow_complete)
    results, errors = [], []

    def worker():
        try:
            results.append(timeline.build("lec"))
        except Exception as e:  # pragma: no cover - the bug this guards against
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(calls) == 1
    assert all(r == results[0] for r in results)


def test_concurrent_writers_use_separate_temp_files(chunks, monkeypatch):
    # Bypass the lock to hit the write path directly: the old shared ".tmp" raced here.
    monkeypatch.setattr(timeline, "generate", lambda c: [Section(title="All", start=0, end=200)])
    errors = []

    def worker():
        try:
            for _ in range(20):
                timeline._generate("lec", CHUNKS)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert [p.name for p in timeline.cache_path("lec").parent.iterdir()] == ["lec.timeline.json"]


# --- review findings: lecture ids ---

@pytest.mark.parametrize("bad", ["../timeline", "../../etc/x", "a/b", "", "x" * 65, "sp ace", "questions.json"])
def test_invalid_lecture_ids_never_touch_files(fake_llm, bad):
    with pytest.raises(ValueError):
        timeline.cache_path(bad)
    assert timeline.build(bad) == []
    with pytest.raises(LookupError):
        timeline.rebuild(bad)
    assert fake_llm.calls == []


def test_timeline_uses_sentence_inside_a_search_chunk_and_replaces_legacy_cache(fake_llm, monkeypatch):
    broad = Chunk(lecture_id="memory", start=94, end=124, text="Strings. Allocate memory. Free memory.")
    monkeypatch.setattr(store, "get_chunks", lambda lid: [broad])
    timed = [TimedSegment(lecture_id="memory", start=94, end=124, text=broad.text, words=[
        {"start": 94, "end": 95, "word": "Strings."},
        {"start": 111.3, "end": 113.08, "word": " Allocate memory."},
        {"start": 117.78, "end": 122.92, "word": " Free memory."},
    ])]
    pipeline._save_timing("memory", timed)
    path = timeline.cache_path("memory")
    path.write_text('[{"title":"Old broad topic", "start":94, "end":124}]')
    fine = timeline.retrieval.timeline_sources("memory")
    fake_llm.reply_timeline("groq", ("Strings", fine[0]), ("Allocation", fine[1]), ("Free", fine[2]))
    sections = timeline.build("memory")
    assert [s.start for s in sections] == [94, 111.3, 117.78]
    assert timeline.build("memory") == sections
    assert len(fake_llm.calls) == 1


def test_changed_sentence_timing_invalidates_cache_even_if_search_chunks_match(fake_llm, monkeypatch):
    sources = [Chunk(lecture_id="lec", start=0, end=111.3, text="Strings."),
               Chunk(lecture_id="lec", start=111.3, end=124, text="Allocate.")]
    monkeypatch.setattr(timeline.retrieval, "timeline_sources", lambda lid: sources)
    fake_llm.reply_timeline("groq", ("Strings", sources[0]), ("Allocation", sources[1]))
    timeline.build("lec")
    sources[1] = sources[1].model_copy(update={"start": 111.64})
    fake_llm.reply_timeline("groq", ("Strings", sources[0]), ("Allocation", sources[1]))
    assert timeline.build("lec")[1].start == 111.64
    assert len(fake_llm.calls) == 2


def test_batched_timeline_preserves_anchors_and_covers_silences(monkeypatch):
    monkeypatch.setattr(timeline, "MAX_TRANSCRIPT_CHARS", 100)
    sources = [Chunk(lecture_id="lec", start=i * 3, end=i * 3 + 1, text="A sentence.") for i in range(10)]
    seen = []

    def complete(prompt, json=False):
        lines = transcript_part(prompt).splitlines()
        seen.extend(line.split(" ")[0] for line in lines)
        return {"sections": [{"title": f"Topic {len(seen)}", "start_id": lines[0].split(" ")[0]}]}

    monkeypatch.setattr(timeline.llm, "complete", complete)
    sections = timeline.generate(sources)
    assert seen == [source_id(c) for c in sources]
    assert sections[0].start == 0 and sections[-1].end == 28
    assert all(a.end == b.start for a, b in zip(sections, sections[1:]))


def test_empty_recovered_speech_does_not_reuse_an_old_topic_cache():
    pipeline._save_timing("silent", [])
    timeline.cache_path("silent").write_text('[{"title":"Old topic", "start":0, "end":30}]')
    assert timeline.build("silent") == []


def test_standalone_portable_timeline_remains_readable():
    path = timeline.cache_path("portable")
    path.parent.mkdir(parents=True)
    path.write_text('[{"title":"Saved topic", "start":0, "end":30}]')
    assert timeline.build("portable") == [Section(title="Saved topic", start=0, end=30)]

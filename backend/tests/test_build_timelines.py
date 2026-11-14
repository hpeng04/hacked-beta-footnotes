import json

import httpx

from app.ingestion import store
from app.schemas import Chunk
from app.search import timeline
from scripts import build_timelines

CHUNKS = [Chunk(lecture_id="lec", start=0, end=100, text="a"), Chunk(lecture_id="lec", start=100, end=200, text="b")]
REPLY = json.dumps({"sections": [{"title": "A", "start": 0}, {"title": "B", "start": 100}]})


def use_chunks(monkeypatch):
    monkeypatch.setattr(store, "get_chunks", lambda lid: CHUNKS if lid == "lec" else [], raising=False)


def test_builds_and_caches(fake_llm, monkeypatch, capsys):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", REPLY)
    assert build_timelines.main(["lec"]) == 0
    assert timeline.cache_path("lec").exists()
    out = capsys.readouterr().out
    assert "2 sections" in out and "1:40  B" in out


def test_skips_cached(fake_llm, monkeypatch, capsys):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", REPLY)
    build_timelines.main(["lec"])
    assert build_timelines.main(["lec"]) == 0
    assert "skipped" in capsys.readouterr().out
    assert len(fake_llm.calls) == 1


def test_force_rebuilds(fake_llm, monkeypatch):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", REPLY, REPLY)
    build_timelines.main(["lec"])
    assert build_timelines.main(["lec", "--force"]) == 0
    assert len(fake_llm.calls) == 2


def test_llm_failure_exits_nonzero(fake_llm, monkeypatch, capsys):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    assert build_timelines.main(["lec"]) == 1
    assert "FAILED (LLM:" in capsys.readouterr().out
    assert not timeline.cache_path("lec").exists()


def test_unknown_lecture_exits_nonzero(fake_llm, monkeypatch, capsys):
    use_chunks(monkeypatch)
    assert build_timelines.main(["nope"]) == 1
    assert "no transcript" in capsys.readouterr().out


def test_continues_after_failure(fake_llm, monkeypatch):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", REPLY)
    assert build_timelines.main(["nope", "lec"]) == 1
    assert timeline.cache_path("lec").exists()


def test_failed_force_rebuild_keeps_old_cache(fake_llm, monkeypatch):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", REPLY)
    build_timelines.main(["lec"])
    before = timeline.cache_path("lec").read_text()
    fake_llm.reply("groq", httpx.ConnectError("down"))
    fake_llm.reply("generativelanguage", httpx.ConnectError("down"))
    assert build_timelines.main(["lec", "--force"]) == 1
    assert timeline.cache_path("lec").read_text() == before


def test_bad_reply_on_force_keeps_old_cache(fake_llm, monkeypatch):
    use_chunks(monkeypatch)
    fake_llm.reply("groq", REPLY)
    build_timelines.main(["lec"])
    before = timeline.cache_path("lec").read_text()
    fake_llm.reply("groq", '{"sections": []}')
    fake_llm.reply("generativelanguage", '{"sections": []}')
    assert build_timelines.main(["lec", "--force"]) == 1
    assert timeline.cache_path("lec").read_text() == before

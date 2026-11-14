"""Ingestion pipeline. Whisper and the vector store are faked; the cache is a temp dir (conftest)."""
import json
from pathlib import Path

import pytest

from app import config
from app.ingestion import pipeline
from app.schemas import Chunk

SEGMENTS = [
    {"start": 0.0, "end": 20.0, "text": "Welcome to the lecture."},
    {"start": 20.0, "end": 35.0, "text": "Today: deadlocks."},
    {"start": 35.0, "end": 50.0, "text": "A deadlock needs four conditions."},
]
CHUNKS = [
    Chunk(lecture_id="lec", start=0.0, end=35.0, text="Welcome to the lecture. Today: deadlocks."),
    Chunk(lecture_id="lec", start=35.0, end=50.0, text="A deadlock needs four conditions."),
]


@pytest.fixture
def fakes(monkeypatch):
    calls = {"transcribe": [], "add_chunks": [], "status_during": []}

    def transcribe(path):
        calls["transcribe"].append(path)
        calls["status_during"].append(pipeline.get_status("lec"))
        return SEGMENTS

    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(pipeline.store, "add_chunks", lambda chunks: calls["add_chunks"].append(list(chunks)))
    return calls


def test_ingest_transcribes_chunks_and_indexes(fakes):
    assert pipeline.ingest(Path("video.mp4"), "lec") == CHUNKS
    assert fakes["transcribe"] == [Path("video.mp4")]
    assert fakes["add_chunks"] == [CHUNKS]


def test_ingest_caches_the_transcript(fakes):
    pipeline.ingest(Path("video.mp4"), "lec")
    assert pipeline.transcript_path("lec") == config.CACHE_DIR / "lec.transcript.json"
    assert json.loads(pipeline.transcript_path("lec").read_text()) == [c.model_dump() for c in CHUNKS]
    assert pipeline.load_transcript("lec") == CHUNKS


def test_second_ingest_skips_whisper_but_reindexes(fakes):
    pipeline.ingest(Path("video.mp4"), "lec")
    assert pipeline.ingest(Path("video.mp4"), "lec") == CHUNKS
    assert len(fakes["transcribe"]) == 1
    assert fakes["add_chunks"] == [CHUNKS, CHUNKS]


def test_status_goes_processing_then_ready(fakes):
    assert pipeline.get_status("lec") is None
    pipeline.ingest(Path("video.mp4"), "lec")
    assert fakes["status_during"] == [{"status": "processing"}]
    assert pipeline.get_status("lec") == {"status": "ready"}


def test_failed_transcription_is_recorded_and_raised(fakes, monkeypatch):
    def boom(path):
        raise RuntimeError("ffmpeg could not extract audio")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    with pytest.raises(RuntimeError):
        pipeline.ingest(Path("video.mp4"), "lec")
    assert pipeline.get_status("lec") == {"status": "failed", "error": "ffmpeg could not extract audio"}
    assert pipeline.load_transcript("lec") is None
    assert fakes["add_chunks"] == []


def test_failed_indexing_keeps_the_transcript_for_a_retry(fakes, monkeypatch):
    def boom(chunks):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(pipeline.store, "add_chunks", boom)
    with pytest.raises(RuntimeError):
        pipeline.ingest(Path("video.mp4"), "lec")
    assert pipeline.get_status("lec")["status"] == "failed"
    assert pipeline.load_transcript("lec") == CHUNKS  # the retry won't run Whisper again


def test_retry_after_failure_ends_ready(fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "transcribe", lambda path: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        pipeline.ingest(Path("video.mp4"), "lec")
    monkeypatch.setattr(pipeline, "transcribe", lambda path: SEGMENTS)
    pipeline.ingest(Path("video.mp4"), "lec")
    assert pipeline.get_status("lec") == {"status": "ready"}


def test_silent_video_is_ready_with_empty_transcript(fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "transcribe", lambda path: [])
    assert pipeline.ingest(Path("video.mp4"), "lec") == []
    assert pipeline.load_transcript("lec") == []
    assert pipeline.get_status("lec") == {"status": "ready"}


def test_unknown_lecture_has_no_transcript_or_status():
    assert pipeline.load_transcript("nope") is None
    assert pipeline.get_status("nope") is None


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x" * 65])
def test_bad_lecture_ids_never_touch_disk(fakes, bad):
    for fn in (pipeline.load_transcript, pipeline.get_status, lambda i: pipeline.ingest(Path("v.mp4"), i)):
        with pytest.raises(ValueError):
            fn(bad)
    assert fakes["transcribe"] == []
    assert not config.CACHE_DIR.exists()


def test_no_temp_files_left_behind(fakes):
    pipeline.ingest(Path("video.mp4"), "lec")
    assert sorted(p.name for p in config.CACHE_DIR.iterdir()) == ["lec.status.json", "lec.transcript.json"]


def test_failed_write_keeps_the_old_file_and_leaves_no_temp_file():
    path = pipeline.transcript_path("lec")
    pipeline._write_json(path, [{"ok": True}])
    with pytest.raises(TypeError):
        pipeline._write_json(path, {"not serializable": object()})
    assert json.loads(path.read_text()) == [{"ok": True}]
    assert [p.name for p in config.CACHE_DIR.iterdir()] == ["lec.transcript.json"]


def test_set_status_omits_error_unless_given():
    pipeline.set_status("lec", pipeline.READY)
    assert pipeline.get_status("lec") == {"status": "ready"}
    pipeline.set_status("lec", pipeline.FAILED, "boom")
    assert pipeline.get_status("lec") == {"status": "failed", "error": "boom"}


# --- force, and what a changed transcript invalidates ---

def write_timeline(lecture_id="lec"):
    path = config.CACHE_DIR / f"{lecture_id}.timeline.json"
    path.write_text('[{"title": "Old", "start": 0, "end": 50}]')
    return path


def test_force_transcribes_again_and_replaces_the_transcript(fakes, monkeypatch):
    pipeline.ingest(Path("video.mp4"), "lec")
    monkeypatch.setattr(pipeline, "transcribe", lambda path: SEGMENTS[:1])
    chunks = pipeline.ingest(Path("video.mp4"), "lec", force=True)
    assert [c.end for c in chunks] == [20.0]
    assert pipeline.load_transcript("lec") == chunks
    assert fakes["add_chunks"][-1] == chunks


def test_failed_forced_run_keeps_the_previous_transcript(fakes, monkeypatch):
    pipeline.ingest(Path("video.mp4"), "lec")
    timeline = write_timeline()
    monkeypatch.setattr(pipeline, "transcribe", lambda path: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        pipeline.ingest(Path("video.mp4"), "lec", force=True)
    assert pipeline.load_transcript("lec") == CHUNKS
    assert timeline.exists()
    assert fakes["add_chunks"] == [CHUNKS]  # the index was not touched either


def test_changed_transcript_drops_the_cached_timeline(fakes, monkeypatch):
    pipeline.ingest(Path("video.mp4"), "lec")
    timeline = write_timeline()
    monkeypatch.setattr(pipeline, "transcribe", lambda path: SEGMENTS[:1])
    pipeline.ingest(Path("video.mp4"), "lec", force=True)
    assert not timeline.exists()


def test_identical_forced_transcript_keeps_the_timeline(fakes):
    pipeline.ingest(Path("video.mp4"), "lec")
    timeline = write_timeline()
    pipeline.ingest(Path("video.mp4"), "lec", force=True)
    assert timeline.exists()


def test_reindexing_a_cached_transcript_keeps_the_timeline(fakes):
    pipeline.ingest(Path("video.mp4"), "lec")
    timeline = write_timeline()
    pipeline.ingest(Path("video.mp4"), "lec")
    assert timeline.exists()


def test_first_transcript_drops_a_leftover_timeline(fakes):
    config.CACHE_DIR.mkdir(parents=True)
    timeline = write_timeline()
    pipeline.ingest(Path("video.mp4"), "lec")
    assert not timeline.exists()


def test_failed_indexing_of_a_forced_run_keeps_transcript_and_timeline(fakes, monkeypatch):
    pipeline.ingest(Path("video.mp4"), "lec")
    timeline = write_timeline()
    monkeypatch.setattr(pipeline, "transcribe", lambda path: SEGMENTS[:1])

    def boom(chunks):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(pipeline.store, "add_chunks", boom)
    with pytest.raises(RuntimeError):
        pipeline.ingest(Path("video.mp4"), "lec", force=True)
    assert pipeline.load_transcript("lec") == CHUNKS
    assert timeline.exists()


def test_forced_run_indexes_before_it_replaces_the_cache(fakes, monkeypatch):
    pipeline.ingest(Path("video.mp4"), "lec")
    monkeypatch.setattr(pipeline, "transcribe", lambda path: SEGMENTS[:1])
    seen = []
    monkeypatch.setattr(pipeline.store, "add_chunks", lambda chunks: seen.append(pipeline.load_transcript("lec")))
    pipeline.ingest(Path("video.mp4"), "lec", force=True)
    assert seen == [CHUNKS]  # the old transcript was still cached while indexing ran
    assert len(pipeline.load_transcript("lec")) == 1

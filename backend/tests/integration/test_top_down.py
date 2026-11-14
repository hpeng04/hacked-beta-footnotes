"""Top-down integration of the ingestion path.

The top module (the FastAPI app and its ingestion router) is tested first with everything
below it stubbed. Each level then swaps one stub for the real module and re-checks the
behaviour through the API:

  level 1  router                                   stubs: pipeline
  level 2  router + pipeline                        stubs: transcribe, chunking, store
  level 3  router + pipeline + chunking             stubs: transcribe, store
  level 4  ... + store (real Chroma) + search side  stubs: transcribe, embedding model
  level 5  ... + transcribe (real ffmpeg)           stubs: Whisper and embedding models

The LLM provider is always faked (tests/conftest.py).
"""
import pytest

from app import config
from app.ingestion import pipeline, store
from app.schemas import Chunk

from .conftest import BIO_SCRIPT, OS_CHUNKS, OS_SCRIPT, OS_TIMELINE_REPLY, get, needs_ffmpeg, post_video, segments


def as_json(lecture_id, chunks):
    return [{"lecture_id": lecture_id, "start": s, "end": e, "text": t} for s, e, t in chunks]


# --------------------------------------------------------------------------------------
class TestLevel1Router:
    """Real: app, ingestion router. Stubbed: the whole pipeline module interface."""

    @pytest.fixture
    def stub(self, monkeypatch, uploads):
        calls, state = [], {"status": {}, "transcripts": {}}

        def set_status(lecture_id, status, error=None):
            calls.append(("set_status", lecture_id, status))
            state["status"][lecture_id] = {"status": status}

        def ingest(path, lecture_id):
            calls.append(("ingest", path, lecture_id, path.read_bytes()))
            if state.get("ingest_error"):
                raise state["ingest_error"]

        monkeypatch.setattr(pipeline, "set_status", set_status)
        monkeypatch.setattr(pipeline, "ingest", ingest)
        monkeypatch.setattr(pipeline, "get_status", lambda lid: state["status"].get(lid))
        monkeypatch.setattr(pipeline, "load_transcript", lambda lid: state["transcripts"].get(lid))
        state["calls"] = calls
        return state

    def test_upload_marks_processing_then_hands_the_saved_file_to_the_pipeline(self, client, stub, uploads):
        lecture_id = post_video(client, b"the bytes", "talk.mov")
        path = uploads / f"{lecture_id}.mov"
        assert stub["calls"] == [
            ("set_status", lecture_id, "processing"),
            ("ingest", path, lecture_id, b"the bytes"),  # the file was complete when ingest ran
        ]

    def test_pipeline_crash_does_not_fail_the_upload(self, client, stub):
        stub["ingest_error"] = RuntimeError("boom")
        r = client.post("/upload", files={"file": ("a.mp4", b"x", "video/mp4")})
        assert r.status_code == 200

    @pytest.mark.parametrize("state, expected", [
        ({"status": "processing"}, {"status": "processing", "error": None}),
        ({"status": "ready"}, {"status": "ready", "error": None}),
        ({"status": "failed", "error": "no audio"}, {"status": "failed", "error": "no audio"}),
    ])
    def test_status_relays_the_pipeline_state(self, client, stub, state, expected):
        stub["status"]["lec"] = state
        assert get(client, "/status", "lec").json() == {"lecture_id": "lec", **expected}

    def test_status_unknown_to_the_pipeline_is_404(self, client, stub):
        assert get(client, "/status", "lec").status_code == 404

    def test_transcript_relays_the_pipeline_chunks(self, client, stub):
        stub["transcripts"]["lec"] = [Chunk(lecture_id="lec", start=0, end=9.5, text="hello")]
        assert get(client, "/transcript", "lec").json() == as_json("lec", [(0.0, 9.5, "hello")])

    def test_transcript_falls_back_to_mock_when_the_pipeline_has_none(self, client, stub):
        assert get(client, "/transcript", "demo").json()[0]["lecture_id"] == "demo"
        assert get(client, "/transcript", "lec").json() == []


# --------------------------------------------------------------------------------------
class TestLevel2Pipeline:
    """Real: router, pipeline (cache and status files). Stubbed: transcribe, chunking, store."""

    STUB_CHUNKS = [(0.0, 40.0, "stub chunk one"), (40.0, 80.0, "stub chunk two")]

    @pytest.fixture
    def stub(self, monkeypatch):
        calls, state = [], {}

        def transcribe(path):
            calls.append(("transcribe", path, pipeline.get_status(path.stem)))
            if state.get("transcribe_error"):
                raise state["transcribe_error"]
            return segments(OS_SCRIPT)

        def chunk_segments(lecture_id, segs):
            calls.append(("chunk_segments", lecture_id, segs))
            return [Chunk(lecture_id=lecture_id, start=s, end=e, text=t) for s, e, t in self.STUB_CHUNKS]

        def add_chunks(chunks):
            calls.append(("add_chunks", list(chunks)))
            if state.get("store_error"):
                raise state["store_error"]

        monkeypatch.setattr(pipeline, "transcribe", transcribe)
        monkeypatch.setattr(pipeline, "chunk_segments", chunk_segments)
        monkeypatch.setattr(store, "add_chunks", add_chunks)
        state["calls"] = calls
        return state

    def test_data_flows_transcribe_to_chunking_to_store(self, client, stub, uploads):
        lecture_id = post_video(client)
        transcribed, chunked, stored = stub["calls"]
        assert transcribed == ("transcribe", uploads / f"{lecture_id}.mp4", {"status": "processing"})
        assert chunked == ("chunk_segments", lecture_id, segments(OS_SCRIPT))
        assert stored == ("add_chunks", [Chunk(lecture_id=lecture_id, start=s, end=e, text=t) for s, e, t in self.STUB_CHUNKS])

    def test_api_reports_ready_and_serves_the_cached_transcript(self, client, stub):
        lecture_id = post_video(client)
        assert get(client, "/status", lecture_id).json()["status"] == "ready"
        assert get(client, "/transcript", lecture_id).json() == as_json(lecture_id, self.STUB_CHUNKS)
        assert (config.CACHE_DIR / f"{lecture_id}.transcript.json").exists()

    def test_transcription_error_reaches_the_api(self, client, stub):
        stub["transcribe_error"] = RuntimeError("ffmpeg could not extract audio")
        lecture_id = post_video(client)
        assert get(client, "/status", lecture_id).json() == {
            "lecture_id": lecture_id, "status": "failed", "error": "ffmpeg could not extract audio",
        }
        assert get(client, "/transcript", lecture_id).json() == []
        assert [c[0] for c in stub["calls"]] == ["transcribe"]  # nothing ran after the failure

    def test_store_error_reaches_the_api_but_keeps_the_transcript(self, client, stub):
        stub["store_error"] = RuntimeError("vector DB unavailable")
        lecture_id = post_video(client)
        assert get(client, "/status", lecture_id).json()["error"] == "vector DB unavailable"
        assert get(client, "/transcript", lecture_id).json() == as_json(lecture_id, self.STUB_CHUNKS)


# --------------------------------------------------------------------------------------
class TestLevel3Chunking:
    """Real: router, pipeline, chunking. Stubbed: transcribe, store."""

    @pytest.fixture
    def stored(self, monkeypatch):
        stored = []
        monkeypatch.setattr(pipeline, "transcribe", lambda path: segments(OS_SCRIPT))
        monkeypatch.setattr(store, "add_chunks", lambda chunks: stored.append(list(chunks)))
        return stored

    def test_segments_are_merged_into_timed_chunks(self, client, stored):
        lecture_id = post_video(client)
        assert get(client, "/transcript", lecture_id).json() == as_json(lecture_id, OS_CHUNKS)

    def test_store_receives_exactly_what_the_api_serves(self, client, stored):
        lecture_id = post_video(client)
        assert [c.model_dump() for c in stored[0]] == get(client, "/transcript", lecture_id).json()

    def test_silent_lecture_is_ready_and_empty(self, client, stored, monkeypatch):
        monkeypatch.setattr(pipeline, "transcribe", lambda path: [])
        lecture_id = post_video(client)
        assert get(client, "/status", lecture_id).json()["status"] == "ready"
        assert get(client, "/transcript", lecture_id).json() == []


# --------------------------------------------------------------------------------------
class TestLevel4Store:
    """Real: router, pipeline, chunking, store with a real Chroma DB, and the search side
    (search router, retrieval, timeline, llm). Stubbed: transcribe, the embedding model."""

    @pytest.fixture(autouse=True)
    def stubs(self, monkeypatch, stub_models):
        scripts = [OS_SCRIPT, BIO_SCRIPT]
        monkeypatch.setattr(pipeline, "transcribe", lambda path: segments(scripts.pop(0)))

    def test_uploaded_lecture_is_searchable(self, client):
        lecture_id = post_video(client)
        hits = get(client, "/search", lecture_id, q="deadlock circular wait").json()
        assert hits[0] == as_json(lecture_id, OS_CHUNKS)[1]
        assert get(client, "/search", lecture_id, q="semaphore resources").json()[0]["start"] == 64.0

    def test_ask_cites_chunks_from_the_vector_db(self, client, fake_llm):
        lecture_id = post_video(client)
        fake_llm.reply_answer("groq", ("Threads wait for each other.",
                                       [next(c for c in pipeline.load_segments(lecture_id) if c.start == 31)]))
        body = client.post("/ask", json={"lecture_id": lecture_id, "question": "what is a deadlock?"}).json()
        assert body["answer"]
        assert body["sources"][0]["start"] == 31.0
        assert "A deadlock happens when two threads" in fake_llm.calls[0]["json"]["messages"][0]["content"]

    def test_timeline_is_built_from_the_stored_chunks(self, client, fake_llm):
        lecture_id = post_video(client)
        fake_llm.reply("groq", OS_TIMELINE_REPLY)
        assert get(client, "/timeline", lecture_id).json() == [
            {"title": "Intro", "start": 0.0, "end": 31.0},
            {"title": "Deadlock", "start": 31.0, "end": 64.0},
            {"title": "Semaphores", "start": 64.0, "end": 80.0},
        ]

    def test_lectures_do_not_leak_into_each_other(self, client):
        os_id, bio_id = post_video(client), post_video(client)
        assert all(h["lecture_id"] == bio_id for h in get(client, "/search", bio_id, q="deadlock threads").json())
        assert all(h["lecture_id"] == os_id for h in get(client, "/search", os_id, q="photosynthesis light").json())
        assert len(get(client, "/search", bio_id, q="light").json()) == 1

    def test_lecture_that_failed_is_not_searchable(self, client, monkeypatch):
        monkeypatch.setattr(pipeline, "transcribe", lambda path: 1 / 0)
        lecture_id = post_video(client)
        assert get(client, "/search", lecture_id, q="deadlock").status_code == 404


# --------------------------------------------------------------------------------------
@needs_ffmpeg
class TestLevel5Transcribe:
    """Real: everything in the repo, with real ffmpeg. Stubbed: the Whisper and embedding models."""

    def test_whisper_gets_16k_mono_audio_of_the_uploaded_video(self, client, stub_models, clip):
        stub_models.whisper.scripts.append(OS_SCRIPT)
        post_video(client, clip.bytes)
        heard, = stub_models.whisper.heard
        assert (heard["channels"], heard["rate"], heard["vad_filter"]) == (1, 16000, True)
        assert heard["seconds"] == pytest.approx(clip.seconds, abs=0.5)

    def test_models_are_loaded_from_config_and_only_once(self, client, stub_models, clip, monkeypatch):
        monkeypatch.setattr(config, "WHISPER_MODEL", "base")
        stub_models.whisper.scripts += [OS_SCRIPT, BIO_SCRIPT]
        post_video(client, clip.bytes)
        post_video(client, clip.bytes)
        assert stub_models.whisper.loaded == [("base", "cpu", "int8")]
        assert stub_models.embedder.loaded == ["all-MiniLM-L6-v2"]

    def test_whisper_output_flows_through_to_search(self, client, stub_models, clip):
        stub_models.whisper.scripts.append(OS_SCRIPT)
        lecture_id = post_video(client, clip.bytes)
        assert get(client, "/status", lecture_id).json()["status"] == "ready"
        assert get(client, "/transcript", lecture_id).json() == as_json(lecture_id, OS_CHUNKS)
        assert get(client, "/search", lecture_id, q="deadlock circular wait").json()[0]["start"] == 31.0

    def test_file_without_audio_fails_before_whisper(self, client, stub_models):
        lecture_id = post_video(client, b"this is not a video", "notes.mp4")
        status = get(client, "/status", lecture_id).json()
        assert status["status"] == "failed" and "could not extract audio" in status["error"]
        assert stub_models.whisper.heard == [] and stub_models.whisper.loaded == []

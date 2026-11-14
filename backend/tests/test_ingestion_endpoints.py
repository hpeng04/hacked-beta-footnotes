"""API checks for Person 1's endpoints: /upload, /status, /transcript. Whisper and embeddings are faked."""
import pytest

from app import config
from app.ingestion import pipeline, router, store
from app.schemas import Chunk

SEGMENTS = [
    {"start": 0.0, "end": 31.0, "text": "A semaphore counts available resources."},
    {"start": 31.0, "end": 40.0, "text": "Deadlock needs circular wait."},
]


@pytest.fixture(autouse=True)
def uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")  # never touch the real uploads
    return tmp_path / "uploads"


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(pipeline, "transcribe", lambda path: SEGMENTS)
    monkeypatch.setattr(store, "_embed", lambda texts: [[float("deadlock" in t.lower()), 1.0] for t in texts])


def post_video(client, name="lecture.mp4", content=b"fake video bytes"):
    return client.post("/upload", files={"file": (name, content, "video/mp4")})


# --- /upload ---

def test_upload_saves_file_and_returns_contract(client, uploads, monkeypatch):
    monkeypatch.setattr(router, "_ingest", lambda path, lecture_id: None)
    r = post_video(client)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"lecture_id", "video_url"}
    assert config.is_valid_lecture_id(body["lecture_id"])
    assert body["video_url"] == f"/media/{body['lecture_id']}.mp4"
    assert (uploads / f"{body['lecture_id']}.mp4").read_bytes() == b"fake video bytes"


def test_upload_schedules_ingestion_of_the_saved_file(client, uploads, monkeypatch):
    calls = []
    monkeypatch.setattr(router, "_ingest", lambda path, lecture_id: calls.append((path, lecture_id)))
    lecture_id = post_video(client).json()["lecture_id"]
    assert calls == [(uploads / f"{lecture_id}.mp4", lecture_id)]


def test_status_is_processing_until_ingestion_finishes(client, monkeypatch):
    monkeypatch.setattr(router, "_ingest", lambda path, lecture_id: None)  # never finishes
    lecture_id = post_video(client).json()["lecture_id"]
    r = client.get("/status", params={"lecture_id": lecture_id})
    assert r.json() == {"lecture_id": lecture_id, "status": "processing", "error": None}
    assert client.get("/transcript", params={"lecture_id": lecture_id}).json() == []


def test_uploads_get_distinct_ids(client, monkeypatch):
    monkeypatch.setattr(router, "_ingest", lambda path, lecture_id: None)
    assert post_video(client).json()["lecture_id"] != post_video(client).json()["lecture_id"]


def test_upload_requires_a_file(client):
    assert client.post("/upload").status_code == 422


# --- upload -> transcript -> search, end to end ---

def test_uploaded_lecture_becomes_searchable(client, fake_models, fake_llm):
    lecture_id = post_video(client).json()["lecture_id"]  # TestClient runs the background task first
    assert client.get("/status", params={"lecture_id": lecture_id}).json()["status"] == "ready"

    chunks = client.get("/transcript", params={"lecture_id": lecture_id}).json()
    assert chunks == [
        {"lecture_id": lecture_id, "start": 0.0, "end": 31.0, "text": "A semaphore counts available resources."},
        {"lecture_id": lecture_id, "start": 31.0, "end": 40.0, "text": "Deadlock needs circular wait."},
    ]

    hits = client.get("/search", params={"lecture_id": lecture_id, "q": "deadlock"}).json()
    assert hits[0]["text"] == "Deadlock needs circular wait."

    fake_llm.reply_answer("groq", ("It needs circular wait.", [pipeline.load_segments(lecture_id)[1]]))
    answer = client.post("/ask", json={"lecture_id": lecture_id, "question": "what causes deadlock?"}).json()
    assert answer["sources"][0]["start"] == 31.0


def test_failed_ingestion_is_reported_not_raised(client, monkeypatch):
    def boom(path):
        raise RuntimeError("ffmpeg could not extract audio from x.mp4")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    r = post_video(client)
    assert r.status_code == 200  # the upload itself worked
    lecture_id = r.json()["lecture_id"]
    assert client.get("/status", params={"lecture_id": lecture_id}).json() == {
        "lecture_id": lecture_id,
        "status": "failed",
        "error": "ffmpeg could not extract audio from x.mp4",
    }


# --- /status ---

def test_status_ready_for_mock_lecture(client):
    assert client.get("/status", params={"lecture_id": "demo"}).json()["status"] == "ready"


def test_status_ready_for_preprocessed_lecture_without_status_file(client):
    pipeline._write_json(pipeline.transcript_path("pre"), [{"lecture_id": "pre", "start": 0, "end": 5, "text": "hi"}])
    assert client.get("/status", params={"lecture_id": "pre"}).json()["status"] == "ready"


@pytest.mark.parametrize("lecture_id", ["nope", "../demo", "a/b", "x" * 65])
def test_status_unknown_lecture_404(client, lecture_id):
    assert client.get("/status", params={"lecture_id": lecture_id}).status_code == 404


def test_status_requires_lecture_id(client):
    assert client.get("/status").status_code == 422


# --- /transcript ---

def test_transcript_demo_still_served_from_mock(client):
    chunks = client.get("/transcript", params={"lecture_id": "demo"}).json()
    assert chunks and set(chunks[0]) == {"lecture_id", "start", "end", "text"}
    assert all(c["lecture_id"] == "demo" for c in chunks)


def test_transcript_prefers_cache_over_mock(client):
    cached = Chunk(lecture_id="demo", start=0.0, end=9.0, text="real transcript")
    pipeline._write_json(pipeline.transcript_path("demo"), [cached.model_dump()])
    assert client.get("/transcript", params={"lecture_id": "demo"}).json() == [cached.model_dump()]


@pytest.mark.parametrize("lecture_id", ["nope", "../transcript", "a/b"])
def test_transcript_unknown_lecture_is_empty(client, lecture_id):
    r = client.get("/transcript", params={"lecture_id": lecture_id})
    assert r.status_code == 200 and r.json() == []

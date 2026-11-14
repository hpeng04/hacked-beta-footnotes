"""Smoke tests for Person 1's ingestion endpoints."""
from app import config
from app.ingestion import router as ingestion_router


def test_upload_saves_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(ingestion_router, "_ingest", lambda path, lecture_id: None)  # no transcription here
    r = client.post("/upload", files={"file": ("clip.mp4", b"fake video", "video/mp4")})
    assert r.status_code == 200
    body = r.json()
    assert body["video_url"] == f"/media/{body['lecture_id']}.mp4"
    assert (tmp_path / f"{body['lecture_id']}.mp4").read_bytes() == b"fake video"


def test_upload_requires_file(client):
    assert client.post("/upload").status_code == 422


def test_transcript_demo(client):
    r = client.get("/transcript", params={"lecture_id": "demo"})
    assert r.status_code == 200
    chunks = r.json()
    assert chunks and all(set(c) == {"lecture_id", "start", "end", "text"} for c in chunks)
    assert all(isinstance(c["start"], float) and c["start"] < c["end"] for c in chunks)

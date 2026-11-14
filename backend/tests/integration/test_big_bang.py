"""Big bang integration: every module assembled at once and exercised as a whole.

Nothing in the repo is stubbed: the FastAPI app, both routers, pipeline, transcribe with real
ffmpeg, chunking, store with a real Chroma DB, retrieval, timeline, the cache on disk and the
scripts all run together. Only what lives outside the repo is replaced:
  - the LLM provider (HTTP, faked in tests/conftest.py)
  - the Whisper and embedding models (stub classes), except in the tests marked `models`,
    which load the real ones:  python -m pytest -m models
"""
import pytest

from app import config
from app.ingestion import pipeline, store, transcribe
from scripts import build_timelines, preprocess

from .conftest import BIO_SCRIPT, OS_CHUNKS, OS_SCRIPT, SPEECH, get, needs_ffmpeg, post_video

pytestmark = needs_ffmpeg


def test_full_lecture_journey(client, stub_models, clip, fake_llm, monkeypatch):
    # Exercise the legacy timeline fallback; shared section routing is covered separately.
    from app.search import presentation
    monkeypatch.setattr(presentation, "section_index", lambda lid: None)
    """Upload -> status -> transcript -> search -> ask -> timeline -> video, as the frontend does it."""
    stub_models.whisper.scripts.append(OS_SCRIPT)

    r = client.post("/upload", files={"file": ("Lecture 1.mp4", clip.bytes, "video/mp4")})
    assert r.status_code == 200
    lecture_id, video_url = r.json()["lecture_id"], r.json()["video_url"]

    assert get(client, "/status", lecture_id).json() == {"lecture_id": lecture_id, "status": "ready", "error": None}

    transcript = get(client, "/transcript", lecture_id).json()
    assert [(c["start"], c["end"], c["text"]) for c in transcript] == OS_CHUNKS
    assert all(c["lecture_id"] == lecture_id for c in transcript)

    hits = get(client, "/search", lecture_id, q="when do threads wait forever in a deadlock").json()
    assert hits[0] == transcript[1]

    fake_llm.reply_answer("groq", ("A deadlock is two threads waiting on each other.",
                                   [next(c for c in pipeline.load_segments(lecture_id) if c.start == 31)]))
    answer = client.post("/ask", json={"lecture_id": lecture_id, "question": "What is a deadlock?"}).json()
    assert "[0:31]" in answer["answer"]
    # Search keeps its broad chunk, but the answer cites the original speech segment.
    assert answer["sources"][0] == {
        "lecture_id": lecture_id, "start": 31.0, "end": 48.0,
        "text": "A deadlock happens when two threads wait on each other forever.",
    }
    assert all(any(c["start"] <= s["start"] < s["end"] <= c["end"] for c in transcript) for s in answer["sources"])

    queue_os_timeline(fake_llm, lecture_id)
    sections = get(client, "/timeline", lecture_id).json()
    assert [s["title"] for s in sections] == ["Intro", "Deadlock", "Semaphores"]
    assert sections[0]["start"] == transcript[0]["start"] and sections[-1]["end"] == transcript[-1]["end"]
    llm_calls = len(fake_llm.calls)
    assert get(client, "/timeline", lecture_id).json() == sections  # second call comes from the cache
    assert len(fake_llm.calls) == llm_calls

    video = client.get(video_url)
    assert video.status_code == 200 and video.content == clip.bytes


def test_uploads_and_the_mock_lecture_coexist(client, stub_models, clip):
    stub_models.whisper.scripts += [OS_SCRIPT, BIO_SCRIPT]
    os_id, bio_id = post_video(client, clip.bytes), post_video(client, clip.bytes)

    assert get(client, "/search", os_id, q="semaphore resources").json()[0]["start"] == 64.0
    bio_hits = get(client, "/search", bio_id, q="chlorophyll light").json()
    assert [h["lecture_id"] for h in bio_hits] == [bio_id]
    assert all(h["lecture_id"] == os_id for h in get(client, "/search", os_id, q="chlorophyll light").json())

    # "demo" has no vector DB rows: it is still served from mock data by keyword search
    assert get(client, "/status", "demo").json()["status"] == "ready"
    assert "deadlock" in get(client, "/search", "demo", q="deadlock").json()[0]["text"].lower()
    assert len(get(client, "/transcript", "demo").json()) > 3


def test_broken_upload_fails_alone(client, stub_models, clip, fake_llm):
    stub_models.whisper.scripts.append(OS_SCRIPT)
    good = post_video(client, clip.bytes)
    bad = post_video(client, b"definitely not a video", "broken.mp4")

    status = get(client, "/status", bad).json()
    assert status["status"] == "failed" and "could not extract audio from" in status["error"]
    assert get(client, "/transcript", bad).json() == []
    assert get(client, "/search", bad, q="deadlock").status_code == 404
    assert client.post("/ask", json={"lecture_id": bad, "question": "deadlock?"}).status_code == 404
    assert get(client, "/timeline", bad).status_code == 404
    assert fake_llm.calls == []  # no LLM quota spent on a lecture that doesn't exist

    assert get(client, "/status", good).json()["status"] == "ready"
    assert get(client, "/search", good, q="deadlock").json()


def test_unknown_lecture_is_404_everywhere(client, stub_models):
    assert get(client, "/status", "nope").status_code == 404
    assert get(client, "/search", "nope", q="x").status_code == 404
    assert get(client, "/timeline", "nope").status_code == 404
    assert get(client, "/transcript", "nope").json() == []


def test_preprocessed_demo_lecture_is_served_by_the_api(client, stub_models, clip, fake_llm, uploads, monkeypatch):
    # Exercise the legacy timeline fallback; shared section routing is covered separately.
    from app.search import presentation
    monkeypatch.setattr(presentation, "section_index", lambda lid: None)
    """The demo workflow: scripts.preprocess, then scripts.build_timelines, then the app."""
    stub_models.whisper.scripts.append(OS_SCRIPT)
    assert preprocess.main([str(clip.path), "os-lecture", "--model", "small"]) == 0
    queue_os_timeline(fake_llm, "os-lecture")
    assert build_timelines.main(["os-lecture"]) == 0
    llm_calls = len(fake_llm.calls)

    assert stub_models.whisper.loaded == [("small", "cpu", "int8")]
    assert get(client, "/status", "os-lecture").json()["status"] == "ready"
    assert [c["start"] for c in get(client, "/transcript", "os-lecture").json()] == [0.0, 31.0, 64.0]
    assert get(client, "/search", "os-lecture", q="deadlock circular wait").json()[0]["start"] == 31.0
    assert [s["title"] for s in get(client, "/timeline", "os-lecture").json()] == ["Intro", "Deadlock", "Semaphores"]
    assert len(fake_llm.calls) == llm_calls  # the timeline came from the cache
    assert client.get("/media/os-lecture.mp4").content == clip.bytes


def test_fresh_clone_needs_only_reindex(client, stub_models, clip, monkeypatch, tmp_path):
    """Committed transcripts plus --reindex restore search without running Whisper again."""
    stub_models.whisper.scripts.append(OS_SCRIPT)
    assert preprocess.main([str(clip.path), "os-lecture"]) == 0
    assert len(stub_models.whisper.heard) == 1

    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "fresh-clone-chroma")  # the vector DB isn't committed
    (config.CACHE_DIR / "os-lecture.status.json").unlink()  # nor are status files
    assert get(client, "/transcript", "os-lecture").json()  # the committed transcript is there
    assert get(client, "/status", "os-lecture").json()["status"] == "ready"
    assert get(client, "/search", "os-lecture", q="deadlock").status_code == 404  # but nothing is indexed

    assert preprocess.main(["--reindex"]) == 0
    assert get(client, "/search", "os-lecture", q="deadlock circular wait").json()[0]["start"] == 31.0
    assert len(stub_models.whisper.heard) == 1  # Whisper did not run again


def test_reprocessing_a_lecture_replaces_it(client, stub_models, clip):
    stub_models.whisper.scripts += [OS_SCRIPT, BIO_SCRIPT]
    assert preprocess.main([str(clip.path), "lec"]) == 0
    assert preprocess.main([str(clip.path), "lec", "--force"]) == 0
    assert [c["text"] for c in get(client, "/transcript", "lec").json()] == [
        "Photosynthesis turns light into chemical energy. Chlorophyll absorbs red and blue light."
    ]
    assert len(get(client, "/search", "lec", q="deadlock light").json()) == 1  # old rows are gone


def test_reprocessing_a_lecture_rebuilds_its_timeline(client, stub_models, clip, fake_llm, monkeypatch):
    # Exercise the legacy timeline fallback; shared section routing is covered separately.
    from app.search import presentation
    monkeypatch.setattr(presentation, "section_index", lambda lid: None)
    stub_models.whisper.scripts += [OS_SCRIPT, BIO_SCRIPT]
    preprocess.main([str(clip.path), "lec"])
    queue_os_timeline(fake_llm, "lec")
    assert build_timelines.main(["lec"]) == 0
    assert get(client, "/timeline", "lec").json()[-1]["end"] == 80.0

    assert preprocess.main([str(clip.path), "lec", "--force"]) == 0
    fake_llm.reply_timeline("groq", ("Photosynthesis", pipeline.load_segments("lec")[0]))
    assert build_timelines.main(["lec"]) == 0  # not "cached, skipped": the old timeline is gone
    assert get(client, "/timeline", "lec").json() == [{"title": "Photosynthesis", "start": 0.0, "end": 45.0}]


def test_failed_reprocessing_keeps_the_lecture_working(client, stub_models, clip, tmp_path):
    stub_models.whisper.scripts.append(OS_SCRIPT)
    preprocess.main([str(clip.path), "lec"])
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    assert preprocess.main([str(broken), "lec", "--force"]) == 1
    assert [c["start"] for c in get(client, "/transcript", "lec").json()] == [0.0, 31.0, 64.0]
    assert get(client, "/search", "lec", q="deadlock circular wait").json()[0]["start"] == 31.0
    assert client.get("/media/lec.mp4").content == clip.bytes


def test_another_video_cannot_silently_take_over_a_lecture(client, stub_models, clip, tmp_path):
    stub_models.whisper.scripts.append(OS_SCRIPT)
    preprocess.main([str(clip.path), "lec"])
    other = tmp_path / "other.mp4"
    other.write_bytes(clip.bytes + b"different")

    assert preprocess.main([str(other), "lec"]) == 1
    assert client.get("/media/lec.mp4").content == clip.bytes


@pytest.mark.models
def test_real_models_end_to_end(client, clip, fake_llm):
    """The same journey with the real Whisper and embedding models (downloads them on first run)."""
    if not clip.spoken:
        pytest.skip("this ffmpeg build cannot synthesize speech (no flite filter)")
    store._model.cache_clear()
    transcribe._whisper.cache_clear()

    lecture_id = post_video(client, clip.bytes)
    status = get(client, "/status", lecture_id).json()
    assert status["status"] == "ready", status["error"]

    transcript = get(client, "/transcript", lecture_id).json()
    text = " ".join(c["text"] for c in transcript).lower()
    spoken_words = set(SPEECH.lower().rstrip(".").split())
    assert len(spoken_words & set(text.replace(".", "").replace(",", "").split())) >= len(spoken_words) // 2, text
    assert all(0 <= c["start"] < c["end"] <= clip.seconds + 1 for c in transcript)

    hits = get(client, "/search", lecture_id, q="threads stuck waiting for each other").json()
    assert hits and hits[0] in transcript


def queue_os_timeline(fake_llm, lecture_id):
    sources = {c.start: c for c in pipeline.load_segments(lecture_id)}
    fake_llm.reply_timeline("groq", *((title, sources[start]) for title, start in [("Intro", 0), ("Deadlock", 31), ("Semaphores", 64)]))

"""scripts.preprocess. Whisper and embeddings are faked; cache, uploads and Chroma are temp dirs."""
import pytest

from app import config
from app.ingestion import pipeline, store, transcribe
from scripts import preprocess

SEGMENTS = [
    {"start": 0.0, "end": 31.0, "text": "Gradient descent follows the slope downhill."},
    {"start": 31.0, "end": 75.0, "text": "The learning rate sets the step size."},
]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(store, "_embed", lambda texts: [[float("rate" in t), 1.0] for t in texts])


@pytest.fixture
def whisper(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "transcribe", lambda path: calls.append(path) or SEGMENTS)
    return calls


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "My Lecture.mp4"
    path.write_bytes(b"video")
    return path


def test_preprocess_copies_transcribes_and_indexes(whisper, video, capsys):
    assert preprocess.main([str(video), "ml-lecture"]) == 0
    target = config.UPLOAD_DIR / "ml-lecture.mp4"
    assert target.read_bytes() == b"video"
    assert whisper == [target]
    assert len(pipeline.load_transcript("ml-lecture")) == 2
    assert store.query("ml-lecture", "learning rate", k=1)[0].start == 31.0
    assert pipeline.get_status("ml-lecture") == {"status": "ready"}
    out = capsys.readouterr().out
    assert "ml-lecture: 2 chunks" in out and "0:31  The learning rate" in out
    assert "/media/ml-lecture.mp4" in out and "build_timelines ml-lecture" in out


def test_second_run_reuses_the_transcript(whisper, video, capsys):
    preprocess.main([str(video), "ml-lecture"])
    assert preprocess.main([str(video), "ml-lecture"]) == 0
    assert len(whisper) == 1
    assert "only re-indexing" in capsys.readouterr().out
    assert len(store.get_chunks("ml-lecture")) == 2


def test_force_transcribes_again(whisper, video):
    preprocess.main([str(video), "ml-lecture"])
    assert preprocess.main([str(video), "ml-lecture", "--force"]) == 0
    assert len(whisper) == 2
    assert len(store.get_chunks("ml-lecture")) == 2


def test_video_already_in_uploads_is_not_copied_onto_itself(whisper):
    config.UPLOAD_DIR.mkdir(parents=True)
    target = config.UPLOAD_DIR / "lec.mp4"
    target.write_bytes(b"video")
    assert preprocess.main([str(target), "lec"]) == 0
    assert target.read_bytes() == b"video"


def test_missing_video_fails(whisper, tmp_path, capsys):
    assert preprocess.main([str(tmp_path / "nope.mp4"), "lec"]) == 1
    assert "no such file" in capsys.readouterr().out
    assert whisper == [] and not config.CACHE_DIR.exists()


@pytest.mark.parametrize("bad", ["../x", "a b", "a/b"])
def test_invalid_lecture_id_fails(whisper, video, bad, capsys):
    assert preprocess.main([str(video), bad]) == 1
    assert "invalid id" in capsys.readouterr().out
    assert whisper == [] and not config.UPLOAD_DIR.exists()


def test_transcription_failure_is_reported(video, monkeypatch, capsys):
    def boom(path):
        raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    assert preprocess.main([str(video), "lec"]) == 1
    assert "lec: FAILED (ffmpeg not found)" in capsys.readouterr().out
    assert pipeline.get_status("lec")["status"] == "failed"


# --- --reindex ---

def write_transcript(lecture_id, texts):
    chunks = [{"lecture_id": lecture_id, "start": i * 30.0, "end": i * 30.0 + 30, "text": t} for i, t in enumerate(texts)]
    pipeline._write_json(pipeline.transcript_path(lecture_id), chunks)


def test_reindex_rebuilds_the_vector_db_from_cache(whisper, capsys):
    write_transcript("os-lecture", ["threads", "locks", "the rate monotonic scheduler"])
    write_transcript("bio-lecture", ["cells"])
    assert store.get_chunks("os-lecture") == []  # fresh clone: transcripts but no vector DB
    assert preprocess.main(["--reindex"]) == 0
    assert [c.text for c in store.get_chunks("os-lecture")] == ["threads", "locks", "the rate monotonic scheduler"]
    assert [c.text for c in store.get_chunks("bio-lecture")] == ["cells"]
    assert whisper == []
    out = capsys.readouterr().out
    assert "os-lecture: 3 chunks indexed" in out and "bio-lecture: 1 chunks indexed" in out


def test_reindex_ignores_other_cache_files(capsys):
    write_transcript("lec", ["a"])
    (config.CACHE_DIR / "lec.timeline.json").write_text("[]")
    (config.CACHE_DIR / "lec.status.json").write_text('{"status": "ready"}')
    assert preprocess.main(["--reindex"]) == 0
    assert capsys.readouterr().out.count("indexed") == 1


def test_reindex_twice_does_not_duplicate():
    write_transcript("lec", ["a", "b"])
    preprocess.main(["--reindex"])
    preprocess.main(["--reindex"])
    assert len(store.get_chunks("lec")) == 2


def test_reindex_with_empty_cache(capsys):
    assert preprocess.main(["--reindex"]) == 0
    assert "no cached transcripts" in capsys.readouterr().out


def test_reindex_reports_a_broken_transcript_and_continues(capsys):
    write_transcript("good", ["a"])
    config.CACHE_DIR.joinpath("bad.transcript.json").write_text("{not json")
    assert preprocess.main(["--reindex"]) == 1
    out = capsys.readouterr().out
    assert "bad: FAILED" in out and "good: 1 chunks indexed" in out


# --- --model ---

def test_model_overrides_whisper_model_for_the_run(video, monkeypatch):
    monkeypatch.setattr(config, "WHISPER_MODEL", "base")
    used = []
    monkeypatch.setattr(pipeline, "transcribe", lambda path: used.append(config.WHISPER_MODEL) or SEGMENTS)
    assert preprocess.main([str(video), "lec", "--model", "small"]) == 0
    assert used == ["small"]


def test_whisper_model_from_config_by_default(video, monkeypatch):
    monkeypatch.setattr(config, "WHISPER_MODEL", "base")
    used = []
    monkeypatch.setattr(pipeline, "transcribe", lambda path: used.append(config.WHISPER_MODEL) or SEGMENTS)
    preprocess.main([str(video), "lec"])
    assert used == ["base"]


# --- --warmup ---

def test_warmup_loads_both_models(monkeypatch, capsys):
    loaded = []
    monkeypatch.setattr(config, "WHISPER_MODEL", "base")
    monkeypatch.setattr(store, "_embed", lambda texts: loaded.append("embed") or [[0.0]])
    monkeypatch.setattr(transcribe, "_whisper", lambda: loaded.append(config.WHISPER_MODEL))
    monkeypatch.setattr(preprocess.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert preprocess.main(["--warmup", "--model", "tiny"]) == 0
    assert loaded == ["embed", "tiny"]
    out = capsys.readouterr().out
    assert "embedding model all-MiniLM-L6-v2: ready" in out and "whisper model tiny: ready" in out
    assert "ffmpeg: found" in out


def test_warmup_reports_every_failure(monkeypatch, capsys):
    def boom(*args):
        raise OSError("download blocked")

    monkeypatch.setattr(store, "_embed", boom)
    monkeypatch.setattr(transcribe, "_whisper", boom)
    monkeypatch.setattr(preprocess.shutil, "which", lambda name: None)
    assert preprocess.main(["--warmup"]) == 1
    out = capsys.readouterr().out
    assert out.count("FAILED (download blocked)") == 2 and "ffmpeg: FAILED" in out


# --- arguments ---

@pytest.mark.parametrize("argv", [[], ["video.mp4"], ["--reindex", "video.mp4", "lec"], ["--warmup", "x.mp4"], ["--reindex", "--warmup"]])
def test_bad_arguments_exit_with_usage(argv):
    with pytest.raises(SystemExit) as e:
        preprocess.main(argv)
    assert e.value.code == 2

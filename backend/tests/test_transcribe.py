"""Transcription. Whisper is faked (no model download); ffmpeg is real and skipped if missing."""
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import config
from app.ingestion import transcribe as t

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class FakeWhisper:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs, Path(path).exists()))
        return (s for s in self.segments), SimpleNamespace(language="en")


def seg(start, end, text, words=None):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


@pytest.fixture
def whisper(monkeypatch):
    fake = FakeWhisper([seg(0, 4.5, " Hello there."), seg(4.5, 6, "   "), seg(6, 9.25, " Second line. ")])
    monkeypatch.setattr(t, "_whisper", lambda: fake)
    return fake


@pytest.fixture
def video(tmp_path):
    """A 2 s clip with video and stereo 44.1 kHz audio."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2",
         "-ac", "2", "-shortest", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_extract_audio_is_16k_mono(video, tmp_path):
    wav = tmp_path / "out.wav"
    t._extract_audio(video, wav)
    with wave.open(str(wav)) as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 16000)
        assert w.getnframes() / w.getframerate() == pytest.approx(2, abs=0.2)


@needs_ffmpeg
def test_extract_audio_rejects_non_media(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_text("not a video")
    with pytest.raises(RuntimeError, match="could not extract audio from notes.mp4"):
        t._extract_audio(bad, tmp_path / "out.wav")


def test_missing_ffmpeg_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(t.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        t._extract_audio(tmp_path / "a.mp4", tmp_path / "a.wav")


def test_missing_video(whisper, tmp_path):
    with pytest.raises(FileNotFoundError):
        t.transcribe(tmp_path / "nope.mp4")
    assert whisper.calls == []


@needs_ffmpeg
def test_transcribe_returns_clean_segments(whisper, video):
    assert t.transcribe(video) == [
        {"start": 0.0, "end": 4.5, "text": "Hello there.", "words": []},
        {"start": 6.0, "end": 9.25, "text": "Second line.", "words": []},
    ]


@needs_ffmpeg
def test_transcribe_cleans_up_the_wav(whisper, video):
    t.transcribe(video)
    (path, kwargs, existed), = whisper.calls
    assert existed and kwargs == {"vad_filter": True, "word_timestamps": True,
                                  "batch_size": config.WHISPER_BATCH_SIZE, "beam_size": config.WHISPER_BEAM_SIZE}
    assert not Path(path).exists()


@needs_ffmpeg
def test_word_alignment_survives_transcription(whisper, video):
    whisper.segments = [seg(110.66, 115.26, " the type. Then we can allocate.", [
        SimpleNamespace(start=110.66, end=111.0, word=" the type."),
        SimpleNamespace(start=111.3, end=111.64, word=" Then we can"),
        SimpleNamespace(start=111.64, end=111.96, word=" allocate."),
    ])]
    result = t.transcribe(video)
    assert result[0]["words"][2] == {"start": 111.64, "end": 111.96, "word": " allocate."}


@needs_ffmpeg
def test_output_feeds_chunking(whisper, video):
    from app.ingestion.chunking import chunk_segments

    chunks = chunk_segments("lec", t.transcribe(video))
    assert [(c.start, c.end, c.text) for c in chunks] == [(0.0, 9.25, "Hello there. Second line.")]


# --- model loading ---

def test_whisper_is_loaded_once_from_config(monkeypatch):
    import sys
    import types

    from app import config

    made = []

    class FakeWhisperModel:
        def __init__(self, name, device, compute_type):
            made.append((name, device, compute_type))

    class FakePipeline:
        def __init__(self, model):
            self.model = model

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel, BatchedInferencePipeline=FakePipeline),
    )
    monkeypatch.setattr(config, "WHISPER_MODEL", "tiny")
    t._whisper.cache_clear()
    try:
        assert t._whisper() is t._whisper()
        assert made == [("tiny", "cpu", "int8")]
        assert isinstance(t._whisper(), FakePipeline)
    finally:
        t._whisper.cache_clear()


# --- Groq Whisper with local fallback ---

class FakeGroq:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, url, headers, data, files, timeout):
        name, f, mime = files["file"]
        self.calls.append({"url": url, "headers": headers, "data": data, "mime": mime, "bytes": len(f.read())})
        if isinstance(self.reply, Exception):
            raise self.reply
        resp = SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.reply)
        return resp


@pytest.fixture
def groq(monkeypatch):
    import httpx

    def use(reply):
        fake = FakeGroq(reply)
        monkeypatch.setattr(config, "GROQ_WHISPER_API_KEY", "gsk-test")
        monkeypatch.setattr(httpx, "post", fake)
        return fake

    return use


@needs_ffmpeg
def test_groq_used_when_key_set(groq, whisper, video):
    fake = groq({
        "segments": [{"start": 0, "end": 1, "text": " Hi there. "}, {"start": 1, "end": 2.5, "text": " Bye."},
                     {"start": 2.5, "end": 3, "text": " "}],
        "words": [{"word": "Hi", "start": 0, "end": 0.4}, {"word": "there.", "start": 0.4, "end": 0.9},
                  {"word": "Bye.", "start": 1.2, "end": 2}],
    })
    assert t.transcribe(video) == [
        {"start": 0.0, "end": 1.0, "text": "Hi there.", "words": [
            {"start": 0.0, "end": 0.4, "word": " Hi"}, {"start": 0.4, "end": 0.9, "word": " there."}]},
        {"start": 1.0, "end": 2.5, "text": "Bye.", "words": [{"start": 1.2, "end": 2.0, "word": " Bye."}]},
    ]
    (call,) = fake.calls
    assert call["data"]["timestamp_granularities[]"] == ["word", "segment"]
    assert call["headers"] == {"Authorization": "Bearer gsk-test"}
    assert call["data"]["model"] == config.GROQ_WHISPER_MODEL and call["data"]["response_format"] == "verbose_json"
    assert call["mime"] == "audio/mpeg" and 0 < call["bytes"] < 50_000  # 2 s at 32 kbps
    assert whisper.calls == []


@needs_ffmpeg
def test_falls_back_to_local_when_groq_fails(groq, whisper, video):
    import httpx

    groq(httpx.ConnectError("down"))
    assert [s["text"] for s in t.transcribe(video)] == ["Hello there.", "Second line."]
    assert len(whisper.calls) == 1


@needs_ffmpeg
def test_falls_back_to_local_when_audio_too_big(groq, whisper, video, monkeypatch):
    fake = groq({"segments": []})
    monkeypatch.setattr(config, "GROQ_WHISPER_MAX_BYTES", 10)
    assert len(t.transcribe(video)) == 2
    assert fake.calls == [] and len(whisper.calls) == 1


def test_no_key_skips_groq(whisper, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(t, "_transcribe_groq", lambda *a: called.append(a))
    monkeypatch.setattr(t, "_transcribe_local", lambda *a: [])
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert t.transcribe(video) == [] and called == []

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


def seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


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
        {"start": 0.0, "end": 4.5, "text": "Hello there."},
        {"start": 6.0, "end": 9.25, "text": "Second line."},
    ]


@needs_ffmpeg
def test_transcribe_cleans_up_the_wav(whisper, video):
    t.transcribe(video)
    (path, kwargs, existed), = whisper.calls
    assert existed and kwargs == {"vad_filter": True, "batch_size": config.WHISPER_BATCH_SIZE, "beam_size": config.WHISPER_BEAM_SIZE}
    assert not Path(path).exists()


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

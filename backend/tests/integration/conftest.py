"""Fixtures for integration tests.

Third-party models never load here: StubWhisperModel and StubSentenceTransformer stand in
for the faster_whisper and sentence_transformers classes. Everything else that a test
integrates is the real code: ffmpeg, Chroma, the cache on disk, the FastAPI app.
"""
import re
import shutil
import subprocess
import sys
import types
import wave
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from app import config
from app.ingestion import store, transcribe
from app.main import app

SPEECH = "A deadlock happens when two threads wait on each other forever."
DIM = 64


class StubSentenceTransformer:
    """Bag-of-words hashing: texts that share words get close vectors."""

    loaded = []  # model names, one per construction

    def __init__(self, name):
        self.loaded.append(name)

    def encode(self, texts, normalize_embeddings=False):
        out = np.zeros((len(texts), DIM))
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % DIM] += 1.0
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


class StubWhisperModel:
    """Returns scripted segments and records the audio file it was handed."""

    loaded = []  # (name, device, compute_type) per construction
    scripts = []  # one list of (start, end, text) per expected transcribe call
    heard = []  # {"channels", "rate", "seconds"} per transcribe call

    def __init__(self, name, device=None, compute_type=None):
        self.loaded.append((name, device, compute_type))

    def transcribe(self, path, vad_filter=False):
        with wave.open(path) as w:
            self.heard.append({
                "channels": w.getnchannels(),
                "rate": w.getframerate(),
                "seconds": w.getnframes() / w.getframerate(),
                "vad_filter": vad_filter,
            })
        script = self.scripts.pop(0)
        return (SimpleNamespace(start=s, end=e, text=t) for s, e, t in script), SimpleNamespace(language="en")


@pytest.fixture
def stub_models(monkeypatch):
    """Replace the two model libraries; the loaders in transcribe.py and store.py run for real."""
    for cls in (StubSentenceTransformer, StubWhisperModel):
        cls.loaded = []
    StubWhisperModel.scripts, StubWhisperModel.heard = [], []
    monkeypatch.setattr(config, "WHISPER_MODEL", config.WHISPER_MODEL)  # scripts' --model must not leak
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=StubSentenceTransformer))
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=StubWhisperModel))
    store._model.cache_clear()
    transcribe._whisper.cache_clear()
    yield SimpleNamespace(whisper=StubWhisperModel, embedder=StubSentenceTransformer)
    store._model.cache_clear()  # don't leak stub instances into other tests
    transcribe._whisper.cache_clear()


@pytest.fixture(autouse=True)
def uploads(monkeypatch, tmp_path):
    """Temp upload dir, also served at /media (the app mounted the real one at import)."""
    directory = tmp_path / "uploads"
    directory.mkdir()
    monkeypatch.setattr(config, "UPLOAD_DIR", directory)
    media = next(r for r in app.routes if getattr(r, "name", None) == "media").app
    monkeypatch.setattr(media, "directory", directory)
    monkeypatch.setattr(media, "all_directories", [directory])
    return directory


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _ffmpeg(*args):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True)


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A short real video file with an audio track. `clip.spoken` says whether it contains SPEECH."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("clip") / "lecture.mp4"
    video = ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=5"]
    spoken = _ffmpeg("-f", "lavfi", "-i", f"flite=text='{SPEECH}':voice=slt", *video, "-shortest", str(path)).returncode == 0
    if not spoken:  # this ffmpeg build has no speech synthesizer: a tone still exercises the audio path
        result = _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=4", *video, "-shortest", str(path))
        assert result.returncode == 0, result.stderr.decode()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return SimpleNamespace(path=path, bytes=path.read_bytes(), spoken=spoken, seconds=float(probe.stdout))


# Scripted "lectures" for the Whisper stub. With 30 s chunks OS_SCRIPT becomes three chunks
# (0-31, 31-64, 64-80) and BIO_SCRIPT one (0-45).
OS_SCRIPT = [
    (0.0, 12.0, " Welcome to operating systems."),
    (12.0, 31.0, " Today we cover concurrency."),
    (31.0, 48.0, " A deadlock happens when two threads wait on each other forever."),
    (48.0, 64.0, " Four conditions are needed, including circular wait."),
    (64.0, 80.0, " A semaphore counts available resources."),
]
OS_CHUNKS = [
    (0.0, 31.0, "Welcome to operating systems. Today we cover concurrency."),
    (31.0, 64.0, "A deadlock happens when two threads wait on each other forever. Four conditions are needed, including circular wait."),
    (64.0, 80.0, "A semaphore counts available resources."),
]
BIO_SCRIPT = [
    (0.0, 20.0, " Photosynthesis turns light into chemical energy."),
    (20.0, 45.0, " Chlorophyll absorbs red and blue light."),
]
OS_TIMELINE_REPLY = '{"sections": [{"title": "Intro", "start": 0}, {"title": "Deadlock", "start": 31}, {"title": "Semaphores", "start": 64}]}'


def segments(script):
    """A script in the shape transcribe() returns."""
    return [{"start": s, "end": e, "text": t.strip()} for s, e, t in script]


def post_video(client, content=b"fake video bytes", name="lecture.mp4"):
    r = client.post("/upload", files={"file": (name, content, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["lecture_id"]


def get(client, path, lecture_id, **params):
    return client.get(path, params={"lecture_id": lecture_id, **params})

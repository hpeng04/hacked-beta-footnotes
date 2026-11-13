"""Person 1: video -> transcript -> chunks -> vector DB, with the transcript cached on disk."""
import json
import os
import tempfile
from pathlib import Path

from .. import config
from ..schemas import Chunk
from . import store
from .chunking import chunk_segments
from .transcribe import transcribe

PROCESSING, READY, FAILED = "processing", "ready", "failed"


def _path(lecture_id: str, kind: str) -> Path:
    if not config.is_valid_lecture_id(lecture_id):
        raise ValueError(f"invalid lecture id: {lecture_id!r}")  # never build paths from bad ids
    return config.CACHE_DIR / f"{lecture_id}.{kind}.json"


def transcript_path(lecture_id: str) -> Path:
    return _path(lecture_id, "transcript")


def status_path(lecture_id: str) -> Path:
    return _path(lecture_id, "status")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
        os.replace(tmp, path)  # atomic: readers never see a half-written file
    except BaseException:
        os.unlink(tmp)
        raise


def load_transcript(lecture_id: str) -> list[Chunk] | None:
    """Cached chunks of a lecture, or None if it hasn't been transcribed."""
    path = transcript_path(lecture_id)
    return [Chunk(**c) for c in json.loads(path.read_text())] if path.exists() else None


def get_status(lecture_id: str) -> dict | None:
    """{"status": "processing" | "ready" | "failed", "error"?: str}, or None for an unknown lecture."""
    path = status_path(lecture_id)
    return json.loads(path.read_text()) if path.exists() else None


def ingest(video_path: Path, lecture_id: str) -> list[Chunk]:
    """Transcribe, chunk and index a lecture. A cached transcript skips Whisper and is only re-indexed.

    Records progress in the status file; on failure stores the error there and re-raises.
    """
    status = status_path(lecture_id)  # validates the id before anything is written
    _write_json(status, {"status": PROCESSING})
    try:
        chunks = load_transcript(lecture_id)
        if chunks is None:
            chunks = chunk_segments(lecture_id, transcribe(Path(video_path)))
            _write_json(transcript_path(lecture_id), [c.model_dump() for c in chunks])
        store.add_chunks(chunks)
    except Exception as e:
        _write_json(status, {"status": FAILED, "error": str(e) or type(e).__name__})
        raise
    _write_json(status, {"status": READY})
    return chunks

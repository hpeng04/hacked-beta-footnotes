"""Person 1: video -> transcript -> chunks -> vector DB, with the transcript cached on disk."""
import json
import os
import tempfile
import threading
from pathlib import Path

from .. import config
from ..schemas import Chunk
from . import store
from .chunking import chunk_segments
from .transcribe import transcribe

PROCESSING, READY, FAILED = "processing", "ready", "failed"
_timing_locks: dict[str, threading.Lock] = {}
_timing_locks_guard = threading.Lock()


def _path(lecture_id: str, kind: str) -> Path:
    if not config.is_valid_lecture_id(lecture_id):
        raise ValueError(f"invalid lecture id: {lecture_id!r}")  # never build paths from bad ids
    return config.CACHE_DIR / f"{lecture_id}.{kind}.json"


def transcript_path(lecture_id: str) -> Path:
    return _path(lecture_id, "transcript")


def status_path(lecture_id: str) -> Path:
    return _path(lecture_id, "status")


def segments_path(lecture_id: str) -> Path:
    return _path(lecture_id, "segments")


def load_segments(lecture_id: str) -> list[Chunk] | None:
    """Original Whisper segments, before merging them into search chunks."""
    path = segments_path(lecture_id)
    return [Chunk(**s) for s in json.loads(path.read_text())] if path.exists() else None


def _segments(lecture_id: str, video_path: Path) -> list[Chunk]:
    return [Chunk(lecture_id=lecture_id, **s) for s in transcribe(video_path) if s["text"].strip()]


def _timing_lock(lecture_id: str):
    with _timing_locks_guard:
        return _timing_locks.setdefault(lecture_id, threading.Lock())


def citation_segments(lecture_id: str) -> list[Chunk] | None:
    """Recover missing segment timing once for recordings ingested by older versions.

    Chunk text alone cannot tell us when a sentence was spoken. Reuse the saved
    recording, without changing the searchable transcript, vector index or status.
    Mock/portable transcripts without a recording retain their available timing.
    """
    segments = load_segments(lecture_id)
    if segments is not None or not config.UPLOAD_DIR.is_dir():
        return segments
    videos = sorted(p for p in config.UPLOAD_DIR.iterdir() if p.is_file() and p.stem == lecture_id)
    if not videos or load_transcript(lecture_id) is None:
        return None
    with _timing_lock(lecture_id):
        segments = load_segments(lecture_id)
        if segments is None:
            segments = _segments(lecture_id, videos[0])
            _write_json(segments_path(lecture_id), [s.model_dump() for s in segments])
        return segments


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


def set_status(lecture_id: str, status: str, error: str | None = None) -> None:
    _write_json(status_path(lecture_id), {"status": status, **({"error": error} if error else {})})


def _save_transcript(lecture_id: str, chunks: list[Chunk], previous: list[Chunk] | None) -> None:
    _write_json(transcript_path(lecture_id), [c.model_dump() for c in chunks])
    if chunks != previous:  # the cached timeline describes the old transcript
        _path(lecture_id, "timeline").unlink(missing_ok=True)


def ingest(video_path: Path, lecture_id: str, force: bool = False) -> list[Chunk]:
    """Transcribe, chunk and index a lecture. A cached transcript skips Whisper and is only re-indexed.

    force transcribes again even if cached. The old transcript and timeline stay until the new
    chunks are indexed, so a failed replacement leaves the lecture as it was.
    Records progress in the status file; on failure stores the error there and re-raises.
    """
    set_status(lecture_id, PROCESSING)  # also validates the id before anything else is written
    try:
        cached = load_transcript(lecture_id)
        if cached is not None and not force:
            chunks = cached
            store.add_chunks(chunks)
        else:
            segments = _segments(lecture_id, Path(video_path))
            chunks = chunk_segments(lecture_id, [s.model_dump(exclude={"lecture_id"}) for s in segments])
            if cached is None:  # nothing to protect: cache first, so a retry skips Whisper
                _save_transcript(lecture_id, chunks, cached)
                _write_json(segments_path(lecture_id), [s.model_dump() for s in segments])
                store.add_chunks(chunks)
            else:  # replacing a working lecture: touch its cache only once indexing worked
                store.add_chunks(chunks)
                _save_transcript(lecture_id, chunks, cached)
                _write_json(segments_path(lecture_id), [s.model_dump() for s in segments])
    except Exception as e:
        set_status(lecture_id, FAILED, str(e) or type(e).__name__)
        raise
    set_status(lecture_id, READY)
    return chunks

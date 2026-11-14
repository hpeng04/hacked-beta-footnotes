"""Person 1: upload, transcript and status endpoints."""
import logging
import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel

from .. import config, mock
from ..schemas import Chunk, UploadResponse
from . import pipeline

router = APIRouter()
log = logging.getLogger(__name__)


class StatusResponse(BaseModel):
    lecture_id: str
    status: Literal["processing", "ready", "failed"]
    error: str | None = None  # set when status is "failed"


def _ingest(video_path: Path, lecture_id: str) -> None:
    try:
        pipeline.ingest(video_path, lecture_id)
    except Exception:  # already recorded in the status file; don't crash the worker
        log.exception("ingestion failed for lecture %s", lecture_id)


def _chunks(lecture_id: str) -> list[Chunk]:
    """Real transcript if the lecture was ingested, else mock data (lecture "demo" and the samples)."""
    if not config.is_valid_lecture_id(lecture_id):
        return []
    cached = pipeline.load_transcript(lecture_id)
    return cached if cached is not None else mock.load_chunks(lecture_id)


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile, background: BackgroundTasks):
    lecture_id = uuid.uuid4().hex[:8]
    name = lecture_id + Path(file.filename or "").suffix
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / name
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    pipeline.set_status(lecture_id, pipeline.PROCESSING)  # so /status answers before the task starts
    background.add_task(_ingest, path, lecture_id)  # transcribing is slow: runs after the response
    return UploadResponse(lecture_id=lecture_id, video_url=f"/media/{name}")


@router.get("/status", response_model=StatusResponse)
def status(lecture_id: str):
    """Poll after /upload until status is "ready" (or "failed")."""
    if config.is_valid_lecture_id(lecture_id):
        if state := pipeline.get_status(lecture_id):
            return StatusResponse(lecture_id=lecture_id, **state)
        if _chunks(lecture_id):  # mock or pre-processed lecture without a status file
            return StatusResponse(lecture_id=lecture_id, status="ready")
    raise HTTPException(404, f"Unknown lecture: {lecture_id}")


@router.get("/transcript", response_model=list[Chunk])
def transcript(lecture_id: str):
    return _chunks(lecture_id)

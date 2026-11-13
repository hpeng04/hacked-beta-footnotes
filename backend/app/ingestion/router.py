"""Person 1: upload and transcript endpoints."""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile

from .. import mock
from ..config import UPLOAD_DIR
from ..schemas import Chunk, UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile):
    lecture_id = uuid.uuid4().hex[:8]
    name = lecture_id + Path(file.filename or "").suffix
    with (UPLOAD_DIR / name).open("wb") as out:
        shutil.copyfileobj(file.file, out)
    # TODO: transcribe -> chunk_segments -> store.add_chunks (run in the background)
    return UploadResponse(lecture_id=lecture_id, video_url=f"/media/{name}")


@router.get("/transcript", response_model=list[Chunk])
def transcript(lecture_id: str):
    # TODO: read the real chunks from the cache or vector DB
    return mock.load_chunks(lecture_id)

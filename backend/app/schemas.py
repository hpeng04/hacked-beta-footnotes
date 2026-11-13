"""Shared contract. Change it only after telling the whole team."""
from pydantic import BaseModel


class Chunk(BaseModel):
    lecture_id: str
    start: float  # seconds
    end: float
    text: str


class UploadResponse(BaseModel):
    lecture_id: str
    video_url: str


class AskRequest(BaseModel):
    lecture_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Chunk]


class Section(BaseModel):
    title: str
    start: float
    end: float

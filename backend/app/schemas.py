"""Shared contract. Change it only after telling the whole team."""
from pydantic import BaseModel, Field


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


class AnswerSentence(BaseModel):
    text: str
    sources: list[Chunk]


class AskResponse(BaseModel):
    answer: str
    sources: list[Chunk]
    sentences: list[AnswerSentence] = Field(default_factory=list)


class Section(BaseModel):
    title: str
    start: float
    end: float

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
    section_ids: list[str] = Field(default_factory=list)  # citations (by section_id) backing this sentence


class ReplayCitation(BaseModel):
    section_id: str
    title: str
    start: float
    end: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Chunk]
    sentences: list[AnswerSentence] = Field(default_factory=list)
    citations: list[ReplayCitation] | None = None


class Section(BaseModel):
    title: str
    start: float
    end: float

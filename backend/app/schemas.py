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
    citation_ids: list[str] = Field(default_factory=list)  # ReplayCitation.id values backing this sentence


class ReplayCitation(BaseModel):
    id: str  # unique within one answer; AnswerSentence.citation_ids refer to it
    section_id: str | None  # the detected topic section, or None when sections are unavailable
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

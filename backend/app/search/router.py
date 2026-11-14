"""Person 2: search, Q&A and timeline endpoints."""
from fastapi import APIRouter, HTTPException

from ..config import is_valid_lecture_id
from ..schemas import AskRequest, AskResponse, Chunk, Section
from . import llm, retrieval, timeline as timeline_mod

router = APIRouter()


def _require_text(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(422, f"{field} must not be empty")
    return value


def _require_lecture(lecture_id: str) -> None:
    if not is_valid_lecture_id(lecture_id) or not retrieval.transcript(lecture_id):
        raise HTTPException(404, f"Unknown lecture: {lecture_id}")


@router.get("/search", response_model=list[Chunk])
def search(lecture_id: str, q: str):
    q = _require_text(q, "q")
    _require_lecture(lecture_id)
    return retrieval.search(lecture_id, q)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = _require_text(req.question, "question")
    _require_lecture(req.lecture_id)
    sources = retrieval.search(req.lecture_id, question)
    sources = retrieval.answer_sources(req.lecture_id, sources)
    return llm.answer_result(question, sources)


@router.get("/timeline", response_model=list[Section])
def timeline(lecture_id: str):
    sections = timeline_mod.build(lecture_id)  # served from cache even without a transcript
    if not sections:
        raise HTTPException(404, f"Unknown lecture: {lecture_id}")
    return sections

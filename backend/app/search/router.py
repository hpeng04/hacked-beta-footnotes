"""Person 2: search, Q&A and timeline endpoints."""
from fastapi import APIRouter, HTTPException

from ..config import is_valid_lecture_id
from ..schemas import AskRequest, AskResponse, Chunk, Section
from . import llm, retrieval, timeline as timeline_mod
from . import presentation

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
    result = llm.answer_result(question, sources, preserve_evidence=True)
    return presentation.with_citations(req.lecture_id, result)


@router.get("/timeline", response_model=list[Section])
def timeline(lecture_id: str):
    index = presentation.section_index(lecture_id)
    sections = ([Section(title=s.title, start=s.start, end=(index.sections[i + 1].start
                 if i + 1 < len(index.sections) else s.end)) for i, s in enumerate(index.sections)]
                if index is not None else timeline_mod.build(lecture_id))
    if not sections:
        raise HTTPException(404, f"Unknown lecture: {lecture_id}")
    return sections

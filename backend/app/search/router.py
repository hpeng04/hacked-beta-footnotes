"""Person 2: search, Q&A and timeline endpoints."""
from fastapi import APIRouter

from ..schemas import AskRequest, AskResponse, Chunk, Section
from . import llm, retrieval, timeline as timeline_mod

router = APIRouter()


@router.get("/search", response_model=list[Chunk])
def search(lecture_id: str, q: str):
    return retrieval.search(lecture_id, q)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    sources = retrieval.search(req.lecture_id, req.question)
    return AskResponse(answer=llm.answer(req.question, sources), sources=sources)


@router.get("/timeline", response_model=list[Section])
def timeline(lecture_id: str):
    return timeline_mod.build(lecture_id)

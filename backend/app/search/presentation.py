"""Resolve display links independently of the sentence evidence used by the answer."""
import logging

from ..ingestion.timing import source_id
from ..schemas import ReplayCitation
from . import llm, segmentation
from .titles import context_title

log = logging.getLogger(__name__)


def section_index(lecture_id):
    try:
        # Never build inside a request: until sections exist, callers use their fallbacks.
        return segmentation.ready(lecture_id)
    except (LookupError, ValueError, RuntimeError, OSError):
        log.warning("Section membership unavailable; using existing timing fallback")
        return None


def with_citations(lecture_id, answer):
    if not answer.sources:
        answer.citations = []
        return answer
    index = section_index(lecture_id)
    membership = index.sentence_membership() if index else {}
    if index and all(source_id(s) in membership for s in answer.sources):
        topics = {s.id: s for s in index.sections}
        grouped = {}
        for source in sorted(answer.sources, key=lambda s: s.start):
            section_id = membership[source_id(source)]
            grouped.setdefault(section_id, source)
        answer.citations = [ReplayCitation(section_id=key, title=topics[key].title,
                                          start=source.start, end=topics[key].end)
                            for key, source in grouped.items()]
    else:
        _, passages = llm.group_citations(answer.sentences, answer.sources)
        answer.citations = [ReplayCitation(section_id=source_id(p), title=context_title([p]), start=p.start, end=p.end)
                            for p in passages]
    # Evidence stays intact; plain-text clients also receive deduplicated links.
    answer.answer = " ".join(s.text for s in answer.sentences) + " " + " ".join(
        f"[{llm.fmt_time(c.start)}]" for c in answer.citations)
    return answer

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
        owner = {source_id(s): membership[source_id(s)] for s in answer.sources}
        grouped = {}
        for source in sorted(answer.sources, key=lambda s: s.start):
            grouped.setdefault(owner[source_id(source)], source)
        answer.citations = [ReplayCitation(id=key, section_id=key, title=topics[key].title,
                                          start=source.start, end=topics[key].end)
                            for key, source in grouped.items()]
    else:
        _, passages = llm.group_citations(answer.sentences, answer.sources)
        # No section to point at: the link is identified by its passage instead.
        answer.citations = [ReplayCitation(id=source_id(p), section_id=None, title=context_title([p]),
                                          start=p.start, end=p.end) for p in passages]
        # Passages are consecutive runs of sources by start time: a source belongs to the last one it follows.
        owner = {source_id(s): next(c.id for c in reversed(answer.citations) if c.start <= s.start)
                 for s in answer.sources}
    order = {c.id: i for i, c in enumerate(answer.citations)}
    for sentence in answer.sentences:
        # Lets clients mark which link supports which sentence, in the order the links are listed.
        sentence.citation_ids = sorted({owner[source_id(s)] for s in sentence.sources}, key=order.__getitem__)
    # Evidence stays intact; plain-text clients also receive deduplicated links. Only real
    # sections are named: a fallback title would just repeat the transcript.
    answer.answer = " ".join(s.text for s in answer.sentences) + " " + " ".join(
        f"[{llm.fmt_time(c.start)} {c.title}]" if c.section_id else f"[{llm.fmt_time(c.start)}]"
        for c in answer.citations)
    return answer

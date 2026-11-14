"""Section membership storage. Topic detection and UI integration are separate stages."""
import hashlib
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .. import config
from ..ingestion import timing
from ..ingestion.pipeline import _write_json
from ..schemas import Chunk

CACHE_VERSION = 1


class Sentence(Chunk):
    model_config = ConfigDict(extra="forbid")
    id: str

    def chunk(self) -> Chunk:
        return Chunk(**self.model_dump(exclude={"id"}))


class SectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)
    sentence_ids: list[str] = Field(min_length=1)


class TopicSection(SectionDraft):
    id: str
    start: float
    end: float


def _section_id(lecture_id: str, sentence_ids: list[str]) -> str:
    # Retitling a section doesn't change its identity; changing membership does.
    payload = json.dumps([lecture_id, sentence_ids], separators=(",", ":"))
    return "sec_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class SectionIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lecture_id: str
    sentences: list[Sentence]
    sections: list[TopicSection]
    fallback: bool = False  # refinement was requested but incomplete; worth retrying later

    @model_validator(mode="after")
    def validate_membership(self):
        if not config.is_valid_lecture_id(self.lecture_id):
            raise ValueError("invalid lecture ID")
        by_id = {}
        previous_start = -1.0
        for sentence in self.sentences:
            if sentence.lecture_id != self.lecture_id:
                raise ValueError("sentence belongs to another lecture")
            if (not math.isfinite(sentence.start) or not math.isfinite(sentence.end)
                    or not 0 <= sentence.start < sentence.end or not sentence.text.strip()):
                raise ValueError("invalid sentence timing or text")
            if sentence.start < previous_start:
                raise ValueError("sentences must be in playback order")
            if sentence.id != timing.source_id(sentence.chunk()) or sentence.id in by_id:
                raise ValueError("invalid or duplicate sentence ID")
            previous_start = sentence.start
            by_id[sentence.id] = sentence

        assigned = [sid for section in self.sections for sid in section.sentence_ids]
        if assigned != list(by_id):
            raise ValueError("sections must assign every sentence exactly once, in playback order")
        previous_end = -1.0
        for section in self.sections:
            members = [by_id[sid] for sid in section.sentence_ids]
            start, end = members[0].start, max(s.end for s in members)
            if (not section.title.strip() or section.id != _section_id(self.lecture_id, section.sentence_ids)
                    or section.start != start or section.end != end):
                raise ValueError("section identity and bounds must match its sentences")
            if start < previous_end:
                raise ValueError("sections must not overlap")
            previous_end = end
        return self

    def sentence_membership(self) -> dict[str, str]:
        return {sid: section.id for section in self.sections for sid in section.sentence_ids}


def build_index(lecture_id: str, sentences: list[Chunk], drafts: list[SectionDraft]) -> SectionIndex:
    """Accept proposed memberships; derive IDs and bounds rather than trusting a model."""
    records = [Sentence(**s.model_dump(), id=timing.source_id(s)) for s in sentences]
    by_id = {s.id: s for s in records}
    sections = []
    for draft in drafts:
        if not draft.sentence_ids or any(sid not in by_id for sid in draft.sentence_ids):
            raise ValueError("section references unknown or missing sentences")
        members = [by_id[sid] for sid in draft.sentence_ids]
        sections.append(TopicSection(
            id=_section_id(lecture_id, draft.sentence_ids), title=draft.title.strip(),
            start=members[0].start, end=max(s.end for s in members), sentence_ids=draft.sentence_ids,
        ))
    return SectionIndex(lecture_id=lecture_id, sentences=records, sections=sections)


def cache_path(lecture_id: str) -> Path:
    if not config.is_valid_lecture_id(lecture_id):
        raise ValueError("invalid lecture ID")
    return config.CACHE_DIR / f"{lecture_id}.sections.json"


def stored_settings(lecture_id: str) -> dict | None:
    """The segmentation settings a cache was built with, so readers can accept any build."""
    try:
        data = json.loads(cache_path(lecture_id).read_text())
    except (FileNotFoundError, ValueError):
        return None
    settings = data.get("segmentation") if isinstance(data, dict) else None
    return settings if isinstance(settings, dict) else None


def _fingerprint(lecture_id: str, sentences: list[Chunk], segmentation: dict[str, JsonValue]) -> str:
    if any(not isinstance(segmentation.get(key), str) or not segmentation[key].strip()
           for key in ("algorithm", "revision")):
        raise ValueError("segmentation settings must include the algorithm and its revision")
    payload = {"lecture_id": lecture_id, "sentences": [s.model_dump() for s in sentences],
               "sentence_version": timing.SENTENCE_EXCERPT_VERSION, "segmentation": segmentation}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_index(index: SectionIndex, *, segmentation: dict[str, JsonValue]) -> None:
    # Revalidate even models mutated after construction. Invalid writes preserve the old cache.
    index = SectionIndex.model_validate(index.model_dump())
    fingerprint = _fingerprint(index.lecture_id, [s.chunk() for s in index.sentences], segmentation)
    _write_json(cache_path(index.lecture_id), {
        "version": CACHE_VERSION, "fingerprint": fingerprint,
        "segmentation": segmentation, "index": index.model_dump(),
    })


def load_index(lecture_id: str, sentences: list[Chunk], *, segmentation: dict[str, JsonValue]) -> SectionIndex | None:
    """None means missing, stale or corrupt; the caller must regenerate sections.

    Callers supply the current sentence snapshot and all segmentation settings,
    including algorithm/model revisions. No transcription or model call occurs here.
    """
    path = cache_path(lecture_id)
    expected = _fingerprint(lecture_id, sentences, segmentation)
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION or data.get("fingerprint") != expected:
            return None
        index = SectionIndex.model_validate(data["index"])
        if index.lecture_id != lecture_id or [s.chunk() for s in index.sentences] != sentences:
            return None
        if data.get("segmentation") != segmentation:
            return None
        return index
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None

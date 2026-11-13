"""Person 2: topic timeline. One LLM call per lecture, cached."""
import json

from .. import config, mock
from ..ingestion import store
from ..schemas import Chunk, Section
from . import llm

MAX_TRANSCRIPT_CHARS = 24_000  # keeps the prompt under free-tier token limits
FALLBACK_SECONDS = 300
MAX_TITLE_CHARS = 60

TIMELINE_PROMPT = """Split this lecture transcript into 3 to 10 topic sections, in order.
Each line starts with the time in seconds when it begins.

Return JSON only: {{"sections": [{{"title": "...", "start": <seconds>}}, ...]}}
- "start" must be one of the line times below. The first section starts at {first}.
- "title": 1 to 4 words naming the topic.

Transcript:
{transcript}"""


def cache_path(lecture_id: str):
    return config.CACHE_DIR / f"{lecture_id}.timeline.json"


def build(lecture_id: str) -> list[Section]:
    path = cache_path(lecture_id)
    if path.exists():
        return [Section(**s) for s in json.loads(path.read_text())]

    chunks = _load_chunks(lecture_id)
    if not chunks:
        return []
    try:
        sections = _parse(llm.complete(_prompt(chunks), json=True), chunks)
    except (llm.LLMError, ValueError, TypeError, KeyError):
        return _fixed_sections(chunks)  # not cached, so the next call retries the LLM

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.model_dump() for s in sections], indent=2))
    return sections


def _load_chunks(lecture_id: str) -> list[Chunk]:
    # store.get_chunks is Person 1's (agreed, not written yet); mock data until then
    get_chunks = getattr(store, "get_chunks", None)
    chunks = []
    if get_chunks:
        try:
            chunks = get_chunks(lecture_id)
        except NotImplementedError:
            pass
    return sorted(chunks or mock.load_chunks(lecture_id), key=lambda c: c.start)


def _prompt(chunks: list[Chunk]) -> str:
    per_chunk = max(80, MAX_TRANSCRIPT_CHARS // len(chunks))
    transcript = "\n".join(f"{c.start:.0f} {c.text[:per_chunk]}" for c in chunks)
    return TIMELINE_PROMPT.format(first=f"{chunks[0].start:.0f}", transcript=transcript)


def _parse(reply: dict, chunks: list[Chunk]) -> list[Section]:
    """Turn the LLM's starts into contiguous sections covering the whole lecture."""
    lecture_start, lecture_end = chunks[0].start, chunks[-1].end
    starts = {}
    for item in reply["sections"]:
        start = min(max(float(item["start"]), lecture_start), lecture_end)
        title = str(item["title"]).strip()[:MAX_TITLE_CHARS]
        if title and start < lecture_end:
            starts.setdefault(start, title)  # first title wins on duplicate starts
    if not starts:
        raise ValueError("no usable sections")

    ordered = sorted(starts.items())
    ordered[0] = (lecture_start, ordered[0][1])  # no gap before the first section
    ends = [s for s, _ in ordered[1:]] + [lecture_end]
    return [Section(title=t, start=s, end=e) for (s, t), e in zip(ordered, ends)]


def _fixed_sections(chunks: list[Chunk]) -> list[Section]:
    start, end = chunks[0].start, chunks[-1].end
    sections, t = [], start
    while t < end:
        sections.append(Section(title=f"Part {len(sections) + 1}", start=t, end=min(t + FALLBACK_SECONDS, end)))
        t += FALLBACK_SECONDS
    return sections

"""Person 2: topic timeline. One LLM call per lecture, cached."""
import json

from .. import config
from ..schemas import Chunk, Section
from . import llm, retrieval

MAX_TRANSCRIPT_CHARS = 24_000  # keeps the prompt under free-tier token limits
FALLBACK_SECONDS = 300
MIN_LINE_CHARS = 100  # merge chunks rather than cut lines shorter than this
MAX_TITLE_CHARS = 60
MIN_SECTIONS, MAX_SECTIONS = 3, 10

TIMELINE_PROMPT = """Split this lecture transcript into {min_sections} to {max_sections} topic sections, in order.
Group related lines: a section usually spans several lines.
Each line starts with the time in seconds when it begins.

Return JSON only: {{"sections": [{{"title": "...", "start": <seconds>}}, ...]}}
- "start" must be one of the line times below. The first section starts at {first}.
- "title": 1 to 4 words naming the topic.

Transcript:
{transcript}"""


def cache_path(lecture_id: str):
    return config.CACHE_DIR / f"{lecture_id}.timeline.json"


LLM_FAILURES = (llm.LLMError, ValueError, TypeError, KeyError)


def build(lecture_id: str) -> list[Section]:
    """Cached timeline, else generate one. Never raises: falls back to fixed sections."""
    path = cache_path(lecture_id)
    if path.exists():
        return [Section(**s) for s in json.loads(path.read_text())]

    chunks = retrieval.transcript(lecture_id)
    if not chunks:
        return []
    try:
        return _generate(lecture_id, chunks)
    except LLM_FAILURES:
        return _fixed_sections(chunks)  # not cached, so the next call retries the LLM


def rebuild(lecture_id: str) -> list[Section]:
    """Generate and cache a fresh timeline, ignoring the cache.

    Raises LookupError if the lecture has no transcript, or one of LLM_FAILURES.
    On failure the existing cache file is left untouched.
    """
    chunks = retrieval.transcript(lecture_id)
    if not chunks:
        raise LookupError(f"no transcript for {lecture_id}")
    return _generate(lecture_id, chunks)


def generate(chunks: list[Chunk]) -> list[Section]:
    """One LLM call, no cache. Raises one of LLM_FAILURES."""
    return _parse(llm.complete(_prompt(chunks), json=True), chunks)


def _generate(lecture_id: str, chunks: list[Chunk]) -> list[Section]:
    sections = generate(chunks)
    path = cache_path(lecture_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([s.model_dump() for s in sections], indent=2))
    tmp.replace(path)  # atomic: readers never see a half-written file
    return sections


def _prompt(chunks: list[Chunk]) -> str:
    return TIMELINE_PROMPT.format(min_sections=MIN_SECTIONS, max_sections=MAX_SECTIONS, first=f"{chunks[0].start:.0f}", transcript=_transcript_text(chunks))


def _transcript_text(chunks: list[Chunk]) -> str:
    """Timestamped transcript lines, at most MAX_TRANSCRIPT_CHARS in total."""
    # Too many chunks: merge neighbours so each line keeps a useful amount of text.
    n = len(chunks)
    n_lines = min(n, MAX_TRANSCRIPT_CHARS // MIN_LINE_CHARS)
    groups = [chunks[i * n // n_lines:(i + 1) * n // n_lines] for i in range(n_lines)]  # even split
    lines = [(f"{g[0].start:.0f} ", " ".join(c.text.strip() for c in g)) for g in groups]
    # Share what is left after timestamps and newlines equally between lines.
    overhead = sum(len(prefix) + 1 for prefix, _ in lines)
    per_line = (MAX_TRANSCRIPT_CHARS - overhead) // len(lines)
    return "\n".join(prefix + text[:per_line] for prefix, text in lines)


def _parse(reply: dict, chunks: list[Chunk]) -> list[Section]:
    """Turn the LLM's starts into contiguous sections covering the whole lecture."""
    lecture_start, lecture_end = chunks[0].start, chunks[-1].end
    chunk_starts = [c.start for c in chunks]
    starts = {}
    for item in reply["sections"]:
        start = min(max(float(item["start"]), lecture_start), lecture_end)
        title = str(item["title"]).strip()[:MAX_TITLE_CHARS]
        if title and start < lecture_end:
            start = min(chunk_starts, key=lambda c: abs(c - start))  # models round times
            starts.setdefault(start, title)  # first title wins on duplicate starts
    if not starts:
        raise ValueError("no usable sections")

    ordered = sorted(starts.items())
    ordered[0] = (lecture_start, ordered[0][1])  # no gap before the first section
    ends = [s for s, _ in ordered[1:]] + [lecture_end]
    return _merge_to_max([Section(title=t, start=s, end=e) for (s, t), e in zip(ordered, ends)])


def _merge_to_max(sections: list[Section]) -> list[Section]:
    """Small models often make one section per line: merge the shortest into a neighbour."""
    sections = list(sections)
    while len(sections) > MAX_SECTIONS:
        i = min(range(len(sections)), key=lambda k: sections[k].end - sections[k].start)
        if i == 0:
            j = 1
        elif i == len(sections) - 1:
            j = i - 1
        else:  # merge into the shorter neighbour, keeping sections balanced
            prev, nxt = sections[i - 1], sections[i + 1]
            j = i - 1 if prev.end - prev.start <= nxt.end - nxt.start else i + 1
        a, b = sorted((i, j))
        keep = max((sections[a], sections[b]), key=lambda x: x.end - x.start)  # longer one names it
        sections[a:b + 1] = [Section(title=keep.title, start=sections[a].start, end=sections[b].end)]
    return sections


def _fixed_sections(chunks: list[Chunk]) -> list[Section]:
    start, end = chunks[0].start, chunks[-1].end
    sections, t = [], start
    while t < end:
        sections.append(Section(title=f"Part {len(sections) + 1}", start=t, end=min(t + FALLBACK_SECONDS, end)))
        t += FALLBACK_SECONDS
    return sections

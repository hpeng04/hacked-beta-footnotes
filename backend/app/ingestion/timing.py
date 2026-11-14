"""Internal transcription metadata; search chunks keep the public Chunk contract."""
import hashlib
import math
import re

from pydantic import BaseModel, Field

from ..schemas import Chunk

SENTENCE_EXCERPT_VERSION = 1  # bump when sentence construction rules change


class Word(BaseModel):
    start: float
    end: float
    word: str


class TimedSegment(Chunk):
    words: list[Word] = Field(default_factory=list)

    def chunk(self) -> Chunk:
        return Chunk(**self.model_dump(exclude={"words"}))


def source_id(chunk: Chunk) -> str:
    """Stable identity independent of retrieval order and rounded display time."""
    key = f"{chunk.lecture_id}\0{chunk.start}\0{chunk.end}\0{chunk.text}"
    return "s_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def sentence_excerpts(segments: list[TimedSegment]) -> list[Chunk]:
    """Join aligned words across segments; never interpolate missing word timing.

    Punctuation ends an excerpt. Long unpunctuated speech is bounded to keep
    citations useful. A silence over two seconds starts a new excerpt.
    Legacy segments without usable alignment retain their original boundaries.
    """
    result: list[Chunk] = []
    pending: list[Word] = []
    lecture_id = segments[0].lecture_id if segments else ""

    def flush():
        if pending:
            result.append(Chunk(lecture_id=lecture_id, start=pending[0].start,
                                end=pending[-1].end, text="".join(w.word for w in pending).strip()))
            pending.clear()

    for segment in sorted(segments, key=lambda s: s.start):
        words = segment.words
        aligned_text = "".join(w.word for w in words).strip()
        valid = bool(words) and aligned_text == segment.text.strip() and all(
            math.isfinite(w.start) and math.isfinite(w.end) and 0 <= w.start < w.end
            and (i == 0 or w.start >= words[i - 1].end)
            for i, w in enumerate(words)
        )
        if not valid:
            flush()
            result.append(segment.chunk())
            continue
        for word in words:
            if pending and (word.start < pending[-1].end or word.start - pending[-1].end > 2
                            or word.end - pending[0].start > 15):
                flush()
            pending.append(word)
            if re.search(r'[.!?。！？]["\u201d\u2019\')\]]*$', word.word.rstrip()):
                flush()
    flush()
    return result

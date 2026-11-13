"""Person 1: merge Whisper segments into search-sized chunks."""
from ..schemas import Chunk


def chunk_segments(lecture_id: str, segments: list[dict], max_seconds: float = 30) -> list[Chunk]:
    """Merge consecutive segments into chunks of about max_seconds.

    A chunk closes with the segment that takes it to max_seconds, so chunks end on
    segment boundaries and may run slightly over. Segments without text are skipped.
    """
    chunks: list[Chunk] = []
    current: list[dict] = []

    def flush():
        if current:
            chunks.append(Chunk(
                lecture_id=lecture_id,
                start=float(current[0]["start"]),
                end=float(current[-1]["end"]),
                text=" ".join(s["text"].strip() for s in current),
            ))
            current.clear()

    for seg in segments:
        if not seg["text"].strip():
            continue
        current.append(seg)
        if seg["end"] - current[0]["start"] >= max_seconds:
            flush()
    flush()
    return chunks

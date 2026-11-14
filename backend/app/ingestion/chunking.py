"""Person 1: merge Whisper segments into search-sized chunks."""
from ..schemas import Chunk


def chunk_segments(lecture_id: str, segments: list[dict], max_seconds: float = 30) -> list[Chunk]:
    chunks = []
    bucket = []

    for seg in segments:
        bucket.append(seg)
        if seg["end"] - bucket[0]["start"] >= max_seconds:
            chunks.append(Chunk(
                lecture_id=lecture_id,
                start=bucket[0]["start"],
                end=bucket[-1]["end"],
                text=" ".join(s["text"].strip() for s in bucket),
            ))
            bucket = []

    if bucket:
        chunks.append(Chunk(
                lecture_id=lecture_id,
                start=bucket[0]["start"],
                end=bucket[-1]["end"],
                text=" ".join(s["text"].strip() for s in bucket),
            ))


    return chunks
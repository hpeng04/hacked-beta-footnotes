"""Person 1: merge Whisper segments into search-sized chunks."""
from ..schemas import Chunk


def chunk_segments(lecture_id: str, segments: list[dict], max_seconds: float = 30) -> list[Chunk]:
    # TODO: merge consecutive segments until a chunk spans about max_seconds
    raise NotImplementedError

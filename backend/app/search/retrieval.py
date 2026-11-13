"""Person 2: retrieval. Keyword matching over mock data until the vector DB is ready."""
from .. import mock
from ..schemas import Chunk


def search(lecture_id: str, query: str, k: int = 5) -> list[Chunk]:
    # TODO: replace with ingestion.store.query
    words = set(query.lower().split())
    scored = [(sum(w in c.text.lower() for w in words), c) for c in mock.load_chunks(lecture_id)]
    return [c for score, c in sorted(scored, key=lambda p: -p[0]) if score > 0][:k]

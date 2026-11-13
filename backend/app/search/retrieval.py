"""Person 2: retrieval. Semantic search via the vector DB, keyword match over mock data as fallback."""
from .. import mock
from ..ingestion import store
from ..schemas import Chunk


def search(lecture_id: str, query: str, k: int = 5) -> list[Chunk]:
    try:
        hits = store.query(lecture_id, query, k)
    except NotImplementedError:  # vector DB not ready yet
        hits = []
    # Empty also covers lectures that only exist as mock data (e.g. "demo")
    return hits or _keyword_search(lecture_id, query, k)


def _keyword_search(lecture_id: str, query: str, k: int) -> list[Chunk]:
    words = set(query.lower().split())
    scored = [(sum(w in c.text.lower() for w in words), c) for c in mock.load_chunks(lecture_id)]
    return [c for score, c in sorted(scored, key=lambda p: -p[0]) if score > 0][:k]

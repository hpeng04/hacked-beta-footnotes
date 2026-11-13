"""Person 2: retrieval. Semantic search via the vector DB, keyword match over mock data as fallback."""
import re

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


def transcript(lecture_id: str) -> list[Chunk]:
    """All chunks of a lecture, in time order. Empty if the lecture is unknown."""
    # store.get_chunks is Person 1's (agreed, not written yet); mock data until then
    get_chunks = getattr(store, "get_chunks", None)
    chunks = []
    if get_chunks:
        try:
            chunks = get_chunks(lecture_id)
        except NotImplementedError:
            pass
    return sorted(chunks or mock.load_chunks(lecture_id), key=lambda c: c.start)


STOPWORDS = set("""a an and are as at be by can do does for from how i in is it of on or so that the
this to was what when where which who why with you""".split())


def _keyword_search(lecture_id: str, query: str, k: int) -> list[Chunk]:
    words = set(re.findall(r"[a-z0-9]+", query.lower())) - STOPWORDS
    scored = [(sum(w in c.text.lower() for w in words), c) for c in mock.load_chunks(lecture_id)]
    return [c for score, c in sorted(scored, key=lambda p: -p[0]) if score > 0][:k]

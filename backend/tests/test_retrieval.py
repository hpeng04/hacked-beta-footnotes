from app.ingestion import store
from app.schemas import Chunk
from app.search import retrieval


def test_finds_matching_chunk():
    hits = retrieval.search("demo", "deadlock")
    assert hits and all("deadlock" in c.text.lower() for c in hits)


def test_best_match_first():
    hits = retrieval.search("demo", "deadlock circular wait")
    assert "circular wait" in hits[0].text


def test_respects_k():
    assert len(retrieval.search("demo", "the a", k=2)) <= 2


def test_no_match_returns_empty():
    assert retrieval.search("demo", "zebra") == []


def test_unknown_lecture_returns_empty():
    assert retrieval.search("nope", "deadlock") == []


# --- vector DB path (store.query is Person 1's; faked here) ---

def test_uses_vector_store_when_available(monkeypatch):
    hit = Chunk(lecture_id="lec1", start=10, end=40, text="semantic hit")
    calls = []
    monkeypatch.setattr(store, "query", lambda lid, q, k: calls.append((lid, q, k)) or [hit])
    assert retrieval.search("lec1", "anything", k=3) == [hit]
    assert calls == [("lec1", "anything", 3)]


def test_falls_back_to_keywords_when_store_empty(monkeypatch):
    monkeypatch.setattr(store, "query", lambda lid, q, k: [])
    hits = retrieval.search("demo", "deadlock")
    assert hits and "deadlock" in hits[0].text.lower()


def test_falls_back_while_store_not_implemented():
    # store.query still raises NotImplementedError until Person 1 lands it
    assert retrieval.search("demo", "mutex")

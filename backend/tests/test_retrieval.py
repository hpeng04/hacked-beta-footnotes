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

"""API contract checks for Person 2's endpoints (see README "Shared contract")."""
CHUNK_KEYS = {"lecture_id", "start", "end", "text"}


def test_search(client):
    r = client.get("/search", params={"lecture_id": "demo", "q": "mutex"})
    assert r.status_code == 200
    body = r.json()
    assert body and set(body[0]) == CHUNK_KEYS


def test_search_requires_params(client):
    assert client.get("/search", params={"lecture_id": "demo"}).status_code == 422


def test_ask(client):
    r = client.post("/ask", json={"lecture_id": "demo", "question": "what is deadlock"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["sources"] and set(body["sources"][0]) == CHUNK_KEYS


def test_ask_no_match(client):
    r = client.post("/ask", json={"lecture_id": "demo", "question": "zebra"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


def test_timeline_shape_and_order(client):
    r = client.get("/timeline", params={"lecture_id": "demo"})
    assert r.status_code == 200
    sections = r.json()
    assert sections and all(set(s) == {"title", "start", "end"} for s in sections)
    for a, b in zip(sections, sections[1:]):
        assert a["end"] == b["start"]  # contiguous, no gaps or overlaps
    assert all(s["start"] < s["end"] for s in sections)

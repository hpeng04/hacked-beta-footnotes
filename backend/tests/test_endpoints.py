"""API contract checks for Person 2's endpoints (see README "Shared contract")."""
import pytest

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


# --- step 6: input guards ---


@pytest.mark.parametrize("q", ["", "   "])
def test_search_rejects_empty_query(client, q):
    r = client.get("/search", params={"lecture_id": "demo", "q": q})
    assert r.status_code == 422


@pytest.mark.parametrize("question", ["", "  \n"])
def test_ask_rejects_empty_question(client, fake_llm, question):
    r = client.post("/ask", json={"lecture_id": "demo", "question": question})
    assert r.status_code == 422
    assert fake_llm.calls == []


def test_ask_strips_question(client, fake_llm):
    fake_llm.reply("groq", "ok")
    client.post("/ask", json={"lecture_id": "demo", "question": "  deadlock  "})
    assert "Question: deadlock\n" in fake_llm.calls[0]["json"]["messages"][0]["content"]


def test_search_unknown_lecture_404(client):
    r = client.get("/search", params={"lecture_id": "nope", "q": "deadlock"})
    assert r.status_code == 404


def test_ask_unknown_lecture_404(client, fake_llm):
    r = client.post("/ask", json={"lecture_id": "nope", "question": "deadlock?"})
    assert r.status_code == 404
    assert fake_llm.calls == []


def test_timeline_unknown_lecture_404(client, fake_llm):
    assert client.get("/timeline", params={"lecture_id": "nope"}).status_code == 404
    assert fake_llm.calls == []


def test_timeline_served_from_cache_without_transcript(client):
    from app.search import timeline
    path = timeline.cache_path("cached-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[{"title": "T", "start": 0, "end": 10}]')
    r = client.get("/timeline", params={"lecture_id": "cached-only"})
    assert r.status_code == 200 and r.json()[0]["title"] == "T"

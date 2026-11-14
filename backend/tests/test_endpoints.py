"""API contract checks for Person 2's endpoints (see README "Shared contract")."""
import pytest

from app.ingestion import store
from app.schemas import Chunk

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


@pytest.fixture
def memory_lecture(monkeypatch):
    chunks = [
        Chunk(lecture_id="c-lecture", start=0, end=31, text="C was created in 1972 and inspired many languages."),
        Chunk(lecture_id="c-lecture", start=31, end=63, text="Variables have a memory address. Pointers store addresses. Free allocated memory to avoid leaks."),
        Chunk(lecture_id="c-lecture", start=63, end=94, text="Install GCC, write a main function, and use printf."),
        Chunk(lecture_id="c-lecture", start=94, end=127, text="Allocate bytes for a string and release the memory with free."),
        Chunk(lecture_id="c-lecture", start=127, end=140, text="Compile your code. Thanks for watching."),
    ]
    monkeypatch.setattr(store, "get_chunks", lambda lid: chunks)
    # A short video can have fewer chunks than the vector query's limit, so all
    # its chunks are retrieved, including those unrelated to the question.
    monkeypatch.setattr(store, "query", lambda lid, q, k: [chunks[i] for i in (1, 3, 0, 4, 2)])
    return chunks


def test_ask_shows_only_cited_parts_of_a_short_video(client, fake_llm, memory_lecture):
    fake_llm.reply("groq", "Variables have a memory address and pointers store addresses [0:31]. Release allocated memory with free [1:34].")
    r = client.post("/ask", json={"lecture_id": "c-lecture", "question": "which part talks about memory?"})
    assert r.status_code == 200
    assert r.json()["sources"] == [memory_lecture[i].model_dump() for i in (1, 3)]
    # Keep the full candidate context available for the model to choose from.
    prompt = fake_llm.calls[0]["json"]["messages"][0]["content"]
    assert all(c.text in prompt for c in memory_lecture)


def test_ask_refusal_does_not_dump_candidate_transcript(client, fake_llm, memory_lecture):
    fake_llm.reply("groq", "The lecture does not cover that.")
    r = client.post("/ask", json={"lecture_id": "c-lecture", "question": "how does photosynthesis work?"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


def test_ask_provider_failure_shows_only_the_fallback_excerpt(client, memory_lecture):
    r = client.post("/ask", json={"lecture_id": "c-lecture", "question": "memory?"})
    assert r.status_code == 200
    assert r.json()["sources"] == [memory_lecture[1].model_dump()]


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


# --- review findings: metadata and unsafe ids are not lectures ---

@pytest.mark.parametrize("lecture_id", ["questions", "sample_questions", "transcript", "timeline", "../timeline", "../mock/transcript"])
def test_non_lectures_return_404(client, fake_llm, lecture_id):
    assert client.get("/search", params={"lecture_id": lecture_id, "q": "test"}).status_code == 404
    assert client.post("/ask", json={"lecture_id": lecture_id, "question": "test"}).status_code == 404
    assert client.get("/timeline", params={"lecture_id": lecture_id}).status_code == 404
    assert fake_llm.calls == []

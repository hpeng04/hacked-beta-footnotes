"""Two requests hitting a cold app at once must not race on creating the Chroma client."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from app.ingestion import store


def test_concurrent_first_requests(client):
    store._client.cache_clear()  # cold: no client for this test's CHROMA_DIR yet
    barrier = Barrier(2)

    def search():
        barrier.wait()
        return client.get("/search", params={"lecture_id": "demo", "q": "deadlock"})

    def ask():
        barrier.wait()
        return client.post("/ask", json={"lecture_id": "demo", "question": "What is deadlock?"})

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(search), pool.submit(ask)]
        assert [f.result().status_code for f in futures] == [200, 200]

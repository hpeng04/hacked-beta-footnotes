"""Vector store. Embeddings are faked (word hashing) so tests need no model download."""
import re
import zlib

import pytest

from app.ingestion import store
from app.schemas import Chunk

DIM = 64
REAL_EMBED = store._embed  # the autouse fixture below replaces it


def fake_embed(texts):
    vectors = []
    for text in texts:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(word.encode()) % DIM] += 1.0
        vectors.append(v)
    return vectors


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(store, "_embed", fake_embed)


def chunk(lecture_id, start, text):
    return Chunk(lecture_id=lecture_id, start=start, end=start + 30, text=text)


A = [
    chunk("a", 60, "A deadlock needs circular wait between threads."),
    chunk("a", 0, "Welcome to the lecture on operating systems."),
    chunk("a", 30, "A mutex protects shared data from race conditions."),
]
B = [chunk("b", 0, "Photosynthesis turns light into chemical energy.")]


def test_get_chunks_sorted_by_start():
    store.add_chunks(A)
    got = store.get_chunks("a")
    assert [c.start for c in got] == [0, 30, 60]
    assert got[1] == A[2]  # text, lecture_id and float timestamps round-trip


def test_query_returns_nearest_first():
    store.add_chunks(A)
    hits = store.query("a", "deadlock circular wait", k=2)
    assert len(hits) == 2
    assert "deadlock" in hits[0].text


def test_lectures_are_isolated():
    store.add_chunks(A + B)
    assert all(c.lecture_id == "b" for c in store.query("b", "deadlock mutex", k=5))
    assert store.get_chunks("b") == B


def test_re_add_replaces_lecture():
    store.add_chunks(A)
    store.add_chunks(A[:1])
    assert store.get_chunks("a") == A[:1]


def test_re_add_leaves_other_lectures():
    store.add_chunks(A + B)
    store.add_chunks(A[:1])
    assert store.get_chunks("b") == B


def test_unknown_lecture_is_empty():
    store.add_chunks(A)
    assert store.query("nope", "deadlock") == []
    assert store.get_chunks("nope") == []


def test_unknown_lecture_does_not_embed(monkeypatch):
    monkeypatch.setattr(store, "_embed", lambda texts: pytest.fail("embedded for an unknown lecture"))
    assert store.query("nope", "deadlock") == []


def test_k_larger_than_lecture():
    store.add_chunks(B)
    assert store.query("b", "light", k=5) == B


def test_add_nothing_is_a_noop():
    store.add_chunks([])
    assert store.get_chunks("a") == []


# --- model loading (the real model is never downloaded in unit tests) ---

class FakeSentenceTransformer:
    made = []

    def __init__(self, name):
        self.made.append(name)

    def encode(self, texts, normalize_embeddings=False):
        import numpy as np

        self.normalize = normalize_embeddings
        return np.ones((len(texts), 3))


@pytest.fixture
def fake_library(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(store, "_embed", REAL_EMBED)
    FakeSentenceTransformer.made = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    store._model.cache_clear()
    yield
    store._model.cache_clear()


def test_model_is_loaded_once_with_the_configured_name(fake_library):
    assert store._model() is store._model()
    assert FakeSentenceTransformer.made == ["all-MiniLM-L6-v2"]


def test_embed_returns_plain_normalized_lists(fake_library):
    vectors = store._embed(["a", "b"])
    assert vectors == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert type(vectors[0][0]) is float  # Chroma gets plain floats, not numpy types
    assert store._model().normalize is True

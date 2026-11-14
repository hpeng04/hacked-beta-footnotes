"""Person 1 writes, Person 2 reads: embeddings and the vector DB."""
import threading
from functools import lru_cache

from .. import config
from ..schemas import Chunk

COLLECTION = "chunks"
_lock = threading.Lock()  # endpoints run in a threadpool; Chroma client creation isn't thread-safe


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # slow import, only when first needed

    return SentenceTransformer(config.EMBEDDING_MODEL)


def _embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=None)
def _client(path: str):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def _collection():
    # CHROMA_DIR is read on every call so tests can point it at a temp dir
    with _lock:
        client = _client(str(config.CHROMA_DIR))
        return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _to_chunk(text: str, meta: dict) -> Chunk:
    return Chunk(lecture_id=meta["lecture_id"], start=meta["start"], end=meta["end"], text=text)


def add_chunks(chunks: list[Chunk]) -> None:
    """Embed and store chunks. Replaces whatever was stored for the same lecture(s)."""
    if not chunks:
        return
    col = _collection()
    by_lecture: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_lecture.setdefault(c.lecture_id, []).append(c)
    for lecture_id, group in by_lecture.items():
        embeddings = _embed([c.text for c in group])  # before the delete: a failure keeps the old rows
        col.delete(where={"lecture_id": lecture_id})
        col.add(
            ids=[f"{lecture_id}:{i}" for i in range(len(group))],
            embeddings=embeddings,
            documents=[c.text for c in group],
            metadatas=[{"lecture_id": lecture_id, "start": float(c.start), "end": float(c.end)} for c in group],
        )


def query(lecture_id: str, text: str, k: int = 5) -> list[Chunk]:
    """The k chunks of this lecture nearest to text, best first. Empty if the lecture isn't stored."""
    col = _collection()
    where = {"lecture_id": lecture_id}
    if not col.get(where=where, limit=1, include=[])["ids"]:
        return []  # unknown lecture: skip loading the embedding model
    res = col.query(query_embeddings=_embed([text]), n_results=k, where=where, include=["documents", "metadatas"])
    return [_to_chunk(d, m) for d, m in zip(res["documents"][0], res["metadatas"][0])]


def get_chunks(lecture_id: str) -> list[Chunk]:
    """All chunks of a lecture in time order. Empty if the lecture isn't stored."""
    res = _collection().get(where={"lecture_id": lecture_id}, include=["documents", "metadatas"])
    chunks = [_to_chunk(d, m) for d, m in zip(res["documents"], res["metadatas"])]
    return sorted(chunks, key=lambda c: c.start)

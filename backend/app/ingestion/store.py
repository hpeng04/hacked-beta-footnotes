"""Person 1 writes, Person 2 reads: embeddings and the vector DB."""
import chromadb
from sentence_transformers import SentenceTransformer

from .. import config
from ..schemas import Chunk

_model = SentenceTransformer(config.EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_collection = _client.get_or_create_collection("chunks")


def add_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    texts = [c.text for c in chunks]
    embeddings = _model.encode(texts).tolist()
    _collection.upsert(
        ids=[f"{c.lecture_id}-{i}" for i, c in enumerate(chunks)],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"lecture_id": c.lecture_id, "start": c.start, "end": c.end} for c in chunks],
    )


def query(lecture_id: str, text: str, k: int = 5) -> list[Chunk]:
    embedding = _model.encode([text]).tolist()
    res = _collection.query(
        query_embeddings=embedding,
        n_results=k,
        where={"lecture_id": lecture_id},
    )
    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append(Chunk(lecture_id=meta["lecture_id"], start=meta["start"], end=meta["end"], text=doc))
    return chunks
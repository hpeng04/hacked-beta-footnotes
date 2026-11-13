"""Person 1 writes, Person 2 reads: embeddings and the vector DB."""
from ..schemas import Chunk


def add_chunks(chunks: list[Chunk]) -> None:
    # TODO: embed with sentence-transformers (config.EMBEDDING_MODEL), store in Chroma (config.CHROMA_DIR)
    raise NotImplementedError


def query(lecture_id: str, text: str, k: int = 5) -> list[Chunk]:
    # TODO: embed the query and return the k nearest chunks for this lecture
    raise NotImplementedError

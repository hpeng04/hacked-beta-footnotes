"""Person 1 writes, Person 2 reads: embeddings and the vector DB."""
import chromadb
from sentence_transformers import SentenceTransformer

from .. import config
from ..schemas import Chunk

_model = SentenceTransformer(config.EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_collection = _client.get_or_create_collection("chunks")


def add_chunks(chunks: list[Chunk]) -> None:
    # TODO next
    raise NotImplementedError


def query(lecture_id: str, text: str, k: int = 5) -> list[Chunk]:
    # TODO later
    raise NotImplementedError
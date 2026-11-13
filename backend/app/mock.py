"""Mock data so search and frontend work before ingestion is done."""
import json

from .config import MOCK_DIR
from .schemas import Chunk, Section


def load_chunks(lecture_id: str) -> list[Chunk]:
    sample = MOCK_DIR / "samples" / f"{lecture_id}.json"  # sample lectures for live LLM testing
    path = sample if sample.exists() else MOCK_DIR / "transcript.json"
    return [Chunk(**c) for c in json.loads(path.read_text()) if c["lecture_id"] == lecture_id]


def load_timeline(lecture_id: str) -> list[Section]:
    path = MOCK_DIR / "timeline.json"
    return [Section(**s) for s in json.loads(path.read_text())]

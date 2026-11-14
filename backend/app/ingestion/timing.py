"""Internal transcription metadata; search chunks keep the public Chunk contract."""
from pydantic import BaseModel, Field

from ..schemas import Chunk


class Word(BaseModel):
    start: float
    end: float
    word: str


class TimedSegment(Chunk):
    words: list[Word] = Field(default_factory=list)

    def chunk(self) -> Chunk:
        return Chunk(**self.model_dump(exclude={"words"}))

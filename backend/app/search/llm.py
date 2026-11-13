"""Person 2: LLM calls."""
from ..schemas import Chunk


def answer(question: str, sources: list[Chunk]) -> str:
    # TODO: prompt the LLM with the question and sources, cite timestamps in the answer
    if not sources:
        return "Nothing in this lecture matches that question."
    return f"(placeholder answer) See {sources[0].start:.0f}s: {sources[0].text}"

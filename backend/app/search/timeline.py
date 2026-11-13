"""Person 2: topic timeline. One LLM call per lecture, cached."""
from .. import mock
from ..schemas import Section


def build(lecture_id: str) -> list[Section]:
    # TODO: return config.CACHE_DIR/<lecture_id>.timeline.json if present,
    # else send the transcript to the LLM, parse the sections and cache them
    return mock.load_timeline(lecture_id)

from app.ingestion.chunking import chunk_segments
from app.schemas import Chunk


def seg(start, end, text="word"):
    return {"start": start, "end": end, "text": text}


def test_empty_input():
    assert chunk_segments("l", []) == []


def test_merges_until_max_seconds():
    segments = [seg(0, 10, "a"), seg(10, 20, "b"), seg(20, 30, "c"), seg(30, 40, "d")]
    assert chunk_segments("l", segments, max_seconds=30) == [
        Chunk(lecture_id="l", start=0, end=30, text="a b c"),
        Chunk(lecture_id="l", start=30, end=40, text="d"),
    ]


def test_remainder_is_flushed():
    chunks = chunk_segments("l", [seg(0, 5, "a"), seg(5, 9, "b")], max_seconds=30)
    assert chunks == [Chunk(lecture_id="l", start=0, end=9, text="a b")]


def test_long_segment_is_its_own_chunk():
    chunks = chunk_segments("l", [seg(0, 5, "a"), seg(5, 90, "long"), seg(90, 95, "b")], max_seconds=30)
    assert [(c.start, c.end, c.text) for c in chunks] == [(0, 90, "a long"), (90, 95, "b")]


def test_skips_empty_segments_and_strips_text():
    segments = [seg(0, 5, "  hello "), seg(5, 10, "   "), seg(10, 15, ""), seg(15, 20, " world")]
    chunks = chunk_segments("l", segments, max_seconds=30)
    assert chunks == [Chunk(lecture_id="l", start=0, end=20, text="hello world")]


def test_empty_segment_does_not_set_chunk_start():
    chunks = chunk_segments("l", [seg(0, 5, ""), seg(5, 10, "a")])
    assert chunks[0].start == 5


def test_chunks_cover_all_text_in_order():
    segments = [seg(i * 4, i * 4 + 4, f"w{i}") for i in range(50)]
    chunks = chunk_segments("lec", segments, max_seconds=30)
    assert " ".join(c.text for c in chunks) == " ".join(s["text"] for s in segments)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))
    assert all(c.lecture_id == "lec" and isinstance(c.start, float) for c in chunks)


def test_gap_between_segments_counts_toward_span():
    # silence between segments still counts: the chunk spans 0-40 on the video
    chunks = chunk_segments("l", [seg(0, 5, "a"), seg(35, 40, "b"), seg(40, 45, "c")], max_seconds=30)
    assert [(c.start, c.end) for c in chunks] == [(0, 40), (40, 45)]

from app.ingestion.timing import TimedSegment, sentence_excerpts, source_id
from app.schemas import Chunk


def segment(words):
    return TimedSegment(lecture_id="sample", start=words[0][0], end=words[-1][1],
                        text="".join(w[2] for w in words).strip(),
                        words=[{"start": s, "end": e, "word": t} for s, e, t in words])


def test_sentence_boundary_inside_segment_uses_actual_word_start():
    excerpts = sentence_excerpts([segment([
        (110.66, 111.0, " the type."), (111.3, 111.38, " Then"),
        (111.38, 111.64, " we can"), (111.64, 113.08, " allocate four bytes to it."),
    ])])
    assert [(s.start, s.end, s.text) for s in excerpts] == [
        (110.66, 111.0, "the type."), (111.3, 113.08, "Then we can allocate four bytes to it."),
    ]


def test_sentence_crosses_whisper_segment_boundary():
    a = segment([(117.78, 119.66, " When you no longer need that memory allocated to")])
    b = segment([(119.66, 122.92, " your program, use free to release it back to RAM.")])
    result = sentence_excerpts([a, b])
    assert len(result) == 1
    assert result[0].start == 117.78 and result[0].end == 122.92
    assert "allocated to your program" in result[0].text


def test_silence_and_unpunctuated_speech_bound_excerpts():
    result = sentence_excerpts([segment([
        (0, 1, " first"), (4, 5, " second"), (5, 19, " third"), (19, 20, " fourth"),
    ])])
    assert [s.start for s in result] == [0, 4, 19]


def test_missing_or_invalid_alignment_keeps_segment_timing():
    legacy = TimedSegment(lecture_id="sample", start=94, end=124, text="One. Two.")
    invalid = segment([(124, 125, " Three."), (124.5, 126, " Four.")])
    assert sentence_excerpts([legacy, invalid]) == [legacy.chunk(), invalid.chunk()]


def test_word_text_mismatch_never_silently_drops_transcript():
    s = segment([(0, 1, " partial")])
    s.text = "A full sentence with more words."
    assert sentence_excerpts([s]) == [s.chunk()]


def test_source_ids_distinguish_same_second_and_survive_reordering():
    a = Chunk(lecture_id="sample", start=1.1, end=1.4, text="Yes.")
    b = Chunk(lecture_id="sample", start=1.5, end=1.9, text="No.")
    assert source_id(a) != source_id(b)
    assert source_id(a) == source_id(a.model_copy())

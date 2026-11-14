import json

import pytest

from app.ingestion import pipeline, timing
from app.schemas import Chunk
from app.search import sections

SETTINGS = {"algorithm": "manual", "revision": "1", "threshold": 0.4}


@pytest.fixture
def sentences():
    return [Chunk(lecture_id="sample", start=s, end=e, text=t) for s, e, t in [
        (51.56, 54.46, "Variables receive memory."),
        (54.46, 57.48, "Pointers store addresses."),
        (57.48, 60.0, "Free unused memory."),
        (111.3, 113.08, "Allocate four bytes."),
    ]]


def drafts(sentences):
    ids = [timing.source_id(s) for s in sentences]
    return [sections.SectionDraft(title="Memory management", sentence_ids=ids[:3]),
            sections.SectionDraft(title="Allocation", sentence_ids=ids[3:])]


def test_builds_exact_membership_and_preserves_float_boundaries(sentences):
    index = sections.build_index("sample", sentences, drafts(sentences))
    assert [(s.start, s.end) for s in index.sections] == [(51.56, 60), (111.3, 113.08)]
    membership = index.sentence_membership()
    assert {membership[timing.source_id(s)] for s in sentences[:3]} == {index.sections[0].id}
    assert membership[timing.source_id(sentences[-1])] == index.sections[1].id
    assert [s.chunk() for s in index.sentences] == sentences


def test_ids_survive_roundtrip_and_retitling_but_change_with_membership(sentences):
    proposed = drafts(sentences)
    first = sections.build_index("sample", sentences, proposed)
    proposed[0].title = "Pointers"
    renamed = sections.build_index("sample", sentences, proposed)
    assert [s.id for s in first.sections] == [s.id for s in renamed.sections]
    proposed[1].sentence_ids.insert(0, proposed[0].sentence_ids.pop())
    changed = sections.build_index("sample", sentences, proposed)
    assert first.sections[0].id != changed.sections[0].id
    sections.save_index(first, segmentation=SETTINGS)
    assert sections.load_index("sample", sentences, segmentation=SETTINGS) == first


@pytest.mark.parametrize("groups", [
    [[0, 1], [3]],          # missing sentence
    [[0, 1, 2], [2, 3]],   # duplicate assignment
    [[0, 2], [1, 3]],      # interleaved sections
    [[3], [0, 1, 2]],      # reversed sections
    [[1, 0, 2], [3]],      # reversed members
])
def test_rejects_invalid_membership(sentences, groups):
    ids = [timing.source_id(s) for s in sentences]
    proposed = [sections.SectionDraft(title="Topic", sentence_ids=[ids[i] for i in g]) for g in groups]
    with pytest.raises(ValueError, match="exactly once"):
        sections.build_index("sample", sentences, proposed)


def test_rejects_unknown_ids_and_empty_sections(sentences):
    with pytest.raises(ValueError, match="unknown"):
        sections.build_index("sample", sentences, [sections.SectionDraft(title="Topic", sentence_ids=["s_missing"])])
    with pytest.raises(ValueError):
        sections.SectionDraft(title="Topic", sentence_ids=[])


def test_rejects_sections_with_overlapping_time_ranges(sentences):
    sentences[2].end = 112.0
    with pytest.raises(ValueError, match="overlap"):
        sections.build_index("sample", sentences, drafts(sentences))


@pytest.mark.parametrize("change", ["lecture", "order", "duplicate", "nan", "negative", "empty"])
def test_rejects_invalid_sentence_snapshots(sentences, change):
    if change == "lecture":
        sentences[0].lecture_id = "other"
    elif change == "order":
        sentences[0], sentences[1] = sentences[1], sentences[0]
    elif change == "duplicate":
        sentences[1] = sentences[0]
    elif change == "nan":
        sentences[0].start = float("nan")
    elif change == "negative":
        sentences[0].start = -1
    else:
        sentences[0].text = " "
    with pytest.raises(ValueError):
        sections.build_index("sample", sentences, drafts(sentences))


def test_silent_recording_has_empty_valid_index():
    index = sections.build_index("silent", [], [])
    sections.save_index(index, segmentation=SETTINGS)
    assert sections.load_index("silent", [], segmentation=SETTINGS) == index
    assert index.sentence_membership() == {}


@pytest.mark.parametrize("change", ["text", "timing", "settings", "revision", "format", "sentences"])
def test_changes_invalidate_cache(sentences, monkeypatch, change):
    index = sections.build_index("sample", sentences, drafts(sentences))
    sections.save_index(index, segmentation=SETTINGS)
    settings = dict(SETTINGS)
    if change == "text":
        sentences[0].text = "Corrected transcript."
    elif change == "timing":
        sentences[0].start += 0.1
    elif change == "settings":
        settings["threshold"] = 0.5
    elif change == "revision":
        settings["revision"] = "2"
    elif change == "format":
        monkeypatch.setattr(sections, "CACHE_VERSION", 2)
    else:
        monkeypatch.setattr(timing, "SENTENCE_EXCERPT_VERSION", 2)
    assert sections.load_index("sample", sentences, segmentation=settings) is None


@pytest.mark.parametrize("bad", ["not json", "[]", '{"version": 0}', '{"version": 1}'])
def test_missing_and_obsolete_caches_request_regeneration(sentences, bad):
    assert sections.load_index("sample", sentences, segmentation=SETTINGS) is None
    path = sections.cache_path("sample")
    path.parent.mkdir(parents=True)
    path.write_text(bad)
    assert sections.load_index("sample", sentences, segmentation=SETTINGS) is None


@pytest.mark.parametrize("field,value", [("start", 0), ("end", 120), ("id", "sec_invalid"),
                                         ("sentence_ids", []), ("title", " ")])
def test_corrupt_membership_cache_is_not_served(sentences, field, value):
    index = sections.build_index("sample", sentences, drafts(sentences))
    sections.save_index(index, segmentation=SETTINGS)
    path = sections.cache_path("sample")
    data = json.loads(path.read_text())
    data["index"]["sections"][0][field] = value
    path.write_text(json.dumps(data))
    assert sections.load_index("sample", sentences, segmentation=SETTINGS) is None


def test_invalid_replacement_preserves_cached_index(sentences):
    index = sections.build_index("sample", sentences, drafts(sentences))
    sections.save_index(index, segmentation=SETTINGS)
    before = sections.cache_path("sample").read_bytes()
    index.sections[0].sentence_ids.pop()
    with pytest.raises(ValueError):
        sections.save_index(index, segmentation=SETTINGS)
    assert sections.cache_path("sample").read_bytes() == before


def test_failed_atomic_write_preserves_cache_and_cleans_temp_file(sentences, monkeypatch):
    index = sections.build_index("sample", sentences, drafts(sentences))
    sections.save_index(index, segmentation=SETTINGS)
    path = sections.cache_path("sample")
    before = path.read_bytes()

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(pipeline.os, "replace", fail)
    with pytest.raises(OSError):
        sections.save_index(index, segmentation=SETTINGS)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_rejects_unsafe_paths_and_unspecified_algorithm(sentences):
    with pytest.raises(ValueError):
        sections.cache_path("../escape")
    with pytest.raises(ValueError):
        sections.load_index("sample", sentences, segmentation={})

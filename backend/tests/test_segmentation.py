import json as jsonlib
import threading
import time

import pytest

from app import config
from app.ingestion.timing import source_id
from app.schemas import Chunk
from app.search import llm, sections, segmentation as seg
from scripts import build_sections


@pytest.fixture
def lecture(monkeypatch):
    sentences = [Chunk(lecture_id="lec", start=i * 5, end=(i + 1) * 5,
                       text=f"Sentence {i}.") for i in range(8)]
    vectors = [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4
    monkeypatch.setattr(seg.store, "_embed", lambda texts: vectors)
    monkeypatch.setattr(seg.retrieval, "timeline_sources", lambda lid: sentences if lid == "lec" else None)
    return sentences, vectors


def decisions(prompt, json=False):
    records = jsonlib.loads(prompt.split("Candidates:\n")[1])
    return {"decisions": [{"candidate_id": r["candidate_id"], "start_id": r["candidate_id"],
                           "title": "Memory" if r["opening"] else "Compilation"} for r in records]}


def test_detects_topic_change_and_keeps_all_sentences(lecture):
    sentences, vectors = lecture
    assert seg.semantic_starts(sentences, vectors, seg.Options()) == [0, 4]
    index = seg.generate("lec", sentences, refine=False)
    assert [s.start for s in index.sections] == [0, 20]
    assert len(index.sentence_membership()) == 8


def test_continuous_topic_and_silence_do_not_force_sections(lecture):
    sentences, _ = lecture
    sentences[-1] = sentences[-1].model_copy(update={"start": 90, "end": 95})
    assert seg.semantic_starts(sentences, [[1.0, 0.0]] * 8, seg.Options()) == [0]


def test_recurring_topic_remains_separate(lecture):
    sentences, vectors = lecture
    sentences += [Chunk(lecture_id="lec", start=i * 5, end=(i + 1) * 5, text=f"Back to memory {i}.") for i in range(8, 12)]
    assert seg.semantic_starts(sentences, vectors + [[1.0, 0.0]] * 4, seg.Options()) == [0, 4, 8]


def test_short_topic_dips_are_suppressed(lecture):
    sentences, _ = lecture
    vectors = [[1.0, 0.0]] * 3 + [[0.0, 1.0]] + [[1.0, 0.0]] * 4
    assert seg.semantic_starts(sentences, vectors, seg.Options()) == [0]
    assert seg.semantic_starts(sentences, vectors, seg.Options(min_seconds=21)) == [0]


def test_overlap_never_becomes_a_section_boundary(lecture):
    sentences, vectors = lecture
    sentences[0].end = 35
    assert seg.semantic_starts(sentences, vectors, seg.Options()) == [0]


def test_refinement_uses_ids_and_original_float_timing(lecture, monkeypatch):
    sentences, _ = lecture
    sentences[3].end = 20.25
    sentences[4].start = 20.25
    monkeypatch.setattr(llm, "complete", decisions)
    index = seg.generate("lec", sentences)
    assert [(s.title, s.start) for s in index.sections] == [("Memory", 0), ("Compilation", 20.25)]


def test_refinement_can_remove_or_move_candidate(lecture, monkeypatch):
    sentences, _ = lecture
    def refine(prompt, json=False):
        result = decisions(prompt)
        result["decisions"][1]["start_id"] = source_id(sentences[5])
        return result
    monkeypatch.setattr(llm, "complete", refine)
    assert seg.generate("lec", sentences).sections[1].start == 25
    def merge(prompt, json=False):
        result = decisions(prompt)
        result["decisions"][1]["start_id"] = None
        return result
    monkeypatch.setattr(llm, "complete", merge)
    assert len(seg.generate("lec", sentences).sections) == 1


@pytest.mark.parametrize("bad", [None, {}, {"decisions": []}, {"decisions": "bad"}])
def test_invalid_model_output_keeps_semantic_boundaries(lecture, monkeypatch, bad):
    sentences, _ = lecture
    monkeypatch.setattr(llm, "complete", lambda *a, **k: bad)
    index = seg.generate("lec", sentences)
    assert [(s.title, s.start) for s in index.sections] == [("Part 1", 0), ("Part 2", 20)]


@pytest.mark.parametrize("error", ["invented", "missing", "duplicate", "dense", "opening"])
def test_bad_boundary_ids_never_change_membership(lecture, monkeypatch, error):
    sentences, _ = lecture
    def invalid(prompt, json=False):
        result = decisions(prompt)
        if error == "invented":
            result["decisions"][1]["start_id"] = "s_unknown"
        elif error == "missing":
            result["decisions"][1].pop("start_id")
        elif error == "duplicate":
            result["decisions"][1] = result["decisions"][0]
        elif error == "dense":
            result["decisions"][1]["start_id"] = source_id(sentences[1])
        else:
            result["decisions"][0]["start_id"] = None
        return result
    monkeypatch.setattr(llm, "complete", invalid)
    assert [s.start for s in seg.generate("lec", sentences).sections] == [0, 20]


def test_cache_reuses_fallback_until_explicit_retry(lecture, monkeypatch):
    calls = []
    def down(*args, **kwargs):
        calls.append(1)
        raise llm.LLMError("offline")
    monkeypatch.setattr(llm, "complete", down)
    first = seg.build("lec")
    monkeypatch.setattr(seg.store, "_embed", lambda *a: pytest.fail("cache should skip embeddings"))
    assert seg.build("lec") == first and len(calls) == 1
    monkeypatch.setattr(seg.store, "_embed", lambda texts: lecture[1])
    monkeypatch.setattr(llm, "complete", decisions)
    assert seg.build("lec", force=True).sections[0].title == "Memory"


def test_settings_and_provider_changes_rebuild(lecture, monkeypatch):
    calls = []
    def complete(prompt, json=False):
        calls.append(1)
        return decisions(prompt)
    monkeypatch.setattr(llm, "complete", complete)
    seg.build("lec")
    seg.build("lec", options=seg.Options(window=2))
    monkeypatch.setattr(config, "LLM_MODEL", "another-model")
    seg.build("lec", options=seg.Options(window=2))
    assert len(calls) == 3


def test_failed_embeddings_preserve_existing_cache(lecture, monkeypatch):
    seg.build("lec", refine=False)
    path = sections.cache_path("lec")
    previous = path.read_bytes()
    monkeypatch.setattr(seg.store, "_embed", lambda texts: [])
    with pytest.raises(ValueError):
        seg.build("lec", refine=False, force=True)
    assert path.read_bytes() == previous


@pytest.mark.parametrize("vectors", [[], [[float("nan"), 0.]] * 8,
                                     [[1., 0.]] * 7 + [[1.]]])
def test_invalid_embeddings_do_not_create_sections(lecture, vectors):
    with pytest.raises(ValueError):
        seg.semantic_starts(lecture[0], vectors, seg.Options())


def test_zero_embeddings_do_not_invent_transitions(lecture):
    assert seg.semantic_starts(lecture[0], [[0., 0.]] * 8, seg.Options()) == [0]


def test_silent_and_unknown_lectures_skip_models(monkeypatch):
    monkeypatch.setattr(seg.retrieval, "timeline_sources", lambda lid: [] if lid == "silent" else None)
    monkeypatch.setattr(seg.store, "_embed", lambda *a: pytest.fail("no model needed"))
    assert seg.build("silent").sections == []
    with pytest.raises(LookupError):
        seg.build("unknown")
    with pytest.raises(ValueError):
        seg.build("../escape")


def test_bounded_requests_cover_every_candidate_without_artificial_boundaries(monkeypatch):
    sentences = [Chunk(lecture_id="long", start=i * 10, end=(i + 1) * 10, text="A sentence. " * 8) for i in range(40)]
    vectors = [[1., 0.] if (i // 4) % 2 == 0 else [0., 1.] for i in range(40)]
    monkeypatch.setattr(seg.store, "_embed", lambda texts: vectors)
    calls = []
    def complete(prompt, json=False):
        assert len(prompt) <= 2500
        calls.append(prompt)
        return decisions(prompt)
    monkeypatch.setattr(llm, "complete", complete)
    index = seg.generate("long", sentences, options=seg.Options(context=1, max_prompt_chars=2500))
    assert len(calls) > 1
    assert [s.start for s in index.sections] == list(range(0, 400, 40))
    assert len(index.sentence_membership()) == 40


def test_concurrent_requests_generate_only_once(lecture, monkeypatch):
    calls = []
    def complete(prompt, json=False):
        calls.append(1)
        time.sleep(0.05)
        return decisions(prompt)
    monkeypatch.setattr(llm, "complete", complete)
    results = []
    threads = [threading.Thread(target=lambda: results.append(seg.build("lec"))) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 3 and len(calls) == 1
    assert all(index == results[0] for index in results)


def test_cli_builds_local_sections_and_reports_unknown_lecture(lecture, capsys):
    assert build_sections.main(["lec", "unknown", "--semantic-only"]) == 1
    output = capsys.readouterr().out
    assert "lec: 2 sections, 8 sentences" in output
    assert "unknown: FAILED" in output

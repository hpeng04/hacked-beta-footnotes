"""Person 2: try the real LLM on the sample lectures in data/mock/samples.

Usage (from backend/, with LLM_* set in .env):
    python -m scripts.live_llm_test [LECTURE_ID ...] [--no-timeline] [--no-ask]

Calls the LLM for real but never writes data/cache. For each lecture it prints the
generated timeline, then asks the questions in data/mock/sample_questions.json and checks
the answer cites the expected chunk (or, for off-topic questions, cites nothing).
"""
import argparse
import json
import re
import sys

from app import config
from app.search import llm, retrieval, timeline

QUESTIONS_FILE = config.MOCK_DIR / "sample_questions.json"
CITATION = re.compile(r"\[(\d+:)?\d+:\d\d\]")


def run_timeline(lecture_id: str) -> bool:
    try:
        sections = timeline.generate(retrieval.transcript(lecture_id))
    except timeline.LLM_FAILURES as e:
        print(f"  timeline: FAILED ({type(e).__name__}: {e})")
        return False
    print(f"  timeline: {len(sections)} sections")
    for s in sections:
        print(f"    {llm.fmt_time(s.start):>7}-{llm.fmt_time(s.end):<7} {s.title}")
    return True


def run_question(lecture_id: str, q: str, expect_start: float | None) -> bool:
    sources = retrieval.search(lecture_id, q)
    reply = llm.answer(q, sources)
    if reply == llm.UNAVAILABLE:
        print(f"  FAIL  {q}\n        LLM unavailable, got fallback answer")
        return False
    if expect_start is None:
        ok = not CITATION.search(reply)
        expected = "no citation"
    else:
        ok = f"[{llm.fmt_time(expect_start)}]" in reply
        expected = f"[{llm.fmt_time(expect_start)}]"
        if not any(c.start == expect_start for c in sources):
            expected += ", not retrieved"  # retrieval missed it, so the LLM never saw it
    print(f"  {'PASS' if ok else 'FAIL'}  {q}  (want {expected})\n        {reply}")
    return ok


def main(argv: list[str] | None = None) -> int:
    questions = json.loads(QUESTIONS_FILE.read_text())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lecture_ids", nargs="*", default=sorted(questions))
    parser.add_argument("--no-timeline", action="store_true")
    parser.add_argument("--no-ask", action="store_true")
    args = parser.parse_args(argv)

    print(f"provider: {config.LLM_PROVIDER} (backup: {config.LLM_BACKUP_PROVIDER or 'none'})")
    passed = total = 0
    for lecture_id in args.lecture_ids:
        print(f"\n== {lecture_id}")
        if not args.no_timeline:
            total += 1
            passed += run_timeline(lecture_id)
        if not args.no_ask:
            for item in questions.get(lecture_id, []):
                total += 1
                passed += run_question(lecture_id, item["q"], item["expect_start"])
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

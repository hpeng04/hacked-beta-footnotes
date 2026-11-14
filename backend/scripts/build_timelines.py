"""Person 2: pre-build and cache topic timelines for the demo lectures.

Usage (from backend/): python -m scripts.build_timelines LECTURE_ID [LECTURE_ID ...] [--force]

Run after Person 1 has ingested the lectures. Writes data/cache/<id>.timeline.json;
commit those files so the demo never depends on the LLM.
"""
import argparse
import sys

from app.search import timeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lecture_ids", nargs="+")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args(argv)

    failed = []
    for lecture_id in args.lecture_ids:
        try:
            if not args.force and timeline.cached(lecture_id) is not None:
                print(f"{lecture_id}: cached, skipped (use --force to rebuild)")
                continue
            sections = timeline.rebuild(lecture_id)  # keeps the old cache if this fails
        except LookupError:
            print(f"{lecture_id}: FAILED (no transcript)")
            failed.append(lecture_id)
            continue
        except timeline.LLM_FAILURES as e:
            print(f"{lecture_id}: FAILED (LLM: {e})")
            failed.append(lecture_id)
            continue
        print(f"{lecture_id}: {len(sections)} sections -> {timeline.cache_path(lecture_id)}")
        for s in sections:
            print(f"  {timeline.llm.fmt_time(s.start):>8}  {s.title}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

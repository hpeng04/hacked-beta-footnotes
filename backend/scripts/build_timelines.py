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
        path = timeline.cache_path(lecture_id)
        if args.force:
            path.unlink(missing_ok=True)
        elif path.exists():
            print(f"{lecture_id}: cached, skipped (use --force to rebuild)")
            continue

        sections = timeline.build(lecture_id)
        if not path.exists():  # build() only caches real LLM results
            reason = "no transcript" if not sections else "LLM failed, got fallback sections"
            print(f"{lecture_id}: FAILED ({reason})")
            failed.append(lecture_id)
            continue
        print(f"{lecture_id}: {len(sections)} sections -> {path}")
        for s in sections:
            print(f"  {timeline.llm.fmt_time(s.start):>8}  {s.title}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

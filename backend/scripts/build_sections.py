"""Build and cache topic membership: python -m scripts.build_sections LECTURE_ID [--force]."""
import argparse

from app.search import segmentation
from app.search.llm import fmt_time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lecture_ids", nargs="+")
    parser.add_argument("--force", action="store_true", help="regenerate, including retrying LLM refinement")
    parser.add_argument("--semantic-only", action="store_true", help="use local embeddings without calling an LLM")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-seconds", type=float, default=12)
    parser.add_argument("--min-sentences", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        options = segmentation.Options(window=args.window, similarity_threshold=args.threshold,
                                       min_seconds=args.min_seconds, min_sentences=args.min_sentences)
    except ValueError as exc:
        parser.error(str(exc))
    failed = False
    for lecture_id in args.lecture_ids:
        try:
            index = segmentation.build(lecture_id, options=options, refine=not args.semantic_only, force=args.force)
        except (LookupError, ValueError, RuntimeError, OSError) as exc:
            print(f"{lecture_id}: FAILED ({exc})")
            failed = True
            continue
        print(f"{lecture_id}: {len(index.sections)} sections, {len(index.sentences)} sentences")
        for section in index.sections:
            print(f"  {fmt_time(section.start)}–{fmt_time(section.end)} {section.title} ({len(section.sentence_ids)} sentences)")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

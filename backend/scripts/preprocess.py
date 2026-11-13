"""Person 1: pre-process demo lectures ahead of time.

Usage (from backend/):
  python -m scripts.preprocess path/to/lecture.mp4 LECTURE_ID [--force]
  python -m scripts.preprocess --reindex

The first form copies the video to data/uploads/<id>.<ext>, transcribes it and indexes it.
The transcript lands in data/cache/<id>.transcript.json: commit that file so nobody has to
run Whisper again. Then build the timeline: python -m scripts.build_timelines LECTURE_ID

--reindex rebuilds the vector DB (not committed) from every cached transcript. Run it after
cloning or pulling new transcripts.
"""
import argparse
import shutil
import sys
from pathlib import Path

from app import config
from app.ingestion import pipeline, store
from app.search.llm import fmt_time

SUFFIX = ".transcript.json"


def preprocess(video: Path, lecture_id: str, force: bool) -> int:
    if not config.is_valid_lecture_id(lecture_id):
        print(f"{lecture_id}: FAILED (invalid id: use letters, digits, - and _)")
        return 1
    if not video.is_file():
        print(f"{lecture_id}: FAILED (no such file: {video})")
        return 1

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = config.UPLOAD_DIR / f"{lecture_id}{video.suffix}"
    if video.resolve() != target.resolve():
        shutil.copyfile(video, target)

    cached = pipeline.transcript_path(lecture_id)
    if force:
        cached.unlink(missing_ok=True)
    elif cached.exists():
        print(f"{lecture_id}: transcript cached, only re-indexing (use --force to transcribe again)")

    try:
        chunks = pipeline.ingest(target, lecture_id)
    except Exception as e:
        print(f"{lecture_id}: FAILED ({e})")
        return 1

    print(f"{lecture_id}: {len(chunks)} chunks -> {cached}")
    for c in chunks[:5]:
        print(f"  {fmt_time(c.start):>8}  {c.text[:70]}")
    print(f"video: /media/{target.name}")
    print(f"next:  python -m scripts.build_timelines {lecture_id}")
    return 0


def reindex() -> int:
    paths = sorted(config.CACHE_DIR.glob(f"*{SUFFIX}"))
    if not paths:
        print(f"no cached transcripts in {config.CACHE_DIR}")
        return 0
    failed = 0
    for path in paths:
        lecture_id = path.name[: -len(SUFFIX)]
        try:
            chunks = pipeline.load_transcript(lecture_id)
            store.add_chunks(chunks)
        except Exception as e:
            print(f"{lecture_id}: FAILED ({e})")
            failed += 1
            continue
        print(f"{lecture_id}: {len(chunks)} chunks indexed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("lecture_id", nargs="?")
    parser.add_argument("--force", action="store_true", help="transcribe again even if cached")
    parser.add_argument("--reindex", action="store_true", help="rebuild the vector DB from cached transcripts")
    args = parser.parse_args(argv)

    if args.reindex:
        if args.video or args.lecture_id:
            parser.error("--reindex takes no video or lecture id")
        return reindex()
    if not args.video or not args.lecture_id:
        parser.error("give a video path and a lecture id, or --reindex")
    return preprocess(args.video, args.lecture_id, args.force)


if __name__ == "__main__":
    sys.exit(main())

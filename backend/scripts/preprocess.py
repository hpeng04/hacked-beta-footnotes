"""Person 1: pre-process demo lectures ahead of time.

Usage (from backend/):
  python -m scripts.preprocess path/to/lecture.mp4 LECTURE_ID [--force] [--model small]
  python -m scripts.preprocess --reindex
  python -m scripts.preprocess --warmup [--model small]

The first form copies the video to data/uploads/<id>.<ext>, transcribes it and indexes it.
The transcript lands in data/cache/<id>.transcript.json: commit that file so nobody has to
run Whisper again. Then build the timeline: python -m scripts.build_timelines LECTURE_ID

--reindex rebuilds the vector DB (not committed) from every cached transcript. Run it after
cloning or pulling new transcripts.

--warmup downloads and loads the embedding model and the Whisper model, so the first upload
of the demo doesn't wait on a download.

--model picks the Whisper model for this run instead of WHISPER_MODEL from .env.
"""
import argparse
import filecmp
import shutil
import sys
import time
from pathlib import Path

from app import config
from app.ingestion import pipeline, store, transcribe
from app.search.llm import fmt_time

SUFFIX = ".transcript.json"


def preprocess(video: Path, lecture_id: str, force: bool) -> int:
    if not config.is_valid_lecture_id(lecture_id):
        print(f"{lecture_id}: FAILED (invalid id: use letters, digits, - and _)")
        return 1
    if not video.is_file():
        print(f"{lecture_id}: FAILED (no such file: {video})")
        return 1

    target = config.UPLOAD_DIR / f"{lecture_id}{video.suffix}"
    cached = pipeline.transcript_path(lecture_id)
    if cached.exists() and not force:
        # The cached transcript belongs to the stored video: never pair it with another one.
        if target.exists() and not filecmp.cmp(video, target, shallow=False):
            print(f"{lecture_id}: FAILED (a different video is already processed under this id: "
                  "use --force to replace it, or another id)")
            return 1
        print(f"{lecture_id}: transcript cached, only re-indexing (use --force to transcribe again)")

    # With --force the new video is transcribed from where it is, and replaces the stored
    # one only after that worked: a failure leaves the old video and transcript in place.
    source = video if force else target
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not force and video.resolve() != target.resolve():
        shutil.copyfile(video, target)

    started = time.monotonic()
    try:
        chunks = pipeline.ingest(source, lecture_id, force=force)
    except Exception as e:
        print(f"{lecture_id}: FAILED ({e})")
        return 1
    if force and video.resolve() != target.resolve():
        shutil.copyfile(video, target)

    print(f"{lecture_id}: {len(chunks)} chunks in {time.monotonic() - started:.0f}s -> {cached}")
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


def warmup() -> int:
    steps = [
        (f"embedding model {config.EMBEDDING_MODEL}", lambda: store._embed(["warm up"])),
        (f"whisper model {config.WHISPER_MODEL}", transcribe._whisper),
    ]
    failed = 0
    for name, load in steps:
        started = time.monotonic()
        try:
            load()
        except Exception as e:
            print(f"{name}: FAILED ({e})")
            failed += 1
            continue
        print(f"{name}: ready in {time.monotonic() - started:.0f}s")
    if shutil.which("ffmpeg") is None:
        print("ffmpeg: FAILED (not found on PATH)")
        failed += 1
    else:
        print("ffmpeg: found")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("lecture_id", nargs="?")
    parser.add_argument("--force", action="store_true", help="transcribe again even if cached")
    parser.add_argument("--reindex", action="store_true", help="rebuild the vector DB from cached transcripts")
    parser.add_argument("--warmup", action="store_true", help="download and load the models")
    parser.add_argument("--model", help="Whisper model for this run (default: WHISPER_MODEL)")
    args = parser.parse_args(argv)

    if args.model:
        config.WHISPER_MODEL = args.model
    if args.reindex or args.warmup:
        if args.video or args.lecture_id or (args.reindex and args.warmup):
            parser.error("--reindex and --warmup take no video or lecture id, and not each other")
        return reindex() if args.reindex else warmup()
    if not args.video or not args.lecture_id:
        parser.error("give a video path and a lecture id, or --reindex, or --warmup")
    return preprocess(args.video, args.lecture_id, args.force)


if __name__ == "__main__":
    sys.exit(main())

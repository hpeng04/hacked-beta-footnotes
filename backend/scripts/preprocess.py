"""Person 1: pre-process a demo lecture ahead of time.

Usage (from backend/): python -m scripts.preprocess path/to/lecture.mp4 lecture_id
"""
import sys


def main(video_path: str, lecture_id: str) -> None:
    # TODO: transcribe -> chunk_segments -> store.add_chunks, save chunks to data/cache
    raise NotImplementedError


if __name__ == "__main__":
    main(*sys.argv[1:3])

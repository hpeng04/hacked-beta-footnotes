"""Person 1: audio extraction and Whisper transcription."""
from pathlib import Path


def transcribe(video_path: Path) -> list[dict]:
    """Return Whisper segments as [{"start": float, "end": float, "text": str}]."""
    # TODO: extract audio with ffmpeg, run faster-whisper (config.WHISPER_MODEL)
    raise NotImplementedError

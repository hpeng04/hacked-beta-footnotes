"""Person 1: audio extraction and Whisper transcription."""
from pathlib import Path
from faster_whisper import WhisperModel

def transcribe(video_path: Path) -> list[dict]:
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path))
    results = []
    for seg in segments:
        results.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return results
"""Person 1: audio extraction and Whisper transcription."""
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from .. import config


def _extract_audio(video_path: Path, wav_path: Path) -> None:
    """Write the video's audio track as 16 kHz mono WAV (what Whisper expects)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found: install it and make sure it is on PATH")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"ffmpeg could not extract audio from {video_path.name}: {result.stderr.strip()[-500:]}")


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # slow import, only when first needed

    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return BatchedInferencePipeline(model=model)


def transcribe(video_path: Path) -> list[dict]:
    """Return Whisper segments as [{"start": float, "end": float, "text": str}]."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    with tempfile.TemporaryDirectory() as tmp:  # the wav is removed even if Whisper fails
        wav_path = Path(tmp) / "audio.wav"
        _extract_audio(video_path, wav_path)
        segments, _info = _whisper().transcribe(
            str(wav_path),
            vad_filter=True,
            batch_size=config.WHISPER_BATCH_SIZE,
            beam_size=config.WHISPER_BEAM_SIZE,
        )
        # segments is a generator: transcription happens here, while the wav still exists
        return [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]

"""Person 1: audio extraction and Whisper transcription.

Tries Groq's hosted Whisper first (fast, free tier) on compressed audio; falls back to
local faster-whisper if there is no key, the file is too big, or the request fails.
"""
import logging
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import httpx

from .. import config

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TIMEOUT = 300.0


def _ffmpeg(video_path: Path, out_path: Path, *codec_args: str) -> None:
    """Write the video's audio track as 16 kHz mono (what Whisper expects)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found: install it and make sure it is on PATH")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
           *codec_args, str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg could not extract audio from {video_path.name}: {result.stderr.strip()[-500:]}")


def _extract_audio(video_path: Path, wav_path: Path) -> None:
    _ffmpeg(video_path, wav_path)


def _compress_audio(video_path: Path, mp3_path: Path) -> None:
    """32 kbps mono MP3: about 14 MB per hour, under Groq's 25 MB limit."""
    _ffmpeg(video_path, mp3_path, "-c:a", "libmp3lame", "-b:a", "32k")


def _clean(segments) -> list[dict]:
    return [
        {
            "start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(),
            "words": [
                {"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]}
                for w in s.get("words") or []
            ],
        }
        for s in segments
        if s["text"].strip()
    ]


def _attach_words(segments: list[dict], words: list[dict]) -> list[dict]:
    """Groq returns one flat word list: give each word to the segment containing its midpoint.

    Words get faster-whisper's leading space so they join back into the segment text.
    """
    out = [{**s, "words": []} for s in segments]
    i = 0
    for w in words:
        mid = (float(w["start"]) + float(w["end"])) / 2
        while i < len(out) - 1 and mid >= float(out[i]["end"]):
            i += 1
        if out:
            out[i]["words"].append({**w, "word": " " + w["word"].strip()})
    return out


def _transcribe_groq(video_path: Path, tmp: Path) -> list[dict]:
    mp3_path = tmp / "audio.mp3"
    _compress_audio(video_path, mp3_path)
    size = mp3_path.stat().st_size
    if size > config.GROQ_WHISPER_MAX_BYTES:
        raise ValueError(f"compressed audio is {size / 1e6:.1f} MB, over Groq's limit")
    with mp3_path.open("rb") as f:
        r = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {config.GROQ_WHISPER_API_KEY}"},
            data={"model": config.GROQ_WHISPER_MODEL, "response_format": "verbose_json", "temperature": "0",
                  "timestamp_granularities[]": ["word", "segment"]},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            timeout=GROQ_TIMEOUT,
        )
    r.raise_for_status()
    body = r.json()
    return _clean(_attach_words(body["segments"], body.get("words") or []))


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # slow import, only when first needed

    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return BatchedInferencePipeline(model=model)


def _transcribe_local(video_path: Path, tmp: Path) -> list[dict]:
    wav_path = tmp / "audio.wav"
    _extract_audio(video_path, wav_path)
    segments, _info = _whisper().transcribe(
        str(wav_path),
        vad_filter=True,
        word_timestamps=True,
        batch_size=config.WHISPER_BATCH_SIZE,
        beam_size=config.WHISPER_BEAM_SIZE,
    )
    # segments is a generator: transcription happens here, while the wav still exists
    return _clean(
        {"start": s.start, "end": s.end, "text": s.text,
         "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]}
        for s in segments
    )


def transcribe(video_path: Path) -> list[dict]:
    """Return segments and word alignment in the original audio's time coordinates."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    with tempfile.TemporaryDirectory() as tmp:  # audio files are removed even if Whisper fails
        if config.GROQ_WHISPER_API_KEY:
            try:
                return _transcribe_groq(video_path, Path(tmp))
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                log.warning("Groq Whisper failed (%s); falling back to local Whisper", e)
        return _transcribe_local(video_path, Path(tmp))

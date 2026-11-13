import os
import re
from pathlib import Path

from dotenv import dotenv_values

BACKEND_DIR = Path(__file__).resolve().parent.parent
# .env wins over shell/environment variables; blank values in .env are ignored
os.environ.update({k: v for k, v in dotenv_values(BACKEND_DIR / ".env").items() if v})

DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
MOCK_DIR = DATA_DIR / "mock"
CHROMA_DIR = DATA_DIR / "chroma"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LECTURE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # also keeps ids safe to use in file names


def is_valid_lecture_id(lecture_id: str) -> bool:
    return bool(LECTURE_ID_RE.fullmatch(lecture_id))


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | gemini | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty = provider default
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BACKUP_PROVIDER = os.getenv("LLM_BACKUP_PROVIDER", "ollama")  # empty = no backup
LLM_BACKUP_MODEL = os.getenv("LLM_BACKUP_MODEL", "")
LLM_BACKUP_API_KEY = os.getenv("LLM_BACKUP_API_KEY", "")

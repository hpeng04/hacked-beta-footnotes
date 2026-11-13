import os
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | gemini | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty = provider default
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BACKUP_PROVIDER = os.getenv("LLM_BACKUP_PROVIDER", "gemini")  # empty = no backup
LLM_BACKUP_MODEL = os.getenv("LLM_BACKUP_MODEL", "")
LLM_BACKUP_API_KEY = os.getenv("LLM_BACKUP_API_KEY", "")

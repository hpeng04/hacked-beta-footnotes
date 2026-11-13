import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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

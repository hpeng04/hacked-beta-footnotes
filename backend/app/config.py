import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
MOCK_DIR = DATA_DIR / "mock"
CHROMA_DIR = DATA_DIR / "chroma"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

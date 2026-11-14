"""Shared fixtures. No test may reach a real LLM: httpx.post is always faked."""
import json
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.ingestion.timing import source_id


class FakeLLM:
    """Stands in for httpx.post. Queue replies per provider; record every request."""

    def __init__(self):
        self.replies = {}  # host substring -> list of str (content) or Exception
        self.calls = []

    def reply(self, host: str, *items):
        self.replies.setdefault(host, []).extend(items)

    def reply_answer(self, host, *sentences):
        self.reply(host, json.dumps({"sentences": [
            {"text": text, "source_ids": [source_id(c) for c in sources]}
            for text, sources in sentences
        ]}))

    def reply_timeline(self, host, *sections):
        self.reply(host, json.dumps({"sections": [
            {"title": title, "start_id": source_id(source)} for title, source in sections
        ]}))

    def __call__(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers})
        for host, queue in self.replies.items():
            if host in url and queue:
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                resp = mock.Mock()
                resp.raise_for_status = lambda: None
                resp.json = lambda c=item: {"choices": [{"message": {"content": c}}]}
                return resp
        raise httpx.ConnectError(f"no fake reply queued for {url}")


@pytest.fixture(autouse=True)
def fake_llm(request, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")  # never touch the real cache
    from app.search import segmentation
    monkeypatch.setattr(segmentation, "_spawn", lambda task: None)  # no threads outliving a test
    monkeypatch.setattr(segmentation, "_retry_at", {})
    monkeypatch.setattr(segmentation, "_loaded", {})
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")  # nor the real vector DB
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("ollama"):
        return None  # live tests use the real network and the settings from backend/.env
    fake = FakeLLM()
    monkeypatch.setattr(httpx, "post", fake)
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "primary-key")
    monkeypatch.setattr(config, "LLM_BACKUP_PROVIDER", "gemini")
    monkeypatch.setattr(config, "LLM_BACKUP_MODEL", "")
    monkeypatch.setattr(config, "LLM_BACKUP_API_KEY", "backup-key")
    monkeypatch.setattr(config, "GROQ_WHISPER_API_KEY", "")  # local Whisper unless a test opts in
    return fake


@pytest.fixture
def client():
    return TestClient(app)

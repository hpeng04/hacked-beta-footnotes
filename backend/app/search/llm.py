"""Person 2: LLM calls.

Groq, Gemini and Ollama all expose an OpenAI-compatible chat endpoint,
so one HTTP client covers every provider. The primary provider is tried
first; on any error the backup is used.
"""
import json as jsonlib

import httpx

from .. import config
from ..schemas import Chunk

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
    "ollama": ("http://localhost:11434/v1", "llama3.1"),
}


class LLMError(RuntimeError):
    pass


def _call(provider: str, api_key: str, model: str, prompt: str, json: bool) -> str:
    base_url, default_model = PROVIDERS[provider]
    body = {
        "model": model or default_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if json:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def complete(prompt: str, json: bool = False) -> str | dict:
    """Send one prompt. With json=True the reply is parsed and returned as a dict."""
    attempts = [(config.LLM_PROVIDER, config.LLM_API_KEY, config.LLM_MODEL)]
    if config.LLM_BACKUP_PROVIDER:
        attempts.append((config.LLM_BACKUP_PROVIDER, config.LLM_BACKUP_API_KEY, config.LLM_BACKUP_MODEL))

    errors = []
    for provider, key, model in attempts:
        try:
            text = _call(provider, key, model, prompt, json)
            return jsonlib.loads(text) if json else text
        except Exception as e:  # rate limit, network, bad JSON: try the next provider
            errors.append(f"{provider}: {e}")
    raise LLMError("; ".join(errors))


def fmt_time(seconds: float) -> str:
    """95 -> "1:35", 3725 -> "1:02:05"."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


ANSWER_PROMPT = """You answer questions about a lecture using only the transcript excerpts below.
Each excerpt starts with its timestamp in [m:ss].

Rules:
- Use only the excerpts. If they do not answer the question, say so in one sentence.
- After every claim, cite the excerpt timestamp exactly as given, e.g. [1:35].
- Be brief: 1 to 4 sentences, plain text, no markdown.

Excerpts:
{sources}

Question: {question}
Answer:"""


def answer(question: str, sources: list[Chunk]) -> str:
    if not sources:
        return "Nothing in this lecture matches that question."
    ordered = sorted(sources, key=lambda c: c.start)
    prompt = ANSWER_PROMPT.format(
        sources="\n".join(f"[{fmt_time(c.start)}] {c.text}" for c in ordered),
        question=question,
    )
    try:
        return complete(prompt).strip()
    except LLMError:  # both providers down: show the best match instead of failing
        top = sources[0]
        return f"[{fmt_time(top.start)}] {top.text}"

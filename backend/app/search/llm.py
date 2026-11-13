"""Person 2: LLM calls.

Groq, Gemini and Ollama all expose an OpenAI-compatible chat endpoint,
so one HTTP client covers every provider. The primary provider is tried
first; on any error the backup is used.
"""
import json as jsonlib
import re
import time

import httpx

from .. import config
from ..schemas import Chunk
from .retrieval import STOPWORDS

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
    "ollama": ("http://localhost:11434/v1", "gemma4:e2b"),
}


TIMEOUTS = {"ollama": 300.0}  # local CPU models are slow on long prompts like the timeline
DEFAULT_TIMEOUT = 60.0
RETRY_STATUSES = {429, 503}  # rate limited / overloaded: worth one retry before the backup
MAX_RETRY_WAIT = 10.0


class LLMError(RuntimeError):
    pass


def _retry_wait(error: Exception) -> float | None:
    """Seconds to wait before retrying the same provider, or None to move on."""
    if not isinstance(error, httpx.HTTPStatusError) or error.response.status_code not in RETRY_STATUSES:
        return None
    try:
        return min(float(error.response.headers.get("retry-after", 2)), MAX_RETRY_WAIT)
    except ValueError:
        return 2.0


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
    r = httpx.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=TIMEOUTS.get(provider, DEFAULT_TIMEOUT))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def complete(prompt: str, json: bool = False) -> str | dict:
    """Send one prompt. With json=True the reply is parsed and returned as a dict."""
    attempts = [(config.LLM_PROVIDER, config.LLM_API_KEY, config.LLM_MODEL)]
    if config.LLM_BACKUP_PROVIDER:
        attempts.append((config.LLM_BACKUP_PROVIDER, config.LLM_BACKUP_API_KEY, config.LLM_BACKUP_MODEL))

    errors = []
    for provider, key, model in attempts:
        for retry in (True, False):
            try:
                text = _call(provider, key, model, prompt, json)
                return jsonlib.loads(text) if json else text
            except Exception as e:  # rate limit, network, bad JSON: retry once or try the next provider
                wait = _retry_wait(e) if retry else None
                if wait is None:
                    errors.append(f"{provider}: {e}")
                    break
                time.sleep(wait)
    raise LLMError("; ".join(errors))


def fmt_time(seconds: float) -> str:
    """95 -> "1:35", 3725 -> "1:02:05"."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


ANSWER_PROMPT = """You answer questions about a lecture using only the transcript excerpts below.
Each excerpt is labelled with the time it starts, like (Excerpt at [1:35]).

Rules:
- Use only the excerpts. If they do not answer the question, say so in one sentence and cite nothing.
- End every sentence with the label of the excerpt it comes from, in square brackets.
  Example: Deadlock needs four conditions [1:35].
- Be brief: 1 to 4 sentences, plain text, no markdown.

{sources}

Question: {question}
Answer:"""

MIN_GROUNDING_OVERLAP = 3  # shared content words before a citation is added or moved


def answer(question: str, sources: list[Chunk]) -> str:
    if not sources:
        return "Nothing in this lecture matches that question."
    ordered = sorted(sources, key=lambda c: c.start)
    prompt = ANSWER_PROMPT.format(
        sources="\n\n".join(f"(Excerpt at [{fmt_time(c.start)}])\n{c.text}" for c in ordered),
        question=question,
    )
    try:
        return _ground_citations(_normalize_citations(complete(prompt).strip()), ordered)
    except LLMError:  # both providers down: show the best match instead of failing
        top = sources[0]
        return f"[{fmt_time(top.start)}] {top.text}"


def _normalize_citations(text: str) -> str:
    """Some models cite as 【1:35】 or (1:35); rewrite to [1:35] so the frontend can link them."""
    return re.sub(r"[【(（]\s*((?:\d+:)?\d+:\d\d)\s*[】)）]", r"[\1]", text)


CITATION = re.compile(r"\[((?:\d+:)?\d+:\d\d)\]")


def _content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", CITATION.sub(" ", text.lower()))) - STOPWORDS


def _ground_citations(reply: str, sources: list[Chunk]) -> str:
    """Repair citations small models get wrong, by word overlap with the sources.

    Per sentence: no citation -> add the best-matching source (only if the reply cites
    nothing at all, so "the lecture doesn't cover X" stays uncited); citations that match
    the sentence far worse than another source -> replace them with that source.
    """
    by_time = {fmt_time(c.start): _content_words(c.text) for c in sources}
    cites_anything = bool(CITATION.search(reply))
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        words = _content_words(sentence)
        scores = {t: len(words & w) for t, w in by_time.items()}
        best = max(scores, key=scores.get)
        cited = CITATION.findall(sentence)
        if scores[best] >= MIN_GROUNDING_OVERLAP:
            if not cited and not cites_anything:
                sentence = re.sub(r"([.!?]?)$", rf" [{best}]\1", sentence, count=1)
            elif cited and 2 * max(scores.get(t, 0) for t in cited) < scores[best]:
                sentence = CITATION.sub(f"[{best}]", sentence)
        out.append(sentence)
    return " ".join(out)

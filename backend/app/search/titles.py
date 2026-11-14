"""Extract short contextual labels when a generated topic title is unavailable."""
import re
from collections import Counter

from .retrieval import STOPWORDS

MAX_WORDS = 6  # labels appear in a one-line timeline tooltip and on replay links


def context_title(sources) -> str:
    texts = [s.text.strip() for s in sources if s.text.strip()]
    if not texts:
        return "No speech"
    tokens = [set(re.findall(r"\w+", text.lower())) - STOPWORDS for text in texts]
    counts = Counter(word for words in tokens for word in words)
    best = max(range(len(texts)), key=lambda i: sum(counts[w] - 1 for w in tokens[i]) / max(1, len(tokens[i])))
    words = texts[best].split()
    return " ".join(words[:MAX_WORDS]).rstrip(".,;:") + ("…" if len(words) > MAX_WORDS else "")


def descriptive_title(title, sources) -> str:
    if not title or re.fullmatch(r"(?:part|section|topic)\s*\d+", title.strip(), re.I):
        return context_title(sources)
    return title

"""
Semantic cache for the eval-service benchmark layer.

Purpose: re-running a benchmark, or overlapping questions across batches,
shouldn't re-hit the system under test (agent-service -> LLM -> retrieval)
for a question that's already been answered. Exact repeats are the common
case (re-running the same suite after an unrelated code change); near-
duplicate phrasing of the same question is the "semantic" part -- caught
via token-set Jaccard similarity rather than an embedding model, since no
network access to a model host is assumed here.

Scoped by document_id: two identical questions about different documents
are correctly treated as different cache entries.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_ARTICLES = {"a", "an", "the"}


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _token_set(text: str) -> set:
    return set(_normalize(text).split())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class CacheStats:
    exact_hits: int = 0
    fuzzy_hits: int = 0
    misses: int = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.fuzzy_hits

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> Optional[float]:
        return self.hits / self.total if self.total else None


@dataclass
class _Entry:
    normalized_text: str
    tokens: set
    value: Any


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self._store: Dict[str, List[_Entry]] = {}
        self.stats = CacheStats()

    def get(self, question_text: str, document_id: Optional[str]) -> Tuple[bool, Any]:
        scope_key = document_id or ""
        normalized = _normalize(question_text)
        tokens = set(normalized.split())
        entries = self._store.get(scope_key, [])

        for entry in entries:
            if entry.normalized_text == normalized:
                self.stats.exact_hits += 1
                return True, entry.value

        best_score = 0.0
        best_entry: Optional[_Entry] = None
        for entry in entries:
            score = _jaccard(tokens, entry.tokens)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            self.stats.fuzzy_hits += 1
            return True, best_entry.value

        self.stats.misses += 1
        return False, None

    def set(self, question_text: str, document_id: Optional[str], value: Any) -> None:
        scope_key = document_id or ""
        normalized = _normalize(question_text)
        entry = _Entry(normalized_text=normalized, tokens=set(normalized.split()), value=value)
        self._store.setdefault(scope_key, []).append(entry)

    def size(self) -> int:
        return sum(len(v) for v in self._store.values())

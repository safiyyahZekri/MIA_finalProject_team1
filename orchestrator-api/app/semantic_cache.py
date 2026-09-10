"""Answer cache for /ask.

A question already answered about the same document returns the validated
answer again instead of running the agent, which costs several model calls
and tens of seconds. Two kinds of match count:

- exact: the same question after lowercasing, punctuation and whitespace;
- reworded: the same content words, in any order, with different function
  words ("What was Jabil's revenue in 2019?" and "Jabil revenue in 2019 --
  what was it?").

Near-synonyms deliberately do not match. In financial filings "revenue" and
"net sales", or "2019" and "2018", are different figures, so a similarity
score that lets one word differ would serve an answer to a question it does
not answer. Every word that carries meaning has to be the same.

Entries are scoped by document_id and cleared whenever the index changes
(an ingest or an extraction correction), so a cached answer never outlives
the evidence it was built on.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Optional

Match = Literal["exact", "reworded"]

# Words that change how a question is phrased, not what it asks. "and", "or",
# "not" and "than" are left out on purpose: they change the answer.
FUNCTION_WORDS = frozenset(
    "a an the of in on for to at by from with as is are was were be been being "
    "what which who whom whose how much many did does do it its s please tell me "
    "show give find there this that these those during".split()
)

_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?%?|[a-z]+")


def _tokens(question: str) -> list[str]:
    # Commas are thousands separators, so "3,875" and "3875" are one number;
    # decimal points stay, so "1.25" and "12.5" remain different.
    return [token.replace(",", "") for token in _TOKEN.findall(question.lower())]


def exact_key(question: str) -> str:
    return " ".join(_tokens(question))


def content_key(question: str) -> frozenset[str]:
    return frozenset(t for t in _tokens(question) if t not in FUNCTION_WORDS)


@dataclass
class _Entry:
    exact: str
    content: frozenset[str]
    value: Any


class SemanticCache:
    def __init__(self, max_entries: int = 500) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self.exact_hits = 0
        self.reworded_hits = 0
        self.misses = 0

    def get(self, question: str, document_id: Optional[str]) -> Optional[tuple[Any, Match]]:
        scope = document_id or ""
        exact = exact_key(question)
        entry = self._entries.get((scope, exact))
        if entry is not None:
            self._entries.move_to_end((scope, exact))
            self.exact_hits += 1
            return entry.value, "exact"
        content = content_key(question)
        # An empty content set would match any question made of function words.
        if content:
            for (entry_scope, key), entry in reversed(self._entries.items()):
                if entry_scope == scope and entry.content == content:
                    self._entries.move_to_end((entry_scope, key))
                    self.reworded_hits += 1
                    return entry.value, "reworded"
        self.misses += 1
        return None

    def put(self, question: str, document_id: Optional[str], value: Any) -> None:
        key = (document_id or "", exact_key(question))
        self._entries[key] = _Entry(key[1], content_key(question), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "exact_hits": self.exact_hits,
            "reworded_hits": self.reworded_hits,
            "misses": self.misses,
        }

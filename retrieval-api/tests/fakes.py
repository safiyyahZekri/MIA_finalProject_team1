from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np


class FakeEmbedder:
    name = "fake-financial-embedder"
    vocabulary = ("revenue", "inventory", "expense", "cash", "employee", "table")

    def _encode(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.asarray(
            [lowered.count(token) for token in self.vocabulary], dtype=np.float32
        )
        if not vector.any():
            vector[-1] = 0.01
        return vector / np.linalg.norm(vector)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._encode(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text)


class FakeReranker:
    name = "fake-overlap-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        return [
            float(len(query_tokens & set(re.findall(r"\w+", passage.lower()))))
            for passage in passages
        ]

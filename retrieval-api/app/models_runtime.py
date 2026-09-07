from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy pretrained embedder; importing the API never downloads model weights."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._name = model_name
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return self._name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._name, device=self._device)
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode_document(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self._load().encode_query(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vector[0], dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._name = model_name
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return self._name

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._name, device=self._device)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        scores = self._load().predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]

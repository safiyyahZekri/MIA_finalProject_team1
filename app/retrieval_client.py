"""
Thin async client for retrieval-api.

Expected retrieval-api contract (documented for the retrieval-api owner too):

  POST {RETRIEVAL_API_URL}/search/vector   {"query": str, "top_k": int, "document_id": str|None}
  POST {RETRIEVAL_API_URL}/search/bm25     {"query": str, "top_k": int, "document_id": str|None}
  POST {RETRIEVAL_API_URL}/search/tables   {"query": str, "top_k": int, "document_id": str|None}
  POST {RETRIEVAL_API_URL}/filter          {"metadata": dict, "top_k": int}

  -> each returns: {"hits": [
        {"document_id": str, "page": int, "section": str|None,
         "content_type": "text"|"table", "text": str, "score": float}
     ]}

Reranking (over-retrieve top-K -> rerank -> top-N) is retrieval-api's job per
the spec, so hits returned here are assumed already reranked.

If retrieval-api is unreachable (e.g. being developed in parallel by a
teammate) and RETRIEVAL_FALLBACK_TO_MOCK is on, we fall back to a tiny local
corpus so this service is independently runnable and demoable.
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from app.config import settings

# --- tiny offline corpus, used only if retrieval-api is unreachable ---
_MOCK_CORPUS = [
    {
        "document_id": "doc_017",
        "page": 1,
        "section": "Income Statement",
        "content_type": "table",
        "text": "Operating income for fiscal year 2020 was $142.5M, up from $125.1M in 2019.",
        "score": 0.91,
    },
    {
        "document_id": "doc_041",
        "page": 2,
        "section": "Operating Expenses",
        "content_type": "table",
        "text": "Total operating expenses were $3,875 thousand in 2021 versus $3,410 thousand in 2020.",
        "score": 0.88,
    },
    {
        "document_id": "doc_022",
        "page": 3,
        "section": "Operating Expenses",
        "content_type": "table",
        "text": "Expense categories that increased year-over-year: Marketing, R&D, Logistics.",
        "score": 0.84,
    },
    {
        "document_id": "doc_005",
        "page": 4,
        "section": "Notes to Financial Statements",
        "content_type": "text",
        "text": "The company reported no restructuring charges during the current fiscal period.",
        "score": 0.40,
    },
]


_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "in", "on", "at", "to", "of", "and",
    "or", "for", "what", "which", "how", "by", "from", "that", "this", "with",
    "as", "it", "its", "did", "do", "does", "are", "be", "than", "much", "many",
}


def _meaningful_words(text: str) -> List[str]:
    cleaned = [w.strip("?,.'\":;") for w in text.lower().split()]
    return [w for w in cleaned if w and w not in _STOPWORDS and len(w) > 2]


def _mock_search(query: str, top_k: int, content_type: Optional[str] = None) -> List[dict]:
    """Keyword-overlap scoring against the tiny offline corpus. Deliberately
    conservative: queries with no real lexical overlap score near zero, so
    the grading step correctly reports insufficient evidence for questions
    the corpus can't answer, instead of always finding "something"."""
    q_words = _meaningful_words(query)
    scored = []
    for chunk in _MOCK_CORPUS:
        if content_type and chunk["content_type"] != content_type:
            continue
        text_lower = chunk["text"].lower()
        overlap = sum(1 for w in q_words if w in text_lower)
        ratio = overlap / max(1, len(q_words))
        score = chunk["score"] * (0.05 + 0.9 * ratio) if overlap else 0.05
        scored.append({**chunk, "score": round(min(score, 0.99), 3)})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]


class RetrievalClient:
    def __init__(self) -> None:
        self.base_url = settings.RETRIEVAL_API_URL.rstrip("/")
        self.timeout = settings.RETRIEVAL_TIMEOUT_S

    async def _post(self, path: str, payload: dict) -> List[dict]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}{path}", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("hits", [])
        except (httpx.HTTPError, ValueError) as exc:
            if settings.RETRIEVAL_FALLBACK_TO_MOCK:
                return None  # signal caller to use mock fallback
            raise RuntimeError(f"retrieval-api call to {path} failed: {exc}") from exc

    async def search_documents(
        self, query: str, top_k: int = None, document_id: Optional[str] = None
    ) -> List[dict]:
        """Corpus-wide semantic (vector) retrieval."""
        top_k = top_k or settings.TOP_K_FINAL
        hits = await self._post(
            "/search/vector",
            {"query": query, "top_k": top_k, "document_id": document_id},
        )
        if hits is None:
            hits = _mock_search(query, top_k)
        return hits

    async def search_bm25(
        self, query: str, top_k: int = None, document_id: Optional[str] = None
    ) -> List[dict]:
        """Non-vector (lexical) retrieval -- required alongside embeddings."""
        top_k = top_k or settings.TOP_K_FINAL
        hits = await self._post(
            "/search/bm25",
            {"query": query, "top_k": top_k, "document_id": document_id},
        )
        if hits is None:
            hits = _mock_search(query, top_k)
        return hits

    async def search_tables(
        self, query: str, top_k: int = None, document_id: Optional[str] = None
    ) -> List[dict]:
        """Table-aware retrieval."""
        top_k = top_k or settings.TOP_K_FINAL
        hits = await self._post(
            "/search/tables",
            {"query": query, "top_k": top_k, "document_id": document_id},
        )
        if hits is None:
            hits = _mock_search(query, top_k, content_type="table")
        return hits

    async def filter_documents(self, metadata: dict, top_k: int = None) -> List[dict]:
        """Direct metadata filter lookup (non-vector mechanism)."""
        top_k = top_k or settings.TOP_K_FINAL
        hits = await self._post("/filter", {"metadata": metadata, "top_k": top_k})
        if hits is None:
            doc_id = metadata.get("document_id")
            hits = [c for c in _MOCK_CORPUS if not doc_id or c["document_id"] == doc_id][:top_k]
        return hits


retrieval_client = RetrievalClient()

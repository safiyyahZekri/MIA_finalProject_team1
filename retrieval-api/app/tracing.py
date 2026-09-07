from __future__ import annotations

import logging
from typing import Any

import httpx

from .models import SearchRequest, SearchResponse

logger = logging.getLogger("retrieval-api.tracing")


class RetrievalTracer:
    """Best-effort bridge to eval-service/Langfuse.

    Tracing is deliberately outside the retrieval engine: failures never alter
    retrieval results and no full chunk/document text is exported.
    """

    def __init__(self, eval_service_url: str = "", timeout_seconds: float = 2.0):
        self.base_url = eval_service_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            response = httpx.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("retrieval trace delivery failed for %s: %s", path, exc)
            return None

    def record(self, request: SearchRequest, response: SearchResponse) -> str | None:
        if not self.enabled:
            return request.trace_id
        owns_trace = request.trace_id is None
        trace_id = request.trace_id
        if owns_trace:
            result = self._post(
                "/trace/start",
                {
                    "name": "retrieval.search",
                    "metadata": {
                        "mode": request.mode.value,
                        "top_k": request.top_k,
                        "candidate_k": request.candidate_k,
                        "rerank": request.rerank,
                    },
                },
            )
            trace_id = result.get("trace_id") if result else None
        if not trace_id:
            return None

        diagnostics = response.diagnostics
        common = {
            "query": request.query,
            "filters": request.filters.model_dump(exclude_none=True),
            "top_k": request.top_k,
        }
        counts = {
            "filter": diagnostics.eligible_count,
            "dense": diagnostics.dense_candidate_count,
            "bm25": diagnostics.lexical_candidate_count,
            "fusion": diagnostics.fused_candidate_count,
            "reranker": diagnostics.reranker_candidate_count,
        }
        for stage in ("filter", "dense", "bm25", "fusion", "reranker"):
            if stage not in diagnostics.stage_latency_ms and not counts[stage]:
                continue
            self._post(
                "/trace/step",
                {
                    "trace_id": trace_id,
                    "name": f"retrieval.{stage}",
                    "input": common if stage in {"filter", "dense", "bm25"} else None,
                    "output": {"candidate_count": counts[stage]},
                    "latency_ms": diagnostics.stage_latency_ms.get(stage, 0.0),
                    "metadata": {"reranking_enabled": response.reranked},
                },
            )
        self._post(
            "/trace/step",
            {
                "trace_id": trace_id,
                "name": "retrieval.final",
                "output": {
                    "hit_count": len(response.hits),
                    "document_ids": [hit.document_id for hit in response.hits],
                },
                "latency_ms": response.latency_ms,
            },
        )
        if owns_trace:
            self._post(
                "/trace/end",
                {
                    "trace_id": trace_id,
                    "output": {
                        "hit_count": len(response.hits),
                        "reranked": response.reranked,
                    },
                },
            )
        return trace_id

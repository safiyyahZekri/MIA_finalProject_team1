from __future__ import annotations

from app.retrieval_benchmark import (
    RetrievalBenchmarkConfig,
    VARIANTS,
    format_comparison,
    run_ablation,
    run_retrieval_benchmark,
)


class FakeResponse:
    def __init__(self, body: dict):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeRetrievalClient:
    def __init__(self):
        self.requests: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict):
        self.requests.append((url, json))
        hits = [
            {
                "document_id": "wrong-document",
                "source_doc_uid": "wrong-document",
                "page": 1,
            },
            {
                "document_id": "gold-document",
                "source_doc_uid": "gold-document",
                "page": 2,
            },
        ]
        return FakeResponse({"hits": hits, "latency_ms": 1.0})


QUESTIONS = [
    {
        "question_id": "q1",
        "question_text": "What was revenue?",
        "source_doc_uid": "gold-document",
        "gold_evidence": [
            {"source_doc_uid": "gold-document", "source_page": 2}
        ],
    }
]


def test_retrieval_benchmark_is_unscoped_and_scores_ranked_hits() -> None:
    client = FakeRetrievalClient()
    report = run_retrieval_benchmark(
        RetrievalBenchmarkConfig(
            retrieval_url="http://retrieval",
            questions=QUESTIONS,
            variant=VARIANTS[2],
            save_results=False,
        ),
        client=client,
    )

    assert "document_id" not in client.requests[0][1]
    assert report["scope_to_gold_document"] is False
    assert report["summary"]["recall_at_1"] == 0.0
    assert report["summary"]["recall_at_5"] == 1.0
    assert report["summary"]["mrr"] == 0.5
    assert report["summary"]["mean_latency_ms"] is not None
    assert report["summary"]["median_latency_ms"] is not None
    assert report["summary"]["p95_latency_ms"] is not None


def test_ablation_runs_all_four_variants_and_computes_deltas() -> None:
    client = FakeRetrievalClient()
    comparison = run_ablation(
        "http://retrieval",
        QUESTIONS,
        client=client,
        save_results=False,
    )

    assert set(comparison["variants"]) == {
        "dense",
        "bm25",
        "hybrid",
        "hybrid_reranker",
    }
    assert len(client.requests) == 4
    assert all("document_id" not in payload for _, payload in client.requests)
    assert comparison["hybrid_reranker_minus_hybrid"]["mrr"] == 0.0
    table = format_comparison(comparison)
    assert "precision_at_10" in table
    assert "hit_rate_at_10" in table

from app.models import SearchDiagnostics, SearchRequest, SearchResponse
from app.tracing import RetrievalTracer


def test_tracing_disabled_is_noop() -> None:
    tracer = RetrievalTracer("")
    request = SearchRequest(query="revenue", trace_id="parent-trace")
    response = SearchResponse(
        query="revenue",
        mode="hybrid",
        candidate_count=0,
        reranked=False,
        latency_ms=1.0,
        hits=[],
    )
    assert tracer.record(request, response) == "parent-trace"


def test_tracer_emits_stage_metadata_without_document_content(monkeypatch) -> None:
    tracer = RetrievalTracer("http://eval")
    delivered = []

    def fake_post(path, payload):
        delivered.append((path, payload))
        return {"trace_id": "trace-1"}

    monkeypatch.setattr(tracer, "_post", fake_post)
    request = SearchRequest(query="revenue", rerank=True)
    response = SearchResponse(
        query="revenue",
        mode="hybrid",
        candidate_count=3,
        reranked=True,
        latency_ms=4.0,
        hits=[],
        diagnostics=SearchDiagnostics(
            eligible_count=3,
            dense_candidate_count=3,
            lexical_candidate_count=3,
            fused_candidate_count=3,
            reranker_candidate_count=3,
            stage_latency_ms={
                "dense": 1.0,
                "bm25": 0.5,
                "fusion": 0.1,
                "reranker": 2.0,
            },
        ),
    )

    assert tracer.record(request, response) == "trace-1"
    assert [payload.get("name") for _, payload in delivered if "name" in payload] == [
        "retrieval.search",
        "retrieval.filter",
        "retrieval.dense",
        "retrieval.bm25",
        "retrieval.fusion",
        "retrieval.reranker",
        "retrieval.final",
    ]
    serialized = str(delivered)
    assert "full document text" not in serialized

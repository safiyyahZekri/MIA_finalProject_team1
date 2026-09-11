"""Request-local evidence fusion across retrieval reformulation attempts."""

import pytest

from app import graph
from app.llm import Classification, EvidenceGrade, ExtractionDirect


def hit(document_id: str, rank: int = 1) -> dict:
    return {
        "document_id": document_id,
        "chunk_id": f"{document_id}-{rank}",
        "page": rank,
        "section": "Evidence",
        "text": f"Evidence from {document_id}",
        "score": 0.9 - rank / 100,
    }


class NeedsBothAttempts:
    last_usage = {}

    def classify(self, question):
        return Classification(question_type="text", search_query="first query")

    def grade(self, question, evidence):
        documents = {item["document_id"] for item in evidence}
        sufficient = {"earlier", "later"} <= documents
        return EvidenceGrade(
            sufficient=sufficient,
            confidence=0.9,
            reason="both passages are required" if sufficient else "missing one passage",
        )

    def reformulate(self, question, previous_query, attempt):
        return "second query"

    def extract(self, question, question_type, evidence):
        return ExtractionDirect(value="supported", evidence_indexes=[1, 2])


async def _run(monkeypatch, enabled: bool):
    monkeypatch.setattr(graph, "get_llm", NeedsBothAttempts)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(graph.settings, "HYBRID_RERANKING", False)
    monkeypatch.setattr(graph.settings, "RETRY_EVIDENCE_FUSION", enabled)
    monkeypatch.setattr(graph.settings, "RETRY_FUSION_MAX_HITS", 10)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 1)

    async def search(query, document_id=None):
        return [hit("earlier" if query == "first query" else "later")]

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_bm25", search)
    return await graph.build_graph().ainvoke({"question": "Need both", "trace": []})


@pytest.mark.asyncio
async def test_retry_fusion_keeps_evidence_that_a_later_attempt_would_replace(
    monkeypatch,
):
    state = await _run(monkeypatch, True)

    assert state["answer"]["answer_type"] == "direct"
    assert {item["document_id"] for item in state["evidence"]} == {
        "earlier",
        "later",
    }
    retrieves = [step for step in state["trace"] if step["step"] == "retrieve"]
    assert retrieves[0]["retry_evidence_fusion"] is False
    assert retrieves[1]["retry_evidence_fusion"] is True
    assert retrieves[1]["raw_attempt_hit_count"] == 1
    assert retrieves[1]["retry_evidence_attempts"] == 2


@pytest.mark.asyncio
async def test_disabled_retry_fusion_preserves_the_existing_replace_behavior(
    monkeypatch,
):
    state = await _run(monkeypatch, False)

    assert state["answer"]["answer_type"] == "insufficient_evidence"
    assert [item["document_id"] for item in state["evidence"]] == ["later"]
    assert "retry_evidence" not in state
    retrieves = [step for step in state["trace"] if step["step"] == "retrieve"]
    assert all(step["retry_evidence_fusion"] is False for step in retrieves)


@pytest.mark.asyncio
async def test_retry_fusion_respects_its_evidence_budget(monkeypatch):
    monkeypatch.setattr(graph, "get_llm", NeedsBothAttempts)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(graph.settings, "HYBRID_RERANKING", False)
    monkeypatch.setattr(graph.settings, "RETRY_EVIDENCE_FUSION", True)
    monkeypatch.setattr(graph.settings, "RETRY_FUSION_MAX_HITS", 6)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 1)

    async def search(query, document_id=None):
        prefix = "first" if query == "first query" else "second"
        return [hit(f"{prefix}-{rank}", rank) for rank in range(1, 6)]

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_bm25", search)
    state = await graph.build_graph().ainvoke({"question": "Need both", "trace": []})

    assert len(state["evidence"]) == 6
    assert {item["document_id"].split("-")[0] for item in state["evidence"]} == {
        "first",
        "second",
    }

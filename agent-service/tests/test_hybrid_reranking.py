import pytest

from app import graph
from app.llm import Classification, EvidenceGrade


class DirectLLM:
    def classify(self, question):
        return Classification(question_type="text", search_query="revenue")

    def grade(self, question, evidence):
        return EvidenceGrade(sufficient=False, confidence=0.9, reason="test complete")


def hit(document_id):
    return {
        "document_id": document_id,
        "chunk_id": document_id,
        "page": 1,
        "section": "Revenue",
        "text": "Revenue",
        "score": 0.8,
    }


@pytest.mark.asyncio
async def test_text_question_uses_only_service_owned_hybrid_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(graph, "get_llm", DirectLLM)
    monkeypatch.setattr(graph.settings, "HYBRID_RERANKING", True)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)

    async def hybrid(query, **kwargs):
        calls.append((query, kwargs))
        return [hit("hybrid")]

    async def forbidden(*args, **kwargs):
        raise AssertionError("agent-side dense/BM25 merge must not run")

    monkeypatch.setattr(graph, "search_hybrid", hybrid)
    monkeypatch.setattr(graph, "search_documents", forbidden)
    monkeypatch.setattr(graph, "search_bm25", forbidden)
    state = await graph.build_graph().ainvoke(
        {"question": "What was revenue?", "trace": []}
    )
    assert calls == [
        (
            "revenue",
            {"top_k": 5, "candidate_k": 30, "document_id": None, "rerank": True},
        )
    ]
    assert [e["document_id"] for e in state["evidence"]] == ["hybrid"]


@pytest.mark.asyncio
async def test_numerical_hybrid_keeps_table_view_and_scope(monkeypatch):
    llm = DirectLLM()
    llm.classify = lambda question: Classification(
        question_type="numerical", search_query="income"
    )
    monkeypatch.setattr(graph, "get_llm", lambda: llm)
    monkeypatch.setattr(graph.settings, "HYBRID_RERANKING", True)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)
    scopes = []

    async def hybrid(query, **kwargs):
        scopes.append(kwargs["document_id"])
        return [hit("hybrid")]

    async def tables(query, document_id=None):
        scopes.append(document_id)
        return [hit("table")]

    async def filtered(metadata):
        return []

    monkeypatch.setattr(graph, "search_hybrid", hybrid)
    monkeypatch.setattr(graph, "search_tables", tables)
    monkeypatch.setattr(graph, "filter_documents", filtered)
    state = await graph.build_graph().ainvoke(
        {"question": "Difference?", "document_id": "scope", "trace": []}
    )
    assert scopes == ["scope", "scope"]
    assert {e["document_id"] for e in state["evidence"]} == {"hybrid", "table"}

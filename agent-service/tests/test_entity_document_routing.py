import pytest
from pydantic import ValidationError

from app import graph
from app import llm as llm_module
from app.entity_routing import (
    choose_entity_document,
    hit_mentions_entity,
    scoped_entity_query,
)
from app.llm import Classification, EvidenceGrade


def hit(document_id, text, score=0.8, chunk="1"):
    return {
        "document_id": document_id,
        "chunk_id": f"{document_id}-{chunk}",
        "page": 1,
        "section": "Document",
        "text": text,
        "content": text,
        "score": score,
    }


def test_entity_match_ignores_corporate_suffix_but_rejects_wrong_company():
    candidate = hit("jabil", "Jabil Circuit, Inc. annual report")
    assert hit_mentions_entity(candidate, "Jabil Inc.")
    assert not hit_mentions_entity(candidate, "Microsoft Corporation")


def test_document_selection_requires_verified_entity_text_and_respects_exclusions():
    wrong = hit("wrong", "Another company revenue", 0.99)
    right = hit("right", "Jabil Inc. annual report", 0.6)
    alternate = hit("alternate", "Jabil Inc. supplementary filing", 0.5)
    assert choose_entity_document("Jabil Inc.", [[wrong, right], [right]]) == "right"
    assert (
        choose_entity_document("Jabil Inc.", [[right, alternate]], {"right"})
        == "alternate"
    )


def test_scoped_query_removes_other_entity_and_keeps_selected_one():
    query = "Compare KEMET net sales with CTS net sales in 2018"
    scoped = scoped_entity_query(query, "KEMET", ["KEMET", "CTS"])
    assert "KEMET" in scoped
    assert "CTS" not in scoped
    assert "net sales" in scoped


def test_classification_entities_are_bounded_clean_and_backward_compatible():
    assert Classification(question_type="text", search_query="q").entities == []
    classified = Classification(
        question_type="table",
        search_query="q",
        entities=[" Jabil Inc. ", "jabil inc.", "CTS"],
    )
    assert classified.entities == ["Jabil Inc.", "CTS"]
    with pytest.raises(ValidationError):
        Classification(
            question_type="text",
            search_query="q",
            entities=["one", "two", "three", "four"],
        )


def test_entity_prompt_and_evidence_label_are_feature_scoped(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "ENTITY_DOCUMENT_ROUTING", False)
    assert llm_module.entity_classification_instruction() == ""
    assert llm_module.evidence_source_label({"document_id": "d", "page": 1}) == "d p1"

    monkeypatch.setattr(llm_module.settings, "ENTITY_DOCUMENT_ROUTING", True)
    assert "explicitly written" in llm_module.entity_classification_instruction()
    assert (
        llm_module.evidence_source_label(
            {"document_id": "d", "page": 1, "routed_entity": "Jabil Inc."}
        )
        == "d p1 verified_entity=Jabil Inc."
    )


class EntityLLM:
    def classify(self, question):
        return Classification(
            question_type="table",
            search_query="deferred tax assets 2019",
            entities=["Jabil Inc."],
        )

    def grade(self, question, evidence):
        return EvidenceGrade(sufficient=False, confidence=0.9, reason="test complete")


@pytest.mark.asyncio
async def test_entity_route_scopes_metric_search_and_preserves_global_results(monkeypatch):
    calls = []
    monkeypatch.setattr(graph, "get_llm", EntityLLM)
    monkeypatch.setattr(graph.settings, "ENTITY_DOCUMENT_ROUTING", True)
    monkeypatch.setattr(graph.settings, "ENTITY_ROUTING_CANDIDATES", 5)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(graph.settings, "HYBRID_RERANKING", False)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)

    async def dense(query, top_k=None, document_id=None):
        calls.append(("dense", query, top_k, document_id))
        if query == "Jabil Inc.":
            return [hit("jabil", "Jabil Circuit Inc annual report")]
        if document_id == "jabil":
            return [hit("jabil", "Deferred tax assets | 2019 | 50,000", chunk="fact")]
        return [hit("wrong", "Deferred tax assets | 2019 | 50,000")]

    async def lexical(query, top_k=None, document_id=None):
        calls.append(("bm25", query, top_k, document_id))
        return [hit("jabil", "Jabil Inc. annual report")]

    async def tables(query, top_k=None, document_id=None):
        calls.append(("tables", query, top_k, document_id))
        if document_id == "jabil":
            return [hit("jabil", "Deferred tax assets | 2019 | 50,000", chunk="table")]
        return [hit("wrong", "Deferred tax assets | 2019 | 50,000")]

    monkeypatch.setattr(graph, "search_documents", dense)
    monkeypatch.setattr(graph, "search_bm25", lexical)
    monkeypatch.setattr(graph, "search_tables", tables)

    state = await graph.build_graph().ainvoke(
        {"question": "For Jabil Inc., which deferred tax assets exceeded 50,000?", "trace": []}
    )

    assert state["entity_document_ids"] == {"Jabil Inc.": "jabil"}
    assert any(call[3] == "jabil" for call in calls)
    assert {item["document_id"] for item in state["evidence"]} == {"jabil", "wrong"}
    assert any(item.get("routed_entity") == "Jabil Inc." for item in state["evidence"])
    route = next(item for item in state["trace"] if item["step"] == "route_entity")
    assert route["document_id"] == "jabil"


@pytest.mark.asyncio
async def test_explicit_document_scope_skips_entity_resolution(monkeypatch):
    monkeypatch.setattr(graph, "get_llm", EntityLLM)
    monkeypatch.setattr(graph.settings, "ENTITY_DOCUMENT_ROUTING", True)
    monkeypatch.setattr(graph.settings, "QUERY_DECOMPOSITION", False)
    monkeypatch.setattr(graph.settings, "HYBRID_RERANKING", False)
    monkeypatch.setattr(graph.settings, "MAX_RETRIES", 0)
    scopes = []

    async def search(query, top_k=None, document_id=None):
        scopes.append(document_id)
        return [hit(document_id, "scoped evidence")]

    async def filtered(metadata):
        return []

    monkeypatch.setattr(graph, "search_documents", search)
    monkeypatch.setattr(graph, "search_tables", search)
    monkeypatch.setattr(graph, "filter_documents", filtered)
    state = await graph.build_graph().ainvoke(
        {"question": "Jabil revenue?", "document_id": "explicit", "trace": []}
    )

    assert scopes == ["explicit", "explicit"]
    assert state["entity_document_ids"] == {}
    assert not any(item["step"] == "route_entity" for item in state["trace"])
